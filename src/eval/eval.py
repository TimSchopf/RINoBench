import math
import numpy as np
from opentelemetry.propagate import extract
from scipy.stats import pearsonr
from sklearn.metrics import f1_score, mean_absolute_error
from tqdm.asyncio import tqdm
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from llm_call import *
from pydantic_models import *
from prompts import *

# computes the coverage of aspects from aspects in the LLM reasoning that are also contained in the gold reasoning
def compute_supported_fraction(comparison_output) -> float:
    """
    Compute the fraction of 'supported_in_expected=True' aspects in the actual aspects,
    divided by the total number of aspects in the expected aspects.

    If the number of supported aspects exceeds the number of expected aspects,
    return 1.0.

    If there are no expected aspects (division by zero), return 0.

    Args:
        comparison_output: Either a NoveltyComparison or RelatedWorkComparison
                           Pydantic model (must have predicted_aspects and gold_aspects).

    Returns:
        float: Fraction between 0.0 and 1.0, or 0.0 if division by zero.
    """
    predicted_aspects = comparison_output.predicted_aspects
    gold_aspects = comparison_output.gold_aspects

    if not gold_aspects:
        return 0.0

    supported_count = sum(1 for aspect in predicted_aspects if aspect.supported_in_gold)
    total_expected = len(gold_aspects)

    fraction = supported_count / total_expected

    return min(fraction, 1.0)

# --- Async wrappers for synchronous functions ---

async def g_eval_alignment_async(predicted_reasoning, gold_reasoning, eval_model):
    """
    Wrap synchronous GEval alignment in an executor to make it async.
    """
    loop = asyncio.get_running_loop()
    func = partial(g_eval_alignment, predicted_reasoning, gold_reasoning, eval_model)
    return await loop.run_in_executor(None, func)


def g_eval_alignment(predicted_output, gold_output, eval_model):
    alignment_metric = GEval(
        name="Alignment",
        criteria=(
            "Determine whether the reasoning in the actual output is generally aligned with the expected output. "
            "This means it follows the same line of argumentation, with the same arguments and comes to the same conclusion."
        ),
        evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
        model=eval_model,
        verbose_mode=False,
    )

    test_case = LLMTestCase(
        input="",
        actual_output=predicted_output,
        expected_output=gold_output
    )


    alignment_metric.measure(test_case)

    return alignment_metric


# --- Full async compute_eval_metrics function to compute all metrics for a single example ---

async def compute_eval_metrics(client: LLMClient, llm_name: str, research_idea: dict, related_works: list[dict], predicted_reasoning: str, gold_reasoning: str, predicted_score: int, gold_score: int, temperature: float | None = 0, max_retries: int = 3, retry_backoff: float = 1.0) -> dict:

    # Step 1: Prepare evaluation prompts
    related_work_prompt = build_evaluation_prompt_existing_works_aspects(
        predicted_reasoning=predicted_reasoning,
        gold_reasoning=gold_reasoning
    )
    novelty_prompt = build_evaluation_prompt_novelty_aspects(
        predicted_reasoning=predicted_reasoning,
        gold_reasoning=gold_reasoning
    )

    # Step 2: Start alignment computation in background (runs concurrently)
    alignment_task = asyncio.create_task(
        g_eval_alignment_async(
            predicted_reasoning=predicted_reasoning,
            gold_reasoning=gold_reasoning,
            eval_model=CustomLLM(client=client.get_base_client(), model_name=llm_name, temperature=temperature)
        )
    )

    # Step 3: Run initial evaluations concurrently
    extracted_validated_known_aspects, extracted_validated_novelty_aspects = await asyncio.gather(
        call_llm_with_retries_async(client=client, llm_name=llm_name, schema=RelatedWorkComparison, user_prompt=related_work_prompt, temperature=temperature, max_retries=max_retries, retry_backoff=retry_backoff),
        call_llm_with_retries_async(client=client, llm_name=llm_name, schema=NoveltyComparison, user_prompt=novelty_prompt, temperature=temperature, max_retries=max_retries, retry_backoff=retry_backoff)
    )

    # Step 4: Prepare fact-check claims
    additional_known_aspects = [aspect.aspect for aspect in extracted_validated_known_aspects.predicted_aspects if not aspect.supported_in_gold]
    additional_novelty_aspects = [aspect.aspect for aspect in extracted_validated_novelty_aspects.predicted_aspects if not aspect.supported_in_gold]

    # Step 5: Run fact-checks concurrently (must wait for evaluation results)
    fact_checked_additional_known_aspects, fact_checked_additional_novelty_aspects = await asyncio.gather(
        call_llm_with_retries_async(client=client, llm_name=llm_name, schema=FactCheckResult,
                                    user_prompt=build_fact_check_prompt(research_idea, related_works, additional_known_aspects), temperature=temperature, max_retries=max_retries, retry_backoff=retry_backoff),
        call_llm_with_retries_async(client=client, llm_name=llm_name, schema=FactCheckResult,
                                    user_prompt=build_fact_check_prompt(research_idea, related_works, additional_novelty_aspects), temperature=temperature, max_retries=max_retries, retry_backoff=retry_backoff)
    )
    # Step 6: Compute coverage metrics
    known_aspects_recall = compute_supported_fraction(extracted_validated_known_aspects)
    novelty_aspects_recall = compute_supported_fraction(extracted_validated_novelty_aspects)

    # Step 7: Compute metrics for fact-checks
    additional_known_aspects_ratio = sum(c.supported for c in fact_checked_additional_known_aspects.checked_aspects) / max(len(extracted_validated_known_aspects.gold_aspects), 1)
    additional_novelty_aspects_ratio = sum(c.supported for c in fact_checked_additional_novelty_aspects.checked_aspects) / max(len(extracted_validated_novelty_aspects.gold_aspects), 1)

    denominator1 = len(extracted_validated_known_aspects.predicted_aspects)
    denominator2 = len(extracted_validated_novelty_aspects.predicted_aspects)

    known_aspects_hallucination_rate = ([c.supported for c in fact_checked_additional_known_aspects.checked_aspects].count(False) / denominator1) if denominator1 != 0 else 0
    novelty_aspects_hallucination_rate = ([c.supported for c in fact_checked_additional_novelty_aspects.checked_aspects].count(False) / denominator2) if denominator2 != 0 else 0

    # Step 8: Await alignment result
    alignment_metric = await alignment_task

    # Step 9: Return metrics
    return {
        "predicted_novelty_score": predicted_score,
        "gold_novelty_score": gold_score,
        "known_aspects_recall": known_aspects_recall,
        "novelty_aspects_recall": novelty_aspects_recall,
        "additional_known_aspects_ratio": additional_known_aspects_ratio,
        "additional_novelty_aspects_ratio": additional_novelty_aspects_ratio,
        "known_aspects_hallucination_rate": known_aspects_hallucination_rate,
        "novelty_aspects_hallucination_rate": novelty_aspects_hallucination_rate,
        "alignment_score": alignment_metric.score,
        "alignment_reason": alignment_metric.reason
    }

async def eval_bench(client: LLMClient, llm_name: str, research_ideas: list[dict], related_works: list[list[dict]], predicted_reasonings: list[str], gold_reasonings: list[str], predicted_scores: list[int], gold_scores: list[int], n_parallel: int = 5, temperature: float | None = 0, max_retries: int = 3, retry_backoff: float = 1.0):
    """
    Run compute_eval_metrics() concurrently across multiple examples, limited to `n_parallel` simultaneous tasks.
    Shows a tqdm progress bar as tasks complete.
    Keeps result order consistent with inputs.
    """
    semaphore = asyncio.Semaphore(n_parallel)
    total = len(research_ideas)

    async def sem_task(idx):
        async with semaphore:
            result = await compute_eval_metrics(
                client=client,
                llm_name=llm_name,
                research_idea=research_ideas[idx],
                related_works=related_works[idx],
                predicted_reasoning=predicted_reasonings[idx],
                gold_reasoning=gold_reasonings[idx],
                predicted_score=predicted_scores[idx],
                gold_score=gold_scores[idx],
                temperature=temperature,
                max_retries=max_retries,
                retry_backoff=retry_backoff
            )
            return idx, result

    tasks = [asyncio.create_task(sem_task(i)) for i in range(total)]
    results = [None] * total  # preserve order

    # tqdm progress bar that updates as tasks finish
    with tqdm(total=total, desc="Computing evaluation metrics") as pbar:
        for coro in asyncio.as_completed(tasks):
            idx, res = await coro
            results[idx] = res
            pbar.update(1)

    # compute benchmark metrics over the entire dataset
    metrics = {}
    y_true = [r['gold_novelty_score'] for r in results]
    y_pred = [r['predicted_novelty_score'] for r in results]

    # compute F1 scores
    labels = np.unique(y_true)
    f1_scores_labels = f1_score(y_true, y_pred, average=None, labels=labels)
    f1_score_macro = f1_score(y_true, y_pred, average='macro')
    f1_score_micro = f1_score(y_true, y_pred, average='micro')
    f1_dict = {int(label): float(score) for label, score in zip(labels, f1_scores_labels)}
    f1_dict['macro'] = float(f1_score_macro)
    #f1_dict['micro'] = float(f1_score_micro)
    metrics['f1_scores'] = f1_dict

    # compute MAE
    mae = mean_absolute_error(y_true, y_pred)
    metrics['mean_absolute_error'] = float(mae)

    # compute Pearson correlation
    #pearson_corr, p_value = pearsonr(y_true, y_pred)
    #metrics['correlation'] = {'Pearson correlation': float(pearson_corr), 'P-value': float(p_value)}

    # average alignment score
    mean_alignment_score = np.mean([r['alignment_score'] for r in results])
    metrics['mean_alignment'] = float(mean_alignment_score)

    # average known aspects recall
    valid_values = [v for v in [r['known_aspects_recall'] for r in results] if not math.isnan(v)]
    mean_known_aspects_recall = np.mean(valid_values) if valid_values else 0
    metrics['mean_known_aspects_recall'] = float(mean_known_aspects_recall)

    # average novelty aspects recall
    valid_values = [v for v in [r['novelty_aspects_recall'] for r in results] if not math.isnan(v)]
    mean_novelty_aspects_recall = np.mean(valid_values) if valid_values else 0
    metrics['mean_novelty_aspects_recall'] = float(mean_novelty_aspects_recall)

    # average additional known aspects ratio
    valid_values = [v for v in [r['additional_known_aspects_ratio'] for r in results] if not math.isnan(v)]
    mean_additional_known_aspects_ratio = np.mean(valid_values) if valid_values else 0
    metrics['mean_additional_known_aspects_ratio'] = float(mean_additional_known_aspects_ratio)

    # average additional novelty aspects ratio
    valid_values = [v for v in [r['additional_novelty_aspects_ratio'] for r in results] if not math.isnan(v)]
    mean_additional_novelty_aspects_ratio = np.mean(valid_values) if valid_values else 0
    metrics['mean_additional_novelty_aspects_ratio'] = float(mean_additional_novelty_aspects_ratio)

    # average known aspects hallucination rate
    valid_values = [v for v in [r['known_aspects_hallucination_rate'] for r in results] if not math.isnan(v)]
    mean_known_aspects_hallucination_rate = np.mean(valid_values) if valid_values else 0
    metrics['mean_known_aspects_hallucination_rate'] = float(mean_known_aspects_hallucination_rate)

    # average novelty aspects hallucination rate
    valid_values = [v for v in [r['novelty_aspects_hallucination_rate'] for r in results] if not math.isnan(v)]
    mean_novelty_aspects_hallucination_rate = np.mean(valid_values) if valid_values else 0
    metrics['mean_novelty_aspects_hallucination_rate'] = float(mean_novelty_aspects_hallucination_rate)

    return metrics