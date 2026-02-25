from datasets import load_dataset
import os
from llm_call import *
from tqdm.asyncio import tqdm

os.environ["HF_TOKEN"] = "..."
os.environ['OPENAI_API_KEY'] = "..."

# load data
ds = load_dataset("...")
labels = load_dataset("...", "class_descriptions")
label_descriptions = [str(l['label'])+": "+l['description'] for l in labels['class_descriptions']]

# load ScaDS.AI API key from file
api_key = ""
path_to_key = os.path.join(os.path.expanduser("~"), ".scadsai-api-key")
if os.path.exists(path_to_key):
    with open(path_to_key) as keyfile:
        api_key = keyfile.readline().strip()
if len(api_key) < 1:
    print("Error: The key file '.scadsai-api-key' did not contain any key. Please make sure the file exists and contains only your API key.")
    exit(1)

# LLM specifics
#llm_base_url = "https://llm.scads.ai/v1"
llm_base_url = "https://api.openai.com/v1"
#llm_name = "meta-llama/Llama-3.1-8B-Instruct"
#llm_name = "meta-llama/Llama-3.3-70B-Instruct"
#llm_name = "meta-llama/Llama-4-Scout-17B-16E-Instruct"
#llm_name = "deepseek-ai/DeepSeek-R1"
#llm_name = "openai/gpt-oss-120b"
#llm_name = "o3-2025-04-16"
llm_name = "gpt-5-2025-08-07"
api_key = os.environ['OPENAI_API_KEY']
client = LLMClient(api_key, llm_base_url)

# ---------------------------
# Prompt Builder
# ---------------------------

def build_judgment_prompt(research_idea: dict, related_works: list, class_descriptions: list) -> str:
    # Join class_descriptions list into a bullet point string, each on a new line with indentation
    class_desc_str = "\n       - " + "\n       - ".join(class_descriptions)

    return f"""
    You are an expert in machine learning research evaluation. You will be given two inputs:

    1. A research idea with objective, problem statement, and solution approach.
    2. A list of related works, each with a title and abstract.

    Your task is to **assess the novelty of the research idea** compared to the related works.

    ### Instructions:
    - Analyze the research idea and summarize its key contributions.
    - Compare it with the related works to identify overlaps and differences.
    - Specifically, assess whether the idea introduces **significant new aspects** not present in existing work, or if it is largely a variation on known approaches.
    - Provide your output as a **JSON object only**, with:
      - `"reasoning"`: a short paragraph (2–4 sentences) explaining the reasoning behind the novelty score.
      - `"novelty_score"`: an integer between 1-5 where:{class_desc_str}

    ### Inputs:

    **Research Idea:**
    {research_idea}

    **Related Works:**
    {related_works}

    ### Output Format:
    ```json
    {{
      "reasoning": "<short explanation>",
      "novelty_score": <1|2|3|4|5>
    }}
    """

# parallel llm prediction calls that keeps the order of results
async def predict_novelty_parallel(client: LLMClient, llm_name: str, research_ideas: list[dict], related_works: list[list[dict]], class_descriptions: list[str], system_prompt: str, temperature: None|int=0, n_parallel_llm_calls: int = 10, return_json=True):
    """
    Run multiple LLM calls concurrently (limited by n_parallel_llm_calls).
    Each call evaluates one research idea against its related works and generates a numeric novelty score together with a textual justification/reasoning of the novelty score.
    Returns results in the same order as the inputs.
    """
    semaphore = asyncio.Semaphore(n_parallel_llm_calls)
    progress_bar = tqdm(total=len(research_ideas), desc=f"{llm_name} Predictions")

    results = [None] * len(research_ideas)  # preserve input order

    async def sem_task(idx):
        async with semaphore:
            rel_works = [
                {k: p[k] for k in ("title", "abstract")}
                for p in related_works[idx]
            ]
            user_prompt = build_judgment_prompt(
                research_idea=research_ideas[idx],
                related_works=rel_works,
                class_descriptions=class_descriptions,
            )
            response = await call_llm_with_retries_async(
                client=client,
                llm_name=llm_name,
                user_prompt=user_prompt,
                temperature=temperature,
                system_prompt=system_prompt,
                return_json=return_json,
                max_retries=1000,
                retry_backoff=1,
            )
            results[idx] = response.choices[0].message.content
            progress_bar.update(1)  # update after each finished call
            return results[idx]

    # Launch tasks for all indices
    tasks = [asyncio.create_task(sem_task(idx)) for idx in range(len(research_ideas))]

    await asyncio.gather(*tasks)
    progress_bar.close()
    return results

predictions = asyncio.run(predict_novelty_parallel(
    client=client,
    llm_name=llm_name,
    research_ideas=ds['test']['research_idea'],
    related_works=[[{k: p[k] for k in ('title', 'abstract')} for p in works] for works in ds['test']['related_works']],
    temperature=1,
    class_descriptions=label_descriptions,
    system_prompt="You are an expert researcher experienced in judging the novelty of a research idea.",
    return_json=True,
    n_parallel_llm_calls=50
))
print("Number of research ideas to evaluate:", ds['test'].num_rows)
print("Number of novelty predictions:", len(predictions))

# Save to disk
save_path = '../data/novelty_predictions/novelty_predictions-'+llm_name.split("/")[-1]+'.json'
with open(save_path, "w", encoding="utf-8") as f:
    json.dump(predictions, f, indent=4)

print(f"Predictions saved to {save_path}")