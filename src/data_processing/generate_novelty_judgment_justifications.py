#!/usr/bin/env python
# coding: utf-8

import json
import os
import instructor
import time
from tqdm import tqdm
from openai import OpenAI
from pydantic import BaseModel, Field
from typing import Optional, List
from concurrent.futures import ThreadPoolExecutor, as_completed

# load data from disk
with open('../data/json/ICLR.cc-22-23-Conferences.json', 'r') as f:
    merged_papers: dict = json.load(f)


# load ScaDS.AI API key from file
api_key = ""
path_to_key = os.path.join(os.path.expanduser("~"), ".scadsai-api-key")
if os.path.exists(path_to_key):
    with open(path_to_key) as keyfile:
        api_key = keyfile.readline().strip()
if len(api_key) < 1:
    print("Error: The key file '.scadsai-api-key' did not contain any key. Please make sure the file exists and contains only your API key.")
    exit(1)



# Initialize the OpenAI client with instructor patching
client = instructor.from_openai(OpenAI(base_url="https://llm.scads.ai/v1",api_key=api_key), mode=instructor.Mode.JSON)
llm = "openai/gpt-oss-120b"



# ---------------------------
# Structured model for novelty justification
# ---------------------------
class NoveltyComments(BaseModel):
    """
    Structured representation of novelty-related judgments from reviews/meta-reviews.
    """

    novelty_comments: List[str] = Field(
        default_factory=list,
        description=(
            "A list of reviewer statements discussing the novelty or technical contributions of the paper. "
            "Each entry should be taken directly from the review/meta-review and reflect explicit commentary. "
            "Do not fabricate any statements or include numerical novelty scores."
        )
    )


# ---------------------------
# Structured model for novelty justification
# ---------------------------
class NoveltyJustification(BaseModel):
    """
    Structured representation of novelty-related judgments from reviews/meta-reviews.
    """

    #novelty_comments: List[str] = Field(
    #    default_factory=list,
    #    description=(
    #        "A list of reviewer statements discussing the novelty or technical contributions of the paper. "
    #        "Each entry should be taken directly from the review/meta-review and reflect explicit commentary. "
    #        "Do not fabricate any statements or include numerical novelty scores."
    #    )
    #)

    novelty_score_justification: str = Field(
        default="",
        description=(
            "A reasoning/justification of a given novelty score, based only on the novelty comments extracted before. "
            "This should explain why a sepcific novelty rating was given."
            "Do not invent judgments. If no novelty judgments are present, return 'No novelty judgment found.'"
            "Write the novelty reasoning/justification as if YOU were the one giving the novelty reasoning/justification and not as if you would synthesize information from reviews. Hence, do NOT inlcude things like 'the reviewers state...', 'the reviewers reagard it less novel', 'is described as novel', 'is viewed as less novel', 'is viewed as incremental', 'is considered as novel', 'is regarded as limited novelty' or similar phrases that indicate the reasoning is from different people. Instead phrase it as direct statement like 'the idea is novel', 'it has minimal novelty' etc."
            "Hence, do NOT include things like 'is highlighted as novel', 'it is regarded as.., the reviewers state..., the reviewers reagrd it.., the method/novelty is described as...., the approach is viewed as..., the contribution is viewed as...' etc."
            "Also do not write it as review style and do not mentaion any authors or papers. Rather concider your resulting text as justification of the judgment of a research idea rather than a paper review."
            "Do NOT summarize/repeat what the idea is about or the contributions again or state what is introduced but only focus on the novelty aspect shortly."
        )
    )


# ---------------------------
# One-shot example
# ---------------------------
example_paper = """

Research idea:

Problem statement: The research problem is to characterize the class of graphs whose edge intersections can be represented as projections of convex polytopes in fixed-dimensional space, and to determine the minimum dimension required for such representations as a function of the graph size.
Objective: The aim is to establish a structural theory for polytope-representable graphs, to identify minimal forbidden subgraphs for low-dimensional cases, and to derive general bounds for the polytope dimension needed to encode arbitrary graphs.
Solution approach: The approach involves framing graph representation as convex set intersection problems, employing tools from geometric combinatorics and extremal graph theory, classifying all graphs representable in two dimensions via forbidden induced subgraphs, extending to higher dimensions through duality and separation arguments, and establishing asymptotic lower and upper bounds on required polytope dimensions using results from communication complexity and metric embedding theory.

The novelty score for this idea is: 3

Reviews:

- This paper proposes a model to classify unknown classes in target domain using adversarial learning and knowledge graphs.
- The idea of using a knowledge graph for unknown classes is novel.
- The domain adaptation method is standard and not highly innovative.
"""

example_output = NoveltyJustification(
    novelty_score_justification=(
        "The approach introduces novelty by applying knowledge graphs to unknown class recognition, "
        "but the reliance on standard domain adaptation techniques reduces the overall novelty."
    )
)


# -----------------------------------------
# Function to process one paper with retries
# -----------------------------------------
def process_paper_with_retries(
    client,
    paper,
    idea,
    example_paper,
    example_output,
    max_retries=3,
    retry_backoff=1.0
):
    def call_llm():
        # Collect all narrative review/meta-review text, excluding numeric novelty fields
        EXCLUDE_FIELDS = {
            "technical_novelty_and_significance",
            "empirical_novelty_and_significance",
            "mean_novelty",
            "novelty_bin",
            "flag_for_ethics_review",
            "recommendation",
            "confidence",
            "correctness",
            "decision"
        }

        texts_to_check = []
        for reply in paper.get("replies", []):
            content = reply.get("content", {})
            for field, text in content.items():
                if field not in EXCLUDE_FIELDS and isinstance(text, str):
                    texts_to_check.append(text)

        # Task prompt
        task_prompt = f"""
            You are analyzing novelty judgments for a research idea. Your task is to write a single, coherent reasoning/justification for the novelty score given below. 

            Rules:
            1. Only use the novelty-related statements explicitly present in the provided reviews. Do NOT invent or infer any additional reasoning.
            2. The justification must always include the novelty claim and the reasoning why the idea is novel or not. 
            3. Write as if you are directly judging the novelty yourself — never phrase it as if others (e.g., reviewers) made the judgments. This also means that you can NOT use phrases like "is described as..." or "is presented as..."
            4. Do not restate or summarize the contributions or describe what the idea does. Focus only on novelty reasoning.
            5. Do NOT mention authors, papers, reviewers, comments, or results. This is not a paper review, but a justification of novelty of an idea. Therefore instead of "this work" use terms like "this idea" or "this approach".
            6. Never state the numerical novelty score itself in the justification. Instead, align the tone and strength of your judgment with the score.
            7. Avoid over-enthusiastic or overly negative wording. Keep the tone balanced, as if from a neutral researcher.
            8. If no explicit novelty judgments are present in the texts, return exactly: "No novelty judgment found."

            Novelty score scale:
            1 = not novel: all aspects already exist in prior work
            2 = marginally novel: minor variation of existing work
            3 = somewhat novel: aspects exist in prior work, might combine known approaches in new ways, apply them to new contexts, or propose incremental updates
            4 = novel: introducing new aspects not present in existing work
            5 = highly innovative and novel: not in existing work and potentially encourages new thinking or opens new research directions
            """ 

        paper_input = f"""
        Research idea:

        Problem statement: {idea['problem_statement']}
        Objective: {idea['objective']}
        Solution approach: {idea['solution_approach']}

        The novelty score for this idea is: {paper.get("novelty_bin")}

        Reviews:

        {texts_to_check}
        """

        # Messages for LLM
        messages = [
            {"role": "system", "content": "You are a human researcher and expert in judging and reasoning about novelty of research ideas."},

            # One-shot example
            {"role": "user", "content": task_prompt + "\n\nPaper:\n" + example_paper},
            {"role": "assistant", "content": example_output.model_dump_json()},

            # Actual paper
            {"role": "user", "content": "\n\nPaper:\n" + paper_input}
        ]
        #print(messages)

        # Actual API call
        return client.chat.completions.create(
            model=llm,
            response_model=NoveltyJustification,
            messages=messages,
            #temperature=0,
            extra_body={"disable_fallbacks": True}
        )

    # Retry loop
    for attempt in range(max_retries):
        try:
            return call_llm()
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(retry_backoff * (2 ** attempt))  # exponential backoff
            else:
                return {"error": f"Failed after {max_retries} attempts: {str(e)}"}

# -----------------------------------------
# Parallel execution
# -----------------------------------------
def fetch_novelty_summaries_parallel(
    client,
    papers,
    ideas,
    example_paper,
    example_output,
    max_parallel=5,
    max_retries=3,
    retry_backoff=1.0,
    sleep_between_calls=0.0
):
    results = [None] * len(papers)

    with ThreadPoolExecutor(max_workers=max_parallel) as executor:
        future_to_index = {
            executor.submit(
                process_paper_with_retries,
                client,
                papers[i],
                ideas[i],
                example_paper,
                example_output,
                max_retries,
                retry_backoff
            ): i for i in range(len(papers))
        }

        # tqdm progress bar over futures
        for future in tqdm(as_completed(future_to_index), total=len(future_to_index), desc="Processing papers"):
            index = future_to_index[future]
            try:
                result = future.result()
                results[index] = result.model_dump() if not isinstance(result, dict) else result
            except Exception as exc:
                results[index] = {"error": f"Unexpected exception: {exc}"}

            if sleep_between_calls > 0:
                time.sleep(sleep_between_calls)

    return results

# -----------------------------------------
# Usage
# -----------------------------------------
reviews = [p['reviews'] for p in merged_papers]
research_ideas = [p['research_idea'] for p in merged_papers]

extracted_novelty_summaries = fetch_novelty_summaries_parallel(
    client,
    reviews,
    research_ideas,
    example_paper,
    example_output,
    max_parallel=50,
    max_retries=300,
    retry_backoff=1.0
)

# add novelty justifications to list of merged papers
for i, p in enumerate(merged_papers):
    p['novelty_score_justification'] = extracted_novelty_summaries[i]['novelty_score_justification']

# Save as a JSON files
with open("../data/json/ICLR.cc-22-23-Conferences.json", 'w') as f:
    json.dump(merged_papers, f, indent=4)  # indent makes it easy to read

