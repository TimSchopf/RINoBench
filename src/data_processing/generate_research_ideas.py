import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm 
import instructor
from openai import OpenAI
from pydantic import BaseModel, Field
from typing import Optional, List

# load data from disk
with open('../data/json/ICLR.cc-2022-Conference_processed_papers.json', 'r') as f:
    iclr22_papers: dict = json.load(f)

with open('../data/json/ICLR.cc-2023-Conference_processed_papers.json', 'r') as f:
    iclr23_papers: dict = json.load(f)

with open('../data/json/ICLR.cc-2022-Conference.json', 'r') as f:
    iclr22_reviews: dict = json.load(f)

with open('../data/json/ICLR.cc-2023-Conference.json', 'r') as f:
    iclr23_reviews: dict = json.load(f)


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


class ResearchIdea(BaseModel):
    """
    Structured representation of a research idea extracted from the title, abstract, and reviewer summaries of a research paper.
    Each field is a single string that may contain multiple sentences. 
    Each string should be written such that if all fields are concatenated, 
    they form a coherent research-proposal-style paragraph. 
    """

    problem_statement: str= Field(
        default="",
        description=(
            "A detailed description of the core research problem(s) or question(s) being addressed. "
            "The description should include all distinct aspect of the problem and generate multiple sentences if necessary. "
            "This should be framed as a pre-research issue, not as something already solved or achieved. "
            "Avoid including findings, results, or conclusions."
        )
    )

    objective: str = Field(
        default="",
        description=(
            "A detailed description stating the aim(s) or intended accomplishments of the research. "
            "The description should include all specify everything that the researchers intend to achieve through the study nd generate multiple sentences if necessary. "
            "Do not include achieved results or performance metrics; focus purely on the intended purpose."
        )
    )

    solution_approach: str = Field(
        default="",
        description=(
            "A detaield description stating the proposed solution approach(es) or method(s) to address the problem and achieve the objective(s). "
            "The description should include all technique(s), framework(s), or strategy/ies the researchers intend to employ. "
            "Avoid describing results, proofs, or outcomes of applying the solution. "
            "This should reflect the plan or conceptual approach rather than post-hoc findings. Therefore, rather descibe 'proposed' approaches and NOT already 'introduced' ones."
        )
    )


# In[6]:


# Task description
research_idea_task_prompt = """You are given the title, abstract, and reviewer summaries of a research paper.

Your task is hierarchical:

1. Identify **research idea facets** present in the all given inputs.  
   - Search in all given inputs. 
   - Exclude results or outcomes that could only exist after performing the research.  
   - Exclude motivation and background discussion.

2. The facets are:
   - "Problem Statement": The research problem or question that the research aims to address.
   - "Objective": The intended goal or purpose of the research.
   - "Solution Approach": The proposed method or approach to solve the problem.

3. For each facet, generate a **single string** that combines all relevant points extracted from all the input, neutral in tone (no "we" or "our"). Use multiple sentences if necessary to avoid leaving any important information out.
   - Remains fully factual and grounded in the paper’s content.
   - Each string should be descriptive and informative.  
   - The tone should be neutral and academic, but smooth and readable. 
   - Do NOT use delimiters (;) etc. Instead generate fully coherent and descriptive sentences.
   - Split into multiple sentences if necessary.

Return only JSON with three fields:
{
  "problem_statement": ["..."] or [],
  "objective": ["..."] or [],
  "solution_approach": ["..."] or []
}
"""


# Few-shot example: paper with an exhaustive research idea
example_paper = """Title: Low-Rank Representation for Pairwise Ranking

Abstract: Learning to rank from pairwise comparisons is challenging. Real-world preferences often exhibit intransitivity, which classical scoring methods fail to capture. We propose a low-rank matrix representation to model rankings and accommodate intransitive preferences. Our approach leverages matrix factorization to embed items in a low-dimensional latent space, enabling prediction of pairwise outcomes while capturing complex relational patterns.

Reviewer Summaries Of The Paper:
- "The paper develops a low-rank representation model for ranking with intransitive preferences."
- "It provides a clear methodology using matrix factorization to model complex pairwise interactions."
- "The study identifies the limitations of classical ranking models and proposes a mathematically grounded alternative."
"""

# Few-shot example: proposal-style, descriptive, single strings
example_paper_output = {
    "problem_statement": (
        "Pairwise comparisons frequently result in intransitive preference patterns, creating challenges for conventional ranking models that assume transitivity. "
        "Learning to rank from these comparisons requires capturing complex relationships among items while handling inconsistent preferences."
    ),
    "objective": (
        "Develop a ranking model that can accurately represent intransitive preference patterns and provide reliable predictions of pairwise outcomes. "
        "Ensure the model captures the underlying structure of real-world preference data."
    ),
    "solution_approach": (
        "Embed items into a low-dimensional latent space using low-rank matrix factorization to capture complex relational interactions. "
        "Leverage this representation to predict pairwise outcomes while reflecting nuanced intransitive preferences, resulting in flexible and interpretable ranking models."
    )
}

extracted_research_ideas = []

for i, paper in enumerate(tqdm(iclr23_reviews[5:15])):  

    # Safely build review summaries list
    reviews_summary_list = "\n".join(
        f"- {reply['content'].get('summary_of_the_paper', 'No summary provided')}"
        for reply in paper.get("replies", [])
        if reply.get("reply_type") == "review"
    )

    # Construct the paper input
    paper_input = f"""Title: {paper['title']}

    Abstract: {paper['abstract']}
    
    Reviewer Summaries Of The Paper:
    {reviews_summary_list}
    """

    # Build messages for LLM call
    messages = [
        {"role": "system", "content": "You are a careful extractive classifier specialized in research idea extraction."},

        # Few-shot example
        {"role": "user", "content": research_idea_task_prompt + "\n\nPaper:\n" + example_paper},
        {"role": "assistant", "content": str(example_paper_output).replace("'", '"')},

        # Actual paper to extract research idea from
        {"role": "user", "content": research_idea_task_prompt + "\n\nPaper:\n" + paper_input}
    ]

    # LLM call
    result = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        response_model=ResearchIdea, 
        messages=messages,
        temperature=0
    )

    # Append extracted research idea
    extracted_research_ideas.append(result.model_dump())

# ---------------------------
# Function to process one paper with retries
# ---------------------------
def process_paper_with_retries(client, paper, example_paper, example_output, max_retries=3, retry_backoff=1.0):
    def call_llm():
        # Safely build review summaries list
        reviews_summary_list = "\n".join(
            f"- {reply['content'].get('summary_of_the_paper', 'No summary provided')}"
            for reply in paper.get("replies", [])
            if reply.get("reply_type") == "review"
        )

        # Construct the paper input
        paper_input = f"""Title: {paper['title']}

            Abstract: {paper['abstract']}

            Reviewer Summaries Of The Paper:
            {reviews_summary_list}
            """

        messages = [
            {"role": "system", "content": "You are a careful extractive classifier specialized in research idea extraction."},

            # Few-shot example
            {"role": "user", "content": research_idea_task_prompt + "\n\nPaper:\n" + example_paper},
            {"role": "assistant", "content": str(example_paper_output).replace("'", '"')},

            # Actual paper to extract research idea from
            {"role": "user", "content": research_idea_task_prompt + "\n\nPaper:\n" + paper_input}
        ]

        return client.chat.completions.create(
            model="openai/gpt-oss-120b",
            response_model=ResearchIdea,
            messages=messages,
            temperature=0
        )

    for attempt in range(max_retries):
        try:
            return call_llm()
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(retry_backoff * (2 ** attempt))
            else:
                return {"error": f"Failed after {max_retries} attempts: {str(e)}"}

# ---------------------------
# Parallel execution
# ---------------------------
def fetch_research_ideas_parallel(client, papers, example_paper, example_output, max_parallel=5, max_retries=3, retry_backoff=1.0, sleep_between_calls=0.0):
    results = [None] * len(papers)

    with ThreadPoolExecutor(max_workers=max_parallel) as executor:
        future_to_index = {
            executor.submit(
                process_paper_with_retries,
                client,
                paper,
                example_paper,
                example_output,
                max_retries,
                retry_backoff
            ): i for i, paper in enumerate(papers)
        }

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

# ---------------------------
# Example usage
# ---------------------------
extracted_research_ideas = fetch_research_ideas_parallel(
    client,
    iclr22_reviews,
    example_paper,
    example_paper_output,
    max_parallel=50,
    max_retries=3,
    retry_backoff=1.0,
    sleep_between_calls=0.0
)

# Save as a JSON files
with open("../data/json/ICLR.cc-2022-Conference-research-ideas.json", 'w') as f:
    json.dump(extracted_research_ideas, f, indent=4)  # indent makes it easy to read

