#!/usr/bin/env python
# coding: utf-8

import json
import os
import time
import concurrent.futures
from tqdm import tqdm 
import instructor
from openai import OpenAI
from pydantic import BaseModel, Field
from typing import List

# load data from disk
with open('../data/json/ICLR.cc-22-23-Conferences.json', 'r') as f:
    merged_papers: dict = json.load(f)

# get raw review texts
def get_raw_review_text(merged_paper):
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
        "decision",
        "title",
        "abstract",
        "comment"
    }

    raw_review_text = []
    for reply in merged_paper.get("reviews", []).get("replies", []):
        content = reply.get("content", {})
        for field, text in content.items():
            if field not in EXCLUDE_FIELDS and isinstance(text, str):
                raw_review_text.append(text)

    return raw_review_text


raw_review_texts = [get_raw_review_text(paper) for paper in merged_papers]

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
# Pydantic Models
# ---------------------------

class Evidence(BaseModel):
    title: str = Field("", description="Exact related work title that supports the claim.")
    passage: str = Field("", description="Exact verbatim passage from the related work that supports the claim.")


class PriorWorkClaims(BaseModel):
    prior_work_claims: List[str] = Field(default_factory=list,
        description="List of ONLY exact explicit claims about PRIOR WORKS or their contributions that DESCRIBE WHAT EXISTS from the **novelty reasoning/judgment**. "
                    "Includes exact explicit claims about prior works such as: what prior works did, methods they used, problems they addressed, contributions they made, etc. "
                    "⚠️ Does NOT include the novelty claim itself (statements describing what is new or unique about the proposed idea). Also, does NOT include statements that describe what is missing, not done, underexplored, or absent in prior works (For example: “Prior work has not ...” is a novelty claim and should NOT be extracted)."
    )

class ClaimSupportCheck(BaseModel):
    evidence: List[Evidence] = Field(
        default_factory=list,
        description="List of evidence from this related work supporting the claim; empty if none."
    )

class ClaimSupportCheck(BaseModel):
    evidence: List[Evidence] = Field(default_factory=list, description="List of evidence; empty if none.")

class PriorWorkEvidenceCheck(BaseModel):
    prior_work_claim: str = Field("", description="The prior work claim being checked.")
    evidence: List[Evidence] = Field(default_factory=list,
        description="Evidence collected from all related works supporting this claim."
    )
    supported: bool = Field(False,
        description="True if at least one evidence item exists, False otherwise."
    )

class NoveltyClaims(BaseModel):
    novelty_claims: List[str] = Field(default_factory=list,
        description="List of novelty claims from the **novelty reasoning/judgment** (statements describing what is new or unique about the proposed idea). This also includes statements that describe what is missing, not done, underexplored, or absent in prior works. "
                    "⚠️ Does NOT include statements novelty reasoning/judgment about PRIOR WORKS or their contributions that DESCRIBE WHAT EXISTS. Also, does NOT include evaluative claims from the novelty reasoning/judgment about the perceived novelty of a claim."
    )

class NoveltyClaimCheck(BaseModel):
    claim: str = Field("", description="The novelty claim.")
    supported: bool = Field(False, description="True if the claim is supported, False otherwise.")
    explanation: str = Field("", description="Explanation of the support decision.")


class ReviewCheck(BaseModel):
    mentions_missing_references: bool = Field(False,
        description=(
                "True if the review explicitly recommends including specific missing references or citations in the paper. False otherwise. "
                "Mentions of related work that merely discuss its relation to the submission, or asking whether a statement "
                "can be proven via a source does NOT count. Mentions of related work that do not mention a specific one do NOT count. Only concrete recommendations to add specific works count."
        )
    )
    missing_references: List[str] = Field(default_factory=list,
        description="If 'mentions_missing_references' is True, a list of all exact copies of mentions of missing references. If False, return empty list."
    )

class NoveltyReasoningFormalCheck(BaseModel):
    criteria_fulfilled: bool = Field(False,
        description="True if all criteria are fulfilled, False otherwise."
    )

# ---------------------------
# Prompts
# ---------------------------

def build_prior_work_claims_prompt(novelty_reasoning: str) -> str:
    return f"""
            You are tasked with analyzing novelty reasoning text.

            Input:
            - Novelty reasoning/judgment:
            {novelty_reasoning}

            Task:
            1. Extract ONLY verbatim claims about PRIOR WORKS or their contributions that describe WHAT EXISTS.
            2. Do NOT include novelty claims (what is new, missing, not done, underexplored).
            3. Return the extracted claims exactly as they appear.

            Output Format (JSON):
            {{
                "prior_work_claims": [
                    "first claim",
                    "second claim",
                    ...
                ]
            }}
            """


def build_claim_support_prompt(claim: str, related_work: dict) -> str:
    return f"""
            You are tasked with verifying if a claim about prior work is supported by a given related work.

            Input:
            - Claim: "{claim}"
            - Related work:
            Title: {related_work['title']}
            Abstract: {related_work['abstract']}

            Task:
            1. Check if the related work provides evidence supporting the claim.
            2. If yes, return the exact title and exact verbatim passage from the abstract that supports it.
            3. If not, return an empty list.

            Output Format (JSON):
            {{
                "evidence": [
                    {{
                        "title": "Exact supporting title",
                        "passage": "Exact supporting passage"
                    }},
                    ...
                ]
            }}
            """


def build_novelty_claims_prompt(novelty_reasoning: str) -> str:
    return f"""
            You are tasked with analyzing novelty reasoning text.

            Input:
            - Novelty reasoning/judgment:
            {novelty_reasoning}

            Task:
            1. Extract ONLY novelty claims (statements describing what is new, unique, missing, not done, underexplored).
            2. Do NOT include claims about prior works describing what EXISTS.
            3. Return the extracted claims verbatim.

            Output Format (JSON):
            {{
                "novelty_claims": [
                    "first novelty claim",
                    "second novelty claim",
                    ...
                ]
            }}
            """


def build_novelty_verification_prompt(novelty_claim: str, research_idea: dict) -> str:
    return f"""
            You are tasked with verifying if a novelty claim is supported by a research idea.

            Input:
            - Novelty claim: "{novelty_claim}"
            - Research idea:
              Problem statement: {research_idea['problem_statement']}
              Objective: {research_idea['objective']}
              Solution approach: {research_idea['solution_approach']}

            Task:
            1. Check if the novelty claim is consistent with and supported by the research idea.
            2. Return True if fully supported, False otherwise, with a brief explanation.

            Output Format (JSON):
            {{
                "claim": "{novelty_claim}",
                "supported": true/false,
                "explanation": "Brief explanation of support decision"
            }}
            """

def build_review_check_prompt(review: str) -> str:
    return f"""
            You are tasked with analyzing a peer review.

            Input:
            - Review text:
            {review}

            Task:
            1. Check if the review explicitly RECOMMENDS including specific missing references or citations in the paper.
            2. Mentions of related work that merely discuss its relation to the submission do NOT count.
            2. Mentions of related work that do not mention a specific one do NOT count.
            3. Asking whether a statement can be proven via a source does NOT count. 
            4. Asking to cite a different version of an already cited article does NOT count. E.g. asking to cite the published version of an article instead of the arXiv version does NOT count. 
            5. Only concrete, explicit recommendations to add specific works count. If simply works are mentioned that might already be included in the paper, this does not count. E.g. if the review discussed the relation or comparison to a reference that is already citet in the paper, this does NOT count. 
            6. Return True if such a recommendation exists, False otherwise.
            7. If True, return all exact copies of mentions of missing references.

            Output Format (JSON):
            {{
                "mentions_missing_references": true/false,
                "missing_references": [
                    "Exact mention of missing reference 1",
                    "Exact mention of missing reference 2",
                    ...
                ]
            }}
            """

def build_novelty_reasoning_formal_check_prompt(novelty_reasoning: str) -> str:
    return f"""
            You are tasked with verifying if a novelty reasoning/judgment fulfills the following formal criteria.

            Input:
            - Novelty reasoning/judgment:
            {novelty_reasoning}

            Task:
            1. Check if the novelty reasoning/judgment is written as if one person judged the novelty. This means a judgment framed as belonging to others (e.g., reviewers). Phrases like “is described as” or “is presented as” are not allowed.
            2. Check if the novelty reasoning/judgment NOT mentions authors, papers, reviewers, comments, or results. Further, it should use neutral terms like “this idea” or “this approach” instead of “this work.”
            3. Check if the novelty reasoning/judgment does NOT contain any explicitly numeric novelty scores.
            4. Check if the novelty reasoning/judgment is valid, meaning no text like 'No novelty judgment found' is present.
            5. Return True if all criteria are fulfilled, False otherwise.


            Output Format (JSON):
            {{
                "criteria_fulfilled": true/false
            }}
            """

# ---------------------------
# LLM Call with Retries
# ---------------------------

def call_llm_with_retries(client, prompt: str, schema, max_retries=300, retry_backoff=1.0):
    def call_llm():
        messages = [
            {"role": "system", "content": "You are an expert in evaluating novelty judgments in research ideas."},
            {"role": "user", "content": prompt},
        ]

        return client.chat.completions.create(
            model=llm,  
            response_model=schema,
            messages=messages,
            #temperature=0,
            extra_body={"disable_fallbacks": True},
        )

    for attempt in range(max_retries):
        try:
            return call_llm()
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(retry_backoff * (2 ** attempt))
            else:
                raise RuntimeError(f"Failed after {max_retries} attempts: {str(e)}")


# ---------------------------
# Pipeline Execution Skeleton
# ---------------------------

def pipeline(client, data: dict, related_works: List[dict], reviews: List[str]):
    max_parallel_calls = 10
    # Step 1: Extract prior work claims
    pw_prompt = build_prior_work_claims_prompt(data['novelty_score_justification'])
    pw_claims: PriorWorkClaims = call_llm_with_retries(client, pw_prompt, PriorWorkClaims)

    # Step 2: For each claim, check each related work (parallelized, preserve order)
    evidence_results = []
    for claim in pw_claims.prior_work_claims:
        def process_related_work(rw):
            cs_prompt = build_claim_support_prompt(claim, rw)
            return call_llm_with_retries(client, cs_prompt, ClaimSupportCheck)

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_parallel_calls) as executor:
            futures = [executor.submit(process_related_work, rw) for rw in related_works]
            results = [f.result() for f in futures]  # preserves order

        combined_evidence = []
        for cs_result in results:
            combined_evidence.extend(cs_result.evidence)

        evidence_results.append(
            PriorWorkEvidenceCheck(
                prior_work_claim=claim,
                evidence=combined_evidence,
                supported=len(combined_evidence) > 0
            )
        )

    # Step 3: Summarize prior work support
    all_supported = None
    if evidence_results:
        all_supported = all(e.supported for e in evidence_results)

    # Step 4: Extract novelty claims
    nc_prompt = build_novelty_claims_prompt(data['novelty_score_justification'])
    novelty_claims: NoveltyClaims = call_llm_with_retries(client, nc_prompt, NoveltyClaims)

    # Step 5: Verify novelty claims against research idea (parallelized, preserve order)
    def verify_novelty(claim):
        nv_prompt = build_novelty_verification_prompt(claim, data['research_idea'])
        return call_llm_with_retries(client, nv_prompt, NoveltyClaimCheck)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_parallel_calls) as executor:
        futures = [executor.submit(verify_novelty, claim) for claim in novelty_claims.novelty_claims]
        novelty_checks = [f.result() for f in futures]  # preserves order

    novelty_supported = None
    if novelty_checks:
        novelty_supported = all(c.supported for c in novelty_checks)

    # Step 6: Check reviews for missing references (parallelized)
    def check_review(review):
        rv_prompt = build_review_check_prompt(review)
        return call_llm_with_retries(client, rv_prompt, ReviewCheck)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_parallel_calls) as executor:
        futures = [executor.submit(check_review, review) for review in reviews]
        review_results = [f.result() for f in futures] # preserves order

    mentions_missing_refs = any(r.mentions_missing_references for r in review_results)

    # Step 7: Check formal requirements of novelty judgment
    nov_check_prompt = build_novelty_reasoning_formal_check_prompt(data['novelty_score_justification'])
    nov_check: NoveltyReasoningFormalCheck = call_llm_with_retries(client, nov_check_prompt, NoveltyReasoningFormalCheck)


    # Step 8: Include decision
    include_example = (
        not mentions_missing_refs
        and all_supported is True
        and novelty_supported is True
        and nov_check.criteria_fulfilled is True
    )

    return {
        "prior_work_evidence": evidence_results,
        "all_prior_work_supported": all_supported,
        "novelty_checks": novelty_checks,
        "novelty_claims_supported_by_idea": novelty_supported,
        "mentions_missing_references": review_results,
        "novelty_check": nov_check,
        "include_example": include_example
    }


filtered_data = []
for i,p in enumerate(tqdm(merged_papers)):
    filtered_data.append(pipeline(client=client, data=p, related_works=p['related_works'], reviews=raw_review_texts[i]))



def to_json_serializable(obj):
    """
    Recursively clean the object:
    - Keep dicts/lists/scalars
    - Drop metadata keys
    - Flatten custom objects into dicts
    """
    # fields that we want to skip entirely
    SKIP_KEYS = {
        "_raw_response",
        "id",
        "choices",
        "created",
        "model",
        "object",
        "service_tier",
        "system_fingerprint",
        "usage",
        "_request_id",
        "annotations",
        "refusal",
        "role",
        "audio",
        "function_call",
        "tool_calls",
        "logprobs"
    }
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    elif isinstance(obj, (list, tuple, set)):
        return [to_json_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {
            k: to_json_serializable(v)
            for k, v in obj.items()
            if k not in SKIP_KEYS
        }
    else:
        if hasattr(obj, "__dict__"):
            return to_json_serializable(obj.__dict__)
        else:
            return str(obj)

# Save as a JSON file
json_filtered_data = [to_json_serializable(d) for d in filtered_data]
with open("../data/json/ICLR.cc-22-23-Conference-filtered-data.json", 'w') as f:
    json.dump(json_filtered_data, f, indent=4)  # indent makes it easy to read

