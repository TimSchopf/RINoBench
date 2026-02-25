
def build_evaluation_prompt_existing_works_aspects(predicted_reasoning: str, gold_reasoning: str) -> str:
    return f"""
        You are an expert in analyzing research reasoning with respect to related work.

        Task:
        1. From the PREDICTED_REASONING, extract all distinct aspects, points, or arguments that describe **similarities or differences with existing work**.
           - Only include statements that explicitly or implicitly reference prior work, methods, or findings.
           - **Do not** consider whether the idea is novel or original.
           - **Do not** include general observations unrelated to comparisons with existing work.
        2. For each extracted aspect from the PREDICTED_REASONING, indicate whether it is **explicitly or implicitly grounded in the GOLD_REASONING**.
        3. Separately, extract all distinct aspects, points, or arguments in the GOLD_REASONING that describe **similarities or differences with existing work**, following the same criteria.
        4. Focus only on the reasoning about related work; ignore any final judgments about novelty, originality, or contribution.

        Return the results in strict JSON format following this schema:

        ---
        PREDICTED_REASONING:
        {predicted_reasoning}

        GOLD_REASONING:
        {gold_reasoning}

        ---
        Output schema:
        {{
          "predicted_aspects": [
            {{
              "aspect": "...",
              "supported_in_gold": true|false
            }}
          ],
          "gold_aspects": [
            "..."
          ]
        }}
    """

def build_evaluation_prompt_novelty_aspects(predicted_reasoning: str, gold_reasoning: str) -> str:
    return f"""
    You are an expert in analyzing novelty reasoning in research ideas.

    Task:
    1. From the PREDICTED_REASONING, extract all distinct aspects, points, or arguments that describe what is **new, original, or innovative** about the idea, method, or approach.
       - Focus only on explicit or implicit claims of novelty (e.g., introducing a new technique, combining methods in a new way, applying an approach in a new domain).
       - **Do not** include statements about comparisons to prior work unless they are explicitly framed as evidence of novelty.
       - **Do not** include the overall judgment (e.g., “this idea is novel” or “this idea is not novel”). Only extract the specific arguments that highlight novelty.
    2. For each extracted aspect from the PREDICTED_REASONING, determine whether it is **explicitly or implicitly grounded in the GOLD_REASONING**.
    3. Separately, extract all distinct aspects, points, or arguments in the GOLD_REASONING that describe novelty in the same way.
    4. Focus strictly on the reasoning about **novelty aspects**, not final evaluations.

    Return the results in strict JSON format following this schema:

    ---
    PREDICTED_REASONING:
    {predicted_reasoning}

    GOLD_REASONING:
    {gold_reasoning}

    ---
    Output schema:
    {{
      "predicted_aspects": [
        {{
          "aspect": "...",
          "supported_in_gold": true|false
        }}
      ],
      "gold_aspects": [
        "..."
      ]
    }}
    """

def build_fact_check_prompt(research_idea: dict, related_works: list[dict], claims: list[str]) -> str:
    # old text: Support also means that if a claim states something is not existing in related works, it should actually not exist in related works but has to be mentioned in a way in the research idea. But only return True if the claim explicitly states something that is not existing in related works and this actually is correct. Otherwise, the claim is not supported and therefore you must return False. If the claim mentions no existing work or similar this implicitly means no related work in the given context and does not generalize to all existing works in the world.
    return f"""
    You are an expert in evaluating the support of claims with a given context.
    You will fact-check whether extracted claims are supported by the provided context consisting of a research idea and related works. The claims you are given were originally extracted from a reasoning process over this context to judge the novelty of the given research idea. We now want to verify whether these claims are actually hallucination-free and supported by the context.

    Context:
    - RESEARCH_IDEA:
    {research_idea}

    - RELATED_WORKS:
    {related_works}

    Claims:
    {claims}

    Task:
    For each claim:
    1. Check whether the claim is factually supported by the information in the context.
    2. Support (True) also means if a claim states something is not existing in prior work and it is actually not existing in the given related works! However, in this case, the claim must be grounded in the research idea.
    3. Consider both explicit statements and reasonable implications.
    4. Return True if it is supported, False otherwise.

    Output format must strictly follow this JSON schema:
    {{
      "checked_claims": [
        {{
          "supported": true|false
        }}
      ]
    }}
    """