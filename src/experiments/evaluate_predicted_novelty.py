from datasets import load_dataset
import os
from llm_call import *
from eval import eval_bench

os.environ["HF_TOKEN"] = "..."
os.environ['OPENAI_API_KEY'] = "..."

# load data
ds = load_dataset("...")
predictor_llm_name = "Llama-3.1-8B-Instruct"
with open('../data/novelty_predictions/novelty_predictions-' + predictor_llm_name + '.json', 'r') as f:
    predictions: dict = json.load(f)

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
eval_llm_base_url = "https://api.openai.com/v1"
#llm_name = "openai/gpt-oss-120b"
eval_llm_name = "gpt-4.1-2025-04-14"
api_key = os.environ['OPENAI_API_KEY']
client = LLMClient(api_key, eval_llm_base_url)

# run evaluation
eval_metrics = asyncio.run(eval_bench(
    client=client,
    llm_name=eval_llm_name,
    research_ideas=ds['test']['research_idea'],
    related_works=[[{k: p[k] for k in ('title', 'abstract')} for p in works] for works in ds['test']['related_works']],
    predicted_reasonings=[p['reasoning'] for p in predictions],
    gold_reasonings=ds['test']['novelty_reasoning'],
    predicted_scores=[p['novelty_score'] for p in predictions],
    gold_scores=ds['test']['novelty_score'],
    n_parallel=10,
    temperature=0,
    max_retries=100,
    retry_backoff=1.0
))

# Save to disk
save_path = '../data/evaluations/novelty_pred_evaluation-' + predictor_llm_name + '.json'
with open(save_path, "w", encoding="utf-8") as f:
    json.dump(eval_metrics, f, indent=4)

print(f"Evaluation results saved to {save_path}")