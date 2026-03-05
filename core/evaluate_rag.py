import json
import time
from pathlib import Path
import mlflow
from datasets import Dataset
from openai import AsyncOpenAI

from langchain_huggingface import HuggingFaceEmbeddings
import asyncio
from ragas.metrics.collections import Faithfulness
from ragas.llms import llm_factory
# --- CONFIGURATION ---
project_root = Path(__file__).resolve().parent.parent
log_file = project_root / "data" / "evaluation-logs" / "eval_logs.jsonl"
mlflow_uri = (project_root / "mlruns").as_uri()

# Create an OpenAI-compatible client for Ollama
client = AsyncOpenAI(
    base_url="http://host.docker.internal:11434/v1",
    api_key="ollama"    # Ollama doesn't require a real key
)
evaluator_llm = llm_factory(
    model="llama3.1:8b",
    provider="openai",
    client=client,
    temperature = 0
)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

faithfulness_scorer = Faithfulness(llm=evaluator_llm)

async def run_offline_audit():
    if not log_file.exists():
        print("📭 No logs found.")
        return

    rows = []
    with open(log_file, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"Skipping corrupted data at line {line_no}")
                continue

    if not rows:
        print("ℹ️ File is empty")
        return

    dataset = Dataset.from_dict({
        "user_input": [r["question"] for r in rows],
        "response": [r["answer"] for r in rows],
        "retrieved_contexts": [r["contexts"] for r in rows],
    })

    # MLflow Tracking Setup
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment("GDPR_Offline_Audit")

    scores = []

    print(f"🚀 Starting Manual Audit for {len(rows)} samples...")

    for i, row in enumerate(rows):
        try:

            result = await faithfulness_scorer.ascore(
                user_input=row["question"],
                response=row["answer"],
                retrieved_contexts=row["contexts"]
            )
            score_value = result.value if hasattr(result, 'value') else result

            print(f"Sample {i + 1}: Faithfulness = {score_value}")
            scores.append(score_value)

        except Exception as e:
            print(f"❌ Error at sample {i + 1}: {e}")
            continue

    if scores:
        avg_score = sum(scores) / len(scores)


        with mlflow.start_run(run_name=f"Manual_Audit_{time.strftime('%Y%m%d-%H%M')}"):
            mlflow.log_metric("faithfulness", float(avg_score))
            print(f"Average Faithfulness logged to MLflow: {avg_score}")
    else:
        print("No scores were generated, skipping MLflow logging.")

    # Archive logs
    archive_path = log_file.with_name(f"processed_{time.strftime('%Y%m%d-%H%M')}.jsonl")
    log_file.rename(archive_path)


if __name__ == "__main__":
    asyncio.run(run_offline_audit())