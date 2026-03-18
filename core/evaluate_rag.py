import json
import time
from pathlib import Path
import mlflow
from datasets import Dataset
from openai import AsyncOpenAI
import os

from langchain_huggingface import HuggingFaceEmbeddings
import asyncio
from ragas.metrics.collections import Faithfulness,AnswerRelevancy
from ragas.llms import llm_factory
from ragas.embeddings.base import embedding_factory
# --- CONFIGURATION ---
project_root = Path(__file__).resolve().parent.parent
log_file = project_root / "data" / "evaluation-logs" / "eval_logs.jsonl"
mlflow_uri = (project_root / "mlruns").as_uri()

# Create an OpenAI-compatible client for Ollama
client = AsyncOpenAI(
    api_key=os.environ["OPENAI_API_KEY"]
)

evaluator_llm = llm_factory(
    model="gpt-4.1-mini",
    client=client,
    temperature=0,
    max_tokens=4096
)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
RAGAS_embeddings = embedding_factory(
    provider="huggingface",
    model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
faithfulness_scorer = Faithfulness(llm=evaluator_llm)
answer_relevancy_scorer = AnswerRelevancy(llm=evaluator_llm, embeddings=RAGAS_embeddings)

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

    faithfulness_scores = []
    relevancy_scores = []

    print(f"Starting Manual Audit for {len(rows)} samples...")

    for i, row in enumerate(rows):
        # Faithfulness
        try:
            result = await faithfulness_scorer.ascore(
                user_input=row["question"],
                response=row["answer"],
                retrieved_contexts=row["contexts"]
            )
            score_value = result.value if hasattr(result, 'value') else result
            faithfulness_scores.append(score_value)
            print(f"Sample {i + 1}: Faithfulness = {score_value:.3f}")
        except Exception as e:
            print(f"Error at sample {i + 1} [Faithfulness]: {e}")

        # Answer Relevancy
        try:
            result = await answer_relevancy_scorer.ascore(
                user_input=row["question"],
                response=row["answer"]
            )
            score_value = result.value if hasattr(result, 'value') else result
            relevancy_scores.append(score_value)
            print(f"Sample {i + 1}: Answer Relevancy = {score_value:.3f}")
        except Exception as e:
            print(f"Error at sample {i + 1} [Answer Relevancy]: {e}")

    if faithfulness_scores or relevancy_scores:
        with mlflow.start_run(run_name=f"Manual_Audit_{time.strftime('%Y%m%d-%H%M')}"):
            if faithfulness_scores:
                avg_faithfulness = sum(faithfulness_scores) / len(faithfulness_scores)
                mlflow.log_metric("faithfulness", float(avg_faithfulness))
                print(f"Average Faithfulness logged to MLflow: {avg_faithfulness:.3f}")

            if relevancy_scores:
                avg_relevancy = sum(relevancy_scores) / len(relevancy_scores)
                mlflow.log_metric("answer_relevancy", float(avg_relevancy))
                print(f"Average Answer Relevancy logged to MLflow: {avg_relevancy:.3f}")
    else:
        print("No scores were generated, skipping MLflow logging.")

    # Archive logs
    archive_path = log_file.with_name(f"processed_{time.strftime('%Y%m%d-%H%M')}.jsonl")
    log_file.rename(archive_path)


if __name__ == "__main__":
    asyncio.run(run_offline_audit())