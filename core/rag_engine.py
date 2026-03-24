from pathlib import Path
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from sentence_transformers import CrossEncoder
from collections import Counter


load_dotenv()

import mlflow
import time
import json

# Path Setup
project_root = Path(__file__).resolve().parent.parent
mlruns_path = project_root / "mlruns"
mlflow.set_tracking_uri(mlruns_path.as_uri())
mlflow.set_experiment("GDPR_Greek_Auditor")
db_path = project_root / "chroma_db"


def load_rag_components():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    vector_db = Chroma(persist_directory=str(db_path), embedding_function=embeddings)
    reranker = CrossEncoder("cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

    return embeddings, vector_db, reranker,  llm

# Then unpack at module level:
embeddings, vector_db, reranker,  llm = load_rag_components()


def rerank_docs(query, docs, top_n=12):
    if not docs:
        return docs

    pairs = [(query, doc.page_content) for doc in docs]
    scores = reranker.predict(pairs)
    scored_docs = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)

    return [doc for score, doc in scored_docs[:top_n]]

def get_answer_with_context(query_text, user_docs=None):
    with mlflow.start_run():
        start_time = time.time()

        # --- LOG PARAMETERS ---
        mlflow.log_param("query_length", len(query_text))
        mlflow.log_param("k_laws", 24)
        mlflow.log_param("has_user_pdf", user_docs is not None)

        context_docs = vector_db.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 24}
        ).invoke(query_text)

        law_counts = Counter(doc.metadata.get("source_law") for doc in context_docs)
        print(f"Retrieved before reranking: {dict(law_counts)}")
        mlflow.log_param("retrieved", len(context_docs))

        context_docs = rerank_docs(query_text, context_docs)

        if user_docs:
            # Create temporary database in RAM for user's PDF
            local_db = DocArrayInMemorySearch.from_documents(user_docs, embeddings)
            local_retriever = local_db.as_retriever(search_kwargs={"k": 5})
            user_specific_chunks = local_retriever.invoke(query_text)

            # Mark user's PDF
            for doc in user_specific_chunks:
                doc.metadata["source_law"] = "Το Έγγραφό σας"

            context_docs.extend(user_specific_chunks)

        # Prompt
        template = """Είσαι ο 'GDPR Greece AI Analyst'. Απάντησε με ακρίβεια, επαγγελματισμό και ΣΥΝΤΟΜΙΑ.

ΟΔΗΓΙΕΣ:
1. ΑΠΑΝΤΗΣΗ: Δώσε μια άμεση απάντηση συνθέτοντας ΣΥΝΔΥΑΣΤΙΚΑ όλα τα αποσπάσματα. Αν ένας νέος νόμος (π.χ. Ν. 5169) κυρώνει μια σύμβαση, εξήγησε τη σημασία του με βάση το γενικό πλαίσιο (GDPR) που παρέχεται.
2. ΣΥΓΚΡΙΣΗ: ΜΟΝΟ αν υπάρχουν αποσπάσματα 'Το Έγγραφό σας', σύγκρινέ τα. Αλλιώς, μην αναφέρεις καθόλου έγγραφα χρήστη.
3. ΔΟΜΗ: Χρησιμοποίησε σύντομα bullet points.
4. HALLUCINATIONS: Αν το θέμα λείπει ΠΑΝΤΕΛΩΣ (π.χ. drones, μαγειρική), δήλωσε άγνοια. Αλλά αν το θέμα υπάρχει (π.χ. διαβίβαση δεδομένων), χρησιμοποίησε όλο το διαθέσιμο Context για να απαντήσεις.
        Context:{context}

        Ερώτηση:{question}
        """

        prompt = ChatPromptTemplate.from_template(template)

        # Chain Execution
        chain = (
                {"context": lambda x: x["context"], "question": lambda x: x["question"]}
                | prompt
                | llm
                | StrOutputParser()
        )

        response = chain.invoke({"context": context_docs, "question": query_text})

        eval_entry = {
            "question": query_text,
            "answer": response,
            "contexts": [doc.page_content for doc in context_docs],
            "timestamp": time.time()
        }

        evaluation_logs_path = project_root / "data" / "evaluation-logs" / "eval_logs.jsonl"
        evaluation_logs_path.parent.mkdir(parents=True, exist_ok=True)

        with open(evaluation_logs_path, "a", encoding="utf-8") as f:
            json.dump(eval_entry, f, ensure_ascii=False)
            f.write("\n")

        # Log metrics
        generation_time = time.time() - start_time
        mlflow.log_metric("latency_sec", generation_time)
        mlflow.log_metric("source_count", len(context_docs))

        # Extract sources
        sources = set()
        for doc in context_docs:
            law = doc.metadata.get("source_law", "Πηγή")
            page = doc.metadata.get("page", 0)
            sources.add(f"{law} (Σελίδα {page + 1})")

        return {"answer": response, "sources": sorted(list(sources))}
