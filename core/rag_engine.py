from pathlib import Path
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_community.retrievers import BM25Retriever
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from sentence_transformers import CrossEncoder
from openai import OpenAI


load_dotenv()

import os
import mlflow
import time
import json

# Path Setup
project_root = Path(__file__).resolve().parent.parent
mlruns_path = project_root / "mlruns"
mlflow.set_tracking_uri(mlruns_path.as_uri())
mlflow.set_experiment("GDPR_Greek_Auditor")
db_path = project_root / "chroma_db"

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
vector_db = Chroma(persist_directory=str(db_path), embedding_function=embeddings)

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
reranker = CrossEncoder("cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")
rewriter_llm = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

def rewrite_query(query: str) -> str:
    response = rewriter_llm.chat.completions.create(
        model="gpt-4.1-mini",
        temperature=0,
        messages=[
            {"role": "system", "content": "Είσαι βοηθός νομικής έρευνας. Ξαναγράψε την ερώτηση του χρήστη ως επίσημο νομικό ερώτημα για αναζήτηση σε ελληνική και ευρωπαϊκή νομοθεσία προστασίας δεδομένων. Απάντησε ΜΟΝΟ με την ξαναγραμμένη ερώτηση."},
            {"role": "user", "content": query}
        ]
    )
    return response.choices[0].message.content.strip()


def get_answer_with_context(query_text, user_docs=None):
    with mlflow.start_run():
        start_time = time.time()

        # --- LOG PARAMETERS ---
        mlflow.log_param("query_length", len(query_text))
        mlflow.log_param("k_laws", 7)
        mlflow.log_param("has_user_pdf", user_docs is not None)

        rewritten_query =  rewrite_query(query_text)
        mlflow.log_param("rewritten_query", rewritten_query[:250])
        print(f"Rewritten query: {rewritten_query}")

        laws_retriever = vector_db.as_retriever(
            search_type="mmr",  # for variety
            search_kwargs={'k': 7}
        )

        # Combine context
        context_docs = laws_retriever.invoke(rewritten_query)

        if user_docs:
            # Create temporary database in RAM for user's PDF
            local_db = DocArrayInMemorySearch.from_documents(user_docs, embeddings)
            local_retriever = local_db.as_retriever(search_kwargs={"k": 5})
            user_specific_chunks = local_retriever.invoke(rewritten_query)

            # Mark user's PDF
            for doc in user_specific_chunks:
                doc.metadata["source_law"] = "Το Έγγραφό σας"

            context_docs.extend(user_specific_chunks)

        # Prompt
        template = """Είσαι ο 'GDPR Greece AI Analyst'. 

ΟΔΗΓΙΕΣ:
1. Απάντησε ΑΠΟΚΛΕΙΣΤΙΚΑ βάσει των αποσπασμάτων που σου παρέχονται στο Context.
2. ΜΗΝ προσθέτεις πληροφορίες, ερμηνείες ή συμπεράσματα που ΔΕΝ υπάρχουν ρητά στο Context.
3. Αν δεν μπορείς να απαντήσεις πλήρως από το Context, δήλωσέ το ρητά: 'Δεν υπάρχει επαρκής πληροφορία στα διαθέσιμα αποσπάσματα.'
4. ΑΝ και ΜΟΝΟ ΑΝ υπάρχουν αποσπάσματα με την ένδειξη 'Το Έγγραφό σας', χρησιμοποίησέ τα για σύγκριση με τη νομοθεσία.
5. Παράθεσε πάντα το άρθρο ή τον νόμο από το οποίο αντλείς την πληροφορία.
    
        Context: {context}

        Ερώτηση: {question}
        """

        prompt = ChatPromptTemplate.from_template(template)

        # Chain Execution
        chain = (
                {"context": lambda x: x["context"], "question": lambda x: x["question"]}
                | prompt
                | llm
                | StrOutputParser()
        )

        response = chain.invoke({"context": context_docs, "question": rewritten_query})

        eval_entry = {
            "question": rewritten_query,
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
