
from pathlib import Path
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
load_dotenv()

import mlflow
import time

def get_answer_with_context(query_text, user_docs=None):
    # Path Setup
    project_root = Path(__file__).resolve().parent.parent
    mlruns_path = project_root / "mlruns"
    mlflow.set_tracking_uri(mlruns_path.as_uri())
    mlflow.set_experiment("GDPR_Greek_Auditor")
    with mlflow.start_run():
        start_time = time.time()  # Για να μετρήσουμε το latency

        # --- LOG PARAMETERS ---
        mlflow.log_param("query_length", len(query_text))
        mlflow.log_param("k_laws", 7)
        mlflow.log_param("has_user_pdf", user_docs is not None)

        project_root = Path(__file__).resolve().parent.parent
        db_path = project_root / "chroma_db"

        # Embeddings (The Librarian)
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )

        #  Load Vector DB
        vector_db = Chroma(persist_directory=str(db_path), embedding_function=embeddings)
        laws_retriever = vector_db.as_retriever(
            search_type="mmr",  # for variety
            search_kwargs={'k': 7}
        )

        # Combine context
        context_docs = laws_retriever.invoke(query_text)

        if user_docs:
            # Create temporary database in RAM for user's PDF
            local_db = DocArrayInMemorySearch.from_documents(user_docs, embeddings)
            local_retriever = local_db.as_retriever(search_kwargs={"k": 5})
            user_specific_chunks = local_retriever.invoke(query_text)

            # Mark user's PDF
            for doc in user_specific_chunks:
                doc.metadata["source_law"] = "Το Έγγραφό σας"

            context_docs.extend(user_specific_chunks)

        #  Initialize Gemini
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

        # Prompt
        template = """Είσαι ο 'GDPR Greece AI Analyst'. 

    ΟΔΗΓΙΕΣ:
    1. Χρησιμοποίησε τα αποσπάσματα από τη Νομοθεσία για να απαντήσεις στην ερώτηση.
    2. ΑΝ και ΜΟΝΟ ΑΝ υπάρχουν αποσπάσματα με την ένδειξη 'Το Έγγραφό σας', χρησιμοποίησέ τα για να κάνεις σύγκριση ή να δώσεις εξειδικευμένες πληροφορίες που αφορούν τον χρήστη.
    3. Αν δεν υπάρχει έγγραφο χρήστη στο Context, απάντησε γενικά με βάση τους νόμους.
    
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

        response = chain.invoke({"context": context_docs, "question": query_text})

        # --- LOG METRICS ---
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


