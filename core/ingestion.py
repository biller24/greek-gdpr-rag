import os
import re
import shutil
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv() # Loads the API key from .env

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)

DATA_PATH = os.path.join(project_root, "data", "core-files")
CHROMA_PATH = os.path.join(project_root, "chroma_db")

def ingest_legislation():
    if os.path.exists(CHROMA_PATH):
        print(f"Clearing old database {CHROMA_PATH}...")
        shutil.rmtree(CHROMA_PATH)
    # Initialize Embeddings Model
    # multilingual, lightweight and fast model
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    # Split into Chunks
    # Greek is "morphologically rich" (long words) and we have legal documents,
    # so we use a decent chunk size.
    all_chunks = []
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=120,
        separators=["\n\n", "\n", "Άρθρο", ".", " "]
    )

    print("🛠️ Processing Legislation...")
    for filename in os.listdir(DATA_PATH):
        if filename.endswith(".pdf"):
            loader = PyMuPDFLoader(os.path.join(DATA_PATH, filename))
            docs = loader.load()

            # Extract Law Number/Year from filename for metadata
            # Pattern matches "N_" followed by digits, then "_" followed by digits
            law_match = re.search(r"[NΝ]_(\d+)_(\d+)", filename)

            if law_match:
                # If the pattern is found, use the professional Greek label
                source_label = f"Ν. {law_match.group(1)}/{law_match.group(2)}"
            else:
                # If the pattern ISN'T found,
                # just use the filename so the code doesn't crash!
                source_label = filename.replace(".pdf", "").replace("-", " ")

            for doc in docs:
                # Cleaning text
                doc.page_content = doc.page_content.replace("\n", " ").replace("  ", " ")
                # Update metadata
                doc.metadata["source_law"] = source_label


            chunks = text_splitter.split_documents(docs)
            all_chunks.extend(chunks)
            print(f"✅ Indexed {source_label} ({len(chunks)} chunks)")


    # Save to ChromaDB
    vector_db = Chroma.from_documents(
        documents=all_chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PATH
    )
    print("Vector Database Created Successfully!")


if __name__ == "__main__":
    ingest_legislation()
