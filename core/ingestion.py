from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv() # Loads the API key from .env

def build_vector_db(pdf_path, db_directory):
    # Load the PDF
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()

    # Split into Chunks
    # Greek is "morphologically rich" (long words) and we have legal documents,
    # so we use a decent chunk size.
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=80)
    chunks = splitter.split_documents(pages)

    # Create the Embeddings Model (The Librarian)
    # multilingual, lightweight and fast model
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    # Save to ChromaDB
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=db_directory
    )
    print("Vector Database Created Successfully!")


if __name__ == "__main__":
    # Find the path to the current script (ingestion.py)
    current_script = Path(__file__).resolve()

    # Go up to the project root (from core/ to greek_banking_rag/)
    project_root = current_script.parent.parent

    # Define the path to the PDF and the DB folder
    pdf_path = project_root / "data" / "fek_5548.pdf"
    db_path = project_root / "chroma_db"

    if pdf_path.exists():
        # Call the function with the absolute path
        build_vector_db(str(pdf_path), str(db_path))
    else:
        print(" Error: The PDF file was not found at the path above.")