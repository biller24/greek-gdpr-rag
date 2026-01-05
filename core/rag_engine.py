
from pathlib import Path
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel
load_dotenv()


def get_answer_with_sources(query_text):
    # Path Setup
    project_root = Path(__file__).resolve().parent.parent
    db_path = project_root / "chroma_db"

    # Embeddings (The Librarian)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    #  Load Vector DB
    vector_db = Chroma(persist_directory=str(db_path), embedding_function=embeddings)
    retriever = vector_db.as_retriever(search_kwargs={"k": 5})

    #  Initialize Gemini
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

    # Note: We use {context} which will be filled by the retriever
    template = """Είσαι ένας εξειδικευμένος Νομικός Σύμβουλος με εμπειρία στην ανάλυση Φύλλων Εφημερίδας της Κυβερνήσεως (ΦΕΚ).
    Απάντησε στην ερώτηση χρησιμοποιώντας ΑΠΟΚΛΕΙΣΤΙΚΑ τα παρακάτω αριθμημένα αποσπάσματα από το ΦΕΚ. 
     ΚΑΝΟΝΕΣ ΑΠΑΝΤΗΣΗΣ:
1. Ακρίβεια: Χρησιμοποίησε την επίσημη ορολογία του κειμένου (π.χ. Άρθρο, Παράγραφος, Εδάφιο).
2. Παραπομπές: Πρέπει να βάζεις την αναφορά σε αγκύλες π.χ. [1] στο τέλος κάθε πρότασης που βασίζεται σε πληροφορία του ΦΕΚ.
3. Δομή: Αν το ΦΕΚ ορίζει προϋποθέσεις ή λίστες, χρησιμοποίησε bullet points για σαφήνεια.
4. Περιορισμός: Αν η πληροφορία δεν υπάρχει στα αποσπάσματα, πες: "Το συγκεκριμένο ΦΕΚ δεν παρέχει πληροφορίες για αυτό το θέμα". Μην χρησιμοποιείς εξωτερικές γνώσεις.

     Context: {context}

     Ερώτηση: {question}
     """

    prompt = ChatPromptTemplate.from_template(template)

    # Create the LCEL Chain
    rag_chain_from_docs = (
            RunnablePassthrough.assign(context=(lambda x: x["context"]))
            | prompt
            | llm
            | StrOutputParser()
    )
    full_chain = RunnableParallel(
        {"context": retriever, "question": RunnablePassthrough()}
    ).assign(answer=rag_chain_from_docs)

    # Execute
    print(f"🔍 Searching for: {query_text}...")
    result = full_chain.invoke(query_text)

    # Extract sources
    # We use a 'set' to avoid repeating the same page number multiple times
    sources = set()
    for doc in result["context"]:
        page_num = doc.metadata.get("page", "Unknown")
        sources.add(f"Σελίδα {page_num + 1}")  # +1 because PDF indexing starts at 0
    return {
        "answer": result["answer"],
        "sources": sorted(list(sources))
    }


if __name__ == "__main__":
    query = "Ποιο είναι το επιτόκιο για στεγαστικό δάνειο;"
    try:
        output = get_answer_with_sources(query)
        print(f"\n--- AI ANSWER ---\n{output['answer']}")
        print(f"\n📌 Πηγές: {', '.join(output['sources'])}")
    except Exception as e:
        print(f"Error: {e}")