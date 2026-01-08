import streamlit as st
import tempfile
import os
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from core.rag_engine import get_answer_with_context

# 1. Page Configuration
st.set_page_config(
    page_title="GDPR Greek Auditor AI",
    page_icon="⚖️",
    layout="wide"
)

# 2. Sidebar - Document Upload & Info
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/law.png", width=80)
    st.title("GDPR Compliance Lab")
    st.markdown("---")

    st.header("📄 Έλεγχος Εγγράφου")
    uploaded_file = st.file_uploader(
        "Ανεβάστε την Πολιτική Απορρήτου ή τους Όρους Χρήσης (PDF)",
        type="pdf",
        help="Το αρχείο αναλύεται προσωρινά για σύγκριση με τη νομοθεσία."
    )

    processed_user_docs = None
    if uploaded_file:
        with st.spinner("🔍 Ανάλυση εγγράφου χρήστη..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name

            loader = PyMuPDFLoader(tmp_path)
            raw_docs = loader.load()

            splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
            processed_user_docs = splitter.split_documents(raw_docs)

            os.remove(tmp_path)
            st.success(f"✅ Το έγγραφο φορτώθηκε ({len(processed_user_docs)} chunks)")

    st.markdown("---")
    st.info("""
    **Βάση Δεδομένων:**
    - EU GDPR 2016/679
    - Ν. 4624/2019 (Ελλάδα)
    - Ν. 5169/2025 (AI & 108+)
    - Ν. 3471/2006 (e-Privacy)
    """)

# 3. Main Chat Interface
st.title("⚖️ Greek GDPR & Compliance Analyst")
st.caption("Ρωτήστε για τη νομοθεσία ή συγκρίνετε το έγγραφό σας με τους επίσημους νόμους.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
if prompt := st.chat_input("π.χ. Είναι το όριο ηλικίας στην πολιτική μου σύννομο για την Ελλάδα;"):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate assistant response
    with st.chat_message("assistant"):
        loading_msg = "Αναζήτηση στη Νομοθεσία..." if not processed_user_docs else "🕵️ Συγκριτικός Έλεγχος (Νόμοι vs Έγγραφο)..."

        with st.spinner(loading_msg):
            try:
                result = get_answer_with_context(prompt, user_docs=processed_user_docs)

                answer = result["answer"]
                sources_list = "\n".join([f"- {s}" for s in result["sources"]])
                full_response = f"{answer}\n\n**📌 Πηγές:**\n{sources_list}"

                st.markdown(full_response)
                st.session_state.messages.append({"role": "assistant", "content": full_response})
            except Exception as e:
                if "429" in str(e):
                    st.warning(
                        "⚠️ Το σύστημα είναι προσωρινά φορτωμένο λόγω των ορίων του δωρεάν API. Παρακαλώ περιμένετε 1 λεπτό και δοκιμάστε ξανά.")
                else:
                    st.error(f"Προέκυψε σφάλμα: {str(e)}")