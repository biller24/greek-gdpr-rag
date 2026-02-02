# ⚖️ Greek GDPR & Legal Compliance AI Auditor

An advanced **RAG (Retrieval-Augmented Generation)** system specializing in Greek and European data protection legislation. The system allows users to query legal databases or upload their own Privacy Policies for automated compliance auditing.



## 🌟 Key Features
- **Dual-Source Retrieval**: Simultaneous search across a permanent Legislation Database (ChromaDB) and temporary user-uploaded documents (In-memory DocArray).
- **MLOps Integration**: Full experiment tracking with **MLflow** to monitor latency, context precision, and model parameters.
- **Legal Reasoning**: The AI performs comparative analysis between General Data Protection Regulation (GDPR) and specific Greek Law derogations (e.g., Law 4624/2019).
- **Dockerized Architecture**: Simplified deployment using **Docker Compose** for both the application and the tracking server.
- **Contextual Awareness**: Utilizes **Maximal Marginal Relevance (MMR)** to ensure diverse and non-redundant legal context.

## 🛠️ Tech Stack
- **LLM**: Google Gemini 2.5 Flash
- **Orchestration**: LangChain (LCEL)
- **Vector Databases**: 
  - **ChromaDB**: Persistent storage for national laws.
  - **DocArray**: In-memory storage for user-uploaded PDFs.
- **Experiment Tracking**: **MLflow**
- **Embeddings**: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- **Containerization**: Docker & Docker Compose
- **Frontend**: Streamlit

## 📂 Legal Knowledge Base
The system is pre-indexed with:
- **EU GDPR 2016/679** (The core European Regulation).
- **Law 4624/2019** (Greek supplementary measures).
- **Law 5169/2025** (New provisions for AI and the "108+" Convention).
- **Law 3471/2006** (Data protection in electronic communications).

## 🚀 Quick Start (Dockerized)
The easiest way to run the project is using Docker Compose:

### 1. Clone the Repository
```bash
git clone [https://github.com/yourusername/greek-gdpr-rag.git](https://github.com/yourusername/greek-gdpr-rag.git)
cd greek-gdpr-rag
```

### 2. Environment Setup
```bash
GOOGLE_API_KEY=your_gemini_api_key_here
```

### 3. Launch with Docker
```bash
docker-compose up --build
```

### 4. Access the Services
- **Streamlit App:** http://localhost:8501
- **MLflow Dashboard:** http://localhost:5000

## 📊 Monitoring & Evaluation (MLflow)
I integrated **MLflow** to transition from a "black box" chatbot to a monitored AI product. Every interaction logs:

* **Metrics**: Response latency (sec), context chunk count.
* **Parameters**: `k-neighbors`, `search_type`, `has_user_pdf`.
* **Artifacts**: Metadata regarding the specific law articles used for grounding.



## 🔧 Technical Optimizations
* **Chunking Strategy**: Implemented `RecursiveCharacterTextSplitter` with a 600-token window and 120-token overlap, customized for the morphologically rich Greek language.
* **Maximal Marginal Relevance (MMR)**: Instead of a standard similarity search, I implemented `search_type="mmr"`. This ensures that the retrieved document chunks are diverse and non-redundant—a critical factor when comparing overlapping provisions across different laws.
* **Advanced Parsing with PyMuPDFLoader**: After extensive testing, I chose `PyMuPDFLoader` (Fitz) over standard loaders. It handles the complex structure of Greek legal documents and tables more effectively while maintaining metadata integrity (e.g., precise page numbering).

* **Context Scaling**: I configured the retriever with `k=7` to provide the LLM with sufficient "legal depth" for complex queries, balancing rich context with noise reduction to ensure concise and accurate responses.
## 📺 Demo & Validation
Μπορείτε να δείτε αναλυτικά παραδείγματα ερωτήσεων και απαντήσεων, καθώς και τον έλεγχο συμμόρφωσης (Compliance Audit) που πραγματοποιεί το σύστημα, στο αρχείο: 👉 [**Test Scenarios & Showcase**](examples/test_scenarios.md)