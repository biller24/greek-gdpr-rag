# ⚖️ Greek GDPR & Legal Compliance AI Auditor

An advanced **RAG (Retrieval-Augmented Generation)** system specializing in Greek and European data protection legislation. This system performs automated compliance auditing by cross-referencing user-uploaded policies against a multi-layered legal knowledge base.

## 🌟 Key Features
- **Two-Stage Retrieval**: Combines **MMR (Maximal Marginal Relevance)** for initial diversity with a **Cross-Encoder Reranker** for high-precision legal grounding.
- **Automated Compliance Audit**: Performs real-time comparative analysis between user-uploaded documents and specific Greek Law derogations (e.g., Law 4624/2019 vs. EU GDPR).
- **Production-Ready API**: Fully asynchronous **FastAPI** backend with endpoints for standard legal queries and document-based auditing.
- **Ragas Evaluation Suite**: An automated offline audit pipeline using **GPT-4.1 mini** as a reasoning judge to measure **Faithfulness** and **Answer Relevancy**.
- **MLOps Observability**: Full lifecycle tracking with **MLflow**, monitoring response latency, source counts, and model parameters.

## 🛠️ Tech Stack
- **LLM**: Google Gemini 2.5 Flash (Generation)
- **Evaluation Judge**: GPT-4.1 mini (Ragas Judge)
- **Reranker**: `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`
- **Vector DBs**: 
  - **ChromaDB**: Persistent storage for national legislation.
  - **DocArray**: In-memory storage for transient user-uploaded PDFs.
- **API Framework**: FastAPI
- **Experiment Tracking**: MLflow
- **Embeddings**: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- **Frontend**: Streamlit

## 📂 Legal Knowledge Base
The system is pre-indexed with:
- **EU GDPR 2016/679** (The core European Regulation).
- **Law 4624/2019** (Greek supplementary measures).
- **Law 5169/2025** (New provisions for AI and the "108+" Convention).
- **Law 3471/2006** (Data protection in electronic communications).

## 🏗️ Advanced RAG Pipeline
This project moves beyond "naive" RAG by implementing a sophisticated pipeline to handle legal complexity:

1. **Diverse Retrieval (MMR)**: Instead of simple similarity, the system uses MMR to retrieve 24 diverse chunks, ensuring overlapping legal provisions are captured.
2. **Cross-Encoding Re-ranking**: A second-stage scorer re-orders chunks based on their direct semantic relationship to the query, selecting the top 12 for the LLM.
3. **Dynamic Context Merging**: Seamlessly joins persistent law data with temporary user document chunks in a single context window.
4. **Contextual Reasoning**: The LLM is prompted to identify contradictions and synthesize a response that prioritizes statutory law over user-provided text.

## 📊 Evaluation & Monitoring (Ragas + MLflow)
To ensure 0% hallucinations, the system includes a dedicated evaluation module.

### Metrics Logged:
* **Faithfulness**: Validates that the answer is derived solely from the retrieved context.
* **Answer Relevancy**: Evaluates how well the response addresses the specific legal question.
* **System Metrics**: Tracking latency (seconds) and chunk distribution (source counts) via MLflow.


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
OPENAI_API_KEY=your_openai_key_for_evaluation
```

### 3. Launch with Docker
```bash
docker-compose up --build
```

### 4. Access the Services
- **Streamlit App:** http://localhost:8501
- **MLflow Dashboard:** http://localhost:5000
- **FastAPI Swagger Docs:** http://localhost:8000/docs

## 📺 Demo & Validation
Μπορείτε να δείτε αναλυτικά παραδείγματα ερωτήσεων και απαντήσεων, καθώς και τον έλεγχο συμμόρφωσης (Compliance Audit) που πραγματοποιεί το σύστημα, στο αρχείο: 👉 [**Test Scenarios & Showcase**](examples/test_scenarios.md)
