from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import Optional
import tempfile
import os
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from core.rag_engine import get_answer_with_context

app = FastAPI(
    title="GDPR Greece AI Auditor",
    description="RAG-powered API for Greek and EU data protection law compliance",
    version="1.0.0"
)

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str
    sources: list[str]

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/audit", response_model=QueryResponse)
def audit(request: QueryRequest):
    try:
        result = get_answer_with_context(request.question)
        return QueryResponse(answer=result["answer"], sources=result["sources"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/audit-with-document", response_model=QueryResponse)
async def audit_with_document(
    question: str,
    file: UploadFile = File(...)
):
    try:
        # Save uploaded PDF temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        # Process PDF
        loader = PyMuPDFLoader(tmp_path)
        raw_docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
        processed_user_docs = splitter.split_documents(raw_docs)
        os.remove(tmp_path)

        result = get_answer_with_context(question, user_docs=processed_user_docs)
        return QueryResponse(answer=result["answer"], sources=result["sources"])

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))