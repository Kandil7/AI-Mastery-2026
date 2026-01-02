
import sys
import os
import logging
from typing import List, Optional
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
import uvicorn
from contextlib import asynccontextmanager

# Add project root to path
# This allows running 'python sprints/week01_rag_production/api.py' from project root
# or from within the folder if we handle paths correctly
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../.."))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.llm.rag import RAGModel, RetrievalStrategy, Document

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Data Models ---
class DocumentRequest(BaseModel):
    id: str
    content: str
    metadata: Optional[dict] = {}

class QueryRequest(BaseModel):
    query: str
    k: int = 3

class SourceDocument(BaseModel):
    id: str
    content: str
    score: float
    rank: int

class QueryResponse(BaseModel):
    query: str
    response: str
    sources: List[SourceDocument]

# --- Global State ---
rag_model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize RAG model
    global rag_model
    logger.info("Initializing Hybrid RAG Model...")
    
    # In a real app, we'd load a persisted index here.
    # For this demo, we initialize it fresh.
    rag_model = RAGModel(
        retriever_strategy=RetrievalStrategy.HYBRID,
        generator_model="gpt2",  # Using small model for demo speed
        dense_weight=0.7,
        sparse_weight=0.3
    )
    
    # Pre-index some sample data for immediate usage
    sample_docs = [
        Document("1", "RAG facilitates searching a large corpus of data.", {"source": "manual"}),
        Document("2", "Hybrid retrieval combines keyword and semantic search.", {"source": "manual"}),
        Document("3", "FastAPI is a modern web framework for building APIs with Python.", {"source": "manual"}),
    ]
    rag_model.add_documents(sample_docs)
    logger.info(f"Indexed {len(sample_docs)} sample documents.")
    
    yield
    
    # Shutdown
    logger.info("Function shut down.")

app = FastAPI(title="Week 1 RAG Demo", lifespan=lifespan)

@app.get("/")
def read_root():
    return {"status": "ok", "message": "Week 1 RAG API is running"}

@app.post("/index")
def add_documents(docs: List[DocumentRequest]):
    global rag_model
    if not rag_model:
        raise HTTPException(status_code=503, detail="Model not initialized")
    
    domain_docs = [Document(d.id, d.content, d.metadata) for d in docs]
    rag_model.add_documents(domain_docs)
    
    return {"message": f"Added {len(docs)} documents", "total_docs": len(rag_model.documents)}

@app.post("/query", response_model=QueryResponse)
def query_rag(request: QueryRequest):
    global rag_model
    if not rag_model:
        raise HTTPException(status_code=503, detail="Model not initialized")
    
    result = rag_model.query(request.query, k=request.k)
    
    # Map to response model
    sources = [
        SourceDocument(
            id=d['id'],
            content=d['content'],
            score=d['score'],
            rank=d['rank']
        ) for d in result['retrieved_documents']
    ]
    
    return QueryResponse(
        query=result['query'],
        response=result['response'],
        sources=sources
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
