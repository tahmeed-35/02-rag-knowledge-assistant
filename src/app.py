"""FastAPI server for the Technical Knowledge Assistant RAG service."""

import os
from pathlib import Path
from fastapi import FastAPI, HTTPException, status
from src.config import settings
from src.schemas import (
    QueryRequest,
    QueryResponse,
    IngestFileRequest,
    IngestResponse,
    HealthResponse,
)
from src.retriever import TechnicalKnowledgeRetriever
from src.ingest import IngestionEngine

app = FastAPI(
    title=settings.app_name,
    version=settings.version,
    description="RAG-Based Technical Knowledge Assistant with ChromaDB and Confidence Refusal.",
)

retriever = TechnicalKnowledgeRetriever()
ingestion_engine = IngestionEngine()


@app.get("/", tags=["General"])
def root():
    return {
        "service": settings.app_name,
        "version": settings.version,
        "docs_url": "/docs",
        "endpoints": {
            "health": "/health",
            "ask": "/ask",
            "ingest": "/ingest",
        },
    }


@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
def health_check():
    total_chunks = retriever.collection.count()
    return HealthResponse(
        status="healthy",
        version=settings.version,
        collection_name=settings.collection_name,
        total_chunks=total_chunks,
    )


@app.post("/ask", response_model=QueryResponse, tags=["Retrieval"])
def ask_question(request: QueryRequest):
    """Answers a technical question using semantic search over indexed documents with source citations."""
    try:
        response = retriever.answer_query(
            query=request.query,
            top_k=request.top_k,
            min_confidence=request.min_confidence,
        )
        return response
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error executing retrieval: {str(exc)}",
        )


@app.post("/ingest", response_model=IngestResponse, tags=["Ingestion"])
def ingest_file(request: IngestFileRequest):
    """Ingests a markdown, text, or PDF file into the vector database."""
    file_path = Path(request.file_path)
    if not file_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"File not found: {request.file_path}",
        )

    try:
        chunks_count = ingestion_engine.ingest_file(str(file_path), request.metadata)
        return IngestResponse(
            status="success",
            document_name=file_path.name,
            chunks_created=chunks_count,
            message=f"Successfully indexed {chunks_count} chunks into collection '{settings.collection_name}'.",
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to ingest file: {str(exc)}",
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.app:app", host="0.0.0.0", port=8000, reload=True)
