"""Request and Response Pydantic Schemas for RAG Knowledge Assistant."""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=2, description="Question to ask the knowledge assistant")
    top_k: Optional[int] = Field(None, ge=1, le=20, description="Number of document chunks to retrieve")
    min_confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="Override confidence threshold")


class Citation(BaseModel):
    source: str = Field(..., description="Document filename or source URI")
    section: Optional[str] = Field(None, description="Section heading or page number")
    chunk_id: str = Field(..., description="Unique chunk identifier")
    similarity_score: float = Field(..., description="Calculated similarity score (0.0 to 1.0)")
    snippet: str = Field(..., description="Excerpt of retrieved text")


class QueryResponse(BaseModel):
    query: str
    answer: str
    refusal: bool = False
    refusal_reason: Optional[str] = None
    confidence: float = Field(..., description="Highest similarity score among retrieved chunks")
    citations: List[Citation] = Field(default_factory=list)


class IngestFileRequest(BaseModel):
    file_path: str = Field(..., description="Relative or absolute path to Markdown/PDF document")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)


class IngestResponse(BaseModel):
    status: str
    document_name: str
    chunks_created: int
    message: str


class HealthResponse(BaseModel):
    status: str
    version: str
    collection_name: str
    total_chunks: int
