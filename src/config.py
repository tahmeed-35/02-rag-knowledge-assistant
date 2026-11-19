"""Configuration settings for the RAG Knowledge Assistant."""

import os
from pydantic import BaseModel, Field


class Settings(BaseModel):
    app_name: str = "Technical Knowledge Assistant"
    version: str = "0.1.0"
    chroma_persist_dir: str = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
    collection_name: str = os.getenv("CHROMA_COLLECTION", "tech_docs")
    embedding_model_name: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "400"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "50"))
    confidence_threshold: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.60"))
    top_k: int = int(os.getenv("TOP_K", "4"))


settings = Settings()
