"""Document parsing, text chunking, and ChromaDB vector store ingestion."""

import os
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings as ChromaSettings
from src.config import settings


def read_document_content(file_path: str) -> str:
    """Reads content from markdown, text, or PDF files."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    ext = path.suffix.lower()
    if ext in [".md", ".txt", ".markdown"]:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    elif ext == ".pdf":
        try:
            from pypdf import PdfReader
            reader = PdfReader(str(path))
            pages = [page.extract_text() or "" for page in reader.pages]
            return "\n\n".join(pages)
        except ImportError:
            raise ImportError("pypdf is required to process PDF files. Run: pip install pypdf")
    else:
        raise ValueError(f"Unsupported file format: {ext}")


def split_text_into_chunks(
    text: str,
    chunk_size: int = 400,
    chunk_overlap: int = 50,
    source_name: str = "document"
) -> List[Dict[str, Any]]:
    """Splits text into overlapping chunks, tracking sections where feasible."""
    paragraphs = text.split("\n\n")
    chunks: List[Dict[str, Any]] = []
    
    current_section = "General"
    current_chunk: List[str] = []
    current_length = 0
    chunk_idx = 0

    for para in paragraphs:
        para_clean = para.strip()
        if not para_clean:
            continue

        # Detect markdown headers
        header_match = re.match(r"^(#{1,6})\s+(.*)$", para_clean)
        if header_match:
            current_section = header_match.group(2).strip()

        words = para_clean.split()
        if current_length + len(words) > chunk_size and current_chunk:
            chunk_text = " ".join(current_chunk)
            chunks.append({
                "chunk_id": f"{source_name}_chunk_{chunk_idx}",
                "text": chunk_text,
                "section": current_section,
                "chunk_index": chunk_idx,
            })
            chunk_idx += 1
            # Keep overlap words
            overlap_words = current_chunk[-chunk_overlap:] if chunk_overlap < len(current_chunk) else current_chunk
            current_chunk = list(overlap_words)
            current_length = len(current_chunk)

        current_chunk.extend(words)
        current_length += len(words)

    if current_chunk:
        chunk_text = " ".join(current_chunk)
        chunks.append({
            "chunk_id": f"{source_name}_chunk_{chunk_idx}",
            "text": chunk_text,
            "section": current_section,
            "chunk_index": chunk_idx,
        })

    return chunks


class IngestionEngine:
    """Manages document chunking and vector storage into ChromaDB."""

    def __init__(self, persist_dir: Optional[str] = None, collection_name: Optional[str] = None):
        self.persist_dir = persist_dir or settings.chroma_persist_dir
        self.collection_name = collection_name or settings.collection_name
        self.client = chromadb.PersistentClient(path=self.persist_dir)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def ingest_file(self, file_path: str, extra_metadata: Optional[Dict[str, Any]] = None) -> int:
        """Ingests a single document file into ChromaDB."""
        path = Path(file_path)
        content = read_document_content(str(path))
        chunks = split_text_into_chunks(
            content,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            source_name=path.stem
        )

        if not chunks:
            return 0

        ids = []
        documents = []
        metadatas = []

        for ch in chunks:
            ids.append(ch["chunk_id"])
            documents.append(ch["text"])
            meta = {
                "source": path.name,
                "section": ch["section"],
                "chunk_index": ch["chunk_index"],
            }
            if extra_metadata:
                meta.update(extra_metadata)
            metadatas.append(meta)

        self.collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas
        )

        return len(chunks)

    def ingest_directory(self, dir_path: str) -> Dict[str, int]:
        """Ingests all supported documents in a directory."""
        directory = Path(dir_path)
        results = {}
        for ext in ["*.md", "*.txt", "*.pdf"]:
            for file_path in directory.glob(ext):
                count = self.ingest_file(str(file_path))
                results[file_path.name] = count
        return results
