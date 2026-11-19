"""Unit tests for text chunking and retriever."""

import pytest
from src.ingest import split_text_into_chunks


def test_chunking_with_headers():
    text = """# Header 1
This is the first paragraph under header 1.

## Subheader A
This is text in subheader A discussing microservices.

# Header 2
This is text under header 2.
"""
    chunks = split_text_into_chunks(text, chunk_size=10, chunk_overlap=2, source_name="test_doc")
    assert len(chunks) >= 2
    assert any(c["section"] == "Header 1" or c["section"] == "Subheader A" for c in chunks)
    assert chunks[0]["chunk_id"].startswith("test_doc_chunk_")


def test_empty_text_chunking():
    chunks = split_text_into_chunks("", chunk_size=100)
    assert chunks == []
