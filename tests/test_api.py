"""API tests for FastAPI endpoints in knowledge assistant."""

import pytest
from src.schemas import QueryRequest, QueryResponse, Citation


def test_query_schema_validation():
    req = QueryRequest(query="What is vector search?", top_k=5)
    assert req.query == "What is vector search?"
    assert req.top_k == 5


def test_query_response_schema():
    citation = Citation(
        source="test.md",
        section="Architecture",
        chunk_id="test_0",
        similarity_score=0.88,
        snippet="Snippet text",
    )
    res = QueryResponse(
        query="test query",
        answer="test answer",
        refusal=False,
        confidence=0.88,
        citations=[citation],
    )
    assert res.confidence == 0.88
    assert len(res.citations) == 1
    assert res.citations[0].source == "test.md"
