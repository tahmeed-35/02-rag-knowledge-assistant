"""Retriever and answer generator with confidence threshold refusal and source citations."""

from typing import List, Tuple, Optional
import chromadb
from src.config import settings
from src.schemas import Citation, QueryResponse


class TechnicalKnowledgeRetriever:
    """Handles semantic search over indexed documents and applies refusal thresholds."""

    def __init__(self, persist_dir: Optional[str] = None, collection_name: Optional[str] = None):
        self.persist_dir = persist_dir or settings.chroma_persist_dir
        self.collection_name = collection_name or settings.collection_name
        self.client = chromadb.PersistentClient(path=self.persist_dir)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Citation]:
        """Queries ChromaDB and calculates cosine similarity scores."""
        k = top_k or settings.top_k
        count = self.collection.count()
        if count == 0:
            return []

        k = min(k, count)
        results = self.collection.query(
            query_texts=[query],
            n_results=k,
            include=["documents", "metadatas", "distances"]
        )

        citations: List[Citation] = []
        if not results["ids"] or not results["ids"][0]:
            return citations

        for chunk_id, doc, meta, dist in zip(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            # Distance in cosine space is between 0 (identical) and 2 (opposite)
            similarity = max(0.0, min(1.0, 1.0 - (dist / 2.0)))
            snippet = (doc[:200] + "...") if len(doc) > 200 else doc
            citations.append(
                Citation(
                    source=meta.get("source", "unknown"),
                    section=meta.get("section", "General"),
                    chunk_id=chunk_id,
                    similarity_score=round(similarity, 3),
                    snippet=snippet,
                )
            )

        # Sort highest similarity first
        citations.sort(key=lambda x: x.similarity_score, reverse=True)
        return citations

    def answer_query(
        self,
        query: str,
        top_k: Optional[int] = None,
        min_confidence: Optional[float] = None
    ) -> QueryResponse:
        """Retrieves context, evaluates confidence against threshold, and forms answer or refusal."""
        threshold = min_confidence if min_confidence is not None else settings.confidence_threshold
        citations = self.retrieve(query, top_k=top_k)

        if not citations:
            return QueryResponse(
                query=query,
                answer="No relevant documentation found in knowledge base.",
                refusal=True,
                refusal_reason="Empty retrieval result or empty knowledge base.",
                confidence=0.0,
                citations=[],
            )

        top_confidence = citations[0].similarity_score

        # Check refusal condition
        if top_confidence < threshold:
            return QueryResponse(
                query=query,
                answer=(
                    f"I cannot reliably answer this question based on current technical documentation. "
                    f"The highest retrieval confidence ({top_confidence:.2f}) is below the required threshold ({threshold:.2f})."
                ),
                refusal=True,
                refusal_reason=f"Top confidence {top_confidence:.2f} is below threshold {threshold:.2f}",
                confidence=top_confidence,
                citations=citations,
            )

        # Answer generation with inline citations
        relevant_snippets = [
            f"[{c.source} | {c.section}] {c.snippet}" for c in citations if c.similarity_score >= threshold
        ]
        context_block = "\n".join(relevant_snippets)
        answer = (
            f"Based on the technical documentation:\n\n{context_block}\n\n"
            f"Sources cited: {', '.join(set(c.source for c in citations if c.similarity_score >= threshold))}."
        )

        return QueryResponse(
            query=query,
            answer=answer,
            refusal=False,
            refusal_reason=None,
            confidence=top_confidence,
            citations=citations,
        )
