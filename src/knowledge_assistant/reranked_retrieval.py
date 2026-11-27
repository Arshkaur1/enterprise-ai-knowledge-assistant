"""Retrieval pipeline with optional cross-encoder reranking."""

import math
from typing import Protocol

from knowledge_assistant.hybrid_retrieval import HybridRetriever
from knowledge_assistant.retrieval import SearchResult, SemanticRetriever


class RerankerProtocol(Protocol):
    """Interface implemented by document reranking services."""

    def rerank(
        self,
        query: str,
        candidates: list[SearchResult],
        top_k: int = 2,
        min_score: float | None = None,
    ) -> list[SearchResult]:
        """Return ranked document candidates."""
        ...


class RerankedRetriever:
    """Compose candidate retrieval with cross-encoder reranking."""

    def __init__(
        self,
        retriever: SemanticRetriever | HybridRetriever,
        reranker: RerankerProtocol,
        candidate_k: int = 5,
        dense_min_score: float | None = 0.35,
        reranker_min_score: float | None = None,
    ) -> None:
        if type(candidate_k) is not int or candidate_k < 1:
            raise ValueError("candidate_k must be a positive integer.")

        if dense_min_score is not None and (
            not math.isfinite(dense_min_score) or not -1 <= dense_min_score <= 1
        ):
            raise ValueError("Dense minimum score must be between -1 and 1.")

        if reranker_min_score is not None and (not math.isfinite(reranker_min_score)):
            raise ValueError("Reranker minimum score must be finite.")

        if not isinstance(
            retriever,
            (SemanticRetriever, HybridRetriever),
        ):
            raise TypeError("Unsupported candidate retriever.")

        self.retriever = retriever
        self.reranker = reranker
        self.candidate_k = candidate_k
        self.dense_min_score = dense_min_score
        self.reranker_min_score = reranker_min_score

    def search_candidates(
        self,
        query: str,
        candidate_k: int | None = None,
    ) -> list[SearchResult]:
        """Retrieve candidates before cross-encoder scoring."""

        if not isinstance(query, str):
            raise TypeError("Query must be a string.")

        query = query.strip()

        if not query:
            raise ValueError("Query cannot be empty.")

        if candidate_k is None:
            candidate_k = self.candidate_k

        if type(candidate_k) is not int or candidate_k < 1:
            raise ValueError("candidate_k must be a positive integer.")

        if isinstance(self.retriever, HybridRetriever):
            return self.retriever.search(
                query=query,
                top_k=candidate_k,
            )

        return self.retriever.search(
            query=query,
            top_k=candidate_k,
            min_score=self.dense_min_score,
        )

    def search(
        self,
        query: str,
        top_k: int = 2,
    ) -> list[SearchResult]:
        """Retrieve, rerank and filter document candidates."""

        if type(top_k) is not int or top_k < 1:
            raise ValueError("top_k must be a positive integer.")

        candidates = self.search_candidates(
            query=query,
            candidate_k=max(
                top_k,
                self.candidate_k,
            ),
        )

        if not candidates:
            return []

        return self.reranker.rerank(
            query=query.strip(),
            candidates=candidates,
            top_k=top_k,
            min_score=self.reranker_min_score,
        )
