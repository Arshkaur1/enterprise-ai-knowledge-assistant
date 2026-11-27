"""Common interfaces and adapters for retrieval strategies."""

from enum import StrEnum
from math import isfinite
from typing import Protocol, runtime_checkable

from knowledge_assistant.retrieval import SearchResult


class RetrievalStrategy(StrEnum):
    """Supported retrieval strategies."""

    DENSE = "dense"
    BM25 = "bm25"
    HYBRID = "hybrid"
    DENSE_RERANKED = "dense_reranked"


class RetrievalScoreType(StrEnum):
    """Semantic meaning of retrieval scores."""

    COSINE_SIMILARITY = "cosine_similarity"
    BM25 = "bm25"
    RRF = "rrf"
    CROSS_ENCODER_LOGIT = "cross_encoder_logit"


@runtime_checkable
class RetrieverProtocol(Protocol):
    """Common interface for all configured retrievers."""

    def search(
        self,
        query: str,
        top_k: int = 2,
    ) -> list[SearchResult]:
        """Return ranked document chunks."""
        ...


class ThresholdRetrieverProtocol(Protocol):
    """Interface for retrievers supporting score thresholds."""

    def search(
        self,
        query: str,
        top_k: int = 2,
        min_score: float | None = None,
    ) -> list[SearchResult]:
        """Return ranked chunks subject to a score threshold."""
        ...


class ThresholdRetrieverAdapter:
    """Bind a retrieval-specific threshold to a retriever."""

    def __init__(
        self,
        retriever: ThresholdRetrieverProtocol,
        min_score: float | None = None,
    ) -> None:
        if min_score is not None and not isfinite(min_score):
            raise ValueError("Minimum score must be finite.")

        self.retriever = retriever
        self.min_score = min_score

    def search(
        self,
        query: str,
        top_k: int = 2,
    ) -> list[SearchResult]:
        """Retrieve chunks using the configured threshold."""

        return self.retriever.search(
            query=query,
            top_k=top_k,
            min_score=self.min_score,
        )


def get_score_type(
    strategy: RetrievalStrategy,
) -> RetrievalScoreType:
    """Return the score semantics for a retrieval strategy."""

    score_types = {
        RetrievalStrategy.DENSE: (RetrievalScoreType.COSINE_SIMILARITY),
        RetrievalStrategy.BM25: RetrievalScoreType.BM25,
        RetrievalStrategy.HYBRID: RetrievalScoreType.RRF,
        RetrievalStrategy.DENSE_RERANKED: (RetrievalScoreType.CROSS_ENCODER_LOGIT),
    }

    return score_types[RetrievalStrategy(strategy)]
