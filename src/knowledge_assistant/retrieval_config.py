"""Validated configuration for retrieval strategies."""

from dataclasses import dataclass
from math import isfinite

from knowledge_assistant.retrieval_interface import (
    RetrievalScoreType,
    RetrievalStrategy,
    get_score_type,
)


@dataclass(frozen=True)
class RetrievalConfig:
    """Strategy-specific retrieval configuration."""

    strategy: RetrievalStrategy = RetrievalStrategy.DENSE

    candidate_k: int = 5

    # BM25 configuration
    sparse_min_score: float | None = None
    bm25_k1: float = 1.5
    bm25_b: float = 0.75

    # Reciprocal rank fusion configuration
    rrf_k: int = 60
    dense_weight: float = 1.0
    sparse_weight: float = 1.0

    # Cross-encoder configuration
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker_min_score: float | None = -9.0
    reranker_device: str = "cpu"
    reranker_batch_size: int = 8
    reranker_max_length: int = 512

    def __post_init__(self) -> None:
        """Validate and normalize retrieval configuration."""

        object.__setattr__(
            self,
            "strategy",
            RetrievalStrategy(self.strategy),
        )

        positive_integer_fields = {
            "candidate_k": self.candidate_k,
            "rrf_k": self.rrf_k,
            "reranker_batch_size": self.reranker_batch_size,
            "reranker_max_length": self.reranker_max_length,
        }

        for name, value in positive_integer_fields.items():
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer.")

        thresholds = {
            "sparse_min_score": self.sparse_min_score,
            "reranker_min_score": self.reranker_min_score,
        }

        for name, value in thresholds.items():
            if value is not None and not isfinite(value):
                raise ValueError(f"{name} must be finite or None.")

        if not isfinite(self.bm25_k1) or self.bm25_k1 <= 0:
            raise ValueError("bm25_k1 must be finite and positive.")

        if not isfinite(self.bm25_b) or not 0 <= self.bm25_b <= 1:
            raise ValueError("bm25_b must be between 0 and 1.")

        weights = {
            "dense_weight": self.dense_weight,
            "sparse_weight": self.sparse_weight,
        }

        for name, value in weights.items():
            if not isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative.")

        if self.dense_weight == 0 and self.sparse_weight == 0:
            raise ValueError("At least one retrieval weight must be positive.")

        string_fields = {
            "reranker_model": self.reranker_model,
            "reranker_device": self.reranker_device,
        }

        for name, value in string_fields.items():
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string.")

            object.__setattr__(
                self,
                name,
                value.strip(),
            )

    @property
    def requires_dense(self) -> bool:
        """Whether dense retrieval infrastructure is needed."""

        return self.strategy in {
            RetrievalStrategy.DENSE,
            RetrievalStrategy.HYBRID,
            RetrievalStrategy.DENSE_RERANKED,
        }

    @property
    def requires_sparse(self) -> bool:
        """Whether lexical retrieval infrastructure is needed."""

        return self.strategy in {
            RetrievalStrategy.BM25,
            RetrievalStrategy.HYBRID,
        }

    @property
    def requires_reranker(self) -> bool:
        """Whether cross-encoder inference is needed."""

        return self.strategy == RetrievalStrategy.DENSE_RERANKED

    @property
    def score_type(self) -> RetrievalScoreType:
        """Semantic meaning of the final retrieval scores."""

        return get_score_type(self.strategy)
