import math

from knowledge_assistant.lexical_retrieval import (
    BM25Retriever,
)
from knowledge_assistant.rank_fusion import (
    reciprocal_rank_fusion,
)
from knowledge_assistant.retrieval import (
    SearchResult,
    SemanticRetriever,
)


class HybridRetriever:
    """Combine semantic and lexical retrieval using RRF."""

    def __init__(
        self,
        dense_retriever: SemanticRetriever,
        sparse_retriever: BM25Retriever,
        candidate_k: int = 5,
        rrf_k: int = 60,
        dense_weight: float = 1.0,
        sparse_weight: float = 1.0,
        dense_min_score: float | None = 0.35,
        sparse_min_score: float | None = None,
    ):

        if type(candidate_k) is not int or candidate_k < 1:
            raise ValueError("candidate_k must be a positive integer.")

        if type(rrf_k) is not int or rrf_k < 1:
            raise ValueError("rrf_k must be a positive integer.")

        weights = [dense_weight, sparse_weight]

        if any(not math.isfinite(weight) or weight < 0 for weight in weights):
            raise ValueError("Weights must be finite and non-negative.")

        if not any(weight > 0 for weight in weights):
            raise ValueError("At least one weight must be positive.")

        if dense_min_score is not None and (
            not math.isfinite(dense_min_score) or not -1 <= dense_min_score <= 1
        ):
            raise ValueError("Dense threshold must be between -1 and 1.")

        if sparse_min_score is not None and not math.isfinite(sparse_min_score):
            raise ValueError("Sparse threshold must be finite.")

        self.dense_retriever = dense_retriever
        self.sparse_retriever = sparse_retriever

        self.candidate_k = candidate_k
        self.rrf_k = rrf_k

        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight

        self.dense_min_score = dense_min_score
        self.sparse_min_score = sparse_min_score

    def search_candidates(
        self,
        query: str,
        candidate_k: int | None = None,
    ) -> dict[str, list[SearchResult]]:

        if not isinstance(query, str):
            raise TypeError("Query must be a string.")

        query = query.strip()

        if not query:
            raise ValueError("Query cannot be empty.")

        if candidate_k is None:
            candidate_k = self.candidate_k

        if type(candidate_k) is not int or candidate_k < 1:
            raise ValueError("candidate_k must be a positive integer.")

        dense_results = []

        if self.dense_weight > 0:
            dense_results = self.dense_retriever.search(
                query=query,
                top_k=candidate_k,
                min_score=self.dense_min_score,
            )

        sparse_results = []

        if self.sparse_weight > 0:
            sparse_results = self.sparse_retriever.search(
                query=query,
                top_k=candidate_k,
                min_score=self.sparse_min_score,
            )

        return {
            "dense": dense_results,
            "sparse": sparse_results,
        }

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[SearchResult]:

        if type(top_k) is not int or top_k < 1:
            raise ValueError("top_k must be a positive integer.")

        candidates = self.search_candidates(
            query=query,
            candidate_k=max(
                top_k,
                self.candidate_k,
            ),
        )

        return reciprocal_rank_fusion(
            rankings=[
                candidates["dense"],
                candidates["sparse"],
            ],
            top_k=top_k,
            k=self.rrf_k,
            weights=[
                self.dense_weight,
                self.sparse_weight,
            ],
        )
