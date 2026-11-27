"""Cross-encoder reranking for retrieved document candidates."""

import math

import numpy as np
from sentence_transformers import CrossEncoder
from torch.nn import Identity

from knowledge_assistant.retrieval import SearchResult

DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class CrossEncoderReranker:
    """Rerank retrieved chunks using joint query-document scoring."""

    def __init__(
        self,
        model_name: str = DEFAULT_RERANKER_MODEL,
        device: str = "cpu",
        batch_size: int = 8,
        max_length: int = 512,
    ) -> None:
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError("Model name cannot be empty.")

        if not isinstance(device, str) or not device.strip():
            raise ValueError("Device cannot be empty.")

        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("Batch size must be a positive integer.")

        if type(max_length) is not int or max_length < 1:
            raise ValueError("Maximum length must be a positive integer.")

        self.model_name = model_name.strip()
        self.device = device.strip()
        self.batch_size = batch_size
        self.max_length = max_length

        self.model = CrossEncoder(
            self.model_name,
            device=self.device,
            max_length=self.max_length,
        )

    def rerank(
        self,
        query: str,
        candidates: list[SearchResult],
        top_k: int = 2,
        min_score: float | None = None,
    ) -> list[SearchResult]:
        """Score, filter and rerank retrieved document chunks."""

        if not isinstance(query, str):
            raise TypeError("Query must be a string.")

        query = query.strip()

        if not query:
            raise ValueError("Query cannot be empty.")

        if type(top_k) is not int or top_k < 1:
            raise ValueError("top_k must be a positive integer.")

        if min_score is not None and not math.isfinite(min_score):
            raise ValueError("Minimum score must be finite.")

        if not candidates:
            return []

        seen_ids: set[str] = set()

        for candidate in candidates:
            chunk_id = candidate.chunk.chunk_id

            if chunk_id in seen_ids:
                raise ValueError(f"Duplicate candidate chunk: {chunk_id}")

            if not candidate.chunk.text.strip():
                raise ValueError(f"Candidate text cannot be empty: {chunk_id}")

            seen_ids.add(chunk_id)

        pairs = [(query, candidate.chunk.text) for candidate in candidates]

        raw_scores = self.model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
            activation_fn=Identity(),
            convert_to_numpy=True,
        )

        scores = np.asarray(raw_scores, dtype=np.float64)

        if scores.shape != (len(candidates),):
            raise ValueError(f"Unexpected cross-encoder score shape: {scores.shape}")

        if not np.all(np.isfinite(scores)):
            raise ValueError("Cross-encoder returned non-finite scores.")

        scored_candidates = [
            (candidate, float(score))
            for candidate, score in zip(
                candidates,
                scores,
                strict=True,
            )
            if min_score is None or score >= min_score
        ]

        scored_candidates.sort(
            key=lambda item: (
                -item[1],
                item[0].chunk.chunk_id,
            )
        )

        return [
            SearchResult(
                rank=rank,
                score=score,
                chunk=candidate.chunk,
            )
            for rank, (candidate, score) in enumerate(
                scored_candidates[:top_k],
                start=1,
            )
        ]
