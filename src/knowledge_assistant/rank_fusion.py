import math
from collections import defaultdict

from knowledge_assistant.models import DocumentChunk
from knowledge_assistant.retrieval import SearchResult


def reciprocal_rank_fusion(
    rankings: list[list[SearchResult]],
    top_k: int = 5,
    k: int = 60,
    weights: list[float] | None = None,
) -> list[SearchResult]:
    """Fuse ranked retrieval results using weighted RRF."""

    if type(top_k) is not int or top_k < 1:
        raise ValueError("top_k must be a positive integer.")

    if type(k) is not int or k < 1:
        raise ValueError("k must be a positive integer.")

    if weights is None:
        weights = [1.0] * len(rankings)

    if len(weights) != len(rankings):
        raise ValueError("Each ranking must have a corresponding weight.")

    if any(not math.isfinite(weight) or weight < 0 for weight in weights):
        raise ValueError("Weights must be finite and non-negative.")

    if rankings and not any(weight > 0 for weight in weights):
        raise ValueError("At least one weight must be positive.")

    fused_scores: dict[str, float] = defaultdict(float)
    chunks: dict[str, DocumentChunk] = {}

    for ranking, weight in zip(rankings, weights):
        if weight == 0:
            continue

        seen_ids = set()
        seen_ranks = set()

        for result in ranking:
            if type(result.rank) is not int or result.rank < 1:
                raise ValueError("Ranks must be positive integers.")

            chunk_id = result.chunk.chunk_id

            if chunk_id in seen_ids:
                raise ValueError(f"Duplicate chunk in ranking: {chunk_id}")

            if result.rank in seen_ranks:
                raise ValueError(f"Duplicate rank: {result.rank}")

            seen_ids.add(chunk_id)
            seen_ranks.add(result.rank)

            if chunk_id in chunks and chunks[chunk_id] != result.chunk:
                raise ValueError(f"Inconsistent chunk data: {chunk_id}")

            chunks[chunk_id] = result.chunk

            fused_scores[chunk_id] += weight / (k + result.rank)

    ordered_ids = sorted(
        fused_scores,
        key=lambda chunk_id: (
            -fused_scores[chunk_id],
            chunk_id,
        ),
    )

    return [
        SearchResult(
            rank=rank,
            score=fused_scores[chunk_id],
            chunk=chunks[chunk_id],
        )
        for rank, chunk_id in enumerate(
            ordered_ids[:top_k],
            start=1,
        )
    ]
