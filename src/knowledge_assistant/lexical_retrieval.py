import math
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from knowledge_assistant.indexing import load_chunks
from knowledge_assistant.models import DocumentChunk
from knowledge_assistant.retrieval import SearchResult
from knowledge_assistant.tokenization import tokenize


class BM25Retriever:
    """In-memory lexical retriever using BM25."""

    def __init__(
        self,
        chunks: list[DocumentChunk],
        k1: float = 1.5,
        b: float = 0.75,
    ):

        if not chunks:
            raise ValueError("At least one document chunk is required.")

        if not math.isfinite(k1) or k1 <= 0:
            raise ValueError("k1 must be finite and positive.")

        if not math.isfinite(b) or not 0 <= b <= 1:
            raise ValueError("b must be between 0 and 1.")

        chunk_ids = [chunk.chunk_id for chunk in chunks]

        if len(set(chunk_ids)) != len(chunk_ids):
            raise ValueError("Duplicate chunk IDs are not allowed.")

        self.chunks = list(chunks)
        self.k1 = k1
        self.b = b

        self.tokenized_corpus = [tokenize(chunk.text) for chunk in self.chunks]

        if any(not tokens for tokens in self.tokenized_corpus):
            raise ValueError("Every chunk must contain searchable tokens.")

        self.index = BM25Okapi(
            self.tokenized_corpus,
            k1=self.k1,
            b=self.b,
        )

        self.corpus_token_sets = [set(tokens) for tokens in self.tokenized_corpus]

    @property
    def chunk_count(self) -> int:
        return len(self.chunks)

    @classmethod
    def from_jsonl(
        cls,
        chunks_path: str | Path,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> "BM25Retriever":

        chunks = load_chunks(Path(chunks_path))

        return cls(
            chunks=chunks,
            k1=k1,
            b=b,
        )

    def search(
        self,
        query: str,
        top_k: int = 5,
        min_score: float | None = None,
    ) -> list[SearchResult]:

        if not isinstance(query, str):
            raise TypeError("Query must be a string.")

        if not query.strip():
            raise ValueError("Query cannot be empty.")

        if type(top_k) is not int or top_k < 1:
            raise ValueError("top_k must be a positive integer.")

        if min_score is not None and not math.isfinite(min_score):
            raise ValueError("min_score must be finite.")

        query_tokens = list(dict.fromkeys(tokenize(query)))

        if not query_tokens:
            return []

        scores = np.asarray(
            self.index.get_scores(query_tokens),
            dtype=np.float64,
        )

        query_token_set = set(query_tokens)

        candidates = []

        for index, score in enumerate(scores):
            if not query_token_set.intersection(self.corpus_token_sets[index]):
                continue

            if not np.isfinite(score):
                continue

            if min_score is not None and score < min_score:
                continue

            candidates.append((index, float(score)))

        candidates.sort(
            key=lambda item: (
                -item[1],
                self.chunks[item[0]].chunk_id,
            )
        )

        return [
            SearchResult(
                rank=rank,
                score=score,
                chunk=self.chunks[index],
            )
            for rank, (index, score) in enumerate(
                candidates[:top_k],
                start=1,
            )
        ]
