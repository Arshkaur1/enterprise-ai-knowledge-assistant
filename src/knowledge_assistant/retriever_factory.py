"""Factory for configurable retrieval strategies."""

import hashlib
import json
from pathlib import Path

from knowledge_assistant.config import RAGConfig
from knowledge_assistant.models import DocumentChunk
from knowledge_assistant.retrieval_interface import (
    RetrievalStrategy,
    RetrieverProtocol,
    ThresholdRetrieverAdapter,
)


def load_artifact_chunks(
    artifact_dir: Path,
) -> list[DocumentChunk]:
    manifest = json.loads(
        (artifact_dir / "index_manifest.json").read_text(encoding="utf-8")
    )

    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported manifest version.")

    raw = (artifact_dir / "chunk_metadata.json").read_bytes()

    if hashlib.sha256(raw).hexdigest() != manifest["metadata_checksum"]:
        raise ValueError("Metadata checksum mismatch.")

    records = json.loads(raw)

    if not isinstance(records, list) or not records:
        raise ValueError("Invalid chunk metadata.")

    chunks = [DocumentChunk(**record) for record in records]

    if len(chunks) != manifest["chunk_count"]:
        raise ValueError("Manifest chunk count mismatch.")

    if len({c.chunk_id for c in chunks}) != len(chunks):
        raise ValueError("Duplicate chunk IDs.")

    return chunks


class RetrieverFactory:
    """Construct the selected retrieval pipeline."""

    @staticmethod
    def create(
        artifact_dir: str | Path,
        config: RAGConfig | None = None,
        embedding_model: str | None = None,
    ) -> RetrieverProtocol:
        config = config or RAGConfig()
        settings = config.retrieval
        strategy = settings.strategy
        artifact_dir = Path(artifact_dir)

        dense = None
        sparse = None

        if settings.requires_dense:
            from knowledge_assistant.embeddings import (
                EmbeddingService,
            )
            from knowledge_assistant.retrieval import (
                SemanticRetriever,
            )

            embedder = (
                EmbeddingService(model_name=embedding_model)
                if embedding_model is not None
                else EmbeddingService()
            )

            dense = SemanticRetriever(
                artifact_dir=artifact_dir,
                embedding_service=embedder,
            )

        if settings.requires_sparse:
            from knowledge_assistant.lexical_retrieval import (
                BM25Retriever,
            )

            chunks = (
                dense.metadata
                if dense is not None
                else load_artifact_chunks(artifact_dir)
            )

            sparse = BM25Retriever(
                chunks=chunks,
                k1=settings.bm25_k1,
                b=settings.bm25_b,
            )

        if strategy == RetrievalStrategy.DENSE:
            if dense is None:
                raise ValueError("Dense retrieval is required for dense retrieval.")

            return ThresholdRetrieverAdapter(
                retriever=dense,
                min_score=config.min_score,
            )

        if strategy == RetrievalStrategy.BM25:
            if sparse is None:
                raise ValueError("Sparse retrieval is required for BM25 retrieval.")

            return ThresholdRetrieverAdapter(
                retriever=sparse,
                min_score=settings.sparse_min_score,
            )

        if strategy == RetrievalStrategy.HYBRID:
            from knowledge_assistant.hybrid_retrieval import (
                HybridRetriever,
            )

            if dense is None:
                raise ValueError("Dense retrieval is required for hybrid retrieval.")
            if sparse is None:
                raise ValueError("Sparse retrieval is required for hybrid retrieval.")

            return HybridRetriever(
                dense_retriever=dense,
                sparse_retriever=sparse,
                candidate_k=settings.candidate_k,
                rrf_k=settings.rrf_k,
                dense_weight=settings.dense_weight,
                sparse_weight=settings.sparse_weight,
                dense_min_score=config.min_score,
                sparse_min_score=settings.sparse_min_score,
            )

        if strategy == RetrievalStrategy.DENSE_RERANKED:
            from knowledge_assistant.reranked_retrieval import (
                RerankedRetriever,
            )
            from knowledge_assistant.reranking import (
                CrossEncoderReranker,
            )

            if dense is None:
                raise ValueError("Dense retrieval is required for reranking.")

            reranker = CrossEncoderReranker(
                model_name=settings.reranker_model,
                device=settings.reranker_device,
                batch_size=settings.reranker_batch_size,
                max_length=settings.reranker_max_length,
            )

            return RerankedRetriever(
                retriever=dense,
                reranker=reranker,
                candidate_k=settings.candidate_k,
                dense_min_score=config.min_score,
                reranker_min_score=settings.reranker_min_score,
            )

        raise ValueError(f"Unsupported strategy: {strategy}")
