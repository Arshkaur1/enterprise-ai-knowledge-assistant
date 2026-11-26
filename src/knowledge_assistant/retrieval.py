import json
from dataclasses import dataclass
from pathlib import Path

from knowledge_assistant.embeddings import EmbeddingService
from knowledge_assistant.indexing import calculate_checksum
from knowledge_assistant.models import DocumentChunk
from knowledge_assistant.vector_store import VectorStore


@dataclass
class SearchResult:
    rank: int
    score: float
    chunk: DocumentChunk


class SemanticRetriever:
    """Retrieve relevant document chunks using FAISS."""

    def __init__(
        self,
        artifact_dir: str | Path,
        embedding_service: EmbeddingService,
    ):
        self.artifact_dir = Path(artifact_dir)
        self.embedding_service = embedding_service

        manifest_path = self.artifact_dir / "index_manifest.json"

        index_path = self.artifact_dir / "vector_index.faiss"

        metadata_path = self.artifact_dir / "chunk_metadata.json"

        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        self._validate_manifest()

        if calculate_checksum(index_path) != self.manifest["index_checksum"]:
            raise ValueError("FAISS index checksum mismatch.")

        if calculate_checksum(metadata_path) != self.manifest["metadata_checksum"]:
            raise ValueError("Metadata checksum mismatch.")

        self.store = VectorStore.load(index_path)

        records = json.loads(metadata_path.read_text(encoding="utf-8"))

        self.metadata = [DocumentChunk(**record) for record in records]

        self._validate_index()

    def _validate_manifest(self) -> None:
        """Validate index configuration compatibility."""

        if self.manifest["schema_version"] != 1:
            raise ValueError("Unsupported manifest version.")

        if self.manifest["embedding_model"] != self.embedding_service.model_name:
            raise ValueError("Embedding model mismatch.")

        if self.manifest["embedding_dimension"] != self.embedding_service.dimension:
            raise ValueError("Embedding dimension mismatch.")

        if self.manifest["index_type"] != "IndexFlatIP":
            raise ValueError("Unsupported index configuration.")

        if not self.manifest["normalized_embeddings"]:
            raise ValueError("Expected normalized embeddings.")

    def _validate_index(self) -> None:
        """Validate vector and metadata consistency."""

        if self.store.size != len(self.metadata):
            raise ValueError("Index and metadata count mismatch.")

        if self.store.size != self.manifest["chunk_count"]:
            raise ValueError("Manifest chunk count mismatch.")

        if self.store.dimension != self.embedding_service.dimension:
            raise ValueError("FAISS dimension mismatch.")

    def search(
        self,
        query: str,
        top_k: int = 5,
        min_score: float | None = None,
    ) -> list[SearchResult]:
        """Return ranked document chunks."""

        if top_k <= 0:
            raise ValueError("top_k must be positive.")

        if min_score is not None and not -1 <= min_score <= 1:
            raise ValueError("min_score must be between -1 and 1.")

        query_embedding = self.embedding_service.embed_query(query)

        scores, indices = self.store.search(
            query_embedding,
            top_k=top_k,
        )

        results = []

        for score, index in zip(
            scores,
            indices,
            strict=True,
        ):
            score = float(score)
            index = int(index)

            if index < 0:
                continue

            if min_score is not None and score < min_score:
                continue

            results.append(
                SearchResult(
                    rank=len(results) + 1,
                    score=score,
                    chunk=self.metadata[index],
                )
            )

        return results
