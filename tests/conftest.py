import json

import numpy as np
import pytest

from knowledge_assistant.indexing import build_index
from knowledge_assistant.models import DocumentChunk


class MockEmbeddingService:
    """Deterministic embedding service for automated tests."""

    model_name = "mock-embedding-model"
    dimension = 3

    def embed_query(self, query: str) -> np.ndarray:
        query = query.lower().strip()

        if not query:
            raise ValueError("Query cannot be empty.")

        if "remote" in query:
            vector = [1.0, 0.0, 0.0]
        elif "password" in query:
            vector = [0.0, 1.0, 0.0]
        elif "expense" in query:
            vector = [0.0, 0.0, 1.0]
        else:
            vector = [-1.0, 0.0, 0.0]

        return np.array(vector, dtype=np.float32)

    def embed_documents(
        self,
        texts: list[str],
    ) -> np.ndarray:
        if not texts:
            return np.empty(
                (0, self.dimension),
                dtype=np.float32,
            )

        return np.stack([self.embed_query(text) for text in texts])


@pytest.fixture
def mock_embedder():
    return MockEmbeddingService()


@pytest.fixture
def sample_chunks():
    return [
        DocumentChunk(
            chunk_id="hr_chunk_0000",
            document_id="hr",
            source="hr_policy.txt",
            file_type="txt",
            category="HR",
            text="Remote work is allowed three days per week.",
        ),
        DocumentChunk(
            chunk_id="it_chunk_0000",
            document_id="it",
            source="it_security.docx",
            file_type="docx",
            category="IT",
            text="Passwords must contain at least 12 characters.",
        ),
        DocumentChunk(
            chunk_id="finance_chunk_0000",
            document_id="finance",
            source="finance_policy.pdf",
            file_type="pdf",
            category="Finance",
            text="Expense claims must be submitted within 30 days.",
            page_number=1,
        ),
    ]


@pytest.fixture
def chunks_file(tmp_path, sample_chunks):
    path = tmp_path / "document_chunks.jsonl"

    with path.open("w", encoding="utf-8") as file:
        for chunk in sample_chunks:
            file.write(json.dumps(chunk.to_dict()) + "\n")

    return path


@pytest.fixture
def indexed_artifacts(
    tmp_path,
    chunks_file,
    mock_embedder,
):
    artifact_dir = tmp_path / "artifacts"

    build_index(
        chunks_path=chunks_file,
        output_dir=artifact_dir,
        embedding_service=mock_embedder,
    )

    return artifact_dir
