import json

import pytest

from knowledge_assistant.retrieval import SemanticRetriever


@pytest.fixture
def retriever(indexed_artifacts, mock_embedder):
    return SemanticRetriever(
        artifact_dir=indexed_artifacts,
        embedding_service=mock_embedder,
    )


@pytest.mark.parametrize(
    ("query", "expected_category"),
    [
        ("What is the remote work policy?", "HR"),
        ("What are the password requirements?", "IT"),
        ("When should I submit expense claims?", "Finance"),
    ],
)
def test_correct_document_retrieval(
    retriever,
    query,
    expected_category,
):
    results = retriever.search(query, top_k=1)

    assert len(results) == 1
    assert results[0].rank == 1
    assert results[0].chunk.category == expected_category
    assert results[0].score == pytest.approx(1.0)


def test_top_k(retriever):
    results = retriever.search(
        "remote work",
        top_k=2,
    )

    assert len(results) == 2


def test_similarity_threshold(retriever):
    results = retriever.search(
        "remote work",
        top_k=3,
        min_score=0.3,
    )

    assert len(results) == 1
    assert results[0].chunk.category == "HR"


def test_out_of_domain_query(retriever):
    results = retriever.search(
        "What is the weather tomorrow?",
        min_score=0.3,
    )

    assert results == []


def test_empty_query(retriever):
    with pytest.raises(ValueError):
        retriever.search("")


def test_invalid_top_k(retriever):
    with pytest.raises(ValueError, match="top_k"):
        retriever.search("remote work", top_k=0)


@pytest.mark.parametrize("threshold", [-1.1, 1.1])
def test_invalid_threshold(retriever, threshold):
    with pytest.raises(ValueError, match="min_score"):
        retriever.search(
            "remote work",
            min_score=threshold,
        )


def test_source_attribution(retriever):
    result = retriever.search(
        "expense claims",
        top_k=1,
    )[0]

    assert result.chunk.source == "finance_policy.pdf"
    assert result.chunk.page_number == 1
    assert result.chunk.chunk_id == "finance_chunk_0000"


def test_embedding_model_mismatch(
    indexed_artifacts,
    mock_embedder,
):
    mock_embedder.model_name = "different-model"

    with pytest.raises(
        ValueError,
        match="Embedding model mismatch",
    ):
        SemanticRetriever(
            indexed_artifacts,
            mock_embedder,
        )


def test_embedding_dimension_mismatch(
    indexed_artifacts,
    mock_embedder,
):
    mock_embedder.dimension = 4

    with pytest.raises(
        ValueError,
        match="Embedding dimension mismatch",
    ):
        SemanticRetriever(
            indexed_artifacts,
            mock_embedder,
        )


def test_corrupted_metadata(
    indexed_artifacts,
    mock_embedder,
):
    path = indexed_artifacts / "chunk_metadata.json"
    path.write_text("[]", encoding="utf-8")

    with pytest.raises(
        ValueError,
        match="Metadata checksum mismatch",
    ):
        SemanticRetriever(
            indexed_artifacts,
            mock_embedder,
        )


def test_corrupted_index(
    indexed_artifacts,
    mock_embedder,
):
    path = indexed_artifacts / "vector_index.faiss"
    path.write_bytes(b"corrupted index")

    with pytest.raises(
        ValueError,
        match="FAISS index checksum mismatch",
    ):
        SemanticRetriever(
            indexed_artifacts,
            mock_embedder,
        )


def test_unsupported_manifest_version(
    indexed_artifacts,
    mock_embedder,
):
    path = indexed_artifacts / "index_manifest.json"

    manifest = json.loads(path.read_text(encoding="utf-8"))

    manifest["schema_version"] = 999

    path.write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="Unsupported manifest version",
    ):
        SemanticRetriever(
            indexed_artifacts,
            mock_embedder,
        )
