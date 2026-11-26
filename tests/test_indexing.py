import json

import pytest

from knowledge_assistant.indexing import (
    build_index,
    calculate_checksum,
    load_chunks,
)
from knowledge_assistant.vector_store import VectorStore


def test_load_chunks(chunks_file):
    chunks = load_chunks(chunks_file)

    assert len(chunks) == 3
    assert chunks[0].category == "HR"
    assert chunks[2].page_number == 1


def test_missing_chunks_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_chunks(tmp_path / "missing.jsonl")


def test_empty_chunks_file(tmp_path):
    path = tmp_path / "empty.jsonl"
    path.write_text("", encoding="utf-8")

    with pytest.raises(
        ValueError,
        match="No document chunks",
    ):
        load_chunks(path)


def test_duplicate_chunk_ids(tmp_path, sample_chunks):
    path = tmp_path / "duplicates.jsonl"

    record = json.dumps(sample_chunks[0].to_dict())

    path.write_text(
        record + "\n" + record + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="Duplicate chunk IDs",
    ):
        load_chunks(path)


def test_build_index(
    tmp_path,
    chunks_file,
    mock_embedder,
):
    output_dir = tmp_path / "artifacts"

    manifest = build_index(
        chunks_path=chunks_file,
        output_dir=output_dir,
        embedding_service=mock_embedder,
    )

    assert manifest["chunk_count"] == 3
    assert manifest["document_count"] == 3
    assert manifest["embedding_dimension"] == 3
    assert manifest["normalized_embeddings"] is True
    assert manifest["similarity_metric"] == "cosine"

    assert (output_dir / "vector_index.faiss").exists()
    assert (output_dir / "chunk_metadata.json").exists()
    assert (output_dir / "index_manifest.json").exists()


def test_index_metadata_alignment(indexed_artifacts):
    store = VectorStore.load(indexed_artifacts / "vector_index.faiss")

    metadata = json.loads(
        (indexed_artifacts / "chunk_metadata.json").read_text(encoding="utf-8")
    )

    assert store.size == len(metadata) == 3
    assert metadata[0]["chunk_id"] == "hr_chunk_0000"
    assert metadata[1]["chunk_id"] == "it_chunk_0000"


def test_manifest_checksums(indexed_artifacts):
    manifest = json.loads(
        (indexed_artifacts / "index_manifest.json").read_text(encoding="utf-8")
    )

    assert manifest["index_checksum"] == calculate_checksum(
        indexed_artifacts / "vector_index.faiss"
    )

    assert manifest["metadata_checksum"] == calculate_checksum(
        indexed_artifacts / "chunk_metadata.json"
    )
