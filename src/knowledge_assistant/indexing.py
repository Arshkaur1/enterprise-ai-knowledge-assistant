import argparse
import hashlib
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

from knowledge_assistant.embeddings import (
    DEFAULT_MODEL,
    EmbeddingService,
)
from knowledge_assistant.models import DocumentChunk
from knowledge_assistant.vector_store import VectorStore

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def calculate_checksum(path: Path) -> str:
    """Calculate the SHA-256 checksum of a file."""

    digest = hashlib.sha256()

    with path.open("rb") as file:
        for block in iter(lambda: file.read(65536), b""):
            digest.update(block)

    return digest.hexdigest()


def load_chunks(path: str | Path) -> list[DocumentChunk]:
    """Load processed document chunks from JSONL."""

    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(f"Chunk file not found: {path}")

    chunks = []

    with path.open(encoding="utf-8") as file:
        for line in file:
            if line.strip():
                record = json.loads(line)
                chunks.append(DocumentChunk(**record))

    if not chunks:
        raise ValueError("No document chunks found.")

    chunk_ids = [chunk.chunk_id for chunk in chunks]

    if len(chunk_ids) != len(set(chunk_ids)):
        raise ValueError("Duplicate chunk IDs detected.")

    return chunks


def build_index(
    chunks_path: str | Path,
    output_dir: str | Path,
    embedding_service: EmbeddingService,
) -> dict:
    """Build and persist a searchable document index."""

    start = perf_counter()

    chunks_path = Path(chunks_path)
    output_dir = Path(output_dir)

    chunks = load_chunks(chunks_path)

    output_dir.mkdir(parents=True, exist_ok=True)

    texts = [chunk.text for chunk in chunks]

    embeddings = embedding_service.embed_documents(texts)

    if len(embeddings) != len(chunks):
        raise ValueError("Embedding count does not match chunks.")

    store = VectorStore(
        dimension=embedding_service.dimension,
    )

    store.add(embeddings)

    index_path = output_dir / "vector_index.faiss"
    metadata_path = output_dir / "chunk_metadata.json"
    manifest_path = output_dir / "index_manifest.json"

    store.save(index_path)

    metadata = [chunk.to_dict() for chunk in chunks]

    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "embedding_model": embedding_service.model_name,
        "embedding_dimension": embedding_service.dimension,
        "normalized_embeddings": True,
        "index_type": "IndexFlatIP",
        "similarity_metric": "cosine",
        "chunk_count": len(chunks),
        "document_count": len({chunk.document_id for chunk in chunks}),
        "source_checksum": calculate_checksum(chunks_path),
        "index_checksum": calculate_checksum(index_path),
        "metadata_checksum": calculate_checksum(metadata_path),
        "build_time_seconds": round(
            perf_counter() - start,
            4,
        ),
    }

    manifest_path.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    logger.info("Indexed %s chunks", len(chunks))
    logger.info("Artifacts saved to %s", output_dir)

    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the enterprise document index.",
    )

    parser.add_argument(
        "--chunks",
        type=Path,
        default=(PROJECT_ROOT / "data/processed/document_chunks.jsonl"),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "artifacts",
    )

    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    embedding_service = EmbeddingService(
        model_name=args.model,
    )

    manifest = build_index(
        chunks_path=args.chunks,
        output_dir=args.output,
        embedding_service=embedding_service,
    )

    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
