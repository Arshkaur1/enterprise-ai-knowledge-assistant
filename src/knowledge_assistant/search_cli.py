import argparse
from pathlib import Path
from time import perf_counter

from knowledge_assistant.embeddings import (
    DEFAULT_MODEL,
    EmbeddingService,
)
from knowledge_assistant.retrieval import SemanticRetriever

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Search enterprise documents.",
    )

    parser.add_argument(
        "query",
        help="Natural language search query.",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--min-score",
        type=float,
        default=None,
    )

    parser.add_argument(
        "--artifacts",
        type=Path,
        default=PROJECT_ROOT / "artifacts",
    )

    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
    )

    args = parser.parse_args()

    embedding_service = EmbeddingService(
        model_name=args.model,
    )

    retriever = SemanticRetriever(
        artifact_dir=args.artifacts,
        embedding_service=embedding_service,
    )

    start = perf_counter()

    results = retriever.search(
        query=args.query,
        top_k=args.top_k,
        min_score=args.min_score,
    )

    latency = perf_counter() - start

    print(f"\nQuery: {args.query}")
    print(f"Results: {len(results)}")
    print(f"Retrieval latency: {latency * 1000:.2f} ms")

    if not results:
        print("\nNo matching documents found.")
        return

    for result in results:
        chunk = result.chunk

        print("\n" + "=" * 70)
        print(f"Rank: {result.rank}")
        print(f"Similarity: {result.score:.4f}")
        print(f"Source: {chunk.source}")
        print(f"Category: {chunk.category}")
        print(f"Chunk ID: {chunk.chunk_id}")

        if chunk.page_number is not None:
            print(f"Page: {chunk.page_number}")

        print(f"\n{chunk.text}")


if __name__ == "__main__":
    main()
