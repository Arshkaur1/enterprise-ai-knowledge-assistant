from pathlib import Path
from time import perf_counter

from knowledge_assistant.lexical_retrieval import (
    BM25Retriever,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


QUESTIONS = [
    "How many days per week can employees work remotely?",
    "What is the minimum password length?",
    "Is multi-factor authentication mandatory?",
    "What is the expense submission deadline?",
    "Who approves claims of 500 currency units or more?",
    "How many paid maternity leave days are provided?",
    "How frequently must employees change their passwords?",
    "What is the company's mileage reimbursement rate?",
    "What is the weather forecast for tomorrow?",
    "MFA",
    "12 characters",
    "500 currency units",
]


def main():

    chunks_path = PROJECT_ROOT / "data" / "processed" / "document_chunks.jsonl"

    started = perf_counter()

    retriever = BM25Retriever.from_jsonl(
        chunks_path=chunks_path,
    )

    build_latency_ms = (perf_counter() - started) * 1000

    print("=" * 70)
    print("BM25 LEXICAL RETRIEVAL")
    print("=" * 70)

    print(f"Chunks: {retriever.chunk_count}")
    print(f"k1: {retriever.k1}")
    print(f"b: {retriever.b}")
    print(f"Build latency: {build_latency_ms:.3f} ms")

    for question in QUESTIONS:
        started = perf_counter()

        results = retriever.search(
            query=question,
            top_k=2,
        )

        latency_ms = (perf_counter() - started) * 1000

        print("\n" + "=" * 70)
        print(f"Query: {question}")
        print(f"Latency: {latency_ms:.3f} ms")

        if not results:
            print("No lexical matches.")
            continue

        for result in results:
            print(f"\nRank: {result.rank} | Score: {result.score:.4f}")

            print(f"Chunk ID: {result.chunk.chunk_id}")

            print(f"Source: {result.chunk.source}")

            print(f"Category: {result.chunk.category}")

            print(f"Text: {result.chunk.text[:250]}...")


if __name__ == "__main__":
    main()
