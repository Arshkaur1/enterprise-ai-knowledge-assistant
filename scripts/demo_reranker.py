from pathlib import Path
from time import perf_counter

from knowledge_assistant.embeddings import EmbeddingService
from knowledge_assistant.reranking import CrossEncoderReranker
from knowledge_assistant.retrieval import SemanticRetriever

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CANDIDATE_K = 5
TOP_K = 2

QUESTIONS = [
    "What is the minimum password length?",
    "How frequently must employees change their passwords?",
    "How many paid maternity leave days are provided?",
    "How many days per week can employees work remotely?",
    "What is the company's mileage reimbursement rate?",
    "What is the expense submission deadline and who approves "
    "claims of 500 currency units or more?",
    "What is the weather forecast for tomorrow?",
]


def main() -> None:
    embedding_service = EmbeddingService()

    retriever = SemanticRetriever(
        artifact_dir=PROJECT_ROOT / "artifacts",
        embedding_service=embedding_service,
    )

    start = perf_counter()

    reranker = CrossEncoderReranker(
        device="cpu",
        batch_size=8,
        max_length=512,
    )

    print(f"Reranker initialization: {perf_counter() - start:.3f} seconds")

    warmup_candidates = retriever.search(
        query=QUESTIONS[0],
        top_k=CANDIDATE_K,
        min_score=None,
    )

    for _ in range(5):
        reranker.rerank(
            query=QUESTIONS[0],
            candidates=warmup_candidates,
            top_k=CANDIDATE_K,
        )

    for question in QUESTIONS:
        print("\n" + "=" * 80)
        print(f"QUESTION: {question}")

        start = perf_counter()

        candidates = retriever.search(
            query=question,
            top_k=CANDIDATE_K,
            min_score=None,
        )

        retrieval_ms = (perf_counter() - start) * 1000

        print("\nDENSE CANDIDATES")

        for result in candidates:
            print(
                f"Rank={result.rank} | "
                f"Score={result.score:.6f} | "
                f"Chunk={result.chunk.chunk_id}"
            )

        start = perf_counter()

        reranked = reranker.rerank(
            query=question,
            candidates=candidates,
            top_k=CANDIDATE_K,
            min_score=None,
        )

        reranking_ms = (perf_counter() - start) * 1000

        print("\nCROSS-ENCODER RANKING")

        for result in reranked:
            print(
                f"Rank={result.rank} | "
                f"Logit={result.score:.6f} | "
                f"Chunk={result.chunk.chunk_id}"
            )

        print("\nTOP FINAL CANDIDATES")

        for result in reranked[:TOP_K]:
            print(f"{result.rank}. {result.chunk.chunk_id} ({result.score:.6f})")

        print(f"\nDense retrieval: {retrieval_ms:.3f} ms")
        print(f"Reranking:       {reranking_ms:.3f} ms")
        print(f"Combined:        {retrieval_ms + reranking_ms:.3f} ms")


if __name__ == "__main__":
    main()
