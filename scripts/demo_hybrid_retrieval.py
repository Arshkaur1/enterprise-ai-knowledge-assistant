import json
from pathlib import Path
from time import perf_counter

from knowledge_assistant.embeddings import EmbeddingService
from knowledge_assistant.hybrid_retrieval import (
    HybridRetriever,
)
from knowledge_assistant.lexical_retrieval import (
    BM25Retriever,
)
from knowledge_assistant.rank_fusion import (
    reciprocal_rank_fusion,
)
from knowledge_assistant.retrieval import (
    SemanticRetriever,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TOP_K = 2
CANDIDATE_K = 5
RRF_K = 60


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
    "What is the annual leave allowance and when is medical documentation required?",
    "What is the remote work allowance and meal reimbursement limit?",
]


def serialize_results(results):

    return [
        {
            "rank": result.rank,
            "score": result.score,
            "chunk_id": result.chunk.chunk_id,
            "source": result.chunk.source,
            "category": result.chunk.category,
            "text": result.chunk.text,
        }
        for result in results
    ]


def display_results(name, results):

    print(f"\n{name}")

    if not results:
        print("  No results.")
        return

    for result in results:
        print(
            f"  Rank {result.rank} | Score {result.score:.6f} | {result.chunk.chunk_id}"
        )


def main():

    embedder = EmbeddingService()

    dense = SemanticRetriever(
        artifact_dir=PROJECT_ROOT / "artifacts",
        embedding_service=embedder,
    )

    sparse = BM25Retriever.from_jsonl(
        PROJECT_ROOT / "data" / "processed" / "document_chunks.jsonl"
    )

    hybrid = HybridRetriever(
        dense_retriever=dense,
        sparse_retriever=sparse,
        candidate_k=CANDIDATE_K,
        rrf_k=RRF_K,
        dense_weight=1.0,
        sparse_weight=1.0,
        dense_min_score=0.35,
        sparse_min_score=None,
    )

    print("Warming up retrieval systems...")

    for _ in range(5):
        hybrid.search(
            "What is the minimum password length?",
            top_k=TOP_K,
        )

    records = []

    for question in QUESTIONS:
        started = perf_counter()

        dense_results = dense.search(
            query=question,
            top_k=CANDIDATE_K,
            min_score=0.35,
        )

        dense_latency = (perf_counter() - started) * 1000

        started = perf_counter()

        sparse_results = sparse.search(
            query=question,
            top_k=CANDIDATE_K,
        )

        sparse_latency = (perf_counter() - started) * 1000

        started = perf_counter()

        fused_results = reciprocal_rank_fusion(
            rankings=[
                dense_results,
                sparse_results,
            ],
            top_k=TOP_K,
            k=RRF_K,
        )

        fusion_latency = (perf_counter() - started) * 1000

        print("\n" + "=" * 70)
        print(f"Query: {question}")

        display_results(
            "DENSE RETRIEVAL",
            dense_results[:TOP_K],
        )

        display_results(
            "BM25 RETRIEVAL",
            sparse_results[:TOP_K],
        )

        display_results(
            "HYBRID RETRIEVAL",
            fused_results,
        )

        total_latency = dense_latency + sparse_latency + fusion_latency

        print(f"\nDense latency: {dense_latency:.3f} ms")

        print(f"BM25 latency: {sparse_latency:.3f} ms")

        print(f"Fusion latency: {fusion_latency:.3f} ms")

        print(f"Combined latency: {total_latency:.3f} ms")

        records.append(
            {
                "query": question,
                "dense_results": serialize_results(dense_results[:TOP_K]),
                "sparse_results": serialize_results(sparse_results[:TOP_K]),
                "hybrid_results": serialize_results(fused_results),
                "dense_candidates": serialize_results(dense_results),
                "sparse_candidates": serialize_results(sparse_results),
                "dense_latency_ms": dense_latency,
                "sparse_latency_ms": sparse_latency,
                "fusion_latency_ms": fusion_latency,
                "combined_latency_ms": total_latency,
            }
        )

    report = {
        "configuration": {
            "top_k": TOP_K,
            "candidate_k": CANDIDATE_K,
            "rrf_k": RRF_K,
            "dense_weight": 1.0,
            "sparse_weight": 1.0,
            "dense_min_score": 0.35,
            "sparse_min_score": None,
            "warmup_requests": 5,
        },
        "total_questions": len(records),
        "results": records,
    }

    output = PROJECT_ROOT / "artifacts" / "evaluation" / "hybrid_retrieval_demo.json"

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(f"\nReport saved to: {output}")


if __name__ == "__main__":
    main()
