"""Demonstrate dense and hybrid retrieval with reranking."""

import json
from pathlib import Path
from time import perf_counter

from knowledge_assistant.embeddings import EmbeddingService
from knowledge_assistant.hybrid_retrieval import HybridRetriever
from knowledge_assistant.lexical_retrieval import BM25Retriever
from knowledge_assistant.reranked_retrieval import RerankedRetriever
from knowledge_assistant.reranking import CrossEncoderReranker
from knowledge_assistant.retrieval import SemanticRetriever

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TOP_K = 2
CANDIDATE_K = 5
DENSE_MIN_SCORE = 0.35

QUESTIONS = [
    (
        "What is the minimum password length?",
        True,
    ),
    (
        "How frequently must employees change their passwords?",
        False,
    ),
    (
        "How many paid maternity leave days are provided?",
        False,
    ),
    (
        "How many days per week can employees work remotely?",
        True,
    ),
    (
        "What is the expense submission deadline and who approves "
        "claims of 500 currency units or more?",
        True,
    ),
    (
        "What is the weather forecast for tomorrow?",
        False,
    ),
]


def main() -> None:
    embedding_service = EmbeddingService()

    dense = SemanticRetriever(
        artifact_dir=PROJECT_ROOT / "artifacts",
        embedding_service=embedding_service,
    )

    sparse = BM25Retriever.from_jsonl(
        PROJECT_ROOT / "data" / "processed" / "document_chunks.jsonl"
    )

    hybrid = HybridRetriever(
        dense_retriever=dense,
        sparse_retriever=sparse,
        candidate_k=CANDIDATE_K,
        dense_min_score=DENSE_MIN_SCORE,
        dense_weight=1.0,
        sparse_weight=1.0,
        rrf_k=60,
    )

    reranker = CrossEncoderReranker(
        device="cpu",
        batch_size=8,
        max_length=512,
    )

    pipelines = {
        "dense_reranked": RerankedRetriever(
            retriever=dense,
            reranker=reranker,
            candidate_k=CANDIDATE_K,
            dense_min_score=DENSE_MIN_SCORE,
            reranker_min_score=None,
        ),
        "hybrid_reranked": RerankedRetriever(
            retriever=hybrid,
            reranker=reranker,
            candidate_k=CANDIDATE_K,
            reranker_min_score=None,
        ),
    }

    records = []

    for name, pipeline in pipelines.items():
        print("\n" + "=" * 80)
        print(f"PIPELINE: {name}")

        for _ in range(5):
            pipeline.search(
                query=QUESTIONS[0][0],
                top_k=TOP_K,
            )

        for question, answerable in QUESTIONS:
            print(f"\nQUESTION: {question}")

            candidates = pipeline.search_candidates(
                query=question,
            )

            start = perf_counter()

            results = pipeline.search(
                query=question,
                top_k=TOP_K,
            )

            latency_ms = (perf_counter() - start) * 1000

            print(f"Expected answerable: {answerable}")
            print(f"Candidates: {len(candidates)}")
            print(f"Final results: {len(results)}")

            for result in results:
                print(
                    f"Rank={result.rank} | "
                    f"Logit={result.score:.6f} | "
                    f"Chunk={result.chunk.chunk_id}"
                )

            print(f"Pipeline latency: {latency_ms:.3f} ms")

            records.append(
                {
                    "pipeline": name,
                    "query": question,
                    "expected_answerable": answerable,
                    "retrieval_nonempty": bool(results),
                    "candidate_count": len(candidates),
                    "candidates": [
                        {
                            "rank": result.rank,
                            "score": result.score,
                            "chunk_id": result.chunk.chunk_id,
                        }
                        for result in candidates
                    ],
                    "reranked_results": [
                        {
                            "rank": result.rank,
                            "logit": result.score,
                            "chunk_id": result.chunk.chunk_id,
                        }
                        for result in results
                    ],
                    "latency_ms": latency_ms,
                }
            )

    report = {
        "configuration": {
            "reranker_model": reranker.model_name,
            "device": reranker.device,
            "batch_size": reranker.batch_size,
            "max_length": reranker.max_length,
            "candidate_k": CANDIDATE_K,
            "top_k": TOP_K,
            "dense_min_score": DENSE_MIN_SCORE,
            "sparse_min_score": None,
            "reranker_min_score": None,
            "dense_weight": 1.0,
            "sparse_weight": 1.0,
            "rrf_k": 60,
            "warmup_requests_per_pipeline": 5,
        },
        "total_questions": len(QUESTIONS),
        "total_pipelines": len(pipelines),
        "results": records,
    }

    output_path = (
        PROJECT_ROOT / "artifacts" / "evaluation" / "reranked_retrieval_demo.json"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print(f"\nReport saved to: {output_path}")


if __name__ == "__main__":
    main()
