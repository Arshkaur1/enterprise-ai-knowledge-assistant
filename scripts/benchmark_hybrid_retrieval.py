import argparse
import json
from pathlib import Path
from statistics import mean
from time import perf_counter

import numpy as np

from knowledge_assistant.embeddings import EmbeddingService
from knowledge_assistant.evaluation import load_evaluation_dataset
from knowledge_assistant.hybrid_retrieval import HybridRetriever
from knowledge_assistant.lexical_retrieval import BM25Retriever
from knowledge_assistant.retrieval import SemanticRetriever

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TOP_K = 2
CANDIDATE_K = 5
DENSE_MIN_SCORE = 0.35
WARMUP_REQUESTS = 5
REPETITIONS = 10


EXPERIMENTS = [
    ("dense_baseline", 1.0, 0.0, 60),
    ("bm25_baseline", 0.0, 1.0, 60),
    ("hybrid_equal", 1.0, 1.0, 60),
    ("hybrid_dense_dominant", 2.0, 1.0, 60),
    ("hybrid_sparse_dominant", 1.0, 2.0, 60),
    ("hybrid_rrf_10", 1.0, 1.0, 10),
    ("hybrid_rrf_100", 1.0, 1.0, 100),
]


def calculate_metrics(
    retrieved_ids: list[str],
    relevant_ids: list[str],
    top_k: int,
) -> dict:
    """Calculate retrieval quality metrics."""

    relevant = set(relevant_ids)
    retrieved = set(retrieved_ids)

    if not relevant:
        return {}

    hits = len(relevant & retrieved)

    reciprocal_rank = next(
        (
            1.0 / rank
            for rank, chunk_id in enumerate(retrieved_ids, start=1)
            if chunk_id in relevant
        ),
        0.0,
    )

    return {
        "precision": hits / top_k,
        "recall": hits / len(relevant),
        "reciprocal_rank": reciprocal_rank,
        "hit": float(hits > 0),
        "complete_recall": float(relevant.issubset(retrieved)),
    }


def summarize(records: list[dict]) -> dict:
    """Aggregate retrieval quality and latency measurements."""

    answerable = [record for record in records if record["answerable"]]

    unanswerable = [record for record in records if not record["answerable"]]

    tp = sum(record["predicted_answerable"] for record in answerable)

    fn = len(answerable) - tp

    fp = sum(record["predicted_answerable"] for record in unanswerable)

    tn = len(unanswerable) - fp

    latencies = [latency for record in records for latency in record["latencies_ms"]]

    metric_names = [
        "precision",
        "recall",
        "reciprocal_rank",
        "hit",
        "complete_recall",
    ]

    retrieval_metrics = {
        name: mean(record["metrics"][name] for record in answerable)
        for name in metric_names
    }

    return {
        "total_queries": len(records),
        "answerable_queries": len(answerable),
        "unanswerable_queries": len(unanswerable),
        "retrieval_metrics": retrieval_metrics,
        "answerability": {
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "accuracy": (tp + tn) / len(records),
            "false_acceptance_rate": (fp / len(unanswerable) if unanswerable else None),
            "false_rejection_rate": (fn / len(answerable) if answerable else None),
        },
        "latency": {
            "measurements": len(latencies),
            "mean_ms": mean(latencies),
            "p50_ms": float(np.percentile(latencies, 50)),
            "p95_ms": float(np.percentile(latencies, 95)),
            "max_ms": max(latencies),
        },
    }


def evaluate_experiment(
    name,
    search,
    examples,
    configuration,
):
    """Evaluate a retrieval configuration."""

    for _ in range(WARMUP_REQUESTS):
        search(examples[0].query)

    records = []

    for example in examples:
        latencies = []
        retrieved_ids = None

        for _ in range(REPETITIONS):
            started = perf_counter()

            results = search(example.query)

            latency_ms = (perf_counter() - started) * 1000

            latencies.append(latency_ms)

            current_ids = [result.chunk.chunk_id for result in results]

            if retrieved_ids is None:
                retrieved_ids = current_ids

            elif current_ids != retrieved_ids:
                raise RuntimeError(
                    f"Non-deterministic retrieval ranking: {example.query_id}"
                )

        assert retrieved_ids is not None

        metrics = calculate_metrics(
            retrieved_ids=retrieved_ids,
            relevant_ids=example.relevant_chunk_ids,
            top_k=TOP_K,
        )

        records.append(
            {
                "query_id": example.query_id,
                "query": example.query,
                "answerable": example.answerable,
                "predicted_answerable": bool(retrieved_ids),
                "relevant_chunk_ids": (example.relevant_chunk_ids),
                "retrieved_chunk_ids": retrieved_ids,
                "metrics": metrics,
                "latencies_ms": latencies,
                "mean_latency_ms": mean(latencies),
            }
        )

    return {
        "experiment": name,
        "configuration": configuration,
        "summary": summarize(records),
        "results": records,
    }


def main():
    """Run comparative dense, sparse and hybrid benchmarks."""

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--split",
        choices=["calibration", "test"],
        default="calibration",
    )

    args = parser.parse_args()

    dataset = load_evaluation_dataset(
        PROJECT_ROOT / "data" / "evaluation" / "retrieval_eval.jsonl"
    )

    examples = [example for example in dataset if example.split == args.split]

    if not examples:
        raise ValueError("Evaluation split is empty.")

    if not any(example.answerable for example in examples):
        raise ValueError("No answerable evaluation queries.")

    embedder = EmbeddingService()

    dense = SemanticRetriever(
        artifact_dir=PROJECT_ROOT / "artifacts",
        embedding_service=embedder,
    )

    sparse = BM25Retriever.from_jsonl(
        PROJECT_ROOT / "data" / "processed" / "document_chunks.jsonl"
    )

    experiments = []

    for name, dense_weight, sparse_weight, rrf_k in EXPERIMENTS:
        print(f"\nRunning: {name}")

        if name == "dense_baseline":

            def dense_search(query):
                return dense.search(
                    query=query,
                    top_k=TOP_K,
                    min_score=DENSE_MIN_SCORE,
                )

            search = dense_search

        elif name == "bm25_baseline":

            def bm25_search(query):
                return sparse.search(
                    query=query,
                    top_k=TOP_K,
                )

            search = bm25_search

        else:
            hybrid = HybridRetriever(
                dense_retriever=dense,
                sparse_retriever=sparse,
                candidate_k=CANDIDATE_K,
                rrf_k=rrf_k,
                dense_weight=dense_weight,
                sparse_weight=sparse_weight,
                dense_min_score=DENSE_MIN_SCORE,
            )

            # FIX B023:
            # Bind the current retriever instance to avoid
            # late binding of the loop variable.
            def hybrid_search(query, retriever=hybrid):
                return retriever.search(
                    query=query,
                    top_k=TOP_K,
                )

            search = hybrid_search

        report = evaluate_experiment(
            name=name,
            search=search,
            examples=examples,
            configuration={
                "top_k": TOP_K,
                "candidate_k": CANDIDATE_K,
                "dense_weight": dense_weight,
                "sparse_weight": sparse_weight,
                "rrf_k": (rrf_k if dense_weight and sparse_weight else None),
                "dense_min_score": DENSE_MIN_SCORE,
            },
        )

        experiments.append(report)

        print(
            json.dumps(
                report["summary"],
                indent=2,
            )
        )

    output = (
        PROJECT_ROOT
        / "artifacts"
        / "evaluation"
        / f"hybrid_benchmark_{args.split}.json"
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.write_text(
        json.dumps(
            {
                "split": args.split,
                "warmup_requests": WARMUP_REQUESTS,
                "repetitions": REPETITIONS,
                "experiments": experiments,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n" + "=" * 110)

    print(
        f"{'Experiment':<27}"
        f"{'P@2':>8}"
        f"{'R@2':>8}"
        f"{'MRR':>8}"
        f"{'Hit@2':>8}"
        f"{'Complete':>10}"
        f"{'FAR':>8}"
        f"{'Mean ms':>10}"
        f"{'P95 ms':>10}"
    )

    for report in experiments:
        summary = report["summary"]
        metrics = summary["retrieval_metrics"]
        latency = summary["latency"]

        far = summary["answerability"]["false_acceptance_rate"]

        print(
            f"{report['experiment']:<27}"
            f"{metrics['precision']:>8.3f}"
            f"{metrics['recall']:>8.3f}"
            f"{metrics['reciprocal_rank']:>8.3f}"
            f"{metrics['hit']:>8.3f}"
            f"{metrics['complete_recall']:>10.3f}"
            f"{far:>8.3f}"
            f"{latency['mean_ms']:>10.3f}"
            f"{latency['p95_ms']:>10.3f}"
        )

    print(f"\nReport saved to: {output}")


if __name__ == "__main__":
    main()
