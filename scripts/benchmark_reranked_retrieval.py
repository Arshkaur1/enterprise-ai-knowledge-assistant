"""Benchmark dense, sparse, hybrid and reranked retrieval."""

import csv
import json
from collections.abc import Callable
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from statistics import mean
from time import perf_counter

import numpy as np

from knowledge_assistant.embeddings import EmbeddingService
from knowledge_assistant.evaluation import load_evaluation_dataset
from knowledge_assistant.hybrid_retrieval import HybridRetriever
from knowledge_assistant.indexing import calculate_checksum
from knowledge_assistant.lexical_retrieval import BM25Retriever
from knowledge_assistant.reranked_retrieval import RerankedRetriever
from knowledge_assistant.reranker_calibration import (
    evaluate_retrieval,
    meets_quality_targets,
)
from knowledge_assistant.reranking import CrossEncoderReranker
from knowledge_assistant.retrieval import SearchResult, SemanticRetriever

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TOP_K = 2
CANDIDATE_K = 5
DENSE_MIN_SCORE = 0.35
DENSE_RERANKER_THRESHOLD = -9.0
EXPANDED_RERANKER_THRESHOLD = -8.75
HYBRID_RERANKER_THRESHOLD = -8.75

RRF_K = 60
WARMUP_REQUESTS = 5
REPETITIONS = 10
P95_LATENCY_TARGET_MS = 100.0

SearchFunction = Callable[[str], list[SearchResult]]


def serialize_results(
    results: list[SearchResult],
) -> list[dict]:
    """Serialize retrieval results."""

    return [
        {
            "rank": result.rank,
            "score": float(result.score),
            "chunk_id": result.chunk.chunk_id,
        }
        for result in results
    ]


def summarize_latency(
    latencies: list[float],
) -> dict:
    """Calculate latency statistics in milliseconds."""

    if not latencies:
        raise ValueError("Latency observations cannot be empty.")

    values = np.asarray(
        latencies,
        dtype=np.float64,
    )

    if not np.all(np.isfinite(values)):
        raise ValueError("Latency values must be finite.")

    if np.any(values < 0):
        raise ValueError("Latency cannot be negative.")

    return {
        "measured_requests": len(latencies),
        "mean_latency_ms": float(np.mean(values)),
        "p50_latency_ms": float(np.percentile(values, 50)),
        "p95_latency_ms": float(np.percentile(values, 95)),
        "max_latency_ms": float(np.max(values)),
        "min_latency_ms": float(np.min(values)),
        "std_latency_ms": float(np.std(values)),
    }


def benchmark_query(
    search: SearchFunction,
    query: str,
    repetitions: int = REPETITIONS,
) -> dict:
    """Measure repeated live retrieval and ranking stability."""

    if type(repetitions) is not int or repetitions < 1:
        raise ValueError("Repetitions must be a positive integer.")

    latencies = []
    reference_results = None
    reference_ids = None
    deterministic_requests = 0

    for _ in range(repetitions):
        start = perf_counter()

        results = search(query)

        latency_ms = (perf_counter() - start) * 1000

        latencies.append(latency_ms)

        serialized = serialize_results(results)

        retrieved_ids = [result["chunk_id"] for result in serialized]

        if reference_results is None:
            reference_results = serialized
            reference_ids = retrieved_ids

        if retrieved_ids == reference_ids:
            deterministic_requests += 1

    return {
        "results": reference_results,
        "latencies_ms": latencies,
        "mean_latency_ms": mean(latencies),
        "deterministic_requests": deterministic_requests,
        "determinism_rate": (deterministic_requests / repetitions),
        "deterministic": (deterministic_requests == repetitions),
    }


def benchmark_experiment(
    name: str,
    search: SearchFunction,
    examples: list,
    configuration: dict,
    candidate_search: SearchFunction | None = None,
    repetitions: int = REPETITIONS,
    warmup_requests: int = WARMUP_REQUESTS,
) -> dict:
    """Benchmark one retrieval configuration."""

    if not examples:
        raise ValueError("Evaluation dataset cannot be empty.")

    print("\n" + "=" * 90)
    print(f"EXPERIMENT: {name}")
    print("=" * 90)

    warmup_query = next(example.query for example in examples if example.answerable)

    for _ in range(warmup_requests):
        search(warmup_query)

    records = []
    all_latencies = []

    for example in examples:
        measured = benchmark_query(
            search=search,
            query=example.query,
            repetitions=repetitions,
        )

        # Candidate diagnostics are intentionally outside
        # the timed retrieval requests.
        candidate_count = (
            len(candidate_search(example.query))
            if candidate_search is not None
            else None
        )

        record = {
            "query_id": example.query_id,
            "query": example.query,
            "answerable": example.answerable,
            "relevant_chunk_ids": list(example.relevant_chunk_ids),
            "candidate_count": candidate_count,
            **measured,
        }

        records.append(record)

        all_latencies.extend(measured["latencies_ms"])

        print(
            f"{example.query_id:<20} "
            f"returned={len(measured['results']):<2} "
            f"mean={measured['mean_latency_ms']:>8.3f} ms "
            f"stable={measured['deterministic']}"
        )

    evaluation = evaluate_retrieval(
        records,
        top_k=TOP_K,
    )

    latency_summary = summarize_latency(all_latencies)

    deterministic_requests = sum(record["deterministic_requests"] for record in records)

    deterministic_queries = sum(record["deterministic"] for record in records)

    candidate_counts = [
        record["candidate_count"]
        for record in records
        if record["candidate_count"] is not None
    ]

    summary = {
        **evaluation["summary"],
        **latency_summary,
        "deterministic_requests": deterministic_requests,
        "deterministic_queries": deterministic_queries,
        "determinism_rate": (deterministic_requests / len(all_latencies)),
        "mean_candidate_count": (mean(candidate_counts) if candidate_counts else None),
        "max_candidate_count": (max(candidate_counts) if candidate_counts else None),
    }

    query_results = [
        {
            **record,
            **metrics,
        }
        for record, metrics in zip(
            records,
            evaluation["results"],
            strict=True,
        )
    ]

    return {
        "experiment": name,
        "configuration": configuration,
        "summary": summary,
        "results": query_results,
    }


def add_acceptance_results(
    experiments: list[dict],
) -> None:
    """Compare experiments against the measured dense baseline."""

    baseline = next(
        experiment["summary"]
        for experiment in experiments
        if experiment["experiment"] == "dense_baseline"
    )

    for experiment in experiments:
        summary = experiment["summary"]

        summary["relative_mean_latency"] = (
            summary["mean_latency_ms"] / baseline["mean_latency_ms"]
        )

        summary["relative_p95_latency"] = (
            summary["p95_latency_ms"] / baseline["p95_latency_ms"]
        )

        summary["additional_mean_latency_ms"] = (
            summary["mean_latency_ms"] - baseline["mean_latency_ms"]
        )

        summary["quality_targets_met"] = meets_quality_targets(
            summary,
            baseline,
        )

        summary["latency_target_met"] = (
            summary["p95_latency_ms"] <= P95_LATENCY_TARGET_MS
        )

        summary["determinism_target_met"] = summary["determinism_rate"] == 1.0

        summary["all_targets_met"] = (
            summary["quality_targets_met"]
            and summary["latency_target_met"]
            and summary["determinism_target_met"]
        )


def print_comparison(
    experiments: list[dict],
) -> None:
    """Display the final quality and latency comparison."""

    print("\n" + "=" * 125)
    print("COMPARATIVE BENCHMARK RESULTS")
    print("=" * 125)

    print(
        f"{'Experiment':<30} "
        f"{'Recall':>7} "
        f"{'MRR':>7} "
        f"{'Micro P':>8} "
        f"{'FAR':>7} "
        f"{'FRR':>7} "
        f"{'Mean ms':>9} "
        f"{'P95 ms':>9} "
        f"{'Stable':>7}"
    )

    for experiment in experiments:
        s = experiment["summary"]

        print(
            f"{experiment['experiment']:<30} "
            f"{s['recall_at_k']:>7.3f} "
            f"{s['mrr']:>7.3f} "
            f"{s['returned_chunk_micro_precision']:>8.3f} "
            f"{s['false_acceptance_rate']:>7.3f} "
            f"{s['false_rejection_rate']:>7.3f} "
            f"{s['mean_latency_ms']:>9.3f} "
            f"{s['p95_latency_ms']:>9.3f} "
            f"{s['determinism_rate']:>7.3f}"
        )

    print("\nACCEPTANCE RESULTS")

    for experiment in experiments:
        s = experiment["summary"]

        print(
            f"{experiment['experiment']:<30} "
            f"quality={s['quality_targets_met']} "
            f"latency={s['latency_target_met']} "
            f"stability={s['determinism_target_met']} "
            f"overall={s['all_targets_met']}"
        )


def write_csv(
    path: Path,
    rows: list[dict],
) -> None:
    """Write experiment records to CSV."""

    if not rows:
        raise ValueError("CSV rows cannot be empty.")

    fieldnames = list(dict.fromkeys(key for row in rows for key in row))

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for row in rows:
            writer.writerow(
                {
                    key: (
                        json.dumps(value, allow_nan=False)
                        if isinstance(value, (list, dict))
                        else value
                    )
                    for key, value in row.items()
                }
            )


def main() -> None:
    start = perf_counter()

    dataset_path = PROJECT_ROOT / "data" / "evaluation" / "retrieval_eval.jsonl"

    chunks_path = PROJECT_ROOT / "data" / "processed" / "document_chunks.jsonl"

    examples = [
        example
        for example in load_evaluation_dataset(dataset_path)
        if example.split == "calibration"
    ]

    if not examples:
        raise ValueError("Calibration dataset is empty.")

    print("COMPARATIVE RETRIEVAL BENCHMARK")
    print(f"Calibration queries: {len(examples)}")
    print(f"Repetitions per query: {REPETITIONS}")

    embedding_service = EmbeddingService()

    dense = SemanticRetriever(
        artifact_dir=PROJECT_ROOT / "artifacts",
        embedding_service=embedding_service,
    )

    sparse = BM25Retriever.from_jsonl(chunks_path)

    hybrid = HybridRetriever(
        dense_retriever=dense,
        sparse_retriever=sparse,
        candidate_k=CANDIDATE_K,
        rrf_k=RRF_K,
        dense_weight=1.0,
        sparse_weight=1.0,
        dense_min_score=DENSE_MIN_SCORE,
        sparse_min_score=None,
    )

    initialization_start = perf_counter()

    reranker = CrossEncoderReranker(
        device="cpu",
        batch_size=8,
        max_length=512,
    )

    initialization_seconds = perf_counter() - initialization_start

    dense_unfiltered = RerankedRetriever(
        retriever=dense,
        reranker=reranker,
        candidate_k=CANDIDATE_K,
        dense_min_score=DENSE_MIN_SCORE,
        reranker_min_score=None,
    )

    dense_calibrated = RerankedRetriever(
        retriever=dense,
        reranker=reranker,
        candidate_k=CANDIDATE_K,
        dense_min_score=DENSE_MIN_SCORE,
        reranker_min_score=DENSE_RERANKER_THRESHOLD,
    )

    dense_expanded = RerankedRetriever(
        retriever=dense,
        reranker=reranker,
        candidate_k=CANDIDATE_K,
        dense_min_score=None,
        reranker_min_score=EXPANDED_RERANKER_THRESHOLD,
    )

    hybrid_reranked = RerankedRetriever(
        retriever=hybrid,
        reranker=reranker,
        candidate_k=CANDIDATE_K,
        reranker_min_score=HYBRID_RERANKER_THRESHOLD,
    )

    def dense_search(query):
        return dense.search(
            query=query,
            top_k=TOP_K,
            min_score=DENSE_MIN_SCORE,
        )

    def sparse_search(query):
        return sparse.search(
            query=query,
            top_k=TOP_K,
        )

    def hybrid_search(query):
        return hybrid.search(
            query=query,
            top_k=TOP_K,
        )

    def make_reranked_search(pipeline):
        def search(query):
            return pipeline.search(
                query=query,
                top_k=TOP_K,
            )

        return search

    def make_candidate_search(pipeline):
        def search(query):
            return pipeline.search_candidates(
                query=query,
                candidate_k=CANDIDATE_K,
            )

        return search

    experiments = []

    baselines = [
        (
            "dense_baseline",
            dense_search,
            {
                "candidate_k": TOP_K,
                "dense_min_score": DENSE_MIN_SCORE,
                "reranking_enabled": False,
                "score_type": "cosine_similarity",
            },
        ),
        (
            "bm25_baseline",
            sparse_search,
            {
                "candidate_k": TOP_K,
                "sparse_min_score": None,
                "reranking_enabled": False,
                "score_type": "bm25",
            },
        ),
        (
            "hybrid_baseline",
            hybrid_search,
            {
                "candidate_k": CANDIDATE_K,
                "dense_min_score": DENSE_MIN_SCORE,
                "sparse_min_score": None,
                "rrf_k": RRF_K,
                "reranking_enabled": False,
                "score_type": "rrf",
            },
        ),
    ]

    for name, search, configuration in baselines:
        experiments.append(
            benchmark_experiment(
                name=name,
                search=search,
                examples=examples,
                configuration={
                    "top_k": TOP_K,
                    **configuration,
                },
            )
        )

    pipelines = {
        "dense_reranked_unfiltered": dense_unfiltered,
        "dense_reranked_calibrated": dense_calibrated,
        "dense_expanded_reranked": dense_expanded,
        "hybrid_reranked": hybrid_reranked,
    }

    for name, pipeline in pipelines.items():
        experiments.append(
            benchmark_experiment(
                name=name,
                search=make_reranked_search(pipeline),
                examples=examples,
                candidate_search=make_candidate_search(pipeline),
                configuration={
                    "top_k": TOP_K,
                    "candidate_k": pipeline.candidate_k,
                    "dense_min_score": (pipeline.dense_min_score),
                    "reranker_min_score": (pipeline.reranker_min_score),
                    "reranking_enabled": True,
                    "score_type": "cross_encoder_logit",
                    "candidate_strategy": (
                        "hybrid"
                        if isinstance(
                            pipeline.retriever,
                            HybridRetriever,
                        )
                        else "dense"
                    ),
                },
            )
        )

    add_acceptance_results(experiments)

    print_comparison(experiments)

    output_dir = PROJECT_ROOT / "artifacts" / "evaluation"

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    report = {
        "created_at": datetime.now(UTC).isoformat(),
        "split": "calibration",
        "configuration": {
            "top_k": TOP_K,
            "candidate_k": CANDIDATE_K,
            "dense_min_score": DENSE_MIN_SCORE,
            "sparse_min_score": None,
            "dense_weight": 1.0,
            "sparse_weight": 1.0,
            "rrf_k": RRF_K,
            "warmup_requests": WARMUP_REQUESTS,
            "repetitions": REPETITIONS,
            "reranker_model": reranker.model_name,
            "reranker_device": reranker.device,
            "reranker_batch_size": reranker.batch_size,
            "reranker_max_length": reranker.max_length,
            "reranker_initialization_seconds": (initialization_seconds),
            "p95_latency_target_ms": (P95_LATENCY_TARGET_MS),
            "bm25_lexical_overlap_required": True,
            "bm25_positive_score_required": False,
            "corpus_checksum": calculate_checksum(chunks_path),
            "dataset_checksum": calculate_checksum(dataset_path),
            "dependency_versions": {
                "torch": version("torch"),
                "transformers": version("transformers"),
                "sentence-transformers": version("sentence-transformers"),
                "faiss-cpu": version("faiss-cpu"),
                "rank-bm25": version("rank-bm25"),
            },
        },
        "total_experiments": len(experiments),
        "total_measured_requests": sum(
            experiment["summary"]["measured_requests"] for experiment in experiments
        ),
        "experiments": experiments,
        "benchmark_runtime_seconds": (perf_counter() - start),
    }

    json_path = output_dir / "reranked_benchmark_calibration.json"

    json_path.write_text(
        json.dumps(
            report,
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )

    summary_rows = [
        {
            "experiment": experiment["experiment"],
            **experiment["configuration"],
            **experiment["summary"],
        }
        for experiment in experiments
    ]

    query_rows = [
        {
            "experiment": experiment["experiment"],
            **record,
        }
        for experiment in experiments
        for record in experiment["results"]
    ]

    summary_path = output_dir / "reranked_benchmark_summary.csv"

    queries_path = output_dir / "reranked_benchmark_queries.csv"

    write_csv(summary_path, summary_rows)
    write_csv(queries_path, query_rows)

    print(f"\nJSON report: {json_path}")
    print(f"Summary CSV: {summary_path}")
    print(f"Query CSV: {queries_path}")

    print(f"Measured requests: {report['total_measured_requests']}")

    print(f"Benchmark runtime: {report['benchmark_runtime_seconds']:.2f}s")


if __name__ == "__main__":
    main()
