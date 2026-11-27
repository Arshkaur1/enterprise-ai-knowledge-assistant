"""Calibrate cross-encoder relevance thresholds."""

import json
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

from knowledge_assistant.embeddings import EmbeddingService
from knowledge_assistant.evaluation import load_evaluation_dataset
from knowledge_assistant.hybrid_retrieval import HybridRetriever
from knowledge_assistant.indexing import calculate_checksum
from knowledge_assistant.lexical_retrieval import BM25Retriever
from knowledge_assistant.reranked_retrieval import RerankedRetriever
from knowledge_assistant.reranker_calibration import (
    build_thresholds,
    evaluate_retrieval,
    filter_ranked_candidates,
    meets_quality_targets,
    pareto_frontier,
    satisfies_recall_constraints,
)
from knowledge_assistant.reranking import CrossEncoderReranker
from knowledge_assistant.retrieval import SemanticRetriever

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TOP_K = 2
CANDIDATE_K = 5
DENSE_MIN_SCORE = 0.35
RRF_K = 60

THRESHOLD_START = -12.0
THRESHOLD_STOP = 10.0
THRESHOLD_STEP = 0.25

WARMUP_REQUESTS = 5


def serialize_results(results):
    """Serialize retrieval results without document text."""

    return [
        {
            "rank": result.rank,
            "score": float(result.score),
            "chunk_id": result.chunk.chunk_id,
        }
        for result in results
    ]


def query_metadata(example):
    """Preserve evaluation labels."""

    return {
        "query_id": example.query_id,
        "query": example.query,
        "answerable": example.answerable,
        "relevant_chunk_ids": list(example.relevant_chunk_ids),
    }


def evaluate_baseline(name, search, examples):
    """Evaluate a retrieval strategy without reranking."""

    records = []

    for example in examples:
        results = search(example.query)

        records.append(
            {
                **query_metadata(example),
                "results": serialize_results(results),
            }
        )

    evaluation = evaluate_retrieval(
        records,
        top_k=TOP_K,
    )

    return {
        "experiment": name,
        "threshold": None,
        "reranking_enabled": False,
        **evaluation,
    }


def cache_reranker_scores(
    name,
    pipeline,
    examples,
):
    """Run cross-encoder inference once per query."""

    records = []

    print(f"\nCaching scores: {name}")

    for example in examples:
        candidates = pipeline.search_candidates(
            query=example.query,
            candidate_k=CANDIDATE_K,
        )

        scored_candidates = (
            pipeline.reranker.rerank(
                query=example.query,
                candidates=candidates,
                top_k=CANDIDATE_K,
                min_score=None,
            )
            if candidates
            else []
        )

        records.append(
            {
                **query_metadata(example),
                "original_candidates": serialize_results(candidates),
                "scored_candidates": serialize_results(scored_candidates),
            }
        )

        best_score = f"{scored_candidates[0].score:.6f}" if scored_candidates else "N/A"

        print(
            f"{example.query_id:<20} "
            f"candidates={len(candidates):<2} "
            f"best_logit={best_score}"
        )

    return records


def sweep_thresholds(name, cached_records):
    """Evaluate all thresholds without further model inference."""

    thresholds = build_thresholds(
        cached_records,
        start=THRESHOLD_START,
        stop=THRESHOLD_STOP,
        step=THRESHOLD_STEP,
    )

    experiments = []

    for threshold in thresholds:
        records = [
            {
                "query_id": record["query_id"],
                "query": record["query"],
                "answerable": record["answerable"],
                "relevant_chunk_ids": (record["relevant_chunk_ids"]),
                "results": filter_ranked_candidates(
                    candidates=record["scored_candidates"],
                    threshold=threshold,
                    top_k=TOP_K,
                ),
            }
            for record in cached_records
        ]

        evaluation = evaluate_retrieval(
            records,
            top_k=TOP_K,
        )

        experiments.append(
            {
                "experiment": name,
                "threshold": threshold,
                "reranking_enabled": True,
                **evaluation,
            }
        )

    print(f"Evaluated {len(thresholds)} thresholds for {name}.")

    return experiments


def selection_key(experiment):
    """Rank configurations satisfying recall constraints."""

    summary = experiment["summary"]

    return (
        summary["false_acceptance_rate"],
        -summary["returned_chunk_micro_precision"],
        -summary["mrr"],
        summary["total_chunks_returned"],
    )


def print_experiment(experiment):
    """Display a compact experiment summary."""

    summary = experiment["summary"]

    threshold = experiment["threshold"]

    threshold_text = f"{threshold:.6f}" if threshold is not None else "None"

    print(
        f"{experiment['experiment']:<27} "
        f"{threshold_text:>12} "
        f"{summary['recall_at_k']:>8.3f} "
        f"{summary['complete_recall_at_k']:>8.3f} "
        f"{summary['mrr']:>8.3f} "
        f"{summary['false_acceptance_rate']:>8.3f} "
        f"{summary['false_rejection_rate']:>8.3f} "
        f"{summary['returned_chunk_micro_precision']:>8.3f}"
    )


def experiment_reference(experiment):
    """Return experiment identity and summary metrics."""

    return {
        "experiment": experiment["experiment"],
        "threshold": experiment["threshold"],
        "reranking_enabled": (experiment["reranking_enabled"]),
        "summary": experiment["summary"],
    }


def main():
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

    answerable_count = sum(example.answerable for example in examples)

    unsupported_count = len(examples) - answerable_count

    print("=" * 90)
    print("CROSS-ENCODER THRESHOLD CALIBRATION")
    print("=" * 90)

    print(f"Calibration queries: {len(examples)}")
    print(f"Answerable: {answerable_count}")
    print(f"Unsupported: {unsupported_count}")

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
        ),
        "dense_expanded_reranked": RerankedRetriever(
            retriever=dense,
            reranker=reranker,
            candidate_k=CANDIDATE_K,
            dense_min_score=None,
        ),
        "hybrid_reranked": RerankedRetriever(
            retriever=hybrid,
            reranker=reranker,
            candidate_k=CANDIDATE_K,
        ),
    }

    warmup_query = next(example.query for example in examples if example.answerable)

    for _ in range(WARMUP_REQUESTS):
        for pipeline in pipelines.values():
            pipeline.search(
                query=warmup_query,
                top_k=TOP_K,
            )

    def dense_search(query):
        return dense.search(
            query=query,
            top_k=TOP_K,
            min_score=DENSE_MIN_SCORE,
        )

    def hybrid_search(query):
        return hybrid.search(
            query=query,
            top_k=TOP_K,
        )

    dense_baseline = evaluate_baseline(
        "dense_baseline",
        dense_search,
        examples,
    )

    hybrid_baseline = evaluate_baseline(
        "hybrid_baseline",
        hybrid_search,
        examples,
    )

    baseline_summary = dense_baseline["summary"]

    all_experiments = [
        dense_baseline,
        hybrid_baseline,
    ]

    score_cache = {}
    strategy_summaries = {}

    for name, pipeline in pipelines.items():
        cached_records = cache_reranker_scores(
            name,
            pipeline,
            examples,
        )

        score_cache[name] = cached_records

        experiments = sweep_thresholds(
            name,
            cached_records,
        )

        all_experiments.extend(experiments)

        recall_preserving = [
            experiment
            for experiment in experiments
            if satisfies_recall_constraints(experiment["summary"])
        ]

        target_meeting = [
            experiment
            for experiment in recall_preserving
            if meets_quality_targets(
                experiment["summary"],
                baseline_summary,
            )
        ]

        best = (
            min(
                recall_preserving,
                key=selection_key,
            )
            if recall_preserving
            else None
        )

        strategy_summaries[name] = {
            "thresholds_evaluated": len(experiments),
            "recall_preserving_count": len(recall_preserving),
            "quality_target_meeting_count": len(target_meeting),
            "best_recall_preserving": (
                experiment_reference(best) if best is not None else None
            ),
        }

    frontier = pareto_frontier(all_experiments)

    recall_preserving = [
        experiment
        for experiment in all_experiments
        if satisfies_recall_constraints(experiment["summary"])
    ]

    quality_target_meeting = [
        experiment
        for experiment in all_experiments
        if meets_quality_targets(
            experiment["summary"],
            baseline_summary,
        )
    ]

    best_recall_preserving = (
        min(
            recall_preserving,
            key=selection_key,
        )
        if recall_preserving
        else None
    )

    best_quality_target_meeting = (
        min(
            quality_target_meeting,
            key=selection_key,
        )
        if quality_target_meeting
        else None
    )

    print("\n" + "=" * 100)
    print("CALIBRATION SUMMARY")
    print("=" * 100)

    print(
        f"{'Experiment':<27} "
        f"{'Threshold':>12} "
        f"{'Recall':>8} "
        f"{'Complete':>8} "
        f"{'MRR':>8} "
        f"{'FAR':>8} "
        f"{'FRR':>8} "
        f"{'Micro P':>8}"
    )

    print_experiment(dense_baseline)
    print_experiment(hybrid_baseline)

    print("\nUNFILTERED RERANKING")

    for experiment in all_experiments:
        if experiment["reranking_enabled"] and experiment["threshold"] is None:
            print_experiment(experiment)

    print("\nBEST RECALL-PRESERVING CONFIGURATIONS")

    for name, summary in strategy_summaries.items():
        best = summary["best_recall_preserving"]

        if best is not None:
            print_experiment(best)
        else:
            print(f"{name}: no recall-preserving configuration found.")

    print(f"\nTotal configurations: {len(all_experiments)}")

    print(f"Recall-preserving configurations: {len(recall_preserving)}")

    print(f"Quality-target-meeting configurations: {len(quality_target_meeting)}")

    print(f"Pareto-optimal configurations: {len(frontier)}")

    if best_quality_target_meeting is not None:
        print("\nBEST QUALITY-TARGET-MEETING CONFIGURATION")
        print_experiment(best_quality_target_meeting)
    else:
        print("\nNo configuration satisfies all retrieval quality targets.")

    report = {
        "created_at": datetime.now(UTC).isoformat(),
        "split": "calibration",
        "configuration": {
            "top_k": TOP_K,
            "candidate_k": CANDIDATE_K,
            "dense_min_score": DENSE_MIN_SCORE,
            "dense_expanded_min_score": None,
            "sparse_min_score": None,
            "rrf_k": RRF_K,
            "dense_weight": 1.0,
            "sparse_weight": 1.0,
            "reranker_model": reranker.model_name,
            "device": reranker.device,
            "batch_size": reranker.batch_size,
            "max_length": reranker.max_length,
            "threshold_start": THRESHOLD_START,
            "threshold_stop": THRESHOLD_STOP,
            "threshold_step": THRESHOLD_STEP,
            "exact_score_boundaries": True,
            "warmup_requests_per_pipeline": (WARMUP_REQUESTS),
            "corpus_checksum": calculate_checksum(chunks_path),
            "dataset_checksum": calculate_checksum(dataset_path),
            "bm25_implementation_checksum": (
                calculate_checksum(
                    PROJECT_ROOT
                    / "src"
                    / "knowledge_assistant"
                    / "lexical_retrieval.py"
                )
            ),
        },
        "dataset_summary": {
            "total_queries": len(examples),
            "answerable_queries": answerable_count,
            "unsupported_queries": unsupported_count,
        },
        "total_configurations": len(all_experiments),
        "recall_preserving_count": len(recall_preserving),
        "quality_target_meeting_count": len(quality_target_meeting),
        "strategy_summaries": strategy_summaries,
        "best_recall_preserving": (
            experiment_reference(best_recall_preserving)
            if best_recall_preserving is not None
            else None
        ),
        "best_quality_target_meeting": (
            experiment_reference(best_quality_target_meeting)
            if best_quality_target_meeting is not None
            else None
        ),
        "quality_target_meeting": [
            experiment_reference(experiment) for experiment in quality_target_meeting
        ],
        "pareto_frontier": [
            experiment_reference(experiment) for experiment in frontier
        ],
        "score_cache": score_cache,
        "experiments": all_experiments,
        "calibration_runtime_seconds": (perf_counter() - start),
    }

    output_path = (
        PROJECT_ROOT / "artifacts" / "evaluation" / "reranker_calibration.json"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(
            report,
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )

    print(f"\nReport saved to: {output_path}")

    print(f"Calibration runtime: {report['calibration_runtime_seconds']:.2f}s")


if __name__ == "__main__":
    main()
