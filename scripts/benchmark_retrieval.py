import csv
import json
from pathlib import Path
from time import perf_counter

from knowledge_assistant.embeddings import EmbeddingService
from knowledge_assistant.evaluation import (
    answerability_metrics,
    latency_metrics,
    load_evaluation_dataset,
    retrieval_metrics,
    summarize_retrieval_metrics,
)
from knowledge_assistant.retrieval import SemanticRetriever

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_PATH = PROJECT_ROOT / "data/evaluation/retrieval_eval.jsonl"

ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
OUTPUT_DIR = ARTIFACT_DIR / "evaluation"

TOP_K_VALUES = [1, 2, 3, 5]
THRESHOLDS = [i / 20 for i in range(13)]

WARMUP_ITERATIONS = 5
REPETITIONS = 10


def write_csv(path, rows):
    if not rows:
        return

    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0]),
        )

        writer.writeheader()
        writer.writerows(rows)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    dataset = load_evaluation_dataset(DATASET_PATH)

    print(f"Loaded {len(dataset)} evaluation queries.")

    # Load model and index only once.
    embedder = EmbeddingService()

    retriever = SemanticRetriever(
        artifact_dir=ARTIFACT_DIR,
        embedding_service=embedder,
    )

    available_ids = {chunk.chunk_id for chunk in retriever.metadata}

    for item in dataset:
        missing = set(item.relevant_chunk_ids) - available_ids

        if missing:
            raise ValueError(f"Unknown chunk IDs for {item.query_id}: {missing}")

    if retriever.store.size < max(TOP_K_VALUES):
        raise ValueError("Index contains fewer chunks than max top-k.")

    # Warm-up.
    print("Running warm-up iterations...")

    for _ in range(WARMUP_ITERATIONS):
        retriever.search(
            dataset[0].query,
            top_k=max(TOP_K_VALUES),
        )

    query_results = []
    latencies = []
    cached_results = {}

    print("Running retrieval benchmark...")

    for item in dataset:
        for repetition in range(REPETITIONS):
            start = perf_counter()

            results = retriever.search(
                item.query,
                top_k=max(TOP_K_VALUES),
            )

            elapsed_ms = (perf_counter() - start) * 1000

            latencies.append(elapsed_ms)

            if repetition == 0:
                cached_results[item.query_id] = results

        print(f"Completed: {item.query_id}")

    # Compare top-k retrieval performance.
    ranking_summary = {}

    for split in ["calibration", "test"]:
        ranking_summary[split] = {}

        for k in TOP_K_VALUES:
            metric_rows = []

            for item in dataset:
                if item.split != split or not item.answerable:
                    continue

                results = cached_results[item.query_id][:k]

                retrieved_ids = [result.chunk.chunk_id for result in results]

                metrics = retrieval_metrics(
                    retrieved_ids,
                    item.relevant_chunk_ids,
                    k,
                )

                metric_rows.append(metrics)

                query_results.append(
                    {
                        "query_id": item.query_id,
                        "query": item.query,
                        "category": item.category,
                        "split": split,
                        "top_k": k,
                        "retrieved_ids": json.dumps(retrieved_ids),
                        "relevant_ids": json.dumps(item.relevant_chunk_ids),
                        **metrics,
                    }
                )

            ranking_summary[split][f"top_{k}"] = summarize_retrieval_metrics(
                metric_rows
            )

    # Select threshold using calibration data only.
    calibration = [item for item in dataset if item.split == "calibration"]

    threshold_results = []

    for threshold in THRESHOLDS:
        expected = []
        predicted = []

        for item in calibration:
            results = cached_results[item.query_id]

            accepted = any(result.score >= threshold for result in results)

            expected.append(item.answerable)
            predicted.append(accepted)

        metrics = answerability_metrics(
            expected,
            predicted,
        )

        threshold_results.append(
            {
                "threshold": threshold,
                **metrics,
            }
        )

    # Prefer maximum F1, then lower false acceptance.
    best = max(
        threshold_results,
        key=lambda row: (
            row["f1"],
            -row["false_acceptance_rate"],
            row["threshold"],
        ),
    )

    selected_threshold = best["threshold"]

    # Evaluate selected threshold on held-out questions.
    test_queries = [item for item in dataset if item.split == "test"]

    expected = []
    predicted = []
    answerability_rows = []
    filtered_metric_rows = []

    for item in test_queries:
        results = cached_results[item.query_id]

        filtered = [result for result in results if result.score >= selected_threshold]

        accepted = bool(filtered)

        expected.append(item.answerable)
        predicted.append(accepted)

        answerability_rows.append(
            {
                "query_id": item.query_id,
                "query": item.query,
                "category": item.category,
                "answerable": item.answerable,
                "accepted": accepted,
                "top_score": (results[0].score if results else None),
                "threshold": selected_threshold,
            }
        )

        if item.answerable:
            filtered_metric_rows.append(
                retrieval_metrics(
                    [result.chunk.chunk_id for result in filtered],
                    item.relevant_chunk_ids,
                    max(TOP_K_VALUES),
                )
            )

    test_answerability = answerability_metrics(
        expected,
        predicted,
    )

    report = {
        "embedding_model": embedder.model_name,
        "embedding_dimension": embedder.dimension,
        "index_checksum": retriever.manifest["index_checksum"],
        "dataset_size": len(dataset),
        "calibration_size": len(calibration),
        "test_size": len(test_queries),
        "indexed_chunks": retriever.store.size,
        "warmup_iterations": WARMUP_ITERATIONS,
        "repetitions_per_query": REPETITIONS,
        "measured_queries": len(latencies),
        "ranking_metrics": ranking_summary,
        "latency_metrics": latency_metrics(latencies),
        "selected_threshold": selected_threshold,
        "calibration_answerability": best,
        "test_answerability": test_answerability,
        "test_filtered_ranking_at_5": (
            summarize_retrieval_metrics(filtered_metric_rows)
        ),
    }

    report_path = OUTPUT_DIR / "retrieval_metrics.json"

    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    write_csv(
        OUTPUT_DIR / "query_results.csv",
        query_results,
    )

    write_csv(
        OUTPUT_DIR / "threshold_analysis.csv",
        threshold_results,
    )

    write_csv(
        OUTPUT_DIR / "answerability_results.csv",
        answerability_rows,
    )

    print("\nBenchmark completed.")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
