import argparse
import csv
import json
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path

from knowledge_assistant.config import RAGConfig
from knowledge_assistant.evaluation import (
    load_evaluation_dataset,
)
from knowledge_assistant.generation_evaluation import (
    evaluate_response,
    summarize_generation_results,
)
from knowledge_assistant.rag import RAGEngine

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description="Evaluate end-to-end RAG generation.")

    parser.add_argument(
        "--split",
        choices=["calibration", "test", "all"],
        default="calibration",
    )

    parser.add_argument(
        "--experiment",
        default="rag_baseline_v1",
    )

    parser.add_argument(
        "--model",
        default=None,
    )

    parser.add_argument(
        "--prompt-version",
        choices=["1.0", "2.0"],
        default=None,
        help="Override the configured prompt version.",
    )

    args = parser.parse_args()

    dataset = load_evaluation_dataset(
        PROJECT_ROOT / "data/evaluation/retrieval_eval.jsonl"
    )

    examples = [
        example
        for example in dataset
        if args.split == "all" or example.split == args.split
    ]

    if not examples:
        raise ValueError("No evaluation queries selected.")

    config = RAGConfig()

    if args.prompt_version is not None:
        config = replace(
            config,
            prompt_version=args.prompt_version,
        )

    engine = RAGEngine.from_artifacts(
        artifact_dir=PROJECT_ROOT / "artifacts",
        config=config,
        llm_model=args.model,
    )

    print("EXPERIMENT:", args.experiment)
    print("SPLIT:", args.split)
    print("MODEL:", engine.llm.model)
    print("QUESTIONS:", len(examples))
    print("PROMPT VERSION:", engine.prompt_builder.version)

    # Warm-up request excluded from evaluation metrics.
    print("\nWarming up pipeline...")

    engine.ask("What is the minimum password length?")

    results = []

    for index, example in enumerate(examples, start=1):
        response = engine.ask(example.query)

        result = evaluate_response(
            example=example,
            response=response,
        )

        results.append(result)

        label = "CORRECT STATUS" if result.answerability_correct else "INCORRECT STATUS"

        print(
            f"[{index}/{len(examples)}] "
            f"{example.query_id} | "
            f"{label} | "
            f"{result.total_latency_ms:.2f} ms"
        )

    summary = summarize_generation_results(results)

    timestamp = datetime.now(UTC)

    run_id = timestamp.strftime("%Y%m%dT%H%M%S%fZ")

    output_dir = (
        PROJECT_ROOT
        / "artifacts/evaluation/generation"
        / args.experiment
        / args.split
        / run_id
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    report = {
        "experiment": args.experiment,
        "created_at": timestamp.isoformat(),
        "split": args.split,
        "model": engine.llm.model,
        "prompt_version": engine.prompt_builder.version,
        "configuration": asdict(config),
        "summary": summary,
        "results": [result.to_dict() for result in results],
    }

    (output_dir / "evaluation_report.json").write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    rows = [result.to_dict() for result in results]

    with (output_dir / "query_results.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0]),
        )

        writer.writeheader()

        for row in rows:
            writer.writerow(
                {
                    key: (json.dumps(value) if isinstance(value, list) else value)
                    for key, value in row.items()
                }
            )

    # Create a human-review worksheet.
    review_fields = [
        "query_id",
        "query",
        "expected_answerable",
        "expected_answer",
        "generated_answer",
        "status",
        "cited_chunk_ids",
        "correctness",
        "grounding",
        "completeness",
        "relevance",
        "abstention_correct",
        "review_notes",
    ]

    with (output_dir / "manual_review.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=review_fields,
        )

        writer.writeheader()

        for result in results:
            writer.writerow(
                {
                    "query_id": result.query_id,
                    "query": result.query,
                    "expected_answerable": (result.expected_answerable),
                    "expected_answer": result.expected_answer,
                    "generated_answer": result.generated_answer,
                    "status": result.status,
                    "cited_chunk_ids": json.dumps(result.cited_chunk_ids),
                    "correctness": "",
                    "grounding": "",
                    "completeness": "",
                    "relevance": "",
                    "abstention_correct": "",
                    "review_notes": "",
                }
            )

    print("\nEVALUATION SUMMARY")
    print(json.dumps(summary, indent=2))

    print("\nReports saved to:", output_dir)


if __name__ == "__main__":
    main()
