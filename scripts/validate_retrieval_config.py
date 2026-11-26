import json
from pathlib import Path

from knowledge_assistant.embeddings import EmbeddingService
from knowledge_assistant.evaluation import (
    answerability_metrics,
    load_evaluation_dataset,
    retrieval_metrics,
    summarize_retrieval_metrics,
)
from knowledge_assistant.retrieval import SemanticRetriever

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TOP_K = 2
MIN_SCORE = 0.35


def main():
    dataset = load_evaluation_dataset(
        PROJECT_ROOT / "data/evaluation/retrieval_eval.jsonl"
    )

    embedder = EmbeddingService()

    retriever = SemanticRetriever(
        artifact_dir=PROJECT_ROOT / "artifacts",
        embedding_service=embedder,
    )

    test_queries = [item for item in dataset if item.split == "test"]

    expected = []
    predicted = []
    ranking_rows = []
    errors = []

    for item in test_queries:
        results = retriever.search(
            query=item.query,
            top_k=TOP_K,
            min_score=MIN_SCORE,
        )

        accepted = bool(results)

        expected.append(item.answerable)
        predicted.append(accepted)

        retrieved_ids = [result.chunk.chunk_id for result in results]

        if item.answerable:
            ranking_rows.append(
                retrieval_metrics(
                    retrieved_ids,
                    item.relevant_chunk_ids,
                    TOP_K,
                )
            )

        if accepted != item.answerable:
            errors.append(
                {
                    "query_id": item.query_id,
                    "query": item.query,
                    "expected_answerable": item.answerable,
                    "accepted": accepted,
                    "retrieved_chunks": [
                        {
                            "chunk_id": result.chunk.chunk_id,
                            "score": result.score,
                        }
                        for result in results
                    ],
                }
            )

    report = {
        "top_k": TOP_K,
        "min_score": MIN_SCORE,
        "test_queries": len(test_queries),
        "ranking_metrics": summarize_retrieval_metrics(ranking_rows),
        "answerability_metrics": answerability_metrics(
            expected,
            predicted,
        ),
        "classification_errors": errors,
    }

    output_path = PROJECT_ROOT / "artifacts/evaluation/selected_config.json"

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
