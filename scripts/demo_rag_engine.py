import json
import logging
from pathlib import Path
from statistics import mean

from knowledge_assistant.config import RAGConfig
from knowledge_assistant.models import AnswerStatus
from knowledge_assistant.rag import RAGEngine

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s | %(name)s | %(message)s",
    )

    config = RAGConfig()

    engine = RAGEngine.from_artifacts(
        artifact_dir=PROJECT_ROOT / "artifacts",
        config=config,
    )

    queries = [
        "How many days per week can employees work remotely?",
        "What is the deadline for submitting expense claims?",
        (
            "What is the minimum password length and how should "
            "unattended devices be secured?"
        ),
        "What is the company's mileage reimbursement rate?",
        "What is the weather forecast for tomorrow?",
        "How many days of annual paid leave do full-time employees receive?",
        "Who must approve an expense claim of 500 currency units or more?",
        "How frequently must employees change their passwords?",
    ]

    responses = []

    print("MODEL:", engine.llm.model)
    print("TOP K:", config.top_k)
    print("MIN SCORE:", config.min_score)

    for query in queries:
        print("\n" + "=" * 80)

        response = engine.ask(query)

        responses.append(response)

        print(
            json.dumps(
                response.to_dict(),
                indent=2,
                ensure_ascii=False,
            )
        )

    generation_responses = [
        response for response in responses if response.generation_latency_ms > 0
    ]

    summary = {
        "total_queries": len(responses),
        "answered": sum(
            response.status == AnswerStatus.ANSWERED for response in responses
        ),
        "insufficient_evidence": sum(
            response.status == AnswerStatus.INSUFFICIENT_EVIDENCE
            for response in responses
        ),
        "invalid_responses": sum(
            response.status == AnswerStatus.INVALID_RESPONSE for response in responses
        ),
        "generation_errors": sum(
            response.status == AnswerStatus.GENERATION_ERROR for response in responses
        ),
        "mean_retrieval_latency_ms": round(
            mean(response.retrieval_latency_ms for response in responses),
            2,
        ),
        "mean_generation_latency_ms": (
            round(
                mean(
                    response.generation_latency_ms for response in generation_responses
                ),
                2,
            )
            if generation_responses
            else None
        ),
        "mean_end_to_end_latency_ms": round(
            mean(response.total_latency_ms for response in responses),
            2,
        ),
        "total_reported_tokens": sum(
            response.total_tokens or 0 for response in responses
        ),
        "responses_with_token_metrics": sum(
            response.total_tokens is not None for response in responses
        ),
        "responses_with_warnings": sum(
            bool(response.warnings) for response in responses
        ),
    }

    print("\n" + "=" * 80)
    print("RAG ENGINE SUMMARY")
    print(json.dumps(summary, indent=2))

    output_path = PROJECT_ROOT / "artifacts/evaluation/rag_engine_demo.json"

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    report = {
        "summary": summary,
        "responses": [response.to_dict() for response in responses],
    }

    output_path.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("\nReport saved to:", output_path)


if __name__ == "__main__":
    main()
