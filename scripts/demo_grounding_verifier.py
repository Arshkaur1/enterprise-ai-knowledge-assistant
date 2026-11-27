import json
from dataclasses import replace
from pathlib import Path
from statistics import mean

from knowledge_assistant.config import RAGConfig
from knowledge_assistant.context_builder import ContextBuilder
from knowledge_assistant.embeddings import EmbeddingService
from knowledge_assistant.generation_schema import build_answer_schema
from knowledge_assistant.grounding_verifier import GroundingVerifier
from knowledge_assistant.llm_client import (
    LLMGenerationError,
    LocalLLM,
)
from knowledge_assistant.prompt_builder import PromptBuilder
from knowledge_assistant.response_validator import (
    ResponseValidationError,
    ResponseValidator,
)
from knowledge_assistant.retrieval import SemanticRetriever

PROJECT_ROOT = Path(__file__).resolve().parents[1]


QUESTIONS = [
    (
        "remote_work",
        "How many days per week can employees work remotely?",
        True,
    ),
    (
        "password_length",
        "What is the minimum password length?",
        True,
    ),
    (
        "expense_deadline",
        "What is the deadline for submitting expense claims?",
        True,
    ),
    (
        "multi_part",
        (
            "What is the expense submission deadline and who "
            "approves claims of 500 currency units or more?"
        ),
        True,
    ),
    (
        "maternity_leave",
        "How many paid maternity leave days are provided?",
        False,
    ),
    (
        "password_rotation",
        "How frequently must employees change their passwords?",
        False,
    ),
    (
        "mileage_rate",
        "What is the company's mileage reimbursement rate?",
        False,
    ),
    (
        "out_of_domain",
        "What is the weather forecast for tomorrow?",
        False,
    ),
    (
        "partial_support",
        (
            "What is the minimum password length and how often "
            "must employees change their passwords?"
        ),
        False,
    ),
]


def main():

    config = replace(
        RAGConfig(),
        top_k=2,
        min_score=0.35,
        temperature=0.0,
        prompt_version="1.0",
    )

    embedder = EmbeddingService()

    retriever = SemanticRetriever(
        artifact_dir=PROJECT_ROOT / "artifacts",
        embedding_service=embedder,
    )

    context_builder = ContextBuilder(max_context_chars=config.max_context_chars)

    llm = LocalLLM()

    prompt_builder = PromptBuilder(version=config.prompt_version)

    response_validator = ResponseValidator(normalize=True)

    grounding_verifier = GroundingVerifier(
        llm=llm,
        config=config,
    )

    def evaluate_question(query_id, question, expected):

        retrieved = retriever.search(
            question,
            top_k=config.top_k,
            min_score=config.min_score,
        )

        context = context_builder.build(retrieved)

        record = {
            "query_id": query_id,
            "question": question,
            "expected_answerable": expected,
            "context": context.text,
            "generated_answer": None,
            "generation_status": None,
            "generation_latency_ms": 0.0,
            "generation_tokens": None,
            "grounding": None,
            "final_status": None,
        }

        if not context.citations:
            record["final_status"] = "insufficient_evidence"

            return record

        try:
            generation = llm.generate_messages(
                messages=prompt_builder.build(
                    question,
                    context,
                ),
                config=config,
                response_schema=build_answer_schema(
                    [citation.citation_id for citation in context.citations]
                ),
            )

            record["generation_latency_ms"] = generation.latency_ms

            record["generation_tokens"] = generation.total_tokens

            answer = response_validator.validate(
                generation.text,
                context,
            )

        except (LLMGenerationError, ResponseValidationError) as exc:
            record["final_status"] = "generation_stage_error"
            record["error_message"] = str(exc)

            return record

        record["generated_answer"] = answer.answer
        record["generation_status"] = answer.status.value

        record["citations"] = [citation.to_dict() for citation in answer.citations]

        if not answer.answered:
            record["final_status"] = answer.status.value

            return record

        grounding = grounding_verifier.verify(
            query=question,
            answer=answer,
            context=context,
        )

        record["grounding"] = grounding.to_dict()

        status_mapping = {
            "supported": "answered",
            "unsupported": "insufficient_evidence",
            "invalid_response": "invalid_response",
            "verification_error": "verification_error",
        }

        record["final_status"] = status_mapping[grounding.status.value]

        return record

    print("Warming up generation and verification...")

    evaluate_question(
        "warmup",
        QUESTIONS[1][1],
        True,
    )

    results = []

    for query_id, question, expected in QUESTIONS:
        record = evaluate_question(
            query_id,
            question,
            expected,
        )

        status = record["final_status"]

        record["correct"] = (expected and status == "answered") or (
            not expected and status == "insufficient_evidence"
        )

        results.append(record)

        print(f"{query_id}: {status} | correct={record['correct']}")

    verified = [item["grounding"] for item in results if item["grounding"] is not None]

    tp = sum(
        item["expected_answerable"] and item["final_status"] == "answered"
        for item in results
    )

    tn = sum(
        not item["expected_answerable"]
        and item["final_status"] == "insufficient_evidence"
        for item in results
    )

    fp = sum(
        not item["expected_answerable"] and item["final_status"] == "answered"
        for item in results
    )

    fn = sum(
        item["expected_answerable"] and item["final_status"] == "insufficient_evidence"
        for item in results
    )

    errors = len(results) - tp - tn - fp - fn

    summary = {
        "total_questions": len(results),
        "correct_predictions": tp + tn,
        "accuracy": (tp + tn) / len(results),
        "true_positives": tp,
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "operational_errors": errors,
        "verification_requests": sum(item["llm_requested"] for item in verified),
        "mean_verification_latency_ms": (
            mean(item["latency_ms"] for item in verified) if verified else 0.0
        ),
        "total_reported_verification_tokens": sum(
            item["total_tokens"] or 0 for item in verified
        ),
        "total_reported_generation_tokens": sum(
            item["generation_tokens"] or 0 for item in results
        ),
    }

    report = {
        "model": llm.model,
        "prompt_version": config.prompt_version,
        "grounding_version": grounding_verifier.prompt_builder.version,
        "summary": summary,
        "results": results,
    }

    output = PROJECT_ROOT / "artifacts" / "evaluation" / "grounding_verifier_demo.json"

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

    print(json.dumps(summary, indent=2))
    print(f"\nReport saved to: {output}")


if __name__ == "__main__":
    main()
