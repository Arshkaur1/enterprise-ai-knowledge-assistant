import json
from dataclasses import replace
from pathlib import Path

from knowledge_assistant.config import RAGConfig
from knowledge_assistant.context_builder import ContextBuilder
from knowledge_assistant.embeddings import EmbeddingService
from knowledge_assistant.evidence_verifier import EvidenceVerifier
from knowledge_assistant.llm_client import LocalLLM
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

    config = RAGConfig(
        top_k=2,
        min_score=0.35,
        temperature=0.0,
        prompt_version="1.0",
    )

    verification_config = replace(
        config,
        max_tokens=512,
    )

    embedding_service = EmbeddingService()

    retriever = SemanticRetriever(
        artifact_dir=PROJECT_ROOT / "artifacts",
        embedding_service=embedding_service,
    )

    context_builder = ContextBuilder(
        max_context_chars=config.max_context_chars,
    )

    llm = LocalLLM()

    verifier = EvidenceVerifier(
        llm=llm,
        config=verification_config,
    )

    results = []

    print("EVIDENCE VERIFICATION DEMO")
    print("MODEL:", llm.model)
    print("VERIFIER VERSION:", verifier.prompt_builder.version)

    for query_id, question, expected_sufficient in QUESTIONS:
        retrieved = retriever.search(
            query=question,
            top_k=config.top_k,
            min_score=config.min_score,
        )

        context = context_builder.build(retrieved)

        result = verifier.verify(
            query=question,
            context=context,
        )

        expected_status = "sufficient" if expected_sufficient else "insufficient"

        correct = result.status.value == expected_status

        record = {
            "query_id": query_id,
            "question": question,
            "expected_sufficient": expected_sufficient,
            "expected_status": expected_status,
            "correct": correct,
            "context": context.text,
            "retrieved_chunk_ids": [item.chunk.chunk_id for item in retrieved],
            "context_citations": [citation.to_dict() for citation in context.citations],
            "verification": result.to_dict(),
        }

        results.append(record)

        print("\n" + "=" * 70)
        print("QUESTION:", question)
        print("EXPECTED:", expected_status)
        print("PREDICTED:", result.status.value)
        print("CORRECT:", correct)
        print("REASON:", result.reason)

        print("\nEVIDENCE:")

        for item in result.evidence:
            print(f"[{item.citation_id}] {item.quote}")

        print("\nMISSING INFORMATION:")
        print(result.missing_information)

        print(f"VERIFICATION LATENCY: {result.latency_ms:.2f} ms")

        print("TOKENS:", result.total_tokens)

        if result.error_message:
            print("ERROR:", result.error_message)

    total = len(results)

    correct_count = sum(item["correct"] for item in results)

    summary = {
        "total_questions": total,
        "correct_predictions": correct_count,
        "accuracy": correct_count / total,
        "sufficient": sum(
            item["verification"]["status"] == "sufficient" for item in results
        ),
        "insufficient": sum(
            item["verification"]["status"] == "insufficient" for item in results
        ),
        "invalid_responses": sum(
            item["verification"]["status"] == "invalid_response" for item in results
        ),
        "verification_errors": sum(
            item["verification"]["status"] == "verification_error" for item in results
        ),
        "mean_verification_latency_ms": sum(
            item["verification"]["latency_ms"] for item in results
        )
        / total,
        "total_reported_tokens": sum(
            item["verification"]["total_tokens"] or 0 for item in results
        ),
    }

    output_path = (
        PROJECT_ROOT / "artifacts" / "evaluation" / "evidence_verifier_demo.json"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    report = {
        "model": llm.model,
        "verifier_version": verifier.prompt_builder.version,
        "summary": summary,
        "results": results,
    }

    output_path.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("\n" + "=" * 70)
    print("VERIFICATION SUMMARY")
    print(json.dumps(summary, indent=2))
    print("\nReport saved to:", output_path)


if __name__ == "__main__":
    main()
