import json

from knowledge_assistant.models import (
    AnswerStatus,
    Citation,
    ContextBundle,
    GenerationResult,
    RAGResponse,
)


def create_citation():
    return Citation(
        citation_id=1,
        chunk_id="hr_policy_chunk_0000",
        document_id="hr_policy",
        source="hr_policy.txt",
        category="HR",
        score=0.75,
    )


def create_response(status=AnswerStatus.ANSWERED):
    answered = status == AnswerStatus.ANSWERED

    return RAGResponse(
        query="How many days can employees work remotely?",
        answer=(
            "Up to three days per week. [1]" if answered else "Insufficient evidence."
        ),
        status=status,
        citations=[create_citation()] if answered else [],
        model="test-model",
        retrieved_count=1 if answered else 0,
        context_chunk_count=1 if answered else 0,
        retrieval_latency_ms=4.0,
        generation_latency_ms=100.0 if answered else 0.0,
        total_latency_ms=105.0 if answered else 5.0,
        prompt_tokens=100 if answered else None,
        completion_tokens=20 if answered else None,
    )


def test_citation_serialization():
    citation = create_citation()
    result = citation.to_dict()

    assert result["citation_id"] == 1
    assert result["source"] == "hr_policy.txt"
    assert result["score"] == 0.75
    assert result["page_number"] is None


def test_pdf_citation():
    citation = Citation(
        citation_id=1,
        chunk_id="finance_policy_p1_chunk_0000",
        document_id="finance_policy",
        source="finance_policy.pdf",
        category="Finance",
        score=0.85,
        page_number=1,
    )

    assert citation.to_dict()["page_number"] == 1


def test_context_bundle():
    text = "[1] Employees may work remotely."

    bundle = ContextBundle(
        text=text,
        citations=[create_citation()],
        retrieved_count=1,
        included_count=1,
        character_count=len(text),
    )

    assert bundle.included_count == 1
    assert bundle.character_count == len(text)
    assert not bundle.truncated


def test_generation_total_tokens():
    result = GenerationResult(
        text="Example answer",
        model="test-model",
        latency_ms=100.0,
        prompt_tokens=150,
        completion_tokens=30,
    )

    assert result.total_tokens == 180


def test_generation_missing_token_metrics():
    result = GenerationResult(
        text="Example answer",
        model="test-model",
        latency_ms=100.0,
    )

    assert result.total_tokens is None


def test_answered_response():
    response = create_response()

    assert response.answered
    assert response.total_tokens == 120
    assert len(response.citations) == 1


def test_insufficient_evidence_response():
    response = create_response(AnswerStatus.INSUFFICIENT_EVIDENCE)

    assert not response.answered
    assert response.citations == []
    assert response.total_tokens is None


def test_response_serialization():
    response = create_response()
    result = response.to_dict()

    assert result["status"] == "answered"
    assert result["answered"] is True
    assert result["total_tokens"] == 120

    assert result["citations"][0]["source"] == "hr_policy.txt"


def test_response_json_compatibility():
    response = create_response()

    serialized = json.dumps(response.to_dict())
    restored = json.loads(serialized)

    assert restored["status"] == "answered"
    assert restored["total_tokens"] == 120


def test_generation_error_status():
    response = create_response(AnswerStatus.GENERATION_ERROR)

    assert not response.answered
    assert response.status.value == "generation_error"
