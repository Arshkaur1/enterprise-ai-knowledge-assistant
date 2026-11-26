import json
import os
from pathlib import Path

import pytest

from knowledge_assistant.config import RAGConfig
from knowledge_assistant.models import AnswerStatus
from knowledge_assistant.rag import RAGEngine

PROJECT_ROOT = Path(__file__).resolve().parents[2]

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.getenv("RUN_INTEGRATION_TESTS") != "1",
        reason="Real-model integration tests are opt-in.",
    ),
]


@pytest.fixture(scope="module")
def real_engine():
    """Initialize the real RAG pipeline once."""

    return RAGEngine.from_artifacts(
        artifact_dir=PROJECT_ROOT / "artifacts",
        config=RAGConfig(
            top_k=2,
            min_score=0.35,
            max_context_chars=4000,
            temperature=0.0,
        ),
    )


@pytest.mark.parametrize(
    "query, expected_text, expected_source",
    [
        (
            "How many days per week can employees work remotely?",
            ("three", "3"),
            "hr_policy.txt",
        ),
        (
            "What is the deadline for submitting expense claims?",
            ("30", "thirty"),
            "finance_policy.pdf",
        ),
        (
            "What is the minimum password length?",
            ("12", "twelve"),
            "it_security.docx",
        ),
        (
            "How many days of annual paid leave do full-time employees receive?",
            ("20", "twenty"),
            "hr_policy.txt",
        ),
    ],
)
def test_answerable_questions(
    real_engine,
    query,
    expected_text,
    expected_source,
):
    response = real_engine.ask(query)

    assert response.status == AnswerStatus.ANSWERED

    assert any(value in response.answer.lower() for value in expected_text)

    assert any(citation.source == expected_source for citation in response.citations)

    assert response.error_message is None


def test_multi_chunk_question(real_engine):
    response = real_engine.ask(
        "What is the minimum password length and how "
        "should unattended devices be secured?"
    )

    assert response.answered
    assert "12" in response.answer or "twelve" in response.answer.lower()
    assert "lock" in response.answer.lower()
    assert len(response.citations) == 2
    assert response.context_chunk_count == 2


def test_out_of_domain_abstention(real_engine):
    response = real_engine.ask("What is the weather forecast for tomorrow?")

    assert response.status == AnswerStatus.INSUFFICIENT_EVIDENCE

    assert response.citations == []
    assert response.retrieved_count == 0
    assert response.generation_latency_ms == 0
    assert response.total_tokens is None


def test_missing_evidence_abstention(real_engine):
    response = real_engine.ask("What is the company's mileage reimbursement rate?")

    assert response.status == AnswerStatus.INSUFFICIENT_EVIDENCE

    assert response.retrieved_count > 0
    assert response.generation_latency_ms > 0
    assert response.citations == []


@pytest.mark.xfail(
    reason=(
        "Known baseline failure: model answers password "
        "complexity instead of abstaining on rotation frequency."
    ),
    strict=False,
)
def test_password_frequency_abstention(real_engine):
    response = real_engine.ask("How frequently must employees change their passwords?")

    assert response.status == AnswerStatus.INSUFFICIENT_EVIDENCE

    assert response.citations == []


def test_pdf_citation_metadata(real_engine):
    response = real_engine.ask("What is the deadline for submitting expense claims?")

    assert response.answered

    assert any(
        citation.source == "finance_policy.pdf" and citation.page_number == 1
        for citation in response.citations
    )


def test_generation_metrics(real_engine):
    response = real_engine.ask("What is the minimum password length?")

    assert response.answered
    assert response.retrieval_latency_ms >= 0
    assert response.generation_latency_ms > 0

    assert response.total_latency_ms >= response.generation_latency_ms

    assert response.prompt_tokens is not None
    assert response.completion_tokens is not None
    assert response.prompt_tokens > 0
    assert response.completion_tokens > 0

    assert response.total_tokens == (
        response.prompt_tokens + response.completion_tokens
    )


def test_context_budget(real_engine):
    response = real_engine.ask("What is the remote work policy?")

    assert response.answered

    assert response.context_characters <= real_engine.config.max_context_chars

    assert response.context_chunk_count <= real_engine.config.top_k


def test_response_json_serialization(real_engine):
    response = real_engine.ask("What is the minimum password length?")

    assert response.answered

    serialized = json.dumps(response.to_dict())
    payload = json.loads(serialized)

    assert payload["status"] == "answered"
    assert payload["answered"] is True
    assert isinstance(payload["citations"], list)
    assert payload["total_tokens"] > 0


def test_repeated_queries(real_engine):
    query = "How many days per week can employees work remotely?"

    first = real_engine.ask(query)
    second = real_engine.ask(query)

    assert first.answered
    assert second.answered

    assert first.citations[0].chunk_id == second.citations[0].chunk_id


def test_independent_questions(real_engine):
    first = real_engine.ask("What is the minimum password length?")

    second = real_engine.ask("How many days per week can employees work remotely?")

    assert first.answered
    assert second.answered

    assert first.query != second.query

    assert any(citation.source == "it_security.docx" for citation in first.citations)

    assert any(citation.source == "hr_policy.txt" for citation in second.citations)
