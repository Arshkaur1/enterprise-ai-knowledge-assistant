import json
from unittest.mock import Mock

import pytest

from knowledge_assistant.config import RAGConfig
from knowledge_assistant.llm_client import LLMGenerationError
from knowledge_assistant.models import (
    AnswerStatus,
    DocumentChunk,
    GenerationResult,
)
from knowledge_assistant.rag import RAGEngine
from knowledge_assistant.retrieval import SearchResult


@pytest.fixture
def retriever():
    chunk = DocumentChunk(
        chunk_id="hr_policy_chunk_0000",
        document_id="hr_policy",
        source="hr_policy.txt",
        file_type="txt",
        category="HR",
        text=("Employees may work remotely up to three days per week."),
    )

    retriever = Mock()

    retriever.search.return_value = [
        SearchResult(
            rank=1,
            score=0.85,
            chunk=chunk,
        )
    ]

    return retriever


@pytest.fixture
def llm():
    llm = Mock()
    llm.model = "test-model"

    llm.generate_messages.return_value = GenerationResult(
        text=json.dumps(
            {
                "status": "answered",
                "answer": "Up to three days per week. [1]",
                "citations": [1],
            }
        ),
        model="test-model",
        latency_ms=100.0,
        prompt_tokens=300,
        completion_tokens=25,
    )

    return llm


@pytest.fixture
def engine(retriever, llm):
    return RAGEngine(
        retriever=retriever,
        llm=llm,
        config=RAGConfig(
            top_k=2,
            min_score=0.35,
            max_context_chars=4000,
        ),
    )


def test_successful_answer(engine):
    response = engine.ask("How many remote working days?")

    assert response.status == AnswerStatus.ANSWERED
    assert response.answered
    assert "three days" in response.answer


def test_citation_resolution(engine):
    response = engine.ask("Remote work policy?")

    assert len(response.citations) == 1
    assert response.citations[0].source == "hr_policy.txt"


def test_token_metrics(engine):
    response = engine.ask("Remote work policy?")

    assert response.prompt_tokens == 300
    assert response.completion_tokens == 25
    assert response.total_tokens == 325


def test_latency_metrics(engine):
    response = engine.ask("Remote work policy?")

    assert response.retrieval_latency_ms >= 0
    assert response.generation_latency_ms == 100.0
    assert response.total_latency_ms >= 0


def test_retrieval_configuration(engine, retriever):
    engine.ask("Remote work policy?")

    retriever.search.assert_called_once_with(
        query="Remote work policy?",
        top_k=2,
        min_score=0.35,
    )


def test_empty_query(engine, retriever, llm):
    with pytest.raises(ValueError):
        engine.ask("   ")

    retriever.search.assert_not_called()
    llm.generate_messages.assert_not_called()


def test_query_whitespace(engine):
    response = engine.ask("  Remote work policy?  ")

    assert response.query == "Remote work policy?"


def test_no_evidence(engine, retriever):
    retriever.search.return_value = []

    response = engine.ask("What is the weather?")

    assert response.status == AnswerStatus.INSUFFICIENT_EVIDENCE
    assert not response.answered
    assert response.citations == []
    assert response.retrieved_count == 0


def test_generation_skipped(engine, retriever, llm):
    retriever.search.return_value = []

    response = engine.ask("What is the weather?")

    llm.generate_messages.assert_not_called()

    assert response.generation_latency_ms == 0
    assert response.total_tokens is None


def test_model_abstention(engine, llm):
    llm.generate_messages.return_value.text = json.dumps(
        {
            "status": "insufficient_evidence",
            "answer": "Information not available.",
            "citations": [],
        }
    )

    response = engine.ask("What is the mileage rate?")

    assert response.status == AnswerStatus.INSUFFICIENT_EVIDENCE
    assert response.citations == []
    assert response.retrieved_count == 1
    assert response.total_tokens == 325


def test_generation_error(engine, llm):
    llm.generate_messages.side_effect = LLMGenerationError("Ollama unavailable")

    response = engine.ask("Remote work policy?")

    assert response.status == AnswerStatus.GENERATION_ERROR
    assert not response.answered
    assert response.error_message is not None
    assert response.citations == []


def test_invalid_json_response(engine, llm):
    llm.generate_messages.return_value.text = "Invalid JSON"

    response = engine.ask("Remote work policy?")

    assert response.status == AnswerStatus.INVALID_RESPONSE
    assert not response.answered
    assert response.total_tokens == 325


def test_unknown_citation(engine, llm):
    llm.generate_messages.return_value.text = json.dumps(
        {
            "status": "answered",
            "answer": "Three days. [99]",
            "citations": [99],
        }
    )

    response = engine.ask("Remote work policy?")

    assert response.status == AnswerStatus.INVALID_RESPONSE
    assert response.citations == []


def test_normalization_warnings(engine, llm):
    llm.generate_messages.return_value.text = json.dumps(
        {
            "status": "answered",
            "answer": "Three days per week.",
            "citations": [1],
        }
    )

    response = engine.ask("Remote work policy?")

    assert response.answered
    assert response.answer.endswith("[1]")
    assert "answer_level_citations_appended" in response.warnings


def test_context_statistics(engine):
    response = engine.ask("Remote work policy?")

    assert response.retrieved_count == 1
    assert response.context_chunk_count == 1
    assert response.context_characters > 0
    assert not response.context_truncated


def test_prompt_version(engine):
    response = engine.ask("Remote work policy?")

    assert response.prompt_version == "1.0"


def test_schema_forwarded(engine, llm):
    engine.ask("Remote work policy?")

    kwargs = llm.generate_messages.call_args.kwargs

    schema = kwargs["response_schema"]

    assert schema["properties"]["citations"]["items"]["enum"] == [1]


def test_system_and_user_messages(engine, llm):
    engine.ask("Remote work policy?")

    messages = llm.generate_messages.call_args.kwargs["messages"]

    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"


def test_response_serialization(engine):
    response = engine.ask("Remote work policy?")

    payload = response.to_dict()

    assert payload["status"] == "answered"
    assert payload["answered"] is True
    assert payload["total_tokens"] == 325

    json.dumps(payload)


def test_context_budget_exhausted(retriever, llm):
    engine = RAGEngine(
        retriever=retriever,
        llm=llm,
        config=RAGConfig(max_context_chars=10),
    )

    response = engine.ask("Remote work policy?")

    assert response.status == AnswerStatus.INSUFFICIENT_EVIDENCE
    assert response.retrieved_count == 1
    assert response.context_chunk_count == 0
    assert response.context_truncated

    llm.generate_messages.assert_not_called()


def test_repeated_queries(engine, retriever, llm):
    first = engine.ask("Remote work policy?")
    second = engine.ask("How many remote working days?")

    assert first.answered
    assert second.answered
    assert first.query != second.query

    assert retriever.search.call_count == 2
    assert llm.generate_messages.call_count == 2


def test_retrieval_failure_propagates(engine, retriever):
    retriever.search.side_effect = RuntimeError("Retrieval failed")

    with pytest.raises(RuntimeError, match="Retrieval failed"):
        engine.ask("Remote work policy?")
