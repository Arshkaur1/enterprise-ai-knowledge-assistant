import json
from unittest.mock import Mock

import pytest

from knowledge_assistant.grounding_verifier import (
    GroundingVerifier,
)
from knowledge_assistant.llm_client import LLMGenerationError
from knowledge_assistant.models import (
    AnswerStatus,
    Citation,
    ContextBundle,
    GenerationResult,
    GroundingStatus,
    ValidatedAnswer,
)


@pytest.fixture
def context():

    text = "Visitors may park for two hours."

    citation = Citation(
        citation_id=1,
        chunk_id="parking_chunk_0",
        document_id="parking",
        source="parking.txt",
        category="General",
        score=0.85,
    )

    return ContextBundle(
        text=text,
        citations=[citation],
        retrieved_count=1,
        included_count=1,
        character_count=len(text),
        source_texts={1: text},
    )


@pytest.fixture
def answer(context):

    return ValidatedAnswer(
        answer="Visitors may park for two hours. [1]",
        status=AnswerStatus.ANSWERED,
        citations=context.citations,
    )


@pytest.fixture
def mock_llm():

    llm = Mock()
    llm.model = "test-model"

    return llm


@pytest.fixture
def verifier(mock_llm):

    return GroundingVerifier(mock_llm)


@pytest.fixture
def supported_payload():

    return {
        "status": "supported",
        "reason": "The answer matches the parking policy.",
        "evidence": [
            {
                "citation_id": 1,
                "quote": "Visitors may park for two hours.",
            }
        ],
        "unsupported_claims": [],
    }


def set_response(llm, payload):

    llm.generate_messages.return_value = GenerationResult(
        text=json.dumps(payload),
        model="test-model",
        latency_ms=100.0,
        prompt_tokens=300,
        completion_tokens=50,
    )


def test_supported_answer(verifier, mock_llm, context, answer, supported_payload):

    set_response(mock_llm, supported_payload)

    result = verifier.verify(
        "How long may visitors park?",
        answer,
        context,
    )

    assert result.status == GroundingStatus.SUPPORTED
    assert result.supported
    assert result.total_tokens == 350
    assert result.llm_requested is True


def test_unsupported_answer(verifier, mock_llm, context, answer):

    set_response(
        mock_llm,
        {
            "status": "unsupported",
            "reason": "Parking cost is not specified.",
            "evidence": [],
            "unsupported_claims": ["Parking cost"],
        },
    )

    result = verifier.verify(
        "What is the parking cost?",
        answer,
        context,
    )

    assert result.status == GroundingStatus.UNSUPPORTED
    assert not result.supported


@pytest.mark.parametrize(
    "quote",
    [
        "Visitors may park for ten hours.",
        "",
        " ",
        "Parking is free.",
    ],
)
def test_invalid_quote(
    verifier,
    mock_llm,
    context,
    answer,
    supported_payload,
    quote,
):

    supported_payload["evidence"][0]["quote"] = quote

    set_response(mock_llm, supported_payload)

    result = verifier.verify(
        "Parking duration?",
        answer,
        context,
    )

    assert result.status == GroundingStatus.INVALID_RESPONSE
    assert result.total_tokens == 350


@pytest.mark.parametrize("citation_id", [0, 2, True, "1"])
def test_invalid_citation(
    verifier,
    mock_llm,
    context,
    answer,
    supported_payload,
    citation_id,
):

    supported_payload["evidence"][0]["citation_id"] = citation_id

    set_response(mock_llm, supported_payload)

    result = verifier.verify(
        "Parking duration?",
        answer,
        context,
    )

    assert result.status == GroundingStatus.INVALID_RESPONSE


def test_supported_without_evidence(
    verifier, mock_llm, context, answer, supported_payload
):

    supported_payload["evidence"] = []

    set_response(mock_llm, supported_payload)

    result = verifier.verify(
        "Parking duration?",
        answer,
        context,
    )

    assert result.status == GroundingStatus.INVALID_RESPONSE


def test_supported_with_unsupported_claims(
    verifier, mock_llm, context, answer, supported_payload
):

    supported_payload["unsupported_claims"] = ["Parking cost"]

    set_response(mock_llm, supported_payload)

    result = verifier.verify(
        "Parking duration?",
        answer,
        context,
    )

    assert result.status == GroundingStatus.INVALID_RESPONSE


def test_generation_error(verifier, mock_llm, context, answer):

    mock_llm.generate_messages.side_effect = LLMGenerationError("Ollama unavailable")

    result = verifier.verify(
        "Parking duration?",
        answer,
        context,
    )

    assert result.status == GroundingStatus.VERIFICATION_ERROR
    assert result.llm_requested is True


def test_missing_source(verifier, mock_llm, context, answer):

    context.source_texts.clear()

    result = verifier.verify(
        "Parking duration?",
        answer,
        context,
    )

    assert result.status == GroundingStatus.INVALID_RESPONSE
    assert result.llm_requested is False

    mock_llm.generate_messages.assert_not_called()


def test_empty_query(verifier, context, answer):

    with pytest.raises(ValueError):
        verifier.verify("", answer, context)


def test_invalid_json(verifier, mock_llm, context, answer):

    mock_llm.generate_messages.return_value = GenerationResult(
        text="not valid JSON",
        model="test-model",
        latency_ms=100,
    )

    result = verifier.verify(
        "Parking duration?",
        answer,
        context,
    )

    assert result.status == GroundingStatus.INVALID_RESPONSE


def test_unsupported_without_claims(verifier, mock_llm, context, answer):

    set_response(
        mock_llm,
        {
            "status": "unsupported",
            "reason": "Insufficient support.",
            "evidence": [],
            "unsupported_claims": [],
        },
    )

    result = verifier.verify(
        "Parking cost?",
        answer,
        context,
    )

    assert result.status == GroundingStatus.INVALID_RESPONSE


def test_abstention_not_verified(verifier, mock_llm, context, answer):

    answer.status = AnswerStatus.INSUFFICIENT_EVIDENCE

    with pytest.raises(ValueError):
        verifier.verify(
            "Parking duration?",
            answer,
            context,
        )

    mock_llm.generate_messages.assert_not_called()


def test_grounding_serialization(
    verifier, mock_llm, context, answer, supported_payload
):

    set_response(mock_llm, supported_payload)

    result = verifier.verify(
        "Parking duration?",
        answer,
        context,
    )

    payload = result.to_dict()

    assert payload["status"] == "supported"
    assert payload["supported"] is True
    assert payload["total_tokens"] == 350
    assert payload["verifier_version"] == "1.0"
