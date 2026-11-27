import json
from unittest.mock import Mock

import pytest

from knowledge_assistant.evidence_verifier import (
    EvidenceVerifier,
)
from knowledge_assistant.llm_client import LLMGenerationError
from knowledge_assistant.models import (
    Citation,
    ContextBundle,
    EvidenceStatus,
    GenerationResult,
)


@pytest.fixture
def context():
    text = "[1] Visitors may park for two hours."

    return ContextBundle(
        text=text,
        source_texts={1: "Visitors may park for two hours."},
        citations=[
            Citation(
                citation_id=1,
                chunk_id="parking_chunk_0",
                document_id="parking",
                source="parking.txt",
                category="General",
                score=0.85,
            )
        ],
        retrieved_count=1,
        included_count=1,
        character_count=len(text),
    )


@pytest.fixture
def empty_context():
    return ContextBundle(
        text="",
        citations=[],
        retrieved_count=0,
        included_count=0,
        character_count=0,
    )


@pytest.fixture
def mock_llm():
    llm = Mock()
    llm.model = "test-model"

    return llm


@pytest.fixture
def verifier(mock_llm):
    return EvidenceVerifier(llm=mock_llm)


@pytest.fixture
def sufficient_payload():
    return {
        "status": "sufficient",
        "reason": "Parking duration is explicitly stated.",
        "evidence": [
            {
                "citation_id": 1,
                "quote": "Visitors may park for two hours.",
            }
        ],
        "missing_information": [],
    }


@pytest.fixture
def insufficient_payload():
    return {
        "status": "insufficient",
        "reason": "Parking cost is missing.",
        "evidence": [],
        "missing_information": ["Parking cost"],
    }


def set_response(llm, payload):
    llm.generate_messages.return_value = GenerationResult(
        text=json.dumps(payload),
        model="test-model",
        latency_ms=100.0,
        prompt_tokens=400,
        completion_tokens=50,
    )


def test_sufficient_evidence(
    verifier,
    mock_llm,
    context,
    sufficient_payload,
):
    set_response(mock_llm, sufficient_payload)

    result = verifier.verify(
        "How long may visitors park?",
        context,
    )

    assert result.status == EvidenceStatus.SUFFICIENT
    assert result.sufficient is True
    assert len(result.evidence) == 1
    assert result.missing_information == []


def test_insufficient_evidence(
    verifier,
    mock_llm,
    context,
    insufficient_payload,
):
    set_response(mock_llm, insufficient_payload)

    result = verifier.verify(
        "How much does parking cost?",
        context,
    )

    assert result.status == EvidenceStatus.INSUFFICIENT
    assert result.sufficient is False
    assert result.missing_information == ["Parking cost"]


def test_partial_evidence(
    verifier,
    mock_llm,
    context,
    insufficient_payload,
):
    insufficient_payload["evidence"] = [
        {
            "citation_id": 1,
            "quote": "Visitors may park for two hours.",
        }
    ]

    set_response(mock_llm, insufficient_payload)

    result = verifier.verify(
        "What are the parking duration and cost?",
        context,
    )

    assert result.status == EvidenceStatus.INSUFFICIENT
    assert len(result.evidence) == 1


def test_empty_context_skips_generation(
    verifier,
    mock_llm,
    empty_context,
):
    result = verifier.verify(
        "Parking duration?",
        empty_context,
    )

    assert result.status == EvidenceStatus.INSUFFICIENT

    mock_llm.generate_messages.assert_not_called()


def test_empty_query(verifier, context):
    with pytest.raises(ValueError):
        verifier.verify(" ", context)


def test_generation_error(
    verifier,
    mock_llm,
    context,
):
    mock_llm.generate_messages.side_effect = LLMGenerationError("Connection failed.")

    result = verifier.verify(
        "Parking duration?",
        context,
    )

    assert result.status == EvidenceStatus.VERIFICATION_ERROR
    assert result.error_message == "Connection failed."


def test_invalid_json(
    verifier,
    mock_llm,
    context,
):
    mock_llm.generate_messages.return_value = GenerationResult(
        text="Not valid JSON",
        model="test-model",
        latency_ms=100.0,
    )

    result = verifier.verify(
        "Parking duration?",
        context,
    )

    assert result.status == EvidenceStatus.INVALID_RESPONSE


@pytest.mark.parametrize(
    "status",
    ["answered", "unknown", None],
)
def test_invalid_status(
    verifier,
    mock_llm,
    context,
    sufficient_payload,
    status,
):
    sufficient_payload["status"] = status

    set_response(mock_llm, sufficient_payload)

    result = verifier.verify(
        "Parking duration?",
        context,
    )

    assert result.status == EvidenceStatus.INVALID_RESPONSE


@pytest.mark.parametrize(
    "citation_id",
    [0, 2, True, "1"],
)
def test_invalid_citation_id(
    verifier,
    mock_llm,
    context,
    sufficient_payload,
    citation_id,
):
    sufficient_payload["evidence"][0]["citation_id"] = citation_id

    set_response(mock_llm, sufficient_payload)

    result = verifier.verify(
        "Parking duration?",
        context,
    )

    assert result.status == EvidenceStatus.INVALID_RESPONSE


def test_empty_quote(
    verifier,
    mock_llm,
    context,
    sufficient_payload,
):
    sufficient_payload["evidence"][0]["quote"] = " "

    set_response(mock_llm, sufficient_payload)

    result = verifier.verify(
        "Parking duration?",
        context,
    )

    assert result.status == EvidenceStatus.INVALID_RESPONSE


def test_duplicate_quote(
    verifier,
    mock_llm,
    context,
    sufficient_payload,
):
    sufficient_payload["evidence"].append(sufficient_payload["evidence"][0].copy())

    set_response(mock_llm, sufficient_payload)

    result = verifier.verify(
        "Parking duration?",
        context,
    )

    assert result.status == EvidenceStatus.INVALID_RESPONSE


def test_sufficient_without_quotes(
    verifier,
    mock_llm,
    context,
    sufficient_payload,
):
    sufficient_payload["evidence"] = []

    set_response(mock_llm, sufficient_payload)

    result = verifier.verify(
        "Parking duration?",
        context,
    )

    assert result.status == EvidenceStatus.INVALID_RESPONSE


def test_sufficient_with_missing_information(
    verifier,
    mock_llm,
    context,
    sufficient_payload,
):
    sufficient_payload["missing_information"] = ["Parking cost"]

    set_response(mock_llm, sufficient_payload)

    result = verifier.verify(
        "Parking duration and cost?",
        context,
    )

    assert result.status == EvidenceStatus.INVALID_RESPONSE


def test_insufficient_without_missing_information(
    verifier,
    mock_llm,
    context,
    insufficient_payload,
):
    insufficient_payload["missing_information"] = []

    set_response(mock_llm, insufficient_payload)

    result = verifier.verify(
        "Parking cost?",
        context,
    )

    assert result.status == EvidenceStatus.INVALID_RESPONSE


def test_extra_response_field(
    verifier,
    mock_llm,
    context,
    sufficient_payload,
):
    sufficient_payload["answer"] = "Two hours."

    set_response(mock_llm, sufficient_payload)

    result = verifier.verify(
        "Parking duration?",
        context,
    )

    assert result.status == EvidenceStatus.INVALID_RESPONSE


def test_missing_response_field(
    verifier,
    mock_llm,
    context,
    sufficient_payload,
):
    del sufficient_payload["reason"]

    set_response(mock_llm, sufficient_payload)

    result = verifier.verify(
        "Parking duration?",
        context,
    )

    assert result.status == EvidenceStatus.INVALID_RESPONSE


def test_verification_telemetry(
    verifier,
    mock_llm,
    context,
    sufficient_payload,
):
    set_response(mock_llm, sufficient_payload)

    result = verifier.verify(
        "Parking duration?",
        context,
    )

    assert result.latency_ms >= 0
    assert result.prompt_tokens == 400
    assert result.completion_tokens == 50
    assert result.total_tokens == 450
    assert result.model == "test-model"
    assert result.verifier_version == "1.1"


def test_structured_schema_passed_to_llm(
    verifier,
    mock_llm,
    context,
    sufficient_payload,
):
    set_response(mock_llm, sufficient_payload)

    verifier.verify(
        "Parking duration?",
        context,
    )

    kwargs = mock_llm.generate_messages.call_args.kwargs

    assert kwargs["response_schema"]["type"] == "object"

    assert kwargs["response_schema"]["properties"]["status"]["enum"] == [
        "sufficient",
        "insufficient",
    ]


def test_invalid_response_preserves_token_metrics(
    verifier,
    mock_llm,
    context,
    sufficient_payload,
):
    sufficient_payload["status"] = "invalid"

    set_response(mock_llm, sufficient_payload)

    result = verifier.verify(
        "Parking duration?",
        context,
    )

    assert result.status == EvidenceStatus.INVALID_RESPONSE
    assert result.total_tokens == 450


def test_fabricated_quote_rejected(
    verifier,
    mock_llm,
    context,
    sufficient_payload,
):

    sufficient_payload["evidence"][0]["quote"] = "Visitors may park for ten hours."

    set_response(mock_llm, sufficient_payload)

    result = verifier.verify(
        "Parking duration?",
        context,
    )

    assert result.status == EvidenceStatus.INVALID_RESPONSE
    assert "Quotation not found" in result.error_message


def test_authentic_quote_accepted(
    verifier,
    mock_llm,
    context,
    sufficient_payload,
):

    set_response(mock_llm, sufficient_payload)

    result = verifier.verify(
        "Parking duration?",
        context,
    )

    assert result.status == EvidenceStatus.SUFFICIENT


def test_invalid_partial_quote_rejected(
    verifier,
    mock_llm,
    context,
    insufficient_payload,
):

    insufficient_payload["evidence"] = [
        {
            "citation_id": 1,
            "quote": "Parking costs five dollars.",
        }
    ]

    set_response(mock_llm, insufficient_payload)

    result = verifier.verify(
        "Parking duration and cost?",
        context,
    )

    assert result.status == EvidenceStatus.INVALID_RESPONSE
