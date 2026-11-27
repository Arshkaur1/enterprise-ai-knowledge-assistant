import json

import pytest

from knowledge_assistant.models import (
    EvidenceQuote,
    EvidenceStatus,
    EvidenceVerificationResult,
)


def test_evidence_quote_serialization():
    quote = EvidenceQuote(
        citation_id=1,
        quote="Visitors may park for two hours.",
    )

    assert quote.to_dict() == {
        "citation_id": 1,
        "quote": "Visitors may park for two hours.",
    }


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (EvidenceStatus.SUFFICIENT, True),
        (EvidenceStatus.INSUFFICIENT, False),
        (EvidenceStatus.INVALID_RESPONSE, False),
        (EvidenceStatus.VERIFICATION_ERROR, False),
    ],
)
def test_sufficient_property(status, expected):
    result = EvidenceVerificationResult(
        status=status,
        reason="Test reason.",
    )

    assert result.sufficient is expected


def test_default_evidence_is_empty():
    result = EvidenceVerificationResult(
        status=EvidenceStatus.INSUFFICIENT,
        reason="Missing information.",
    )

    assert result.evidence == []
    assert result.missing_information == []
    assert result.warnings == []


def test_default_lists_are_independent():
    first = EvidenceVerificationResult(
        status=EvidenceStatus.INSUFFICIENT,
        reason="First result.",
    )

    second = EvidenceVerificationResult(
        status=EvidenceStatus.INSUFFICIENT,
        reason="Second result.",
    )

    first.missing_information.append("Parking cost")

    assert second.missing_information == []


def test_total_tokens():
    result = EvidenceVerificationResult(
        status=EvidenceStatus.SUFFICIENT,
        reason="Evidence found.",
        prompt_tokens=300,
        completion_tokens=50,
    )

    assert result.total_tokens == 350


@pytest.mark.parametrize(
    ("prompt_tokens", "completion_tokens"),
    [
        (None, None),
        (300, None),
        (None, 50),
    ],
)
def test_missing_token_metrics(
    prompt_tokens,
    completion_tokens,
):
    result = EvidenceVerificationResult(
        status=EvidenceStatus.SUFFICIENT,
        reason="Evidence found.",
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )

    assert result.total_tokens is None


def test_result_serialization():
    result = EvidenceVerificationResult(
        status=EvidenceStatus.SUFFICIENT,
        reason="Direct evidence found.",
        evidence=[
            EvidenceQuote(
                citation_id=1,
                quote="The library opens at 9 AM.",
            )
        ],
        latency_ms=100.0,
        prompt_tokens=200,
        completion_tokens=30,
        model="test-model",
        verifier_version="1.0",
    )

    payload = result.to_dict()

    assert payload["status"] == "sufficient"
    assert payload["sufficient"] is True
    assert payload["total_tokens"] == 230
    assert payload["evidence"][0]["citation_id"] == 1
    assert payload["model"] == "test-model"
    assert payload["verifier_version"] == "1.0"


def test_result_is_json_serializable():
    result = EvidenceVerificationResult(
        status=EvidenceStatus.INSUFFICIENT,
        reason="Required information is missing.",
        missing_information=["Membership cost"],
    )

    payload = json.loads(json.dumps(result.to_dict()))

    assert payload["status"] == "insufficient"
    assert payload["missing_information"] == ["Membership cost"]


def test_verification_error_metadata():
    result = EvidenceVerificationResult(
        status=EvidenceStatus.VERIFICATION_ERROR,
        reason="Verification failed.",
        error_message="Connection failed.",
    )

    assert result.sufficient is False
    assert result.error_message == "Connection failed."
