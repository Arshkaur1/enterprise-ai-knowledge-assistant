import json

import pytest

from knowledge_assistant.models import (
    AnswerStatus,
    Citation,
    ContextBundle,
)
from knowledge_assistant.response_validator import (
    ABSTENTION_MESSAGE,
    ResponseValidationError,
    ResponseValidator,
)


@pytest.fixture
def context():
    citations = [
        Citation(
            citation_id=1,
            chunk_id="hr_chunk_0000",
            document_id="hr",
            source="hr_policy.txt",
            category="HR",
            score=0.85,
        ),
        Citation(
            citation_id=2,
            chunk_id="it_chunk_0000",
            document_id="it",
            source="it_security.docx",
            category="IT",
            score=0.75,
        ),
    ]

    text = (
        "[1] Remote work is allowed three days per week.\n\n"
        "[2] Passwords require at least 12 characters."
    )

    return ContextBundle(
        text=text,
        citations=citations,
        retrieved_count=2,
        included_count=2,
        character_count=len(text),
    )


def make_response(
    answer="Three days per week. [1]",
    status="answered",
    citations=None,
):
    return json.dumps(
        {
            "status": status,
            "answer": answer,
            "citations": ([1] if citations is None else citations),
        }
    )


def test_valid_answer(context):
    result = ResponseValidator().validate(
        make_response(),
        context,
    )

    assert result.answered
    assert result.status == AnswerStatus.ANSWERED
    assert result.answer == "Three days per week. [1]"
    assert result.warnings == []


def test_citation_metadata_resolution(context):
    result = ResponseValidator().validate(
        make_response(),
        context,
    )

    assert result.citations[0].source == "hr_policy.txt"
    assert result.citations[0].score == 0.85


def test_multiple_citations(context):
    raw = make_response(
        answer="Three days [1] and 12 characters [2].",
        citations=[1, 2],
    )

    result = ResponseValidator().validate(raw, context)

    assert len(result.citations) == 2
    assert result.warnings == []


def test_missing_inline_citations_normalized(context):
    raw = make_response(
        answer="Three days per week.",
    )

    result = ResponseValidator().validate(raw, context)

    assert result.answer == "Three days per week. [1]"
    assert result.warnings == ["answer_level_citations_appended"]


def test_multiple_answer_level_citations(context):
    raw = make_response(
        answer="Three days and 12 characters.",
        citations=[1, 2],
    )

    result = ResponseValidator().validate(raw, context)

    assert result.answer.endswith("[1] [2]")


def test_strict_mode_missing_inline_citations(context):
    raw = make_response(answer="Three days per week.")

    with pytest.raises(ResponseValidationError):
        ResponseValidator(normalize=False).validate(
            raw,
            context,
        )


def test_valid_abstention(context):
    raw = make_response(
        status="insufficient_evidence",
        answer="Information unavailable.",
        citations=[],
    )

    result = ResponseValidator().validate(raw, context)

    assert not result.answered
    assert result.citations == []
    assert result.answer == ABSTENTION_MESSAGE
    assert result.warnings == []


def test_abstention_citations_removed(context):
    raw = make_response(
        status="insufficient_evidence",
        answer="Mileage rate is not specified.",
        citations=[1],
    )

    result = ResponseValidator().validate(raw, context)

    assert result.citations == []
    assert result.warnings == ["abstention_citations_removed"]


def test_strict_mode_abstention_citations(context):
    raw = make_response(
        status="insufficient_evidence",
        answer="Information unavailable.",
        citations=[1],
    )

    with pytest.raises(ResponseValidationError):
        ResponseValidator(normalize=False).validate(
            raw,
            context,
        )


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "Not JSON",
        "[]",
        "null",
        '{"status":"answered"}',
        '{"status":"answered","answer":"Example","citations":[1],"extra":true}',
        '{"status":"answered","status":"answered","answer":"Example","citations":[1]}',
    ],
)
def test_invalid_json_or_structure(raw, context):
    with pytest.raises(ResponseValidationError):
        ResponseValidator().validate(raw, context)


@pytest.mark.parametrize(
    "status",
    ["unknown", "generation_error", None],
)
def test_invalid_status(status, context):
    raw = make_response(status=status)

    with pytest.raises(ResponseValidationError):
        ResponseValidator().validate(raw, context)


@pytest.mark.parametrize(
    "answer",
    ["", "   ", None, 123],
)
def test_invalid_answer(answer, context):
    raw = make_response(answer=answer)

    with pytest.raises(ResponseValidationError):
        ResponseValidator().validate(raw, context)


@pytest.mark.parametrize(
    "citations",
    [
        "1",
        1,
        [True],
        ["1"],
        [0],
        [-1],
        [1.5],
        [1, 1],
        [99],
    ],
)
def test_invalid_citations(citations, context):
    raw = make_response(citations=citations)

    with pytest.raises(ResponseValidationError):
        ResponseValidator().validate(raw, context)


def test_answer_without_citations(context):
    raw = make_response(
        answer="Three days per week.",
        citations=[],
    )

    with pytest.raises(ResponseValidationError):
        ResponseValidator().validate(raw, context)


def test_unknown_inline_citation(context):
    raw = make_response(
        answer="Three days per week. [99]",
        citations=[1],
    )

    with pytest.raises(ResponseValidationError):
        ResponseValidator().validate(raw, context)


def test_mismatched_citations(context):
    raw = make_response(
        answer="Three days per week. [2]",
        citations=[1],
    )

    with pytest.raises(ResponseValidationError):
        ResponseValidator().validate(raw, context)


def test_empty_context_rejects_answer():
    context = ContextBundle(
        text="",
        citations=[],
        retrieved_count=0,
        included_count=0,
        character_count=0,
    )

    with pytest.raises(ResponseValidationError):
        ResponseValidator().validate(
            make_response(),
            context,
        )


def test_duplicate_context_citations(context):
    context.citations.append(context.citations[0])

    with pytest.raises(ResponseValidationError):
        ResponseValidator().validate(
            make_response(),
            context,
        )


def test_validated_answer_serialization(context):
    result = ResponseValidator().validate(
        make_response(),
        context,
    )

    payload = result.to_dict()

    assert payload["status"] == "answered"
    assert payload["answered"] is True
    assert payload["citations"][0]["citation_id"] == 1
    assert payload["warnings"] == []

    json.dumps(payload)
