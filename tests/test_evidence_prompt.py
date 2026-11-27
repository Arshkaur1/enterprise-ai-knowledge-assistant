import json

import pytest

from knowledge_assistant.evidence_prompt import (
    EVIDENCE_SYSTEM_PROMPT,
    EvidencePromptBuilder,
)
from knowledge_assistant.models import (
    Citation,
    ContextBundle,
)


@pytest.fixture
def context():
    text = "[1] Visitors may park for two hours."

    return ContextBundle(
        text=text,
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


def test_verifier_prompt_version():
    assert EvidencePromptBuilder().version == "1.1"


def test_prompt_message_count(context):
    messages = EvidencePromptBuilder().build(
        "How long may visitors park?",
        context,
    )

    assert len(messages) == 2


def test_prompt_roles(context):
    messages = EvidencePromptBuilder().build(
        "Parking duration?",
        context,
    )

    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"


def test_system_prompt(context):
    messages = EvidencePromptBuilder().build(
        "Parking duration?",
        context,
    )

    assert messages[0]["content"] == EVIDENCE_SYSTEM_PROMPT


def test_prompt_payload(context):
    messages = EvidencePromptBuilder().build(
        "Parking duration?",
        context,
    )

    payload = json.loads(messages[1]["content"])

    assert payload["question"] == "Parking duration?"
    assert payload["document_evidence"] == context.text
    assert payload["available_citations"] == [1]


def test_empty_query(context):
    with pytest.raises(ValueError):
        EvidencePromptBuilder().build(" ", context)


def test_query_whitespace_removed(context):
    messages = EvidencePromptBuilder().build(
        "  Parking duration?  ",
        context,
    )

    payload = json.loads(messages[1]["content"])

    assert payload["question"] == "Parking duration?"


def test_prompt_requires_exact_quotations():
    assert "Extract exact quotations" in EVIDENCE_SYSTEM_PROMPT


def test_prompt_prohibits_answer_generation():
    assert "Do not answer the question itself" in (EVIDENCE_SYSTEM_PROMPT)


def test_prompt_contains_abstention_example():
    assert '"status": "insufficient"' in EVIDENCE_SYSTEM_PROMPT
