import json

import pytest

from knowledge_assistant.config import RAGConfig
from knowledge_assistant.models import ContextBundle
from knowledge_assistant.prompt_builder import (
    SYSTEM_PROMPT,
    PromptBuilder,
)
from knowledge_assistant.prompts import (
    PROMPT_REGISTRY,
    SYSTEM_PROMPT_V1,
    SYSTEM_PROMPT_V2,
    get_system_prompt,
)


@pytest.fixture
def context():
    return ContextBundle(
        text="[1] Visitors may park for two hours.",
        citations=[],
        retrieved_count=1,
        included_count=0,
        character_count=36,
    )


def test_default_prompt_version():
    assert PromptBuilder().version == "1.0"


def test_original_prompt_preserved():
    assert SYSTEM_PROMPT_V1 == SYSTEM_PROMPT


def test_version_one_registered():
    assert "1.0" in PROMPT_REGISTRY


def test_version_two_registered():
    assert "2.0" in PROMPT_REGISTRY


def test_prompts_are_different():
    assert SYSTEM_PROMPT_V1 != SYSTEM_PROMPT_V2


@pytest.mark.parametrize("version", ["1.0", "2.0"])
def test_prompt_lookup(version):
    assert get_system_prompt(version) == PROMPT_REGISTRY[version]


def test_unknown_prompt_version():
    with pytest.raises(ValueError):
        get_system_prompt("99.0")


def test_builder_version_two():
    builder = PromptBuilder(version="2.0")

    assert builder.version == "2.0"


def test_builder_rejects_unknown_version():
    with pytest.raises(ValueError):
        PromptBuilder(version="invalid")


def test_version_one_system_message(context):
    messages = PromptBuilder("1.0").build(
        "How long may visitors park?",
        context,
    )

    assert messages[0]["content"] == SYSTEM_PROMPT_V1


def test_version_two_system_message(context):
    messages = PromptBuilder("2.0").build(
        "How long may visitors park?",
        context,
    )

    assert messages[0]["content"] == SYSTEM_PROMPT_V2


def test_user_payload_unchanged(context):
    query = "How long may visitors park?"

    v1 = PromptBuilder("1.0").build(query, context)
    v2 = PromptBuilder("2.0").build(query, context)

    assert v1[1] == v2[1]


def test_payload_is_valid_json(context):
    messages = PromptBuilder("2.0").build(
        "How long may visitors park?",
        context,
    )

    payload = json.loads(messages[1]["content"])

    assert payload["document_evidence"] == context.text
    assert payload["question"] == "How long may visitors park?"


def test_version_two_evidence_instructions():
    assert "Related information is not sufficient evidence" in (SYSTEM_PROMPT_V2)


def test_version_two_abstention_examples():
    assert '"status":"insufficient_evidence"' in SYSTEM_PROMPT_V2


def test_version_two_citation_examples():
    assert '"citations":[1,2]' in SYSTEM_PROMPT_V2


def test_version_two_empty_query(context):
    with pytest.raises(ValueError):
        PromptBuilder("2.0").build(" ", context)


@pytest.mark.parametrize("version", ["1.0", "2.0"])
def test_config_supported_versions(version):
    config = RAGConfig(prompt_version=version)

    assert config.prompt_version == version


def test_config_invalid_version():
    with pytest.raises(ValueError):
        RAGConfig(prompt_version="invalid")
