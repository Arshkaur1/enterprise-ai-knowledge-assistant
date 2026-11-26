from unittest.mock import Mock

import ollama
import pytest

from knowledge_assistant.config import RAGConfig
from knowledge_assistant.generation_schema import (
    build_answer_schema,
)
from knowledge_assistant.llm_client import (
    LLMGenerationError,
    LocalLLM,
)


@pytest.fixture
def fake_client(monkeypatch):
    response = Mock()

    response.message.content = (
        '{"status":"answered","answer":"Three days per week. [1]","citations":[1]}'
    )

    response.prompt_eval_count = 300
    response.eval_count = 25
    response.done_reason = "stop"

    client = Mock()
    client.chat.return_value = response

    monkeypatch.setattr(
        ollama,
        "Client",
        lambda host: client,
    )

    return client


@pytest.fixture
def messages():
    return [
        {
            "role": "system",
            "content": "Answer using supplied evidence.",
        },
        {
            "role": "user",
            "content": "How many remote working days?",
        },
    ]


def test_structured_generation(fake_client, messages):
    llm = LocalLLM(model="test-model")

    result = llm.generate_messages(messages)

    assert result.model == "test-model"
    assert "Three days" in result.text
    assert result.latency_ms >= 0


def test_token_metrics(fake_client, messages):
    result = LocalLLM().generate_messages(messages)

    assert result.prompt_tokens == 300
    assert result.completion_tokens == 25
    assert result.total_tokens == 325


def test_generation_options(fake_client, messages):
    config = RAGConfig(
        temperature=0.2,
        max_tokens=128,
        context_window=2048,
    )

    LocalLLM().generate_messages(
        messages,
        config=config,
    )

    kwargs = fake_client.chat.call_args.kwargs

    assert kwargs["options"]["temperature"] == 0.2
    assert kwargs["options"]["num_predict"] == 128
    assert kwargs["options"]["num_ctx"] == 2048


def test_schema_passed_to_ollama(fake_client, messages):
    schema = build_answer_schema([1])

    LocalLLM().generate_messages(
        messages,
        response_schema=schema,
    )

    kwargs = fake_client.chat.call_args.kwargs

    assert kwargs["format"] == schema


def test_default_json_mode(fake_client, messages):
    LocalLLM().generate_messages(messages)

    kwargs = fake_client.chat.call_args.kwargs

    assert kwargs["format"] == "json"


def test_streaming_disabled(fake_client, messages):
    LocalLLM().generate_messages(messages)

    kwargs = fake_client.chat.call_args.kwargs

    assert kwargs["stream"] is False


def test_messages_preserved(fake_client, messages):
    LocalLLM().generate_messages(messages)

    kwargs = fake_client.chat.call_args.kwargs

    assert kwargs["messages"] == messages


def test_empty_messages(fake_client):
    with pytest.raises(ValueError):
        LocalLLM().generate_messages([])

    fake_client.chat.assert_not_called()


def test_invalid_role(fake_client):
    messages = [
        {
            "role": "invalid",
            "content": "Test",
        }
    ]

    with pytest.raises(ValueError):
        LocalLLM().generate_messages(messages)


def test_empty_message_content(fake_client):
    messages = [
        {
            "role": "user",
            "content": "   ",
        }
    ]

    with pytest.raises(ValueError):
        LocalLLM().generate_messages(messages)


def test_empty_model_response(fake_client, messages):
    fake_client.chat.return_value.message.content = ""

    with pytest.raises(LLMGenerationError):
        LocalLLM().generate_messages(messages)


def test_missing_model_response(fake_client, messages):
    fake_client.chat.return_value.message.content = None

    with pytest.raises(LLMGenerationError):
        LocalLLM().generate_messages(messages)


def test_connection_failure(fake_client, messages):
    fake_client.chat.side_effect = ConnectionError("Connection refused")

    with pytest.raises(
        LLMGenerationError,
        match="Ollama generation failed",
    ):
        LocalLLM().generate_messages(messages)


def test_missing_token_metrics(fake_client, messages):
    response = fake_client.chat.return_value

    response.prompt_eval_count = None
    response.eval_count = None

    result = LocalLLM().generate_messages(messages)

    assert result.total_tokens is None


def test_generation_token_limit(fake_client, messages):
    fake_client.chat.return_value.done_reason = "length"

    with pytest.raises(
        LLMGenerationError,
        match="output token limit",
    ):
        LocalLLM().generate_messages(messages)
