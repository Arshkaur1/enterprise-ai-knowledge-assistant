from dataclasses import FrozenInstanceError

import pytest

from knowledge_assistant.config import RAGConfig


def test_default_rag_config():
    config = RAGConfig()

    assert config.top_k > 0
    assert -1 <= config.min_score <= 1
    assert config.max_context_chars > 0
    assert config.temperature >= 0
    assert config.max_tokens > 0
    assert config.max_tokens < config.context_window


def test_custom_rag_config():
    config = RAGConfig(
        top_k=3,
        min_score=0.4,
        max_context_chars=2000,
        temperature=0.2,
        max_tokens=128,
        context_window=2048,
    )

    assert config.top_k == 3
    assert config.min_score == 0.4
    assert config.temperature == 0.2


@pytest.mark.parametrize(
    "kwargs",
    [
        {"top_k": 0},
        {"top_k": -1},
        {"min_score": 1.1},
        {"min_score": -1.1},
        {"min_score": float("nan")},
        {"max_context_chars": 0},
        {"temperature": -0.1},
        {"temperature": float("nan")},
        {"temperature": float("inf")},
        {"max_tokens": 0},
        {"context_window": 0},
        {"max_tokens": 4096, "context_window": 4096},
    ],
)
def test_invalid_rag_config(kwargs):
    with pytest.raises(ValueError):
        RAGConfig(**kwargs)


def test_config_is_immutable():
    config = RAGConfig()

    with pytest.raises(FrozenInstanceError):
        config.top_k = 5  # pyright: ignore[reportAttributeAccessIssue]
