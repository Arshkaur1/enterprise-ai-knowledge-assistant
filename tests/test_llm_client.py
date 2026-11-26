
from unittest.mock import Mock

import ollama

from knowledge_assistant.llm_client import LocalLLM

def test_llm_generate(monkeypatch):

    fake_response = Mock()
    fake_response.message.content = "RAG combines retrieval and generation."

    fake_client = Mock()
    fake_client.chat.return_value = fake_response

    monkeypatch.setattr(
        ollama,
        "Client",
        lambda host: fake_client,
    )

    llm = LocalLLM(model="test-model")

    result = llm.generate("What is RAG?")

    assert result == "RAG combines retrieval and generation."

    fake_client.chat.assert_called_once()

def test_empty_prompt():

    llm = LocalLLM()

    try:
        llm.generate("   ")
        assert False, "Expected ValueError"
    except ValueError:
        pass