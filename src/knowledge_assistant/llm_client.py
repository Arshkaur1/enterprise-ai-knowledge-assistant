from time import perf_counter

import ollama

from knowledge_assistant.config import (
    DEFAULT_MODEL,
    OLLAMA_HOST,
    RAGConfig,
)
from knowledge_assistant.models import GenerationResult


class LLMGenerationError(RuntimeError):
    """Raised when language model generation fails."""


class LocalLLM:
    """Client for local language model inference using Ollama."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        host: str = OLLAMA_HOST,
    ):
        self.model = model
        self.client = ollama.Client(host=host)

    def generate(self, prompt: str) -> str:
        """Generate text using the original interface."""

        if not prompt.strip():
            raise ValueError("Prompt cannot be empty")

        response = self.client.chat(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        )

        content = response.message.content

        if not isinstance(content, str) or not content.strip():
            raise LLMGenerationError("Model returned an empty response.")

        return content

    def generate_messages(
        self,
        messages: list[dict[str, str]],
        config: RAGConfig | None = None,
        response_schema: dict | None = None,
    ) -> GenerationResult:
        """Generate a structured response with inference metrics."""

        if not messages:
            raise ValueError("Messages cannot be empty.")

        for message in messages:
            if message.get("role") not in {
                "system",
                "user",
                "assistant",
            }:
                raise ValueError("Invalid message role.")

            content = message.get("content")

            if not isinstance(content, str) or not content.strip():
                raise ValueError("Message content cannot be empty.")

        config = config or RAGConfig()

        options = {
            "temperature": config.temperature,
            "num_predict": config.max_tokens,
            "num_ctx": config.context_window,
        }

        start = perf_counter()

        try:
            response = self.client.chat(
                model=self.model,
                messages=messages,
                options=options,
                format=(response_schema if response_schema is not None else "json"),
                stream=False,
            )

        except (ollama.ResponseError, ConnectionError) as exc:
            raise LLMGenerationError(f"Ollama generation failed: {exc}") from exc

        latency_ms = (perf_counter() - start) * 1000

        content = response.message.content

        if not isinstance(content, str) or not content.strip():
            raise LLMGenerationError("Model returned an empty response.")

        if getattr(response, "done_reason", None) == "length":
            raise LLMGenerationError("Generation reached the output token limit.")

        return GenerationResult(
            text=content,
            model=self.model,
            latency_ms=latency_ms,
            prompt_tokens=getattr(
                response,
                "prompt_eval_count",
                None,
            ),
            completion_tokens=getattr(
                response,
                "eval_count",
                None,
            ),
        )
