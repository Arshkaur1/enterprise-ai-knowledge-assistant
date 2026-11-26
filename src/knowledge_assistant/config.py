import os
from dataclasses import dataclass
from math import isfinite

from dotenv import load_dotenv

load_dotenv()

# Ollama configuration
OLLAMA_HOST = os.getenv(
    "OLLAMA_HOST",
    "http://localhost:11434",
)

DEFAULT_MODEL = os.getenv(
    "OLLAMA_MODEL",
    "qwen2.5:1.5b",
)

# Document processing configuration
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))

# Retrieval configuration
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "2"))
RAG_MIN_SCORE = float(os.getenv("RAG_MIN_SCORE", "0.35"))

# Context configuration
RAG_MAX_CONTEXT_CHARS = int(os.getenv("RAG_MAX_CONTEXT_CHARS", "4000"))

# Generation configuration
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.0"))

LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "256"))

LLM_CONTEXT_WINDOW = int(os.getenv("LLM_CONTEXT_WINDOW", "4096"))


@dataclass(frozen=True)
class RAGConfig:
    """Validated configuration for the RAG pipeline."""

    top_k: int = RAG_TOP_K
    min_score: float = RAG_MIN_SCORE

    max_context_chars: int = RAG_MAX_CONTEXT_CHARS

    temperature: float = LLM_TEMPERATURE
    max_tokens: int = LLM_MAX_TOKENS
    context_window: int = LLM_CONTEXT_WINDOW

    def __post_init__(self) -> None:
        if self.top_k <= 0:
            raise ValueError("top_k must be positive.")

        if not -1 <= self.min_score <= 1:
            raise ValueError("min_score must be between -1 and 1.")

        if self.max_context_chars <= 0:
            raise ValueError("max_context_chars must be positive.")

        if not isfinite(self.temperature) or self.temperature < 0:
            raise ValueError("temperature must be finite and non-negative.")

        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive.")

        if self.context_window <= 0:
            raise ValueError("context_window must be positive.")

        if self.max_tokens >= self.context_window:
            raise ValueError("max_tokens must be smaller than context_window.")
