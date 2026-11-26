
import os

from dotenv import load_dotenv

load_dotenv()

OLLAMA_HOST = os.getenv(
    "OLLAMA_HOST",
    "http://localhost:11434",
)

DEFAULT_MODEL = os.getenv(
    "OLLAMA_MODEL",
    "qwen2.5:1.5b",
)

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))