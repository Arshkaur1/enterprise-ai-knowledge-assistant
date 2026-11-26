import ollama

from knowledge_assistant.config import DEFAULT_MODEL, OLLAMA_HOST


class LocalLLM:
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        host: str = OLLAMA_HOST,
    ):
        self.model = model
        self.client = ollama.Client(host=host)

    def generate(self, prompt: str) -> str:

        if not prompt.strip():
            raise ValueError("Prompt cannot be empty")

        response = self.client.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
        )

        return response.message.content  # pyright: ignore[reportReturnType]
