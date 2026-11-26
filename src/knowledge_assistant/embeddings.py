import numpy as np
from sentence_transformers import SentenceTransformer

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class EmbeddingService:
    """Generate normalized dense text embeddings."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str = "cpu",
        batch_size: int = 32,
    ):
        if batch_size <= 0:
            raise ValueError("Batch size must be positive.")

        self.model_name = model_name
        self.batch_size = batch_size

        self.model = SentenceTransformer(
            model_name,
            device=device,
        )

        if hasattr(self.model, "get_embedding_dimension"):
            dimension = self.model.get_embedding_dimension()
        else:
            dimension = self.model.get_sentence_embedding_dimension()
        if dimension is None:
            raise ValueError("The model does not expose an embedding dimension.")
        self.dimension = dimension

    def embed_documents(
        self,
        texts: list[str],
    ) -> np.ndarray:
        """Generate embeddings for multiple documents."""

        if not texts:
            return np.empty(
                (0, self.dimension),
                dtype=np.float32,
            )

        if any(not text.strip() for text in texts):
            raise ValueError("Documents cannot be empty.")

        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        return np.ascontiguousarray(
            embeddings,
            dtype=np.float32,
        )

    def embed_query(self, query: str) -> np.ndarray:
        """Generate an embedding for a single query."""

        if not query.strip():
            raise ValueError("Query cannot be empty.")

        return self.embed_documents([query])[0]
