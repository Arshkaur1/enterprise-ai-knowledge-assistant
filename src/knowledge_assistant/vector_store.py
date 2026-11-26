from pathlib import Path

import faiss
import numpy as np


class VectorStore:
    """FAISS-based exact vector similarity search."""

    def __init__(self, dimension: int = 384):
        if dimension <= 0:
            raise ValueError("Dimension must be positive.")

        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)

    @property
    def size(self) -> int:
        """Return the number of indexed vectors."""

        return self.index.ntotal

    def add(self, embeddings: np.ndarray) -> None:
        """Add normalized document embeddings."""

        embeddings = np.ascontiguousarray(
            embeddings,
            dtype=np.float32,
        )

        if (
            embeddings.ndim != 2
            or embeddings.shape[1] != self.dimension
        ):
            raise ValueError("Invalid embedding dimensions.")

        if not np.isfinite(embeddings).all():
            raise ValueError("Embeddings must be finite.")

        if len(embeddings) == 0:
            return

        if not np.allclose(
            np.linalg.norm(embeddings, axis=1),
            1.0,
            atol=1e-4,
        ):
            raise ValueError("Embeddings must be normalized.")

        self.index.add(embeddings)

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return similarity scores and vector indices."""

        if top_k <= 0:
            raise ValueError("top_k must be positive.")

        query = np.asarray(
            query_embedding,
            dtype=np.float32,
        )

        if query.shape != (self.dimension,):
            raise ValueError("Invalid query dimensions.")

        if not np.isfinite(query).all():
            raise ValueError("Query must be finite.")

        if not np.isclose(
            np.linalg.norm(query),
            1.0,
            atol=1e-4,
        ):
            raise ValueError("Query must be normalized.")

        if self.size == 0:
            return (
                np.array([], dtype=np.float32),
                np.array([], dtype=np.int64),
            )

        query = np.ascontiguousarray(
            query.reshape(1, -1)
        )

        top_k = min(top_k, self.size)

        scores, indices = self.index.search(
            query,
            top_k,
        )

        return scores[0], indices[0]

    def save(self, path: str | Path) -> None:
        """Persist the FAISS index to disk."""

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(path))

    @classmethod
    def load(cls, path: str | Path) -> "VectorStore":
        """Load a previously saved FAISS index."""

        index = faiss.read_index(str(path))

        if not isinstance(index, faiss.IndexFlatIP):
            raise TypeError("Unsupported index type.")

        store = cls(dimension=index.d)
        store.index = index

        return store