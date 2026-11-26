import faiss
import numpy as np
import pytest

from knowledge_assistant.vector_store import VectorStore


@pytest.fixture
def vectors():
    return np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def test_add_vectors(vectors):
    store = VectorStore(dimension=3)
    store.add(vectors)

    assert store.size == 3


def test_similarity_search(vectors):
    store = VectorStore(dimension=3)
    store.add(vectors)

    scores, indices = store.search(
        vectors[0],
        top_k=1,
    )

    assert indices[0] == 0
    assert scores[0] == pytest.approx(1.0)


def test_invalid_dimensions():
    store = VectorStore(dimension=3)

    with pytest.raises(ValueError):
        store.add(np.ones((2, 4)))


def test_empty_index():
    store = VectorStore(dimension=3)

    scores, indices = store.search(
        np.array([1.0, 0.0, 0.0]),
        top_k=3,
    )

    assert len(scores) == 0
    assert len(indices) == 0


def test_top_k_exceeds_index_size(vectors):
    store = VectorStore(dimension=3)
    store.add(vectors)

    scores, indices = store.search(
        vectors[0],
        top_k=10,
    )

    assert len(scores) == 3
    assert len(indices) == 3


def test_invalid_top_k(vectors):
    store = VectorStore(dimension=3)
    store.add(vectors)

    with pytest.raises(ValueError):
        store.search(vectors[0], top_k=0)


def test_save_and_load(vectors, tmp_path):
    store = VectorStore(dimension=3)
    store.add(vectors)

    path = tmp_path / "test_index.faiss"

    store.save(path)

    loaded_store = VectorStore.load(path)

    assert loaded_store.size == store.size
    assert loaded_store.dimension == store.dimension

    scores, indices = loaded_store.search(
        vectors[0],
        top_k=1,
    )

    assert indices[0] == 0
    assert scores[0] == pytest.approx(1.0)


def test_unnormalized_vectors():
    store = VectorStore(dimension=3)

    with pytest.raises(ValueError):
        store.add(np.array([[2.0, 0.0, 0.0]]))


def test_load_unsupported_index_type(tmp_path):
    index = faiss.IndexFlatL2(3)

    path = tmp_path / "unsupported_index.faiss"

    faiss.write_index(index, str(path))

    with pytest.raises(
        TypeError,
        match="Unsupported index type",
    ):
        VectorStore.load(path)
