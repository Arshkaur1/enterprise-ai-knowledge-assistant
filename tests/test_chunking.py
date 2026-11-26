
import pytest

from knowledge_assistant.chunking import chunk_text

def test_empty_text():
    assert chunk_text("") == []

def test_text_is_split():
    text = " ".join(["machine learning"] * 100)

    chunks = chunk_text(
        text,
        chunk_size=100,
        overlap=20,
    )

    assert len(chunks) > 1

def test_invalid_chunk_size():
    with pytest.raises(ValueError):
        chunk_text("Hello", chunk_size=0)

def test_invalid_overlap():
    with pytest.raises(ValueError):
        chunk_text("Hello", chunk_size=100, overlap=100)


def test_overlap_preserves_context():
    text = " ".join(["data"] * 100)

    chunks = chunk_text(
        text,
        chunk_size=100,
        overlap=20,
    )

    assert len(chunks) > 1

    # Consecutive chunks should share content.
    for previous, current in zip(chunks, chunks[1:]):
        previous_words = previous.split()
        current_words = current.split()

        assert set(previous_words) & set(current_words)


    

