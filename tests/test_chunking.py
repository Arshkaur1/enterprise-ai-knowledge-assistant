
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


def test_exact_overlap():
    text = "A" * 250

    chunks = chunk_text(
        text,
        chunk_size=100,
        overlap=20,
    )

    assert len(chunks) == 3

    assert chunks[0][-20:] == chunks[1][:20]

    assert chunks[1][-20:] == chunks[2][:20]


def test_chunk_size_limit():
    text = "A" * 1000

    chunks = chunk_text(
        text,
        chunk_size=100,
        overlap=20,
    )

    assert all(len(chunk) <= 100 for chunk in chunks)


def test_reconstruction_with_overlap():
    text = "A" * 250

    chunks = chunk_text(
        text,
        chunk_size=100,
        overlap=20,
    )

    reconstructed = chunks[0] + "".join(
        chunk[20:] for chunk in chunks[1:]
    )

    assert reconstructed == text
