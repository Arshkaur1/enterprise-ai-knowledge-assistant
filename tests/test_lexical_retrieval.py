from dataclasses import replace

import pytest

from knowledge_assistant.lexical_retrieval import (
    BM25Retriever,
)
from knowledge_assistant.models import DocumentChunk


@pytest.fixture
def lexical_chunks():

    documents = [
        (
            "hr_001",
            "HR",
            "Employees may work remotely three days per week.",
        ),
        (
            "it_001",
            "IT",
            "Passwords must contain at least 12 characters.",
        ),
        (
            "finance_001",
            "Finance",
            "Expense claims require itemized receipts.",
        ),
    ]

    return [
        DocumentChunk(
            chunk_id=chunk_id,
            document_id=chunk_id,
            source=f"{chunk_id}.txt",
            file_type="txt",
            category=category,
            text=text,
            chunk_index=0,
        )
        for chunk_id, category, text in documents
    ]


@pytest.fixture
def retriever(lexical_chunks):
    return BM25Retriever(lexical_chunks)


def test_chunk_count(retriever):
    assert retriever.chunk_count == 3


def test_remote_work_search(retriever):

    results = retriever.search(
        "remote remotely work",
        top_k=1,
    )

    assert len(results) == 1
    assert results[0].chunk.chunk_id == "hr_001"


def test_password_search(retriever):

    results = retriever.search(
        "passwords 12 characters",
        top_k=1,
    )

    assert results[0].chunk.chunk_id == "it_001"


def test_expense_search(retriever):

    results = retriever.search(
        "expense receipts",
        top_k=1,
    )

    assert results[0].chunk.chunk_id == "finance_001"


def test_no_matching_terms(retriever):
    assert retriever.search("astronomy") == []


def test_punctuation_only_query(retriever):
    assert retriever.search("!!!") == []


def test_case_insensitive_search(retriever):

    lower = retriever.search("passwords")
    upper = retriever.search("PASSWORDS")

    assert lower == upper


def test_contiguous_ranks(retriever):

    results = retriever.search(
        "employees passwords expense",
        top_k=3,
    )

    assert [result.rank for result in results] == [
        1,
        2,
        3,
    ]


def test_top_k_limit(retriever):

    results = retriever.search(
        "employees passwords expense",
        top_k=2,
    )

    assert len(results) == 2


def test_top_k_larger_than_corpus(retriever):

    results = retriever.search(
        "employees passwords expense",
        top_k=100,
    )

    assert len(results) == 3


def test_descending_scores(retriever):

    results = retriever.search(
        "employees passwords 12 expense",
        top_k=3,
    )

    scores = [result.score for result in results]

    assert scores == sorted(
        scores,
        reverse=True,
    )


def test_score_threshold(retriever):

    results = retriever.search(
        "passwords",
        min_score=1000.0,
    )

    assert results == []


def test_repeated_query_tokens(retriever):

    single = retriever.search("passwords")
    repeated = retriever.search("passwords passwords passwords")

    assert single == repeated


def test_deterministic_results(retriever):

    first = retriever.search("employees passwords expense")

    second = retriever.search("employees passwords expense")

    assert first == second


def test_empty_corpus():

    with pytest.raises(ValueError):
        BM25Retriever([])


def test_duplicate_chunks(lexical_chunks):

    with pytest.raises(ValueError):
        BM25Retriever([lexical_chunks[0], lexical_chunks[0]])


def test_empty_chunk(lexical_chunks):

    chunk = replace(
        lexical_chunks[0],
        text=" ",
    )

    with pytest.raises(ValueError):
        BM25Retriever([chunk])


@pytest.mark.parametrize(
    "k1",
    [0, -1, float("nan"), float("inf")],
)
def test_invalid_k1(lexical_chunks, k1):

    with pytest.raises(ValueError):
        BM25Retriever(
            lexical_chunks,
            k1=k1,
        )


@pytest.mark.parametrize(
    "b",
    [-0.1, 1.1, float("nan"), float("inf")],
)
def test_invalid_b(lexical_chunks, b):

    with pytest.raises(ValueError):
        BM25Retriever(
            lexical_chunks,
            b=b,
        )


@pytest.mark.parametrize(
    "top_k",
    [0, -1, True, 1.5],
)
def test_invalid_top_k(retriever, top_k):

    with pytest.raises(ValueError):
        retriever.search(
            "passwords",
            top_k=top_k,
        )


@pytest.mark.parametrize(
    "query",
    ["", " ", "\n"],
)
def test_empty_query(retriever, query):

    with pytest.raises(ValueError):
        retriever.search(query)


@pytest.mark.parametrize(
    "threshold",
    [float("nan"), float("inf")],
)
def test_invalid_threshold(retriever, threshold):

    with pytest.raises(ValueError):
        retriever.search(
            "passwords",
            min_score=threshold,
        )


def test_invalid_query_type(retriever):

    with pytest.raises(TypeError):
        retriever.search(None)
