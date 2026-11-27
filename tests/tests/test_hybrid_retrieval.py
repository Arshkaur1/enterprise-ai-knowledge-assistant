from unittest.mock import Mock

import pytest

from knowledge_assistant.hybrid_retrieval import (
    HybridRetriever,
)
from knowledge_assistant.models import DocumentChunk
from knowledge_assistant.retrieval import SearchResult


def make_result(chunk_id, rank, score):

    return SearchResult(
        rank=rank,
        score=score,
        chunk=DocumentChunk(
            chunk_id=chunk_id,
            document_id=chunk_id,
            source=f"{chunk_id}.txt",
            file_type="txt",
            category="General",
            text=f"Document {chunk_id}",
            chunk_index=0,
        ),
    )


@pytest.fixture
def dense():

    retriever = Mock()

    retriever.search.return_value = [
        make_result("a", 1, 0.85),
        make_result("b", 2, 0.75),
    ]

    return retriever


@pytest.fixture
def sparse():

    retriever = Mock()

    retriever.search.return_value = [
        make_result("b", 1, 7.5),
        make_result("c", 2, 3.5),
    ]

    return retriever


@pytest.fixture
def hybrid(dense, sparse):

    return HybridRetriever(
        dense_retriever=dense,
        sparse_retriever=sparse,
    )


def test_hybrid_ranking(hybrid):

    results = hybrid.search(
        "example query",
        top_k=3,
    )

    assert [result.chunk.chunk_id for result in results] == ["b", "a", "c"]


def test_top_k_limit(hybrid):

    results = hybrid.search(
        "example query",
        top_k=1,
    )

    assert len(results) == 1


def test_both_retrievers_called(hybrid, dense, sparse):

    hybrid.search("example query")

    dense.search.assert_called_once()
    sparse.search.assert_called_once()


def test_separate_thresholds(hybrid, dense, sparse):

    hybrid.search("example query")

    dense.search.assert_called_once_with(
        query="example query",
        top_k=5,
        min_score=0.35,
    )

    sparse.search.assert_called_once_with(
        query="example query",
        top_k=5,
        min_score=None,
    )


def test_candidate_pool_expansion(hybrid, dense, sparse):

    hybrid.search(
        "example query",
        top_k=10,
    )

    assert dense.search.call_args.kwargs["top_k"] == 10
    assert sparse.search.call_args.kwargs["top_k"] == 10


def test_empty_dense_results(hybrid, dense):

    dense.search.return_value = []

    results = hybrid.search("example query")

    assert [result.chunk.chunk_id for result in results] == ["b", "c"]


def test_empty_sparse_results(hybrid, sparse):

    sparse.search.return_value = []

    results = hybrid.search("example query")

    assert [result.chunk.chunk_id for result in results] == ["a", "b"]


def test_both_empty(hybrid, dense, sparse):

    dense.search.return_value = []
    sparse.search.return_value = []

    assert hybrid.search("example query") == []


def test_dense_only(dense, sparse):

    hybrid = HybridRetriever(
        dense_retriever=dense,
        sparse_retriever=sparse,
        sparse_weight=0.0,
    )

    results = hybrid.search("example query")

    assert len(results) == 2
    sparse.search.assert_not_called()


def test_sparse_only(dense, sparse):

    hybrid = HybridRetriever(
        dense_retriever=dense,
        sparse_retriever=sparse,
        dense_weight=0.0,
    )

    results = hybrid.search("example query")

    assert len(results) == 2
    dense.search.assert_not_called()


def test_candidate_diagnostics(hybrid):

    candidates = hybrid.search_candidates("example query")

    assert set(candidates) == {
        "dense",
        "sparse",
    }

    assert len(candidates["dense"]) == 2
    assert len(candidates["sparse"]) == 2


def test_query_normalization(hybrid, dense, sparse):

    hybrid.search("  example query  ")

    assert dense.search.call_args.kwargs["query"] == "example query"

    assert sparse.search.call_args.kwargs["query"] == "example query"


@pytest.mark.parametrize("query", ["", " ", "\n"])
def test_empty_query(hybrid, query):

    with pytest.raises(ValueError):
        hybrid.search(query)


@pytest.mark.parametrize("top_k", [0, -1, True, 1.5])
def test_invalid_top_k(hybrid, top_k):

    with pytest.raises(ValueError):
        hybrid.search(
            "example query",
            top_k=top_k,
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"candidate_k": 0},
        {"candidate_k": True},
        {"rrf_k": 0},
        {"dense_weight": -1},
        {"sparse_weight": float("nan")},
        {"dense_weight": 0, "sparse_weight": 0},
        {"dense_min_score": 1.5},
        {"sparse_min_score": float("inf")},
    ],
)
def test_invalid_configuration(dense, sparse, kwargs):

    with pytest.raises(ValueError):
        HybridRetriever(
            dense_retriever=dense,
            sparse_retriever=sparse,
            **kwargs,
        )


def test_invalid_query_type(hybrid):

    with pytest.raises(TypeError):
        hybrid.search(None)
