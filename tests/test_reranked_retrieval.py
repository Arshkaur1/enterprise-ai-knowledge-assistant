"""Unit tests for the reranked retrieval pipeline."""

from typing import cast
from unittest.mock import Mock

import pytest

from knowledge_assistant.hybrid_retrieval import HybridRetriever
from knowledge_assistant.models import DocumentChunk
from knowledge_assistant.reranked_retrieval import (
    RerankedRetriever,
    RerankerProtocol,
)
from knowledge_assistant.retrieval import (
    SearchResult,
    SemanticRetriever,
)


@pytest.fixture
def dense():
    return Mock(spec=SemanticRetriever)


@pytest.fixture
def hybrid():
    return Mock(spec=HybridRetriever)


@pytest.fixture
def reranker():
    return Mock(spec=RerankerProtocol)


@pytest.fixture
def candidates():
    return [
        SearchResult(
            rank=index + 1,
            score=0.9 - index * 0.1,
            chunk=DocumentChunk(
                chunk_id=f"chunk_{index}",
                document_id="test_document",
                source="test.txt",
                file_type="txt",
                category="General",
                text=f"Document content {index}",
                chunk_index=index,
            ),
        )
        for index in range(5)
    ]


@pytest.fixture
def pipeline(dense, reranker):
    return RerankedRetriever(
        retriever=dense,
        reranker=reranker,
    )


def test_default_configuration(pipeline):
    assert pipeline.candidate_k == 5
    assert pipeline.dense_min_score == 0.35
    assert pipeline.reranker_min_score is None


@pytest.mark.parametrize("candidate_k", [0, -1, True, 1.5])
def test_invalid_candidate_k(dense, reranker, candidate_k):
    with pytest.raises(ValueError, match="candidate_k"):
        RerankedRetriever(
            dense,
            reranker,
            candidate_k=candidate_k,
        )


@pytest.mark.parametrize(
    "threshold",
    [
        -1.1,
        1.1,
        float("nan"),
        float("inf"),
        float("-inf"),
    ],
)
def test_invalid_dense_threshold(dense, reranker, threshold):
    with pytest.raises(ValueError, match="Dense"):
        RerankedRetriever(
            dense,
            reranker,
            dense_min_score=threshold,
        )


@pytest.mark.parametrize(
    "threshold",
    [float("nan"), float("inf"), float("-inf")],
)
def test_invalid_reranker_threshold(dense, reranker, threshold):
    with pytest.raises(ValueError, match="Reranker"):
        RerankedRetriever(
            dense,
            reranker,
            reranker_min_score=threshold,
        )


def test_unsupported_retriever(reranker):
    with pytest.raises(TypeError, match="Unsupported"):
        RerankedRetriever(
            retriever=cast(SemanticRetriever | HybridRetriever, object()),
            reranker=reranker,
        )


def test_dense_candidate_retrieval(pipeline, dense, candidates):
    dense.search.return_value = candidates

    results = pipeline.search_candidates("query")

    assert results == candidates

    dense.search.assert_called_once_with(
        query="query",
        top_k=5,
        min_score=0.35,
    )


def test_hybrid_candidate_retrieval(hybrid, reranker, candidates):
    hybrid.search.return_value = candidates

    pipeline = RerankedRetriever(hybrid, reranker)

    results = pipeline.search_candidates("query")

    assert results == candidates

    hybrid.search.assert_called_once_with(
        query="query",
        top_k=5,
    )


def test_custom_candidate_count(pipeline, dense, candidates):
    dense.search.return_value = candidates[:3]

    pipeline.search_candidates(
        "query",
        candidate_k=3,
    )

    dense.search.assert_called_once_with(
        query="query",
        top_k=3,
        min_score=0.35,
    )


def test_dense_threshold_can_be_disabled(dense, reranker, candidates):
    dense.search.return_value = candidates

    pipeline = RerankedRetriever(
        dense,
        reranker,
        dense_min_score=None,
    )

    pipeline.search_candidates("query")

    dense.search.assert_called_once_with(
        query="query",
        top_k=5,
        min_score=None,
    )


def test_dense_search_calls_reranker(pipeline, dense, reranker, candidates):
    dense.search.return_value = candidates
    reranker.rerank.return_value = candidates[:2]

    results = pipeline.search("query")

    assert results == candidates[:2]

    reranker.rerank.assert_called_once_with(
        query="query",
        candidates=candidates,
        top_k=2,
        min_score=None,
    )


def test_hybrid_search_calls_reranker(hybrid, reranker, candidates):
    hybrid.search.return_value = candidates
    reranker.rerank.return_value = candidates[:2]

    pipeline = RerankedRetriever(hybrid, reranker)

    results = pipeline.search("query")

    assert results == candidates[:2]

    hybrid.search.assert_called_once_with(
        query="query",
        top_k=5,
    )

    reranker.rerank.assert_called_once_with(
        query="query",
        candidates=candidates,
        top_k=2,
        min_score=None,
    )


def test_reranker_threshold_forwarded(dense, reranker, candidates):
    dense.search.return_value = candidates
    reranker.rerank.return_value = candidates[:1]

    pipeline = RerankedRetriever(
        dense,
        reranker,
        reranker_min_score=-8.5,
    )

    pipeline.search("query")

    reranker.rerank.assert_called_once_with(
        query="query",
        candidates=candidates,
        top_k=2,
        min_score=-8.5,
    )


def test_empty_candidates_skip_reranker(pipeline, dense, reranker):
    dense.search.return_value = []

    assert pipeline.search("query") == []

    reranker.rerank.assert_not_called()


def test_reranker_can_filter_all_candidates(pipeline, dense, reranker, candidates):
    dense.search.return_value = candidates
    reranker.rerank.return_value = []

    assert pipeline.search("query") == []


def test_candidate_count_expands_for_large_top_k(pipeline, dense, reranker, candidates):
    dense.search.return_value = candidates
    reranker.rerank.return_value = candidates

    pipeline.search("query", top_k=8)

    dense.search.assert_called_once_with(
        query="query",
        top_k=8,
        min_score=0.35,
    )

    reranker.rerank.assert_called_once_with(
        query="query",
        candidates=candidates,
        top_k=8,
        min_score=None,
    )


def test_query_whitespace_normalization(pipeline, dense, reranker, candidates):
    dense.search.return_value = candidates
    reranker.rerank.return_value = candidates[:2]

    pipeline.search("  example query  ")

    dense.search.assert_called_once_with(
        query="example query",
        top_k=5,
        min_score=0.35,
    )

    reranker.rerank.assert_called_once_with(
        query="example query",
        candidates=candidates,
        top_k=2,
        min_score=None,
    )


@pytest.mark.parametrize("query", ["", " ", "\n"])
def test_empty_query(pipeline, dense, query):
    with pytest.raises(ValueError, match="Query"):
        pipeline.search(query)

    dense.search.assert_not_called()


@pytest.mark.parametrize("query", [None, 123, []])
def test_invalid_query_type(pipeline, query):
    with pytest.raises(TypeError, match="Query"):
        pipeline.search(query)


@pytest.mark.parametrize("top_k", [0, -1, True, 1.5])
def test_invalid_top_k(pipeline, dense, top_k):
    with pytest.raises(ValueError, match="top_k"):
        pipeline.search("query", top_k=top_k)

    dense.search.assert_not_called()


@pytest.mark.parametrize("candidate_k", [0, -1, True, 1.5])
def test_invalid_candidate_override(pipeline, candidate_k):
    with pytest.raises(ValueError, match="candidate_k"):
        pipeline.search_candidates(
            "query",
            candidate_k=candidate_k,
        )


def test_retrieval_failure_propagates(pipeline, dense, reranker):
    dense.search.side_effect = RuntimeError("Retrieval failed")

    with pytest.raises(RuntimeError, match="Retrieval failed"):
        pipeline.search("query")

    reranker.rerank.assert_not_called()


def test_reranking_failure_propagates(pipeline, dense, reranker, candidates):
    dense.search.return_value = candidates

    reranker.rerank.side_effect = RuntimeError("Reranking failed")

    with pytest.raises(RuntimeError, match="Reranking failed"):
        pipeline.search("query")
