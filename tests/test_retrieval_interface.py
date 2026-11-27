"""Tests for retrieval interfaces and adapters."""

import json
from typing import cast
from unittest.mock import Mock

import pytest

from knowledge_assistant.retrieval_interface import (
    RetrievalScoreType,
    RetrievalStrategy,
    RetrieverProtocol,
    ThresholdRetrieverAdapter,
    get_score_type,
)


@pytest.mark.parametrize(
    ("strategy", "expected"),
    [
        (RetrievalStrategy.DENSE, "dense"),
        (RetrievalStrategy.BM25, "bm25"),
        (RetrievalStrategy.HYBRID, "hybrid"),
        (
            RetrievalStrategy.DENSE_RERANKED,
            "dense_reranked",
        ),
    ],
)
def test_strategy_values(strategy, expected):
    assert strategy.value == expected
    assert str(strategy) == expected
    assert strategy == expected


def test_invalid_strategy_rejected():
    with pytest.raises(ValueError):
        RetrievalStrategy("unknown")


def test_strategy_json_serialization():
    result = json.dumps(
        {
            "strategy": RetrievalStrategy.DENSE,
        }
    )

    assert json.loads(result) == {
        "strategy": "dense",
    }


@pytest.mark.parametrize(
    ("strategy", "expected"),
    [
        (
            RetrievalStrategy.DENSE,
            RetrievalScoreType.COSINE_SIMILARITY,
        ),
        (
            RetrievalStrategy.BM25,
            RetrievalScoreType.BM25,
        ),
        (
            RetrievalStrategy.HYBRID,
            RetrievalScoreType.RRF,
        ),
        (
            RetrievalStrategy.DENSE_RERANKED,
            RetrievalScoreType.CROSS_ENCODER_LOGIT,
        ),
    ],
)
def test_score_type_mapping(strategy, expected):
    assert get_score_type(strategy) == expected


def test_score_type_invalid_strategy():
    with pytest.raises(ValueError):
        get_score_type(cast(RetrievalStrategy, "invalid"))


def test_adapter_passes_dense_threshold():
    retriever = Mock()
    retriever.search.return_value = []

    adapter = ThresholdRetrieverAdapter(
        retriever=retriever,
        min_score=0.35,
    )

    result = adapter.search(
        query="annual leave",
        top_k=2,
    )

    retriever.search.assert_called_once_with(
        query="annual leave",
        top_k=2,
        min_score=0.35,
    )

    assert result == []


def test_adapter_passes_none_threshold():
    retriever = Mock()
    retriever.search.return_value = []

    adapter = ThresholdRetrieverAdapter(
        retriever=retriever,
        min_score=None,
    )

    adapter.search(
        query="expense policy",
        top_k=5,
    )

    retriever.search.assert_called_once_with(
        query="expense policy",
        top_k=5,
        min_score=None,
    )


def test_adapter_preserves_results():
    retriever = Mock()

    expected = [Mock(), Mock()]
    retriever.search.return_value = expected

    adapter = ThresholdRetrieverAdapter(
        retriever=retriever,
        min_score=0.35,
    )

    actual = adapter.search("test query")

    assert actual is expected


def test_adapter_preserves_empty_results():
    retriever = Mock()
    retriever.search.return_value = []

    adapter = ThresholdRetrieverAdapter(retriever)

    assert adapter.search("unsupported query") == []


def test_adapter_default_top_k():
    retriever = Mock()

    adapter = ThresholdRetrieverAdapter(
        retriever=retriever,
        min_score=0.35,
    )

    adapter.search("test query")

    retriever.search.assert_called_once_with(
        query="test query",
        top_k=2,
        min_score=0.35,
    )


@pytest.mark.parametrize(
    "threshold",
    [
        float("nan"),
        float("inf"),
        float("-inf"),
    ],
)
def test_adapter_rejects_nonfinite_threshold(threshold):
    with pytest.raises(
        ValueError,
        match="Minimum score must be finite",
    ):
        ThresholdRetrieverAdapter(
            retriever=Mock(),
            min_score=threshold,
        )


@pytest.mark.parametrize(
    "threshold",
    [-10.0, -1.0, 0.0, 0.35, 1.0, 10.0],
)
def test_adapter_accepts_finite_threshold(threshold):
    adapter = ThresholdRetrieverAdapter(
        retriever=Mock(),
        min_score=threshold,
    )

    assert adapter.min_score == threshold


def test_adapter_propagates_retrieval_errors():
    retriever = Mock()
    retriever.search.side_effect = RuntimeError("Retrieval failed.")

    adapter = ThresholdRetrieverAdapter(retriever)

    with pytest.raises(
        RuntimeError,
        match="Retrieval failed",
    ):
        adapter.search("test query")


def test_adapter_implements_retriever_protocol():
    adapter = ThresholdRetrieverAdapter(
        retriever=Mock(),
    )

    assert isinstance(adapter, RetrieverProtocol)


def test_structural_protocol_implementation():
    class CustomRetriever:
        def search(self, query, top_k=2):
            return []

    assert isinstance(
        CustomRetriever(),
        RetrieverProtocol,
    )


def test_object_without_search_fails_protocol():
    assert not isinstance(
        object(),
        RetrieverProtocol,
    )


def test_adapter_instances_keep_separate_thresholds():
    retriever = Mock()

    first = ThresholdRetrieverAdapter(
        retriever=retriever,
        min_score=0.35,
    )

    second = ThresholdRetrieverAdapter(
        retriever=retriever,
        min_score=None,
    )

    first.search("query one")
    second.search("query two")

    calls = retriever.search.call_args_list

    assert calls[0].kwargs["min_score"] == 0.35
    assert calls[1].kwargs["min_score"] is None
