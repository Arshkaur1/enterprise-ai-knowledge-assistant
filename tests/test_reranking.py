"""Unit tests for cross-encoder reranking."""

from unittest.mock import patch

import numpy as np
import pytest
from torch.nn import Identity

from knowledge_assistant.models import DocumentChunk
from knowledge_assistant.reranking import (
    DEFAULT_RERANKER_MODEL,
    CrossEncoderReranker,
)
from knowledge_assistant.retrieval import SearchResult


def make_result(
    chunk_id: str,
    text: str = "Example document content.",
    rank: int = 1,
    score: float = 0.5,
) -> SearchResult:
    chunk = DocumentChunk(
        chunk_id=chunk_id,
        document_id="test_document",
        source="test.txt",
        file_type="txt",
        category="General",
        text=text,
        chunk_index=0,
    )

    return SearchResult(
        rank=rank,
        score=score,
        chunk=chunk,
    )


@pytest.fixture
def mock_model():
    with patch("knowledge_assistant.reranking.CrossEncoder") as model_class:
        yield model_class


@pytest.fixture
def reranker(mock_model):
    return CrossEncoderReranker()


@pytest.fixture
def candidates():
    return [
        make_result("chunk_a", rank=1, score=0.9),
        make_result("chunk_b", rank=2, score=0.7),
        make_result("chunk_c", rank=3, score=0.5),
    ]


def test_default_configuration(mock_model):
    reranker = CrossEncoderReranker()

    assert reranker.model_name == DEFAULT_RERANKER_MODEL
    assert reranker.device == "cpu"
    assert reranker.batch_size == 8
    assert reranker.max_length == 512

    mock_model.assert_called_once_with(
        DEFAULT_RERANKER_MODEL,
        device="cpu",
        max_length=512,
    )


def test_custom_configuration(mock_model):
    reranker = CrossEncoderReranker(
        model_name="custom-model",
        device="mps",
        batch_size=4,
        max_length=256,
    )

    assert reranker.model_name == "custom-model"
    assert reranker.device == "mps"
    assert reranker.batch_size == 4
    assert reranker.max_length == 256

    mock_model.assert_called_once_with(
        "custom-model",
        device="mps",
        max_length=256,
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"model_name": ""},
        {"model_name": "   "},
        {"device": ""},
        {"device": "   "},
        {"batch_size": 0},
        {"batch_size": -1},
        {"batch_size": True},
        {"batch_size": 1.5},
        {"max_length": 0},
        {"max_length": -1},
        {"max_length": False},
        {"max_length": 2.5},
    ],
)
def test_invalid_configuration(mock_model, kwargs):
    with pytest.raises(ValueError):
        CrossEncoderReranker(**kwargs)

    mock_model.assert_not_called()


def test_reranks_by_model_score(reranker, candidates):
    reranker.model.predict.return_value = np.array([-5.0, 3.0, 1.0])

    results = reranker.rerank(
        "Example query",
        candidates,
        top_k=3,
    )

    assert [r.chunk.chunk_id for r in results] == [
        "chunk_b",
        "chunk_c",
        "chunk_a",
    ]

    assert [r.rank for r in results] == [1, 2, 3]
    assert [r.score for r in results] == [3.0, 1.0, -5.0]


def test_top_k_limits_results(reranker, candidates):
    reranker.model.predict.return_value = [1.0, 3.0, 2.0]

    results = reranker.rerank(
        "query",
        candidates,
        top_k=2,
    )

    assert len(results) == 2
    assert results[0].chunk.chunk_id == "chunk_b"
    assert results[1].chunk.chunk_id == "chunk_c"


def test_top_k_exceeds_candidate_count(reranker, candidates):
    reranker.model.predict.return_value = [1.0, 2.0, 3.0]

    results = reranker.rerank(
        "query",
        candidates,
        top_k=10,
    )

    assert len(results) == 3


def test_negative_scores_are_preserved(reranker, candidates):
    reranker.model.predict.return_value = [
        -8.0,
        -2.0,
        -11.0,
    ]

    results = reranker.rerank(
        "query",
        candidates,
        top_k=3,
    )

    assert [r.score for r in results] == [
        -2.0,
        -8.0,
        -11.0,
    ]


def test_minimum_score_filter(reranker, candidates):
    reranker.model.predict.return_value = [
        -8.0,
        2.0,
        -3.0,
    ]

    results = reranker.rerank(
        "query",
        candidates,
        top_k=3,
        min_score=-4.0,
    )

    assert [r.score for r in results] == [2.0, -3.0]
    assert [r.rank for r in results] == [1, 2]


def test_threshold_is_inclusive(reranker, candidates):
    reranker.model.predict.return_value = [
        -5.0,
        -4.0,
        -6.0,
    ]

    results = reranker.rerank(
        "query",
        candidates,
        top_k=3,
        min_score=-5.0,
    )

    assert [r.score for r in results] == [-4.0, -5.0]


def test_all_candidates_filtered(reranker, candidates):
    reranker.model.predict.return_value = [
        -10.0,
        -9.0,
        -8.0,
    ]

    assert (
        reranker.rerank(
            "query",
            candidates,
            min_score=0.0,
        )
        == []
    )


def test_empty_candidates_skip_inference(reranker):
    assert reranker.rerank("query", []) == []

    reranker.model.predict.assert_not_called()


def test_deterministic_tie_breaking(reranker, candidates):
    reranker.model.predict.return_value = [2.0, 2.0, 2.0]

    results = reranker.rerank(
        "query",
        list(reversed(candidates)),
        top_k=3,
    )

    assert [r.chunk.chunk_id for r in results] == [
        "chunk_a",
        "chunk_b",
        "chunk_c",
    ]


def test_original_results_are_not_modified(reranker, candidates):
    reranker.model.predict.return_value = [1.0, 3.0, 2.0]

    original_scores = [r.score for r in candidates]
    original_ranks = [r.rank for r in candidates]

    results = reranker.rerank(
        "query",
        candidates,
        top_k=3,
    )

    assert [r.score for r in candidates] == original_scores
    assert [r.rank for r in candidates] == original_ranks
    assert results[0] is not candidates[1]
    assert results[0].chunk is candidates[1].chunk


def test_predict_arguments(reranker, candidates):
    reranker.model.predict.return_value = [1.0, 2.0, 3.0]

    reranker.rerank("  query  ", candidates)

    args, kwargs = reranker.model.predict.call_args

    assert args[0] == [("query", candidate.chunk.text) for candidate in candidates]

    assert kwargs["batch_size"] == 8
    assert kwargs["show_progress_bar"] is False
    assert kwargs["convert_to_numpy"] is True
    assert isinstance(kwargs["activation_fn"], Identity)


@pytest.mark.parametrize("query", ["", " ", "\n\t"])
def test_empty_query(reranker, candidates, query):
    with pytest.raises(ValueError, match="Query"):
        reranker.rerank(query, candidates)

    reranker.model.predict.assert_not_called()


@pytest.mark.parametrize("query", [None, 123, []])
def test_invalid_query_type(reranker, candidates, query):
    with pytest.raises(TypeError, match="Query"):
        reranker.rerank(query, candidates)


@pytest.mark.parametrize("top_k", [0, -1, True, 1.5])
def test_invalid_top_k(reranker, candidates, top_k):
    with pytest.raises(ValueError, match="top_k"):
        reranker.rerank(
            "query",
            candidates,
            top_k=top_k,
        )


@pytest.mark.parametrize(
    "threshold",
    [float("nan"), float("inf"), float("-inf")],
)
def test_invalid_threshold(reranker, candidates, threshold):
    with pytest.raises(ValueError, match="finite"):
        reranker.rerank(
            "query",
            candidates,
            min_score=threshold,
        )


def test_duplicate_candidates(reranker, candidates):
    with pytest.raises(ValueError, match="Duplicate"):
        reranker.rerank(
            "query",
            [candidates[0], candidates[0]],
        )

    reranker.model.predict.assert_not_called()


def test_empty_candidate_text(reranker):
    with pytest.raises(ValueError, match="empty"):
        reranker.rerank(
            "query",
            [make_result("empty_chunk", text="   ")],
        )

    reranker.model.predict.assert_not_called()


@pytest.mark.parametrize(
    "scores",
    [
        [1.0, 2.0],
        [1.0, 2.0, 3.0, 4.0],
        [[1.0], [2.0], [3.0]],
        1.0,
    ],
)
def test_invalid_score_shape(reranker, candidates, scores):
    reranker.model.predict.return_value = scores

    with pytest.raises(ValueError, match="score shape"):
        reranker.rerank("query", candidates)


@pytest.mark.parametrize(
    "scores",
    [
        [float("nan"), 1.0, 2.0],
        [float("inf"), 1.0, 2.0],
        [float("-inf"), 1.0, 2.0],
    ],
)
def test_nonfinite_model_scores(reranker, candidates, scores):
    reranker.model.predict.return_value = scores

    with pytest.raises(ValueError, match="non-finite"):
        reranker.rerank("query", candidates)


def test_model_failure_propagates(reranker, candidates):
    reranker.model.predict.side_effect = RuntimeError("Inference failed")

    with pytest.raises(RuntimeError, match="Inference failed"):
        reranker.rerank("query", candidates)
