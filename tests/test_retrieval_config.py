"""Tests for retrieval strategy configuration."""

import json
from dataclasses import FrozenInstanceError, asdict
from typing import cast

import pytest

from knowledge_assistant.config import RAGConfig
from knowledge_assistant.retrieval_config import RetrievalConfig
from knowledge_assistant.retrieval_interface import (
    RetrievalScoreType,
    RetrievalStrategy,
)


def test_default_configuration():
    config = RetrievalConfig()

    assert config.strategy == RetrievalStrategy.DENSE
    assert config.candidate_k == 5
    assert config.sparse_min_score is None
    assert config.bm25_k1 == 1.5
    assert config.bm25_b == 0.75
    assert config.rrf_k == 60
    assert config.dense_weight == 1.0
    assert config.sparse_weight == 1.0
    assert config.reranker_min_score == -9.0
    assert config.reranker_device == "cpu"
    assert config.reranker_batch_size == 8
    assert config.reranker_max_length == 512
    assert config.reranker_model == ("cross-encoder/ms-marco-MiniLM-L-6-v2")


@pytest.mark.parametrize(
    "strategy",
    list(RetrievalStrategy),
)
def test_strategy_normalization(strategy):
    config = RetrievalConfig(
        strategy=strategy.value,
    )

    assert config.strategy is strategy


def test_invalid_strategy_rejected():
    with pytest.raises(ValueError):
        RetrievalConfig(strategy=cast(RetrievalStrategy, "invalid"))


def test_configuration_is_immutable():
    config = RetrievalConfig()

    with pytest.raises(FrozenInstanceError):
        config.candidate_k = 10  # type: ignore


@pytest.mark.parametrize(
    "field_name",
    [
        "candidate_k",
        "rrf_k",
        "reranker_batch_size",
        "reranker_max_length",
    ],
)
@pytest.mark.parametrize(
    "value",
    [0, -1, True, 1.5],
)
def test_invalid_positive_integer(field_name, value):
    with pytest.raises(
        ValueError,
        match=field_name,
    ):
        RetrievalConfig(
            **{field_name: value},
        )


@pytest.mark.parametrize(
    "field_name",
    [
        "sparse_min_score",
        "reranker_min_score",
    ],
)
@pytest.mark.parametrize(
    "value",
    [
        float("nan"),
        float("inf"),
        float("-inf"),
    ],
)
def test_nonfinite_threshold_rejected(field_name, value):
    with pytest.raises(
        ValueError,
        match=field_name,
    ):
        RetrievalConfig(
            **{field_name: value},
        )


@pytest.mark.parametrize(
    "threshold",
    [None, -10.0, 0.0, 10.0],
)
def test_valid_thresholds(threshold):
    config = RetrievalConfig(
        sparse_min_score=threshold,
        reranker_min_score=threshold,
    )

    assert config.sparse_min_score == threshold
    assert config.reranker_min_score == threshold


@pytest.mark.parametrize(
    "value",
    [
        0.0,
        -1.0,
        float("nan"),
        float("inf"),
    ],
)
def test_invalid_bm25_k1(value):
    with pytest.raises(
        ValueError,
        match="bm25_k1",
    ):
        RetrievalConfig(bm25_k1=value)


@pytest.mark.parametrize(
    "value",
    [
        -0.1,
        1.1,
        float("nan"),
        float("inf"),
    ],
)
def test_invalid_bm25_b(value):
    with pytest.raises(
        ValueError,
        match="bm25_b",
    ):
        RetrievalConfig(bm25_b=value)


@pytest.mark.parametrize(
    "value",
    [0.0, 0.75, 1.0],
)
def test_valid_bm25_b(value):
    config = RetrievalConfig(bm25_b=value)

    assert config.bm25_b == value


@pytest.mark.parametrize(
    "field_name",
    ["dense_weight", "sparse_weight"],
)
@pytest.mark.parametrize(
    "value",
    [
        -1.0,
        float("nan"),
        float("inf"),
    ],
)
def test_invalid_fusion_weight(field_name, value):
    with pytest.raises(
        ValueError,
        match=field_name,
    ):
        RetrievalConfig(
            **{field_name: value},
        )


def test_both_zero_weights_rejected():
    with pytest.raises(
        ValueError,
        match="At least one retrieval weight",
    ):
        RetrievalConfig(
            dense_weight=0.0,
            sparse_weight=0.0,
        )


@pytest.mark.parametrize(
    ("dense_weight", "sparse_weight"),
    [
        (1.0, 0.0),
        (0.0, 1.0),
        (2.0, 1.0),
    ],
)
def test_valid_fusion_weights(
    dense_weight,
    sparse_weight,
):
    config = RetrievalConfig(
        dense_weight=dense_weight,
        sparse_weight=sparse_weight,
    )

    assert config.dense_weight == dense_weight
    assert config.sparse_weight == sparse_weight


@pytest.mark.parametrize(
    "field_name",
    [
        "reranker_model",
        "reranker_device",
    ],
)
@pytest.mark.parametrize(
    "value",
    ["", "   ", None, 123],
)
def test_invalid_string_configuration(field_name, value):
    with pytest.raises(
        ValueError,
        match=field_name,
    ):
        RetrievalConfig(
            **{field_name: value},
        )


def test_string_configuration_normalized():
    config = RetrievalConfig(
        reranker_model=" custom-model ",
        reranker_device=" cpu ",
    )

    assert config.reranker_model == "custom-model"
    assert config.reranker_device == "cpu"


@pytest.mark.parametrize(
    ("strategy", "dense", "sparse", "reranker"),
    [
        (RetrievalStrategy.DENSE, True, False, False),
        (RetrievalStrategy.BM25, False, True, False),
        (RetrievalStrategy.HYBRID, True, True, False),
        (
            RetrievalStrategy.DENSE_RERANKED,
            True,
            False,
            True,
        ),
    ],
)
def test_strategy_dependencies(
    strategy,
    dense,
    sparse,
    reranker,
):
    config = RetrievalConfig(strategy=strategy)

    assert config.requires_dense is dense
    assert config.requires_sparse is sparse
    assert config.requires_reranker is reranker


@pytest.mark.parametrize(
    ("strategy", "score_type"),
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
def test_score_type(strategy, score_type):
    config = RetrievalConfig(strategy=strategy)

    assert config.score_type is score_type


def test_configuration_json_serialization():
    config = RetrievalConfig(
        strategy=RetrievalStrategy.DENSE_RERANKED,
    )

    serialized = json.dumps(
        asdict(config),
        allow_nan=False,
    )

    result = json.loads(serialized)

    assert result["strategy"] == "dense_reranked"
    assert result["reranker_min_score"] == -9.0


def test_rag_config_default_retrieval():
    config = RAGConfig()

    assert isinstance(
        config.retrieval,
        RetrievalConfig,
    )

    assert config.retrieval.strategy == RetrievalStrategy.DENSE


def test_rag_config_backward_compatibility():
    config = RAGConfig(
        top_k=3,
        min_score=0.4,
        temperature=0.0,
    )

    assert config.top_k == 3
    assert config.min_score == 0.4
    assert config.temperature == 0.0
    assert config.retrieval.strategy == RetrievalStrategy.DENSE


def test_rag_config_custom_retrieval():
    retrieval = RetrievalConfig(
        strategy=RetrievalStrategy.HYBRID,
        candidate_k=10,
    )

    config = RAGConfig(
        retrieval=retrieval,
    )

    assert config.retrieval is retrieval
    assert config.retrieval.candidate_k == 10


def test_rag_config_rejects_invalid_retrieval():
    with pytest.raises(
        TypeError,
        match="retrieval must be a RetrievalConfig",
    ):
        RAGConfig(retrieval=None)  # type: ignore[arg-type]


def test_rag_configs_have_independent_defaults():
    first = RAGConfig()
    second = RAGConfig()

    assert first.retrieval == second.retrieval
    assert first.retrieval is not second.retrieval


def test_nested_configuration_serialization():
    config = RAGConfig(
        retrieval=RetrievalConfig(
            strategy=RetrievalStrategy.BM25,
        ),
    )

    serialized = json.dumps(
        asdict(config),
        allow_nan=False,
    )

    result = json.loads(serialized)

    assert result["retrieval"]["strategy"] == "bm25"
