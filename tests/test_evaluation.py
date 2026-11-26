import pytest

from knowledge_assistant.evaluation import (
    answerability_metrics,
    latency_metrics,
    retrieval_metrics,
    summarize_retrieval_metrics,
)


def test_perfect_retrieval():
    result = retrieval_metrics(
        ["a", "b"],
        ["a", "b"],
        k=2,
    )

    assert result["precision"] == 1.0
    assert result["recall"] == 1.0
    assert result["reciprocal_rank"] == 1.0
    assert result["hit"] == 1.0
    assert result["complete_recall"] == 1.0


def test_partial_retrieval():
    result = retrieval_metrics(
        ["x", "a", "y"],
        ["a", "b"],
        k=3,
    )

    assert result["precision"] == pytest.approx(1 / 3)
    assert result["recall"] == 0.5
    assert result["reciprocal_rank"] == 0.5
    assert result["hit"] == 1.0
    assert result["complete_recall"] == 0.0


def test_no_relevant_results():
    result = retrieval_metrics(
        ["x", "y"],
        ["a"],
        k=2,
    )

    assert all(value == 0.0 for value in result.values())


def test_empty_retrieval():
    result = retrieval_metrics([], ["a"], k=3)

    assert result["precision"] == 0.0
    assert result["recall"] == 0.0


def test_invalid_k():
    with pytest.raises(ValueError):
        retrieval_metrics(["a"], ["a"], k=0)


def test_empty_ground_truth():
    with pytest.raises(ValueError):
        retrieval_metrics(["a"], [], k=1)


def test_duplicate_retrieved_ids():
    with pytest.raises(ValueError):
        retrieval_metrics(["a", "a"], ["a"], k=2)


def test_macro_average():
    rows = [
        retrieval_metrics(["a"], ["a"], k=1),
        retrieval_metrics(["x"], ["b"], k=1),
    ]

    summary = summarize_retrieval_metrics(rows)

    assert summary["precision"] == 0.5
    assert summary["recall"] == 0.5
    assert summary["mrr"] == 0.5
    assert summary["hit_rate"] == 0.5


def test_latency_metrics():
    result = latency_metrics([10.0, 20.0, 30.0])

    assert result["mean_ms"] == 20.0
    assert result["p50_ms"] == 20.0
    assert result["p95_ms"] == pytest.approx(29.0)
    assert result["max_ms"] == 30.0
    assert result["queries_per_second"] == pytest.approx(50.0)


def test_invalid_latency():
    with pytest.raises(ValueError):
        latency_metrics([10.0, -1.0])


def test_answerability_metrics():
    result = answerability_metrics(
        expected=[True, True, False, False],
        predicted=[True, False, True, False],
    )

    assert result["true_positive"] == 1
    assert result["true_negative"] == 1
    assert result["false_positive"] == 1
    assert result["false_negative"] == 1
    assert result["accuracy"] == 0.5
    assert result["f1"] == 0.5
    assert result["false_acceptance_rate"] == 0.5
    assert result["false_rejection_rate"] == 0.5


def test_invalid_classification_inputs():
    with pytest.raises(ValueError):
        answerability_metrics([True], [])
