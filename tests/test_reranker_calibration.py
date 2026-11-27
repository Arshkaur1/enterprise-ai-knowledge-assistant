"""Unit tests for reranker threshold calibration."""

import math

import pytest

from knowledge_assistant.reranker_calibration import (
    build_thresholds,
    dominates,
    evaluate_retrieval,
    filter_ranked_candidates,
    meets_quality_targets,
    pareto_frontier,
    satisfies_recall_constraints,
)


@pytest.fixture
def candidates():
    return [
        {"chunk_id": "a", "score": -8.0, "rank": 1},
        {"chunk_id": "b", "score": 3.0, "rank": 2},
        {"chunk_id": "c", "score": -3.0, "rank": 3},
    ]


@pytest.fixture
def records():
    return [
        {
            "query_id": "q1",
            "query": "Answerable question",
            "answerable": True,
            "relevant_chunk_ids": ["a"],
            "results": [
                {"chunk_id": "a", "score": 3.0},
            ],
        },
        {
            "query_id": "q2",
            "query": "Multi-part question",
            "answerable": True,
            "relevant_chunk_ids": ["b", "c"],
            "results": [
                {"chunk_id": "b", "score": 4.0},
                {"chunk_id": "c", "score": 2.0},
            ],
        },
        {
            "query_id": "q3",
            "query": "Unsupported question",
            "answerable": False,
            "relevant_chunk_ids": [],
            "results": [
                {"chunk_id": "a", "score": 1.0},
            ],
        },
        {
            "query_id": "q4",
            "query": "Out-of-domain question",
            "answerable": False,
            "relevant_chunk_ids": [],
            "results": [],
        },
    ]


def test_unfiltered_ranking(candidates):
    results = filter_ranked_candidates(
        candidates,
        threshold=None,
        top_k=3,
    )

    assert [r["chunk_id"] for r in results] == [
        "b",
        "c",
        "a",
    ]

    assert [r["rank"] for r in results] == [1, 2, 3]


def test_inclusive_threshold(candidates):
    results = filter_ranked_candidates(
        candidates,
        threshold=-3.0,
    )

    assert [r["chunk_id"] for r in results] == [
        "b",
        "c",
    ]


def test_exact_boundary_excludes_score(candidates):
    results = filter_ranked_candidates(
        candidates,
        threshold=math.nextafter(-3.0, math.inf),
    )

    assert [r["chunk_id"] for r in results] == ["b"]


def test_all_candidates_filtered(candidates):
    assert (
        filter_ranked_candidates(
            candidates,
            threshold=4.0,
        )
        == []
    )


def test_original_candidates_unchanged(candidates):
    original = [candidate.copy() for candidate in candidates]

    filter_ranked_candidates(
        candidates,
        threshold=None,
    )

    assert candidates == original


def test_deterministic_ties():
    candidates = [
        {"chunk_id": "b", "score": 2.0},
        {"chunk_id": "a", "score": 2.0},
    ]

    results = filter_ranked_candidates(
        candidates,
        threshold=None,
    )

    assert [r["chunk_id"] for r in results] == [
        "a",
        "b",
    ]


def test_regular_threshold_grid():
    thresholds = build_thresholds([])

    assert len(thresholds) == 90
    assert thresholds[0] is None
    assert thresholds[1] == -12.0
    assert thresholds[-1] == 10.0


def test_observed_score_boundaries(candidates):
    thresholds = build_thresholds([{"scored_candidates": candidates}])

    for candidate in candidates:
        score = candidate["score"]

        assert isinstance(score, (int, float))
        assert score in thresholds
        assert (
            math.nextafter(
                score,
                math.inf,
            )
            in thresholds
        )


def test_invalid_threshold_step():
    with pytest.raises(ValueError):
        build_thresholds([], step=0.0)


def test_confusion_matrix(records):
    summary = evaluate_retrieval(records)["summary"]

    assert summary["true_positives"] == 2
    assert summary["true_negatives"] == 1
    assert summary["false_positives"] == 1
    assert summary["false_negatives"] == 0

    assert summary["answerability_accuracy"] == 0.75
    assert summary["false_acceptance_rate"] == 0.5
    assert summary["false_rejection_rate"] == 0.0


def test_retrieval_metrics(records):
    summary = evaluate_retrieval(records)["summary"]

    assert summary["precision_at_k"] == 0.75
    assert summary["recall_at_k"] == 1.0
    assert summary["complete_recall_at_k"] == 1.0
    assert summary["mrr"] == 1.0
    assert summary["hit_at_k"] == 1.0


def test_returned_chunk_precision(records):
    summary = evaluate_retrieval(records)["summary"]

    assert summary["returned_chunk_micro_precision"] == 1.0
    assert summary["global_returned_chunk_precision"] == 0.75


def test_partial_evidence_recall(records):
    records[1]["results"] = [
        {"chunk_id": "b", "score": 4.0},
    ]

    summary = evaluate_retrieval(records)["summary"]

    assert summary["recall_at_k"] == 0.75
    assert summary["complete_recall_at_k"] == 0.5
    assert summary["false_rejection_rate"] == 0.0

    assert not satisfies_recall_constraints(summary)


def test_false_rejection(records):
    records[0]["results"] = []

    summary = evaluate_retrieval(records)["summary"]

    assert summary["false_negatives"] == 1
    assert summary["false_rejection_rate"] == 0.5


def test_wrong_nonempty_context(records):
    records[0]["results"] = [
        {"chunk_id": "irrelevant", "score": 3.0},
    ]

    summary = evaluate_retrieval(records)["summary"]

    assert summary["false_rejection_rate"] == 0.0
    assert summary["recall_at_k"] == 0.5

    assert not satisfies_recall_constraints(summary)


def test_recall_constraints(records):
    summary = evaluate_retrieval(records)["summary"]

    assert satisfies_recall_constraints(summary)


def test_quality_targets(records):
    baseline = evaluate_retrieval(records)["summary"]

    records[2]["results"] = []

    improved = evaluate_retrieval(records)["summary"]

    assert meets_quality_targets(
        improved,
        baseline,
    )

    assert not meets_quality_targets(
        baseline,
        baseline,
    )


def test_pareto_dominance(records):
    baseline = evaluate_retrieval(records)

    records[2]["results"] = []

    improved = evaluate_retrieval(records)

    assert dominates(improved, baseline)
    assert not dominates(baseline, improved)


def test_pareto_frontier(records):
    baseline = evaluate_retrieval(records)

    records[2]["results"] = []

    improved = evaluate_retrieval(records)

    assert pareto_frontier(
        [
            baseline,
            improved,
        ]
    ) == [improved]


def test_identical_metrics_do_not_dominate(records):
    experiment = evaluate_retrieval(records)

    assert not dominates(
        experiment,
        experiment,
    )


def test_empty_evaluation_rejected():
    with pytest.raises(ValueError):
        evaluate_retrieval([])


def test_duplicate_retrieved_ids_rejected(records):
    records[0]["results"] = [
        {"chunk_id": "a"},
        {"chunk_id": "a"},
    ]

    with pytest.raises(ValueError, match="Duplicate"):
        evaluate_retrieval(records)
