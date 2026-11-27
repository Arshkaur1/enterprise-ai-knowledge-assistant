"""Metrics and threshold calibration for reranked retrieval."""

import math
from collections.abc import Sequence

import numpy as np


def filter_ranked_candidates(
    candidates: list[dict],
    threshold: float | None,
    top_k: int = 2,
) -> list[dict]:
    """Filter cached reranker scores and assign contiguous ranks."""

    if type(top_k) is not int or top_k < 1:
        raise ValueError("top_k must be a positive integer.")

    if threshold is not None and not math.isfinite(threshold):
        raise ValueError("Threshold must be finite.")

    selected = [
        candidate
        for candidate in candidates
        if threshold is None or candidate["score"] >= threshold
    ]

    selected.sort(
        key=lambda candidate: (
            -candidate["score"],
            candidate["chunk_id"],
        )
    )

    return [
        {
            **candidate,
            "rank": rank,
        }
        for rank, candidate in enumerate(
            selected[:top_k],
            start=1,
        )
    ]


def build_thresholds(
    records: Sequence[dict],
    start: float = -12.0,
    stop: float = 10.0,
    step: float = 0.25,
) -> list[float | None]:
    """Combine a regular threshold grid with score boundaries."""

    if not all(math.isfinite(value) for value in (start, stop, step)):
        raise ValueError("Threshold parameters must be finite.")

    if step <= 0 or stop < start:
        raise ValueError("Invalid threshold range.")

    grid = np.arange(
        start,
        stop + step / 2,
        step,
    )

    thresholds = {round(float(value), 10) for value in grid if value <= stop}

    thresholds.add(float(stop))

    for record in records:
        for candidate in record["scored_candidates"]:
            score = float(candidate["score"])

            if not math.isfinite(score):
                raise ValueError("Cached candidate score must be finite.")

            thresholds.add(score)

            # Filtering is inclusive, so evaluate the smallest
            # representable threshold above the observed score.
            boundary = math.nextafter(score, math.inf)

            if math.isfinite(boundary):
                thresholds.add(boundary)

    return [None, *sorted(thresholds)]


def evaluate_retrieval(
    records: Sequence[dict],
    top_k: int = 2,
) -> dict:
    """Evaluate retrieval quality and nonempty retrieval outcomes."""

    if not records:
        raise ValueError("Evaluation records cannot be empty.")

    if type(top_k) is not int or top_k < 1:
        raise ValueError("top_k must be a positive integer.")

    tp = tn = fp = fn = 0

    precision_values = []
    recall_values = []
    reciprocal_ranks = []
    hit_values = []
    complete_recall_values = []

    relevant_returned = 0
    answerable_chunks_returned = 0
    total_chunks_returned = 0

    query_results = []

    for record in records:
        expected_answerable = record["answerable"]

        relevant_ids = set(record["relevant_chunk_ids"])

        retrieved_ids = [candidate["chunk_id"] for candidate in record["results"]]

        if len(retrieved_ids) > top_k:
            raise ValueError("Retrieved count exceeds top_k.")

        if len(retrieved_ids) != len(set(retrieved_ids)):
            raise ValueError("Duplicate retrieved chunk IDs.")

        predicted_answerable = bool(retrieved_ids)

        total_chunks_returned += len(retrieved_ids)

        if expected_answerable:
            if not relevant_ids:
                raise ValueError("Answerable query has no relevance labels.")

            if predicted_answerable:
                tp += 1
            else:
                fn += 1

        elif predicted_answerable:
            fp += 1
        else:
            tn += 1

        metrics = None

        if expected_answerable:
            hits = len(set(retrieved_ids) & relevant_ids)

            precision = hits / top_k
            recall = hits / len(relevant_ids)

            reciprocal_rank = next(
                (
                    1.0 / rank
                    for rank, chunk_id in enumerate(
                        retrieved_ids,
                        start=1,
                    )
                    if chunk_id in relevant_ids
                ),
                0.0,
            )

            hit = float(hits > 0)

            complete_recall = float(hits == len(relevant_ids))

            precision_values.append(precision)
            recall_values.append(recall)
            reciprocal_ranks.append(reciprocal_rank)
            hit_values.append(hit)
            complete_recall_values.append(complete_recall)

            relevant_returned += hits
            answerable_chunks_returned += len(retrieved_ids)

            metrics = {
                "precision_at_k": precision,
                "recall_at_k": recall,
                "reciprocal_rank": reciprocal_rank,
                "hit_at_k": hit,
                "complete_recall_at_k": complete_recall,
            }

        query_results.append(
            {
                "query_id": record["query_id"],
                "query": record["query"],
                "answerable": expected_answerable,
                "predicted_answerable": predicted_answerable,
                "relevant_chunk_ids": sorted(relevant_ids),
                "retrieved_chunk_ids": retrieved_ids,
                "metrics": metrics,
            }
        )

    if not precision_values:
        raise ValueError("No answerable evaluation queries.")

    if tn + fp == 0:
        raise ValueError("No unsupported evaluation queries.")

    answerability_precision = tp / (tp + fp) if tp + fp else 0.0

    answerability_recall = tp / (tp + fn)

    answerability_f1 = (
        2
        * answerability_precision
        * answerability_recall
        / (answerability_precision + answerability_recall)
        if answerability_precision + answerability_recall
        else 0.0
    )

    summary = {
        "total_queries": len(records),
        "answerable_queries": tp + fn,
        "unsupported_queries": tn + fp,
        "true_positives": tp,
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "answerability_accuracy": ((tp + tn) / len(records)),
        "answerability_precision": answerability_precision,
        "answerability_recall": answerability_recall,
        "answerability_f1": answerability_f1,
        "false_acceptance_rate": fp / (fp + tn),
        "false_rejection_rate": fn / (fn + tp),
        "precision_at_k": float(np.mean(precision_values)),
        "recall_at_k": float(np.mean(recall_values)),
        "mrr": float(np.mean(reciprocal_ranks)),
        "hit_at_k": float(np.mean(hit_values)),
        "complete_recall_at_k": float(np.mean(complete_recall_values)),
        "returned_chunk_micro_precision": (
            relevant_returned / answerable_chunks_returned
            if answerable_chunks_returned
            else 0.0
        ),
        "global_returned_chunk_precision": (
            relevant_returned / total_chunks_returned if total_chunks_returned else 0.0
        ),
        "relevant_chunks_returned": relevant_returned,
        "answerable_chunks_returned": (answerable_chunks_returned),
        "total_chunks_returned": total_chunks_returned,
    }

    return {
        "summary": summary,
        "results": query_results,
    }


def satisfies_recall_constraints(
    summary: dict,
) -> bool:
    """Check whether all labelled evidence is preserved."""

    return (
        math.isclose(
            summary["recall_at_k"],
            1.0,
            abs_tol=1e-9,
        )
        and math.isclose(
            summary["complete_recall_at_k"],
            1.0,
            abs_tol=1e-9,
        )
        and math.isclose(
            summary["false_rejection_rate"],
            0.0,
            abs_tol=1e-9,
        )
    )


def meets_quality_targets(
    summary: dict,
    baseline: dict,
) -> bool:
    """Check retrieval quality targets, excluding latency."""

    tolerance = 1e-9

    return (
        satisfies_recall_constraints(summary)
        and summary["false_acceptance_rate"] <= 0.25
        and summary["mrr"] + tolerance >= baseline["mrr"]
        and summary["precision_at_k"] + tolerance >= baseline["precision_at_k"]
        and summary["returned_chunk_micro_precision"] + tolerance
        >= baseline["returned_chunk_micro_precision"]
    )


def objective_vector(experiment: dict) -> tuple:
    """Convert all optimization objectives to maximization."""

    summary = experiment["summary"]

    return (
        summary["complete_recall_at_k"],
        summary["recall_at_k"],
        -summary["false_rejection_rate"],
        -summary["false_acceptance_rate"],
        summary["returned_chunk_micro_precision"],
        summary["mrr"],
    )


def dominates(first: dict, second: dict) -> bool:
    """Return whether first Pareto-dominates second."""

    first_values = objective_vector(first)
    second_values = objective_vector(second)

    comparisons = list(
        zip(
            first_values,
            second_values,
            strict=True,
        )
    )

    tolerance = 1e-9

    no_worse = all(a >= b - tolerance for a, b in comparisons)

    strictly_better = any(a > b + tolerance for a, b in comparisons)

    return no_worse and strictly_better


def pareto_frontier(
    experiments: Sequence[dict],
) -> list[dict]:
    """Return nondominated retrieval configurations."""

    return [
        experiment
        for index, experiment in enumerate(experiments)
        if not any(
            dominates(other, experiment)
            for other_index, other in enumerate(experiments)
            if other_index != index
        )
    ]
