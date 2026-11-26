import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class EvaluationQuery:
    query_id: str
    query: str
    category: str
    answerable: bool
    relevant_chunk_ids: list[str]
    expected_answer: str | None
    split: str


def load_evaluation_dataset(
    path: str | Path,
) -> list[EvaluationQuery]:
    """Load and validate labelled evaluation questions."""

    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(path)

    records = []

    with path.open(encoding="utf-8") as file:
        for line in file:
            if line.strip():
                records.append(EvaluationQuery(**json.loads(line)))

    if not records:
        raise ValueError("Evaluation dataset is empty.")

    query_ids = [item.query_id for item in records]

    if len(query_ids) != len(set(query_ids)):
        raise ValueError("Duplicate evaluation query IDs.")

    for item in records:
        if not item.query.strip():
            raise ValueError("Evaluation query cannot be empty.")

        if item.split not in {"calibration", "test"}:
            raise ValueError("Invalid evaluation split.")

        if item.answerable != bool(item.relevant_chunk_ids):
            raise ValueError("Inconsistent answerability labels.")

        if item.answerable and not item.expected_answer:
            raise ValueError("Missing expected answer.")

        if not item.answerable and item.expected_answer is not None:
            raise ValueError("Unexpected answer for unanswerable query.")

    return records


def retrieval_metrics(
    retrieved_ids: list[str],
    relevant_ids: list[str],
    k: int,
) -> dict:
    """Calculate ranking metrics for an answerable query."""

    if k <= 0:
        raise ValueError("k must be positive.")

    relevant = set(relevant_ids)

    if not relevant:
        raise ValueError("Relevant IDs cannot be empty.")

    retrieved = retrieved_ids[:k]

    if len(retrieved) != len(set(retrieved)):
        raise ValueError("Duplicate retrieved chunk IDs.")

    hits = len(set(retrieved) & relevant)

    reciprocal_rank = 0.0

    for rank, chunk_id in enumerate(retrieved, start=1):
        if chunk_id in relevant:
            reciprocal_rank = 1.0 / rank
            break

    return {
        "precision": hits / k,
        "recall": hits / len(relevant),
        "reciprocal_rank": reciprocal_rank,
        "hit": float(hits > 0),
        "complete_recall": float(hits == len(relevant)),
    }


def summarize_retrieval_metrics(
    rows: list[dict],
) -> dict:
    """Calculate macro-averaged retrieval metrics."""

    if not rows:
        raise ValueError("No metric records provided.")

    return {
        "precision": float(np.mean([row["precision"] for row in rows])),
        "recall": float(np.mean([row["recall"] for row in rows])),
        "mrr": float(np.mean([row["reciprocal_rank"] for row in rows])),
        "hit_rate": float(np.mean([row["hit"] for row in rows])),
        "complete_recall_rate": float(
            np.mean([row["complete_recall"] for row in rows])
        ),
    }


def latency_metrics(
    latencies_ms: list[float],
) -> dict:
    """Summarize observed retrieval latency."""

    if not latencies_ms:
        raise ValueError("No latency measurements provided.")

    values = np.asarray(latencies_ms, dtype=np.float64)

    if not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError("Invalid latency measurements.")

    return {
        "mean_ms": float(np.mean(values)),
        "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
        "max_ms": float(np.max(values)),
        "queries_per_second": float(len(values) / (np.sum(values) / 1000)),
    }


def answerability_metrics(
    expected: list[bool],
    predicted: list[bool],
) -> dict:
    """Evaluate answerable versus unanswerable classification."""

    if len(expected) != len(predicted) or not expected:
        raise ValueError("Invalid classification inputs.")

    pairs = list(zip(expected, predicted, strict=True))

    tp = sum(actual and pred for actual, pred in pairs)
    tn = sum(not actual and not pred for actual, pred in pairs)
    fp = sum(not actual and pred for actual, pred in pairs)
    fn = sum(actual and not pred for actual, pred in pairs)

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0

    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    return {
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "accuracy": (tp + tn) / len(expected),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_acceptance_rate": (fp / (fp + tn) if fp + tn else 0.0),
        "false_rejection_rate": (fn / (fn + tp) if fn + tp else 0.0),
    }
