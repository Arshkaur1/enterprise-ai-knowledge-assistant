"""Tests for comparative retrieval benchmarking."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "benchmark_reranked_retrieval.py"
)

SPEC = importlib.util.spec_from_file_location(
    "benchmark_reranked_retrieval",
    SCRIPT_PATH,
)

assert SPEC is not None and SPEC.loader is not None
benchmark = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(benchmark)


def make_result(
    chunk_id: str,
    rank: int = 1,
    score: float = 0.5,
):
    return SimpleNamespace(
        rank=rank,
        score=score,
        chunk=SimpleNamespace(
            chunk_id=chunk_id,
        ),
    )


def test_serialize_results():
    results = [
        make_result("chunk_a"),
        make_result(
            "chunk_b",
            rank=2,
            score=-2.0,
        ),
    ]

    serialized = benchmark.serialize_results(results)

    assert serialized == [
        {
            "rank": 1,
            "score": 0.5,
            "chunk_id": "chunk_a",
        },
        {
            "rank": 2,
            "score": -2.0,
            "chunk_id": "chunk_b",
        },
    ]


def test_latency_summary():
    summary = benchmark.summarize_latency([1.0, 2.0, 3.0, 4.0, 5.0])

    assert summary["measured_requests"] == 5
    assert summary["mean_latency_ms"] == 3.0
    assert summary["p50_latency_ms"] == 3.0
    assert summary["p95_latency_ms"] == pytest.approx(4.8)
    assert summary["max_latency_ms"] == 5.0
    assert summary["min_latency_ms"] == 1.0


def test_empty_latency_rejected():
    with pytest.raises(ValueError):
        benchmark.summarize_latency([])


@pytest.mark.parametrize(
    "value",
    [float("nan"), float("inf"), -1.0],
)
def test_invalid_latency_rejected(value):
    with pytest.raises(ValueError):
        benchmark.summarize_latency([value])


def test_deterministic_retrieval():
    def search(query):
        return [make_result("chunk_a")]

    result = benchmark.benchmark_query(
        search=search,
        query="test query",
        repetitions=5,
    )

    assert result["deterministic"]
    assert result["deterministic_requests"] == 5
    assert result["determinism_rate"] == 1.0
    assert len(result["latencies_ms"]) == 5


def test_nondeterministic_retrieval():
    responses = iter(
        [
            [make_result("chunk_a")],
            [make_result("chunk_b")],
            [make_result("chunk_a")],
        ]
    )

    def search(query):
        return next(responses)

    result = benchmark.benchmark_query(
        search=search,
        query="test query",
        repetitions=3,
    )

    assert not result["deterministic"]
    assert result["deterministic_requests"] == 2
    assert result["determinism_rate"] == pytest.approx(2 / 3)


def test_ranking_order_affects_determinism():
    responses = iter(
        [
            [
                make_result("a"),
                make_result("b", rank=2),
            ],
            [
                make_result("b"),
                make_result("a", rank=2),
            ],
        ]
    )

    def search(query):
        return next(responses)

    result = benchmark.benchmark_query(
        search=search,
        query="test query",
        repetitions=2,
    )

    assert not result["deterministic"]


def test_score_variation_does_not_change_determinism():
    responses = iter(
        [
            [make_result("a", score=0.500001)],
            [make_result("a", score=0.500002)],
        ]
    )

    def search(query):
        return next(responses)

    result = benchmark.benchmark_query(
        search=search,
        query="test query",
        repetitions=2,
    )

    assert result["deterministic"]


def test_empty_results_are_deterministic():
    def search(query):
        return []

    result = benchmark.benchmark_query(
        search=search,
        query="unsupported query",
        repetitions=3,
    )

    assert result["results"] == []
    assert result["deterministic"]
    assert result["determinism_rate"] == 1.0


@pytest.mark.parametrize(
    "repetitions",
    [0, -1, True, 1.5],
)
def test_invalid_repetitions(repetitions):
    with pytest.raises(ValueError):
        benchmark.benchmark_query(
            search=lambda query: [],
            query="test query",
            repetitions=repetitions,
        )


def test_benchmark_experiment():
    examples = [
        SimpleNamespace(
            query_id="q1",
            query="answerable",
            answerable=True,
            relevant_chunk_ids=["chunk_a"],
        ),
        SimpleNamespace(
            query_id="q2",
            query="unsupported",
            answerable=False,
            relevant_chunk_ids=[],
        ),
    ]

    calls = []

    def search(query):
        calls.append(query)

        if query == "answerable":
            return [make_result("chunk_a")]

        return []

    result = benchmark.benchmark_experiment(
        name="test",
        search=search,
        examples=examples,
        configuration={},
        repetitions=3,
        warmup_requests=2,
    )

    summary = result["summary"]

    assert len(calls) == 8
    assert summary["measured_requests"] == 6
    assert summary["recall_at_k"] == 1.0
    assert summary["false_acceptance_rate"] == 0.0
    assert summary["false_rejection_rate"] == 0.0
    assert summary["determinism_rate"] == 1.0


def test_candidate_diagnostics_are_separate():
    examples = [
        SimpleNamespace(
            query_id="q1",
            query="answerable",
            answerable=True,
            relevant_chunk_ids=["chunk_a"],
        ),
        SimpleNamespace(
            query_id="q2",
            query="unsupported",
            answerable=False,
            relevant_chunk_ids=[],
        ),
    ]

    candidate_calls = []

    def search(query):
        if query == "answerable":
            return [make_result("chunk_a")]

        return []

    def candidate_search(query):
        candidate_calls.append(query)

        return [
            make_result("chunk_a"),
            make_result("chunk_b", rank=2),
        ]

    result = benchmark.benchmark_experiment(
        name="test",
        search=search,
        examples=examples,
        configuration={},
        candidate_search=candidate_search,
        repetitions=3,
        warmup_requests=1,
    )

    assert len(candidate_calls) == 2
    assert result["summary"]["measured_requests"] == 6
    assert result["summary"]["mean_candidate_count"] == 2
    assert result["summary"]["max_candidate_count"] == 2


def test_write_csv(tmp_path):
    path = tmp_path / "results.csv"

    benchmark.write_csv(
        path,
        [
            {
                "experiment": "dense",
                "latency": 3.0,
            },
            {
                "experiment": "reranked",
                "latency": 30.0,
                "threshold": -9.0,
            },
        ],
    )

    content = path.read_text(encoding="utf-8")

    assert "experiment,latency,threshold" in content
    assert "dense,3.0" in content
    assert "reranked,30.0,-9.0" in content


def test_empty_csv_rejected(tmp_path):
    with pytest.raises(ValueError):
        benchmark.write_csv(
            tmp_path / "empty.csv",
            [],
        )
