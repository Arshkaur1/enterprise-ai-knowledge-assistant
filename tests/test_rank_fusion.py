from dataclasses import replace

import pytest

from knowledge_assistant.models import DocumentChunk
from knowledge_assistant.rank_fusion import (
    reciprocal_rank_fusion,
)
from knowledge_assistant.retrieval import SearchResult


def make_result(
    chunk_id: str,
    rank: int,
    score: float = 1.0,
) -> SearchResult:

    chunk = DocumentChunk(
        chunk_id=chunk_id,
        document_id=chunk_id,
        source=f"{chunk_id}.txt",
        file_type="txt",
        category="General",
        text=f"Document {chunk_id}",
        chunk_index=0,
    )

    return SearchResult(
        rank=rank,
        score=score,
        chunk=chunk,
    )


@pytest.fixture
def rankings():

    return [
        [
            make_result("a", 1),
            make_result("b", 2),
            make_result("c", 3),
        ],
        [
            make_result("b", 1),
            make_result("a", 2),
            make_result("d", 3),
        ],
    ]


def test_fusion_order(rankings):

    results = reciprocal_rank_fusion(
        rankings,
        top_k=4,
    )

    assert [result.chunk.chunk_id for result in results] == ["a", "b", "c", "d"]


def test_fusion_score(rankings):

    results = reciprocal_rank_fusion(rankings)

    assert results[0].score == pytest.approx(1 / 61 + 1 / 62)


def test_single_ranking():

    results = reciprocal_rank_fusion([[make_result("a", 1)]])

    assert results[0].score == pytest.approx(1 / 61)


def test_empty_rankings():

    assert reciprocal_rank_fusion([]) == []


def test_empty_candidate_lists():

    assert reciprocal_rank_fusion([[], []]) == []


def test_one_empty_ranking():

    results = reciprocal_rank_fusion([[], [make_result("a", 1)]])

    assert len(results) == 1
    assert results[0].chunk.chunk_id == "a"


def test_top_k_limit(rankings):

    results = reciprocal_rank_fusion(
        rankings,
        top_k=2,
    )

    assert len(results) == 2


def test_contiguous_output_ranks(rankings):

    results = reciprocal_rank_fusion(rankings)

    assert [result.rank for result in results] == [1, 2, 3, 4]


def test_original_scores_ignored():

    rankings = [
        [
            make_result("a", 1, score=0.01),
            make_result("b", 2, score=1000),
        ]
    ]

    results = reciprocal_rank_fusion(rankings)

    assert results[0].chunk.chunk_id == "a"


def test_weighted_fusion(rankings):

    results = reciprocal_rank_fusion(
        rankings,
        weights=[2.0, 1.0],
    )

    assert results[0].chunk.chunk_id == "a"

    assert results[0].score == pytest.approx(2 / 61 + 1 / 62)


def test_zero_weight(rankings):

    results = reciprocal_rank_fusion(
        rankings,
        weights=[0.0, 1.0],
    )

    assert [result.chunk.chunk_id for result in results] == ["b", "a", "d"]


def test_custom_k():

    results = reciprocal_rank_fusion(
        [[make_result("a", 1)]],
        k=10,
    )

    assert results[0].score == pytest.approx(1 / 11)


def test_duplicate_chunk_rejected():

    ranking = [
        make_result("a", 1),
        make_result("a", 2),
    ]

    with pytest.raises(ValueError):
        reciprocal_rank_fusion([ranking])


def test_duplicate_rank_rejected():

    ranking = [
        make_result("a", 1),
        make_result("b", 1),
    ]

    with pytest.raises(ValueError):
        reciprocal_rank_fusion([ranking])


def test_inconsistent_chunk_data():

    first = make_result("a", 1)
    second = make_result("a", 1)

    second = replace(
        second,
        chunk=replace(
            second.chunk,
            text="Different content",
        ),
    )

    with pytest.raises(ValueError):
        reciprocal_rank_fusion([[first], [second]])


@pytest.mark.parametrize("top_k", [0, -1, True, 1.5])
def test_invalid_top_k(rankings, top_k):

    with pytest.raises(ValueError):
        reciprocal_rank_fusion(
            rankings,
            top_k=top_k,
        )


@pytest.mark.parametrize("k", [0, -1, True, 1.5])
def test_invalid_k(rankings, k):

    with pytest.raises(ValueError):
        reciprocal_rank_fusion(
            rankings,
            k=k,
        )


@pytest.mark.parametrize(
    "weights",
    [
        [1.0],
        [1.0, -1.0],
        [0.0, 0.0],
        [1.0, float("nan")],
        [1.0, float("inf")],
    ],
)
def test_invalid_weights(rankings, weights):

    with pytest.raises(ValueError):
        reciprocal_rank_fusion(
            rankings,
            weights=weights,
        )


@pytest.mark.parametrize("rank", [0, -1, True, 1.5])
def test_invalid_input_rank(rank):

    with pytest.raises(ValueError):
        reciprocal_rank_fusion([[make_result("a", rank)]])


def test_deterministic_tie_breaking():

    results = reciprocal_rank_fusion(
        [
            [make_result("z", 1)],
            [make_result("a", 1)],
        ]
    )

    assert [result.chunk.chunk_id for result in results] == ["a", "z"]
