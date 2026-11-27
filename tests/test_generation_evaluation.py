from dataclasses import replace

import pytest

from knowledge_assistant.evaluation import EvaluationQuery
from knowledge_assistant.generation_evaluation import (
    calculate_citation_metrics,
    evaluate_response,
    summarize_generation_results,
)
from knowledge_assistant.models import (
    AnswerStatus,
    Citation,
    RAGResponse,
)


@pytest.fixture
def example():
    return EvaluationQuery(
        query_id="hr_001",
        query="How many remote working days?",
        category="HR",
        answerable=True,
        relevant_chunk_ids=["hr_chunk_0"],
        expected_answer="Three days per week.",
        split="calibration",
    )


@pytest.fixture
def response():
    return RAGResponse(
        query="How many remote working days?",
        answer="Three days per week. [1]",
        status=AnswerStatus.ANSWERED,
        citations=[
            Citation(
                citation_id=1,
                chunk_id="hr_chunk_0",
                document_id="hr_policy",
                source="hr_policy.txt",
                category="HR",
                score=0.85,
            )
        ],
        model="test-model",
        retrieved_count=1,
        context_chunk_count=1,
        retrieval_latency_ms=5.0,
        generation_latency_ms=100.0,
        total_latency_ms=106.0,
        prompt_tokens=300,
        completion_tokens=25,
        prompt_version="1.0",
    )


@pytest.mark.parametrize(
    "cited,relevant,precision,recall",
    [
        (["a"], ["a"], 1.0, 1.0),
        (["a", "b"], ["a"], 0.5, 1.0),
        (["a"], ["a", "b"], 1.0, 0.5),
        (["b"], ["a"], 0.0, 0.0),
        ([], ["a"], 0.0, 0.0),
        (["a", "a"], ["a"], 1.0, 1.0),
    ],
)
def test_citation_metrics(
    cited,
    relevant,
    precision,
    recall,
):
    metrics = calculate_citation_metrics(
        cited,
        relevant,
    )

    assert metrics["precision"] == precision
    assert metrics["recall"] == recall


def test_no_relevant_chunks():
    metrics = calculate_citation_metrics([], [])

    assert metrics["precision"] is None
    assert metrics["recall"] is None
    assert metrics["f1"] is None


def test_citation_f1():
    metrics = calculate_citation_metrics(
        ["a", "b"],
        ["a"],
    )

    assert metrics["f1"] == pytest.approx(2 / 3)


def test_correct_answer(example, response):
    result = evaluate_response(example, response)

    assert result.answerability_correct
    assert result.predicted_answerable
    assert result.citation_precision == 1.0
    assert result.citation_recall == 1.0


def test_false_acceptance(example, response):
    example = replace(
        example,
        answerable=False,
        relevant_chunk_ids=[],
        expected_answer=None,
    )

    result = evaluate_response(example, response)

    assert not result.answerability_correct
    assert result.predicted_answerable


def test_false_rejection(example, response):
    response = replace(
        response,
        status=AnswerStatus.INSUFFICIENT_EVIDENCE,
        citations=[],
    )

    result = evaluate_response(example, response)

    assert not result.answerability_correct
    assert not result.predicted_answerable


def test_correct_abstention(example, response):
    example = replace(
        example,
        answerable=False,
        relevant_chunk_ids=[],
        expected_answer=None,
    )

    response = replace(
        response,
        status=AnswerStatus.INSUFFICIENT_EVIDENCE,
        citations=[],
    )

    result = evaluate_response(example, response)

    assert result.answerability_correct


def test_operational_error_not_correct_abstention(
    example,
    response,
):
    example = replace(
        example,
        answerable=False,
        relevant_chunk_ids=[],
        expected_answer=None,
    )

    response = replace(
        response,
        status=AnswerStatus.GENERATION_ERROR,
        citations=[],
    )

    result = evaluate_response(example, response)

    assert not result.answerability_correct


def test_evaluation_metadata(example, response):
    result = evaluate_response(example, response)

    assert result.query_id == "hr_001"
    assert result.model == "test-model"
    assert result.total_tokens == 325
    assert result.prompt_version == "1.0"


def test_serialization(example, response):
    result = evaluate_response(example, response)

    payload = result.to_dict()

    assert payload["expected_answerable"] is True
    assert payload["citation_precision"] == 1.0


def test_summary_metrics(example, response):
    result = evaluate_response(example, response)

    summary = summarize_generation_results([result])

    assert summary["total_queries"] == 1
    assert summary["answerability"]["accuracy"] == 1.0
    assert summary["answerability"]["f1"] == 1.0
    assert summary["operations"]["answered"] == 1


def test_summary_latency(example, response):
    result = evaluate_response(example, response)

    summary = summarize_generation_results([result])

    assert summary["latency"]["mean_retrieval_ms"] == 5.0
    assert summary["latency"]["mean_generation_ms"] == 100.0
    assert summary["latency"]["p95_end_to_end_ms"] == 106.0


def test_summary_tokens(example, response):
    result = evaluate_response(example, response)

    summary = summarize_generation_results([result])

    assert summary["tokens"]["total_reported_tokens"] == 325


def test_summary_warnings(example, response):
    response.warnings = ["answer_level_citations_appended"]

    result = evaluate_response(example, response)

    summary = summarize_generation_results([result])

    assert summary["operations"]["responses_with_warnings"] == 1


def test_empty_results():
    with pytest.raises(ValueError):
        summarize_generation_results([])


def test_summary_confusion_matrix(example, response):
    positive = evaluate_response(example, response)

    negative = replace(
        positive,
        expected_answerable=False,
        predicted_answerable=False,
        status="insufficient_evidence",
    )

    false_positive = replace(
        positive,
        expected_answerable=False,
    )

    false_negative = replace(
        positive,
        predicted_answerable=False,
        status="insufficient_evidence",
    )

    summary = summarize_generation_results(
        [
            positive,
            negative,
            false_positive,
            false_negative,
        ]
    )

    metrics = summary["answerability"]

    assert metrics["true_positives"] == 1
    assert metrics["true_negatives"] == 1
    assert metrics["false_positives"] == 1
    assert metrics["false_negatives"] == 1
    assert metrics["accuracy"] == 0.5
    assert metrics["f1"] == 0.5


def test_summary_operational_failure(example, response):
    response = replace(
        response,
        status=AnswerStatus.GENERATION_ERROR,
        citations=[],
    )

    result = evaluate_response(example, response)

    summary = summarize_generation_results([result])

    assert summary["answerability"]["accuracy"] == 0.0
    assert summary["answerability"]["true_negatives"] == 0
    assert summary["operations"]["generation_errors"] == 1
    assert summary["answerability"]["classification_coverage"] == 0.0


def test_summary_no_generation(example, response):
    response = replace(
        response,
        status=AnswerStatus.INSUFFICIENT_EVIDENCE,
        generation_latency_ms=0.0,
        prompt_tokens=None,
        completion_tokens=None,
        citations=[],
    )

    result = evaluate_response(example, response)

    summary = summarize_generation_results([result])

    assert summary["latency"]["mean_generation_ms"] is None
    assert summary["tokens"]["mean_prompt_tokens"] is None
    assert summary["operations"]["generation_requests"] == 0
