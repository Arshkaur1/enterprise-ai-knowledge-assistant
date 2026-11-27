from dataclasses import asdict, dataclass
from statistics import mean

from knowledge_assistant.evaluation import EvaluationQuery
from knowledge_assistant.models import AnswerStatus, RAGResponse


@dataclass
class GenerationEvaluationResult:
    """Evaluation result for an individual RAG query."""

    query_id: str
    query: str
    category: str
    split: str

    expected_answerable: bool
    expected_answer: str | None

    predicted_answerable: bool
    generated_answer: str
    status: str

    answerability_correct: bool

    relevant_chunk_ids: list[str]
    cited_chunk_ids: list[str]

    citation_precision: float | None
    citation_recall: float | None
    citation_f1: float | None

    retrieval_latency_ms: float
    generation_latency_ms: float
    total_latency_ms: float

    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None

    retrieved_count: int
    context_chunk_count: int
    context_characters: int
    context_truncated: bool

    warnings: list[str]
    model: str
    prompt_version: str | None

    def to_dict(self) -> dict:
        return asdict(self)


def calculate_citation_metrics(
    cited_chunk_ids: list[str],
    relevant_chunk_ids: list[str],
) -> dict[str, float | None]:
    """Measure overlap between cited and relevant chunks."""

    cited = set(cited_chunk_ids)
    relevant = set(relevant_chunk_ids)

    if not relevant:
        return {
            "precision": None,
            "recall": None,
            "f1": None,
        }

    hits = len(cited & relevant)

    precision = hits / len(cited) if cited else 0.0
    recall = hits / len(relevant)

    f1 = (
        2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def evaluate_response(
    example: EvaluationQuery,
    response: RAGResponse,
) -> GenerationEvaluationResult:
    """Evaluate a generated response against labelled data."""

    predicted_answerable = response.answered

    operational_error = response.status in {
        AnswerStatus.INVALID_RESPONSE,
        AnswerStatus.GENERATION_ERROR,
    }

    answerability_correct = (
        not operational_error and predicted_answerable == example.answerable
    )

    cited_chunk_ids = [citation.chunk_id for citation in response.citations]

    citation_metrics = calculate_citation_metrics(
        cited_chunk_ids=cited_chunk_ids,
        relevant_chunk_ids=example.relevant_chunk_ids,
    )

    return GenerationEvaluationResult(
        query_id=example.query_id,
        query=example.query,
        category=example.category,
        split=example.split,
        expected_answerable=example.answerable,
        expected_answer=example.expected_answer,
        predicted_answerable=predicted_answerable,
        generated_answer=response.answer,
        status=response.status.value,
        answerability_correct=answerability_correct,
        relevant_chunk_ids=example.relevant_chunk_ids,
        cited_chunk_ids=cited_chunk_ids,
        citation_precision=citation_metrics["precision"],
        citation_recall=citation_metrics["recall"],
        citation_f1=citation_metrics["f1"],
        retrieval_latency_ms=response.retrieval_latency_ms,
        generation_latency_ms=response.generation_latency_ms,
        total_latency_ms=response.total_latency_ms,
        prompt_tokens=response.prompt_tokens,
        completion_tokens=response.completion_tokens,
        total_tokens=response.total_tokens,
        retrieved_count=response.retrieved_count,
        context_chunk_count=response.context_chunk_count,
        context_characters=response.context_characters,
        context_truncated=response.context_truncated,
        warnings=response.warnings,
        model=response.model,
        prompt_version=response.prompt_version,
    )


def safe_divide(
    numerator: float,
    denominator: float,
) -> float:
    return numerator / denominator if denominator else 0.0


def optional_mean(values: list) -> float | None:
    available = [value for value in values if value is not None]

    return mean(available) if available else None


def summarize_generation_results(
    results: list[GenerationEvaluationResult],
) -> dict:
    """Aggregate generation quality and operational metrics."""

    if not results:
        raise ValueError("Evaluation results cannot be empty.")

    total = len(results)

    error_statuses = {
        AnswerStatus.INVALID_RESPONSE.value,
        AnswerStatus.GENERATION_ERROR.value,
    }

    valid_results = [
        result for result in results if result.status not in error_statuses
    ]

    tp = sum(
        result.expected_answerable and result.predicted_answerable
        for result in valid_results
    )

    tn = sum(
        not result.expected_answerable and not result.predicted_answerable
        for result in valid_results
    )

    fp = sum(
        not result.expected_answerable and result.predicted_answerable
        for result in valid_results
    )

    fn = sum(
        result.expected_answerable and not result.predicted_answerable
        for result in valid_results
    )

    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)

    f1 = safe_divide(
        2 * precision * recall,
        precision + recall,
    )

    generation_results = [
        result for result in results if result.generation_latency_ms > 0
    ]

    answerable_results = [result for result in results if result.expected_answerable]

    latencies = sorted(result.total_latency_ms for result in results)

    import numpy as np

    return {
        "total_queries": total,
        "answerability": {
            "true_positives": tp,
            "true_negatives": tn,
            "false_positives": fp,
            "false_negatives": fn,
            "accuracy": safe_divide(tp + tn, total),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "false_acceptance_rate": safe_divide(fp, fp + tn),
            "false_rejection_rate": safe_divide(fn, fn + tp),
            "classification_coverage": safe_divide(len(valid_results), total),
        },
        "citations": {
            "macro_precision": optional_mean(
                [result.citation_precision for result in answerable_results]
            ),
            "macro_recall": optional_mean(
                [result.citation_recall for result in answerable_results]
            ),
            "macro_f1": optional_mean(
                [result.citation_f1 for result in answerable_results]
            ),
        },
        "operations": {
            "answered": sum(result.predicted_answerable for result in results),
            "abstained": sum(
                result.status == AnswerStatus.INSUFFICIENT_EVIDENCE.value
                for result in results
            ),
            "invalid_responses": sum(
                result.status == AnswerStatus.INVALID_RESPONSE.value
                for result in results
            ),
            "generation_errors": sum(
                result.status == AnswerStatus.GENERATION_ERROR.value
                for result in results
            ),
            "operational_success_rate": safe_divide(len(valid_results), total),
            "generation_requests": len(generation_results),
            "responses_with_warnings": sum(bool(result.warnings) for result in results),
            "context_truncations": sum(result.context_truncated for result in results),
        },
        "latency": {
            "mean_retrieval_ms": mean(
                result.retrieval_latency_ms for result in results
            ),
            "mean_generation_ms": optional_mean(
                [result.generation_latency_ms for result in generation_results]
            ),
            "mean_end_to_end_ms": mean(latencies),
            "p50_end_to_end_ms": float(np.percentile(latencies, 50)),
            "p95_end_to_end_ms": float(np.percentile(latencies, 95)),
            "max_end_to_end_ms": max(latencies),
        },
        "tokens": {
            "total_reported_tokens": sum(
                result.total_tokens or 0 for result in results
            ),
            "mean_prompt_tokens": optional_mean(
                [result.prompt_tokens for result in results]
            ),
            "mean_completion_tokens": optional_mean(
                [result.completion_tokens for result in results]
            ),
            "responses_with_token_metrics": sum(
                result.total_tokens is not None for result in results
            ),
        },
    }
