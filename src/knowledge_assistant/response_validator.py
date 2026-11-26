import json
import re

from knowledge_assistant.models import (
    AnswerStatus,
    ContextBundle,
    ValidatedAnswer,
)

INLINE_CITATION_PATTERN = re.compile(r"\[(\d+)\]")

REQUIRED_FIELDS = {
    "status",
    "answer",
    "citations",
}

ABSTENTION_MESSAGE = (
    "The provided document evidence is insufficient to answer this question."
)


class ResponseValidationError(ValueError):
    """Raised when an LLM response violates the output contract."""


def reject_duplicate_keys(pairs: list[tuple]) -> dict:
    """Reject JSON objects containing duplicate keys."""

    result = {}

    for key, value in pairs:
        if key in result:
            raise ResponseValidationError(f"Duplicate JSON field: {key}")

        result[key] = value

    return result


class ResponseValidator:
    """Validate and normalize grounded LLM responses."""

    def __init__(
        self,
        normalize: bool = True,
    ):
        self.normalize = normalize

    @staticmethod
    def _parse_response(raw_response: str) -> dict:
        if not isinstance(raw_response, str):
            raise ResponseValidationError("Response must be a string.")

        if not raw_response.strip():
            raise ResponseValidationError("Response cannot be empty.")

        try:
            payload = json.loads(
                raw_response,
                object_pairs_hook=reject_duplicate_keys,
            )

        except json.JSONDecodeError as exc:
            raise ResponseValidationError("Invalid JSON response.") from exc

        if not isinstance(payload, dict):
            raise ResponseValidationError("Response must be a JSON object.")

        if set(payload) != REQUIRED_FIELDS:
            raise ResponseValidationError("Unexpected or missing response fields.")

        return payload

    @staticmethod
    def _validate_fields(
        payload: dict,
    ) -> tuple[AnswerStatus, str, list[int]]:
        status = payload["status"]
        answer = payload["answer"]
        citation_ids = payload["citations"]

        if status not in (
            AnswerStatus.ANSWERED.value,
            AnswerStatus.INSUFFICIENT_EVIDENCE.value,
        ):
            raise ResponseValidationError("Invalid answer status.")

        if not isinstance(answer, str) or not answer.strip():
            raise ResponseValidationError("Answer must be a non-empty string.")

        if not isinstance(citation_ids, list):
            raise ResponseValidationError("Citations must be a list.")

        if any(
            type(citation_id) is not int or citation_id <= 0
            for citation_id in citation_ids
        ):
            raise ResponseValidationError("Citation IDs must be positive integers.")

        if len(citation_ids) != len(set(citation_ids)):
            raise ResponseValidationError("Duplicate citation IDs.")

        return (
            AnswerStatus(status),
            answer.strip(),
            citation_ids,
        )

    def validate(
        self,
        raw_response: str,
        context: ContextBundle,
    ) -> ValidatedAnswer:
        """Validate output and resolve trusted citation metadata."""

        payload = self._parse_response(raw_response)

        status, answer, citation_ids = self._validate_fields(payload)

        warnings = []

        citation_map = {
            citation.citation_id: citation for citation in context.citations
        }

        if len(citation_map) != len(context.citations):
            raise ResponseValidationError("Context contains duplicate citation IDs.")

        inline_ids = {int(match) for match in INLINE_CITATION_PATTERN.findall(answer)}

        referenced_ids = set(citation_ids) | inline_ids

        unknown_ids = referenced_ids - set(citation_map)

        if unknown_ids:
            raise ResponseValidationError(
                f"Unknown citation IDs: {sorted(unknown_ids)}"
            )

        # Handle insufficient-evidence responses.
        if status == AnswerStatus.INSUFFICIENT_EVIDENCE:
            if citation_ids or inline_ids:
                if not self.normalize:
                    raise ResponseValidationError(
                        "Abstention must not contain citations."
                    )

                warnings.append("abstention_citations_removed")

            return ValidatedAnswer(
                answer=ABSTENTION_MESSAGE,
                status=status,
                citations=[],
                warnings=warnings,
            )

        # An answered response must reference evidence.
        if not citation_ids:
            raise ResponseValidationError("Answered response requires citations.")

        structured_ids = set(citation_ids)

        if inline_ids and inline_ids != structured_ids:
            raise ResponseValidationError(
                "Inline and structured citations do not match."
            )

        # Normalize missing inline references.
        if not inline_ids:
            if not self.normalize:
                raise ResponseValidationError("Answer is missing inline citations.")

            references = " ".join(f"[{citation_id}]" for citation_id in citation_ids)

            answer = f"{answer} {references}"

            warnings.append("answer_level_citations_appended")

        citations = [citation_map[citation_id] for citation_id in citation_ids]

        return ValidatedAnswer(
            answer=answer,
            status=status,
            citations=citations,
            warnings=warnings,
        )
