from knowledge_assistant.models import (
    ContextBundle,
    EvidenceQuote,
)


class QuotationValidationError(ValueError):
    """Raised when an evidence quotation cannot be verified."""


def normalize_whitespace(text: str) -> str:
    """Normalize whitespace without changing factual content."""

    return " ".join(text.split())


class QuotationValidator:
    """Validate quotation authenticity and source attribution."""

    def validate_quote(
        self,
        evidence: EvidenceQuote,
        context: ContextBundle,
    ) -> None:

        citation_id = evidence.citation_id

        if type(citation_id) is not int or citation_id < 1:
            raise QuotationValidationError("Citation ID must be a positive integer.")

        available_ids = {citation.citation_id for citation in context.citations}

        if citation_id not in available_ids:
            raise QuotationValidationError(f"Unknown citation ID: {citation_id}")

        if citation_id not in context.source_texts:
            raise QuotationValidationError(
                f"Source text unavailable for citation {citation_id}."
            )

        if not isinstance(evidence.quote, str):
            raise QuotationValidationError("Quotation must be a string.")

        quote = normalize_whitespace(evidence.quote)

        if not quote:
            raise QuotationValidationError("Quotation cannot be empty.")

        source = normalize_whitespace(context.source_texts[citation_id])

        if quote not in source:
            raise QuotationValidationError(
                f"Quotation not found in cited source {citation_id}."
            )

    def validate(
        self,
        evidence: list[EvidenceQuote],
        context: ContextBundle,
    ) -> None:

        for item in evidence:
            self.validate_quote(item, context)
