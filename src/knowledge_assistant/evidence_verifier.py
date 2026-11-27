import json
from time import perf_counter

from knowledge_assistant.config import RAGConfig
from knowledge_assistant.evidence_prompt import (
    EvidencePromptBuilder,
)
from knowledge_assistant.evidence_schema import (
    build_evidence_schema,
)
from knowledge_assistant.llm_client import (
    LLMGenerationError,
    LocalLLM,
)
from knowledge_assistant.models import (
    ContextBundle,
    EvidenceQuote,
    EvidenceStatus,
    EvidenceVerificationResult,
)
from knowledge_assistant.quotation_validator import (
    QuotationValidationError,
    QuotationValidator,
)
from knowledge_assistant.response_validator import (
    reject_duplicate_keys,
)


class EvidenceValidationError(ValueError):
    """Raised when verifier output violates its contract."""


class EvidenceVerifier:
    """Assess whether retrieved evidence supports a question."""

    REQUIRED_FIELDS = frozenset(
        {
            "status",
            "reason",
            "evidence",
            "missing_information",
        }
    )

    def __init__(
        self,
        llm: LocalLLM,
        config: RAGConfig | None = None,
    ):
        self.llm = llm
        self.config = config or RAGConfig()
        self.prompt_builder = EvidencePromptBuilder()
        self.quotation_validator = QuotationValidator()

    def _parse_response(
        self,
        raw_response: str,
        context: ContextBundle,
    ) -> EvidenceVerificationResult:

        try:
            payload = json.loads(
                raw_response,
                object_pairs_hook=reject_duplicate_keys,
            )

        except (ValueError, TypeError) as exc:
            raise EvidenceValidationError("Verifier returned invalid JSON.") from exc

        if not isinstance(payload, dict):
            raise EvidenceValidationError("Verifier response must be a JSON object.")

        if set(payload) != self.REQUIRED_FIELDS:
            raise EvidenceValidationError("Verifier response contains invalid fields.")

        status = payload["status"]

        if status not in ("sufficient", "insufficient"):
            raise EvidenceValidationError("Invalid evidence status.")

        reason = payload["reason"]

        if not isinstance(reason, str) or not reason.strip():
            raise EvidenceValidationError(
                "Verification reason must be a non-empty string."
            )

        raw_evidence = payload["evidence"]

        if not isinstance(raw_evidence, list):
            raise EvidenceValidationError("Evidence must be a list.")

        available_ids = {citation.citation_id for citation in context.citations}

        evidence = []
        seen_quotes = set()

        for item in raw_evidence:
            if not isinstance(item, dict):
                raise EvidenceValidationError("Evidence items must be objects.")

            if set(item) != {"citation_id", "quote"}:
                raise EvidenceValidationError("Invalid evidence item fields.")

            citation_id = item["citation_id"]
            quote = item["quote"]

            if type(citation_id) is not int:
                raise EvidenceValidationError("Citation ID must be an integer.")

            if citation_id not in available_ids:
                raise EvidenceValidationError(f"Unknown citation ID: {citation_id}")

            if not isinstance(quote, str) or not quote.strip():
                raise EvidenceValidationError("Evidence quotation cannot be empty.")

            quote = quote.strip()
            quote_key = (citation_id, quote)

            if quote_key in seen_quotes:
                raise EvidenceValidationError("Duplicate evidence quotation.")

            seen_quotes.add(quote_key)

            evidence.append(
                EvidenceQuote(
                    citation_id=citation_id,
                    quote=quote,
                )
            )

        missing = payload["missing_information"]

        if not isinstance(missing, list):
            raise EvidenceValidationError("Missing information must be a list.")

        if any(not isinstance(item, str) or not item.strip() for item in missing):
            raise EvidenceValidationError(
                "Missing information must contain non-empty strings."
            )

        missing = [item.strip() for item in missing]

        if len(set(missing)) != len(missing):
            raise EvidenceValidationError("Duplicate missing information.")

        if status == "sufficient":
            if not evidence:
                raise EvidenceValidationError(
                    "Sufficient evidence requires quotations."
                )

            if missing:
                raise EvidenceValidationError(
                    "Sufficient evidence cannot contain missing facts."
                )

        elif not missing:
            raise EvidenceValidationError(
                "Insufficient evidence must identify missing facts."
            )

        try:
            self.quotation_validator.validate(
                evidence=evidence,
                context=context,
            )
        except QuotationValidationError as exc:
            raise EvidenceValidationError(str(exc)) from exc

        return EvidenceVerificationResult(
            status=EvidenceStatus(status),
            reason=reason.strip(),
            evidence=evidence,
            missing_information=missing,
        )

    def verify(
        self,
        query: str,
        context: ContextBundle,
    ) -> EvidenceVerificationResult:

        query = query.strip()

        if not query:
            raise ValueError("Query cannot be empty.")

        started = perf_counter()

        if not context.citations or not context.text.strip():
            return EvidenceVerificationResult(
                status=EvidenceStatus.INSUFFICIENT,
                reason="No document evidence is available.",
                missing_information=[query],
                latency_ms=(perf_counter() - started) * 1000,
                model=self.llm.model,
                verifier_version=self.prompt_builder.version,
            )

        messages = self.prompt_builder.build(
            query,
            context,
        )

        schema = build_evidence_schema(
            [citation.citation_id for citation in context.citations]
        )

        try:
            generation = self.llm.generate_messages(
                messages=messages,
                config=self.config,
                response_schema=schema,
            )

        except LLMGenerationError as exc:
            return EvidenceVerificationResult(
                status=EvidenceStatus.VERIFICATION_ERROR,
                reason="Evidence verification failed.",
                latency_ms=(perf_counter() - started) * 1000,
                model=self.llm.model,
                verifier_version=self.prompt_builder.version,
                error_message=str(exc),
            )

        try:
            result = self._parse_response(
                generation.text,
                context,
            )

        except EvidenceValidationError as exc:
            result = EvidenceVerificationResult(
                status=EvidenceStatus.INVALID_RESPONSE,
                reason="Verifier returned an invalid response.",
                error_message=str(exc),
            )

        result.latency_ms = (perf_counter() - started) * 1000

        result.prompt_tokens = generation.prompt_tokens
        result.completion_tokens = generation.completion_tokens
        result.model = generation.model
        result.verifier_version = self.prompt_builder.version

        return result
