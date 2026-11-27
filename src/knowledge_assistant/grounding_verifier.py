import json
from dataclasses import replace
from time import perf_counter

from knowledge_assistant.config import RAGConfig
from knowledge_assistant.grounding_prompt import (
    GroundingPromptBuilder,
)
from knowledge_assistant.grounding_schema import (
    build_grounding_schema,
)
from knowledge_assistant.llm_client import (
    LLMGenerationError,
    LocalLLM,
)
from knowledge_assistant.models import (
    ContextBundle,
    EvidenceQuote,
    GroundingResult,
    GroundingStatus,
    ValidatedAnswer,
)
from knowledge_assistant.quotation_validator import (
    QuotationValidationError,
    QuotationValidator,
)
from knowledge_assistant.response_validator import (
    reject_duplicate_keys,
)


class GroundingValidationError(ValueError):
    """Invalid grounding verifier response."""


class GroundingVerifier:
    REQUIRED_FIELDS = frozenset(
        {
            "status",
            "reason",
            "evidence",
            "unsupported_claims",
        }
    )

    def __init__(
        self,
        llm: LocalLLM,
        config: RAGConfig | None = None,
    ):

        self.llm = llm

        base_config = config or RAGConfig()

        self.config = replace(
            base_config,
            temperature=0.0,
            max_tokens=min(
                512,
                base_config.context_window - 1,
            ),
        )

        self.prompt_builder = GroundingPromptBuilder()
        self.quotation_validator = QuotationValidator()

    def _parse_response(
        self,
        raw_response: str,
        context: ContextBundle,
        allowed_ids: set[int],
    ) -> GroundingResult:

        try:
            payload = json.loads(
                raw_response,
                object_pairs_hook=reject_duplicate_keys,
            )

        except (ValueError, TypeError) as exc:
            raise GroundingValidationError("Invalid grounding JSON.") from exc

        if not isinstance(payload, dict):
            raise GroundingValidationError("Grounding response must be an object.")

        if set(payload) != self.REQUIRED_FIELDS:
            raise GroundingValidationError("Unexpected grounding response fields.")

        status = payload["status"]
        reason = payload["reason"]

        if status not in ("supported", "unsupported"):
            raise GroundingValidationError("Invalid grounding status.")

        if not isinstance(reason, str) or not reason.strip():
            raise GroundingValidationError("Grounding reason cannot be empty.")

        raw_evidence = payload["evidence"]

        if not isinstance(raw_evidence, list):
            raise GroundingValidationError("Evidence must be a list.")

        evidence = []
        seen_quotes = set()

        for item in raw_evidence:
            if not isinstance(item, dict) or set(item) != {"citation_id", "quote"}:
                raise GroundingValidationError("Invalid evidence object.")

            citation_id = item["citation_id"]
            quote = item["quote"]

            if type(citation_id) is not int or citation_id not in allowed_ids:
                raise GroundingValidationError("Invalid grounding citation ID.")

            if not isinstance(quote, str) or not quote.strip():
                raise GroundingValidationError("Evidence quote cannot be empty.")

            key = (citation_id, " ".join(quote.split()))

            if key in seen_quotes:
                raise GroundingValidationError("Duplicate evidence quotation.")

            seen_quotes.add(key)

            evidence.append(
                EvidenceQuote(
                    citation_id=citation_id,
                    quote=quote.strip(),
                )
            )

        claims = payload["unsupported_claims"]

        if not isinstance(claims, list):
            raise GroundingValidationError("Unsupported claims must be a list.")

        if any(not isinstance(claim, str) or not claim.strip() for claim in claims):
            raise GroundingValidationError("Invalid unsupported claim.")

        claims = [claim.strip() for claim in claims]

        if len(set(claims)) != len(claims):
            raise GroundingValidationError("Duplicate unsupported claims.")

        if status == "supported":
            if not evidence or claims:
                raise GroundingValidationError(
                    "Supported answers require evidence and no unsupported claims."
                )

        elif not claims:
            raise GroundingValidationError(
                "Unsupported answers require unsupported claims."
            )

        try:
            self.quotation_validator.validate(
                evidence=evidence,
                context=context,
            )

        except QuotationValidationError as exc:
            raise GroundingValidationError(str(exc)) from exc

        return GroundingResult(
            status=GroundingStatus(status),
            reason=reason.strip(),
            evidence=evidence,
            unsupported_claims=claims,
        )

    def verify(
        self,
        query: str,
        answer: ValidatedAnswer,
        context: ContextBundle,
    ) -> GroundingResult:

        if not query.strip():
            raise ValueError("Query cannot be empty.")

        if not answer.answered:
            raise ValueError("Only answered responses can be verified.")

        started = perf_counter()

        allowed_ids = {citation.citation_id for citation in answer.citations}

        context_ids = {citation.citation_id for citation in context.citations}

        def finalize(
            result: GroundingResult,
        ) -> GroundingResult:

            result.latency_ms = (perf_counter() - started) * 1000

            result.model = result.model or self.llm.model

            result.verifier_version = self.prompt_builder.version

            return result

        if (
            not allowed_ids
            or not allowed_ids.issubset(context_ids)
            or any(not context.source_texts.get(cid, "").strip() for cid in allowed_ids)
        ):
            return finalize(
                GroundingResult(
                    status=GroundingStatus.INVALID_RESPONSE,
                    reason="Cited source evidence is unavailable.",
                    error_message="Missing or invalid cited source.",
                )
            )

        messages = self.prompt_builder.build(
            query=query,
            answer=answer,
            context=context,
        )

        schema = build_grounding_schema(sorted(allowed_ids))

        try:
            generation = self.llm.generate_messages(
                messages=messages,
                config=self.config,
                response_schema=schema,
            )

        except LLMGenerationError as exc:
            return finalize(
                GroundingResult(
                    status=GroundingStatus.VERIFICATION_ERROR,
                    reason="Grounding verification failed.",
                    error_message=str(exc),
                    llm_requested=True,
                )
            )

        try:
            result = self._parse_response(
                raw_response=generation.text,
                context=context,
                allowed_ids=allowed_ids,
            )

        except GroundingValidationError as exc:
            result = GroundingResult(
                status=GroundingStatus.INVALID_RESPONSE,
                reason="Invalid grounding verifier response.",
                error_message=str(exc),
            )

        result.prompt_tokens = generation.prompt_tokens
        result.completion_tokens = generation.completion_tokens
        result.model = generation.model
        result.llm_requested = True

        return finalize(result)
