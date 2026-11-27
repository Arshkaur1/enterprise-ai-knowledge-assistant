from dataclasses import asdict, dataclass, field
from enum import StrEnum


@dataclass
class DocumentUnit:
    """
    Represents one extracted unit of a source document.

    For PDF files, a unit represents one page.
    For TXT and DOCX files, a unit represents the document.
    """

    document_id: str
    source: str
    file_type: str
    category: str
    text: str

    page_number: int | None = None


@dataclass
class DocumentChunk:
    """
    Represents a processed chunk ready for embedding.
    """

    chunk_id: str
    document_id: str
    source: str
    file_type: str
    category: str
    text: str

    page_number: int | None = None
    chunk_index: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


class AnswerStatus(StrEnum):
    """Possible outcomes of a RAG request."""

    ANSWERED = "answered"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    INVALID_RESPONSE = "invalid_response"
    GENERATION_ERROR = "generation_error"


@dataclass
class Citation:
    """Reference to a retrieved source document."""

    citation_id: int
    chunk_id: str
    document_id: str
    source: str
    category: str
    score: float

    page_number: int | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ContextBundle:
    text: str
    citations: list[Citation]
    retrieved_count: int
    included_count: int
    character_count: int
    truncated: bool = False

    source_texts: dict[int, str] = field(default_factory=dict)


@dataclass
class GenerationResult:
    """Language model output and inference statistics."""

    text: str
    model: str
    latency_ms: float

    prompt_tokens: int | None = None
    completion_tokens: int | None = None

    @property
    def total_tokens(self) -> int | None:
        if self.prompt_tokens is None or self.completion_tokens is None:
            return None

        return self.prompt_tokens + self.completion_tokens


@dataclass
class RAGResponse:
    """Structured response returned by the RAG pipeline."""

    query: str
    answer: str
    status: AnswerStatus

    citations: list[Citation]

    model: str

    retrieved_count: int
    context_chunk_count: int

    retrieval_latency_ms: float
    generation_latency_ms: float
    total_latency_ms: float

    prompt_tokens: int | None = None
    completion_tokens: int | None = None

    error_message: str | None = None
    warnings: list[str] = field(default_factory=list)

    context_characters: int = 0
    context_truncated: bool = False
    prompt_version: str | None = None

    @property
    def answered(self) -> bool:
        return self.status == AnswerStatus.ANSWERED

    @property
    def total_tokens(self) -> int | None:
        if self.prompt_tokens is None or self.completion_tokens is None:
            return None

        return self.prompt_tokens + self.completion_tokens

    def to_dict(self) -> dict:
        result = asdict(self)

        result["status"] = self.status.value
        result["answered"] = self.answered
        result["total_tokens"] = self.total_tokens

        return result


@dataclass
class ValidatedAnswer:
    """Validated and normalized language model response."""

    answer: str
    status: AnswerStatus
    citations: list[Citation]

    warnings: list[str] = field(default_factory=list)

    @property
    def answered(self) -> bool:
        return self.status == AnswerStatus.ANSWERED

    def to_dict(self) -> dict:
        result = asdict(self)

        result["status"] = self.status.value
        result["answered"] = self.answered

        return result


class EvidenceStatus(StrEnum):
    SUFFICIENT = "sufficient"
    INSUFFICIENT = "insufficient"
    INVALID_RESPONSE = "invalid_response"
    VERIFICATION_ERROR = "verification_error"


@dataclass
class EvidenceQuote:
    """An extracted quotation from a retrieved source."""

    citation_id: int
    quote: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class EvidenceVerificationResult:
    """Structured result of evidence sufficiency verification."""

    status: EvidenceStatus
    reason: str
    evidence: list[EvidenceQuote] = field(default_factory=list)
    missing_information: list[str] = field(default_factory=list)

    latency_ms: float = 0.0
    prompt_tokens: int | None = None
    completion_tokens: int | None = None

    model: str | None = None
    verifier_version: str | None = None
    error_message: str | None = None

    warnings: list[str] = field(default_factory=list)

    @property
    def sufficient(self) -> bool:
        return self.status == EvidenceStatus.SUFFICIENT

    @property
    def total_tokens(self) -> int | None:
        if self.prompt_tokens is None or self.completion_tokens is None:
            return None

        return self.prompt_tokens + self.completion_tokens

    def to_dict(self) -> dict:
        result = asdict(self)

        result["status"] = self.status.value
        result["sufficient"] = self.sufficient
        result["total_tokens"] = self.total_tokens

        return result


class GroundingStatus(StrEnum):
    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"
    INVALID_RESPONSE = "invalid_response"
    VERIFICATION_ERROR = "verification_error"


@dataclass
class GroundingResult:
    status: GroundingStatus
    reason: str

    evidence: list[EvidenceQuote] = field(default_factory=list)

    unsupported_claims: list[str] = field(default_factory=list)

    latency_ms: float = 0.0
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    model: str | None = None
    verifier_version: str | None = None
    error_message: str | None = None
    llm_requested: bool = False

    @property
    def supported(self) -> bool:
        return self.status == GroundingStatus.SUPPORTED

    @property
    def total_tokens(self) -> int | None:

        if self.prompt_tokens is None or self.completion_tokens is None:
            return None

        return self.prompt_tokens + self.completion_tokens

    def to_dict(self) -> dict:

        result = asdict(self)

        result["status"] = self.status.value
        result["supported"] = self.supported
        result["total_tokens"] = self.total_tokens

        return result
