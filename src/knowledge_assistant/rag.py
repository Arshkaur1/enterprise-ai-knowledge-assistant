import logging
from pathlib import Path
from time import perf_counter

from knowledge_assistant.config import RAGConfig
from knowledge_assistant.context_builder import ContextBuilder
from knowledge_assistant.embeddings import EmbeddingService
from knowledge_assistant.generation_schema import (
    build_answer_schema,
)
from knowledge_assistant.llm_client import (
    LLMGenerationError,
    LocalLLM,
)
from knowledge_assistant.models import (
    AnswerStatus,
    Citation,
    RAGResponse,
)
from knowledge_assistant.prompt_builder import PromptBuilder
from knowledge_assistant.response_validator import (
    ABSTENTION_MESSAGE,
    ResponseValidationError,
    ResponseValidator,
)
from knowledge_assistant.retrieval import SemanticRetriever
from knowledge_assistant.retrieval_interface import RetrieverProtocol
from knowledge_assistant.retriever_factory import RetrieverFactory

logger = logging.getLogger(__name__)


class RAGEngine:
    """Orchestrate retrieval-augmented answer generation."""

    def __init__(
        self,
        retriever: SemanticRetriever | RetrieverProtocol,
        llm: LocalLLM,
        config: RAGConfig | None = None,
    ):
        self.config = config or RAGConfig()

        self.retriever = retriever
        self.llm = llm
        self._configured_retrieval = False

        self.context_builder = ContextBuilder(
            max_context_chars=self.config.max_context_chars
        )

        self.prompt_builder = PromptBuilder(
            version=self.config.prompt_version,
        )

        self.validator = ResponseValidator(normalize=True)

    @classmethod
    def from_artifacts(
        cls,
        artifact_dir: str | Path,
        config: RAGConfig | None = None,
        embedding_model: str | None = None,
        llm_model: str | None = None,
    ) -> "RAGEngine":
        """Initialize a RAG engine from persisted artifacts."""

        embedder = (
            EmbeddingService(model_name=embedding_model)
            if embedding_model is not None
            else EmbeddingService()
        )

        retriever = SemanticRetriever(
            artifact_dir=artifact_dir,
            embedding_service=embedder,
        )

        llm = LocalLLM(model=llm_model) if llm_model is not None else LocalLLM()

        return cls(
            retriever=retriever,
            llm=llm,
            config=config,
        )

    @classmethod
    def from_configured_artifacts(
        cls,
        artifact_dir: str | Path,
        config: RAGConfig | None = None,
        embedding_model: str | None = None,
        llm_model: str | None = None,
    ) -> "RAGEngine":
        config = config or RAGConfig()

        retriever = RetrieverFactory.create(
            artifact_dir=artifact_dir,
            config=config,
            embedding_model=embedding_model,
        )

        llm = LocalLLM(model=llm_model) if llm_model is not None else LocalLLM()

        engine = cls(
            retriever=retriever,
            llm=llm,
            config=config,
        )

        engine._configured_retrieval = True

        return engine

    def ask(self, query: str) -> RAGResponse:
        """Execute the complete RAG pipeline."""

        query = query.strip()

        if not query:
            raise ValueError("Query cannot be empty.")

        total_start = perf_counter()

        # Stage 1: Semantic retrieval
        retrieval_start = perf_counter()

        if getattr(self, "_configured_retrieval", False):
            results = self.retriever.search(
                query=query,
                top_k=self.config.top_k,
            )
        else:
            results = self.retriever.search(
                query=query,
                top_k=self.config.top_k,
            )

        retrieval_latency_ms = (perf_counter() - retrieval_start) * 1000

        # Stage 2: Context construction
        context = self.context_builder.build(results)

        def make_response(
            answer: str,
            status: AnswerStatus,
            citations: list[Citation] | None = None,
            generation_latency_ms: float = 0.0,
            prompt_tokens: int | None = None,
            completion_tokens: int | None = None,
            error_message: str | None = None,
            warnings: list[str] | None = None,
        ) -> RAGResponse:
            """Construct a consistent response for every outcome."""

            response = RAGResponse(
                query=query,
                answer=answer,
                status=status,
                citations=(citations if citations is not None else []),
                model=self.llm.model,
                retrieved_count=context.retrieved_count,
                context_chunk_count=context.included_count,
                retrieval_latency_ms=retrieval_latency_ms,
                generation_latency_ms=generation_latency_ms,
                total_latency_ms=(perf_counter() - total_start) * 1000,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                error_message=error_message,
                warnings=(warnings if warnings is not None else []),
                context_characters=context.character_count,
                context_truncated=context.truncated,
                prompt_version=self.prompt_builder.version,
            )

            logger.info(
                "RAG completed: status=%s, chunks=%d, "
                "retrieval_ms=%.2f, generation_ms=%.2f, "
                "total_ms=%.2f",
                response.status.value,
                response.context_chunk_count,
                response.retrieval_latency_ms,
                response.generation_latency_ms,
                response.total_latency_ms,
            )

            return response

        # Stage 3: Early abstention
        if not context.citations:
            return make_response(
                answer=ABSTENTION_MESSAGE,
                status=AnswerStatus.INSUFFICIENT_EVIDENCE,
            )

        # Stage 4: Prompt and schema construction
        messages = self.prompt_builder.build(
            query=query,
            context=context,
        )

        schema = build_answer_schema(
            [citation.citation_id for citation in context.citations]
        )

        # Stage 5: LLM generation
        generation_start = perf_counter()

        try:
            generation = self.llm.generate_messages(
                messages=messages,
                config=self.config,
                response_schema=schema,
            )

        except LLMGenerationError:
            logger.exception("LLM generation failed.")

            return make_response(
                answer=("Unable to generate an answer at this time."),
                status=AnswerStatus.GENERATION_ERROR,
                generation_latency_ms=(perf_counter() - generation_start) * 1000,
                error_message="LLM generation failed.",
            )

        # Stage 6: Response validation
        try:
            validated = self.validator.validate(
                raw_response=generation.text,
                context=context,
            )

        except ResponseValidationError:
            logger.exception("Response validation failed.")

            return make_response(
                answer=("The generated response could not be validated."),
                status=AnswerStatus.INVALID_RESPONSE,
                generation_latency_ms=generation.latency_ms,
                prompt_tokens=generation.prompt_tokens,
                completion_tokens=generation.completion_tokens,
                error_message="Response validation failed.",
            )

        # Stage 7: Final structured response
        return make_response(
            answer=validated.answer,
            status=validated.status,
            citations=validated.citations,
            generation_latency_ms=generation.latency_ms,
            prompt_tokens=generation.prompt_tokens,
            completion_tokens=generation.completion_tokens,
            warnings=validated.warnings,
        )
