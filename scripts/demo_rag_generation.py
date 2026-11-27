import json
from pathlib import Path

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
from knowledge_assistant.prompt_builder import PromptBuilder
from knowledge_assistant.response_validator import (
    ResponseValidationError,
    ResponseValidator,
)
from knowledge_assistant.retrieval import SemanticRetriever

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main():
    config = RAGConfig()

    embedder = EmbeddingService()

    retriever = SemanticRetriever(
        artifact_dir=PROJECT_ROOT / "artifacts",
        embedding_service=embedder,
    )

    context_builder = ContextBuilder(max_context_chars=config.max_context_chars)

    prompt_builder = PromptBuilder()

    llm = LocalLLM()
    validator = ResponseValidator(normalize=True)

    queries = [
        "How many days per week can employees work remotely?",
        "What is the deadline for submitting expense claims?",
        (
            "What is the minimum password length and how should "
            "unattended devices be secured?"
        ),
        "What is the company's mileage reimbursement rate?",
        "What is the weather forecast for tomorrow?",
    ]

    print("MODEL:", llm.model)
    print("PROMPT VERSION:", prompt_builder.version)

    for query in queries:
        print("\n" + "=" * 80)
        print("QUERY:", query)

        results = retriever.search(
            query=query,
            top_k=config.top_k,
            min_score=config.min_score,
        )

        context = context_builder.build(results)

        print("RETRIEVED CHUNKS:", context.retrieved_count)
        print("INCLUDED CHUNKS:", context.included_count)

        if not context.citations:
            print(
                json.dumps(
                    {
                        "status": "insufficient_evidence",
                        "answer": ("No relevant evidence was retrieved."),
                        "citations": [],
                    },
                    indent=2,
                )
            )

            print("LLM generation skipped.")
            continue

        messages = prompt_builder.build(
            query=query,
            context=context,
        )

        schema = build_answer_schema(
            [citation.citation_id for citation in context.citations]
        )

        try:
            result = llm.generate_messages(
                messages=messages,
                config=config,
                response_schema=schema,
            )

        except LLMGenerationError as exc:
            print("GENERATION ERROR:", exc)
            continue

        print("\nRAW MODEL RESPONSE:")
        print(result.text)

        try:
            validated = validator.validate(
                raw_response=result.text,
                context=context,
            )

            print("\nVALIDATED MODEL RESPONSE:")

            print(
                json.dumps(
                    validated.to_dict(),
                    indent=2,
                )
            )

        except ResponseValidationError as exc:
            print("\nRESPONSE VALIDATION FAILED:")
            print(str(exc))

        print("\nGENERATION METRICS:")

        print(
            json.dumps(
                {
                    "model": result.model,
                    "latency_ms": round(
                        result.latency_ms,
                        2,
                    ),
                    "prompt_tokens": result.prompt_tokens,
                    "completion_tokens": (result.completion_tokens),
                    "total_tokens": result.total_tokens,
                },
                indent=2,
            )
        )

        print("\nAVAILABLE SOURCE METADATA:")

        print(
            json.dumps(
                [citation.to_dict() for citation in context.citations],
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
