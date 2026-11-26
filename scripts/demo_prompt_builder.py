import json
from pathlib import Path

from knowledge_assistant.config import RAGConfig
from knowledge_assistant.context_builder import ContextBuilder
from knowledge_assistant.embeddings import EmbeddingService
from knowledge_assistant.prompt_builder import PromptBuilder
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

    queries = [
        "How many days per week can employees work remotely?",
        "What is the minimum password length and how should "
        "unattended devices be secured?",
        "What is the company's mileage reimbursement rate?",
        "What is the weather forecast for tomorrow?",
    ]

    print("PROMPT VERSION:", prompt_builder.version)

    for query in queries:
        results = retriever.search(
            query=query,
            top_k=config.top_k,
            min_score=config.min_score,
        )

        context = context_builder.build(results)

        messages = prompt_builder.build(
            query=query,
            context=context,
        )

        print("\n" + "=" * 80)
        print("QUERY:", query)

        print("\nSYSTEM MESSAGE:")
        print(messages[0]["content"])

        print("\nUSER MESSAGE:")
        print(messages[1]["content"])

        total_chars = sum(len(message["content"]) for message in messages)

        print("\nPROMPT STATISTICS:")

        print(
            json.dumps(
                {
                    "prompt_version": prompt_builder.version,
                    "message_count": len(messages),
                    "context_characters": context.character_count,
                    "prompt_content_characters": total_chars,
                    "available_citations": [
                        citation.citation_id for citation in context.citations
                    ],
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
