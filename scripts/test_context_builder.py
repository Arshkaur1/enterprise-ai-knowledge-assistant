import json
from dataclasses import asdict
from pathlib import Path

from knowledge_assistant.config import RAGConfig
from knowledge_assistant.context_builder import ContextBuilder
from knowledge_assistant.embeddings import EmbeddingService
from knowledge_assistant.retrieval import SemanticRetriever

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main():
    config = RAGConfig()

    embedder = EmbeddingService()

    retriever = SemanticRetriever(
        artifact_dir=PROJECT_ROOT / "artifacts",
        embedding_service=embedder,
    )

    builder = ContextBuilder(max_context_chars=config.max_context_chars)

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

    for query in queries:
        results = retriever.search(
            query=query,
            top_k=config.top_k,
            min_score=config.min_score,
        )

        bundle = builder.build(results)

        print("\n" + "=" * 80)
        print("QUERY:", query)

        print("\nCONTEXT:")
        print(bundle.text or "[No evidence retrieved]")

        print("\nCONTEXT METADATA:")
        print(
            json.dumps(
                {
                    "retrieved_count": bundle.retrieved_count,
                    "included_count": bundle.included_count,
                    "character_count": bundle.character_count,
                    "truncated": bundle.truncated,
                    "citations": [asdict(citation) for citation in bundle.citations],
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
