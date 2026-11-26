import argparse
import json
import logging
import sys
from pathlib import Path

from knowledge_assistant.config import RAGConfig
from knowledge_assistant.models import (
    AnswerStatus,
    RAGResponse,
)
from knowledge_assistant.rag import RAGEngine

HELP_MESSAGE = """
Available commands:

  /help     Show available commands
  /config   Display current configuration
  /exit     Exit the assistant
  /quit     Exit the assistant

Enter any question to search the enterprise documents.
""".strip()


def build_parser() -> argparse.ArgumentParser:
    """Build the question-answering CLI argument parser."""

    defaults = RAGConfig()

    parser = argparse.ArgumentParser(
        description="Enterprise AI Knowledge Assistant",
    )

    parser.add_argument(
        "query",
        nargs="?",
        help="Question to answer. Omit for interactive mode.",
    )

    parser.add_argument(
        "--artifacts",
        type=Path,
        default=Path("artifacts"),
        help="Directory containing retrieval artifacts.",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=defaults.top_k,
        help="Maximum number of retrieved chunks.",
    )

    parser.add_argument(
        "--min-score",
        type=float,
        default=defaults.min_score,
        help="Minimum retrieval similarity score.",
    )

    parser.add_argument(
        "--max-context-chars",
        type=int,
        default=defaults.max_context_chars,
        help="Maximum evidence context characters.",
    )

    parser.add_argument(
        "--model",
        default=None,
        help="Ollama model name.",
    )

    parser.add_argument(
        "--embedding-model",
        default=None,
        help="Embedding model name.",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=defaults.temperature,
        help="LLM generation temperature.",
    )

    parser.add_argument(
        "--max-tokens",
        type=int,
        default=defaults.max_tokens,
        help="Maximum generated tokens.",
    )

    parser.add_argument(
        "--context-window",
        type=int,
        default=defaults.context_window,
        help="LLM context window.",
    )

    parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Return machine-readable JSON.",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed metrics and execution logs.",
    )

    return parser


def format_response(
    response: RAGResponse,
    verbose: bool = False,
) -> str:
    """Format a RAG response for terminal display."""

    lines = [
        "",
        f"Assistant: {response.answer}",
    ]

    if response.citations:
        lines.append("")
        lines.append("Sources:")

        for citation in response.citations:
            source = f"[{citation.citation_id}] {citation.source}"

            if citation.page_number is not None:
                source += f" | Page {citation.page_number}"

            source += f" | {citation.category} | Similarity: {citation.score:.4f}"

            lines.append(source)

    lines.append("")

    metrics = (
        f"Status: {response.status.value} | Latency: {response.total_latency_ms:.2f} ms"
    )

    if response.total_tokens is not None:
        metrics += f" | Tokens: {response.total_tokens}"

    lines.append(metrics)

    if verbose:
        lines.extend(
            [
                "",
                "Execution details:",
                f"  Model: {response.model}",
                f"  Prompt version: {response.prompt_version}",
                f"  Retrieved chunks: {response.retrieved_count}",
                f"  Included chunks: {response.context_chunk_count}",
                f"  Context characters: {response.context_characters}",
                f"  Context truncated: {response.context_truncated}",
                (f"  Retrieval latency: {response.retrieval_latency_ms:.2f} ms"),
                (f"  Generation latency: {response.generation_latency_ms:.2f} ms"),
                f"  Prompt tokens: {response.prompt_tokens}",
                f"  Completion tokens: {response.completion_tokens}",
            ]
        )

        if response.warnings:
            lines.append("  Warnings:")

            for warning in response.warnings:
                lines.append(f"    - {warning}")

        if response.error_message:
            lines.append(f"  Error: {response.error_message}")

    return "\n".join(lines)


def display_response(
    response: RAGResponse,
    json_output: bool = False,
    verbose: bool = False,
) -> None:
    """Print a response in the requested output format."""

    if json_output:
        print(
            json.dumps(
                response.to_dict(),
                indent=2,
                ensure_ascii=False,
            )
        )

    else:
        print(
            format_response(
                response,
                verbose=verbose,
            )
        )


def display_config(engine: RAGEngine) -> None:
    """Display active RAG configuration."""

    config = engine.config

    print(
        json.dumps(
            {
                "model": engine.llm.model,
                "top_k": config.top_k,
                "min_score": config.min_score,
                "max_context_chars": config.max_context_chars,
                "temperature": config.temperature,
                "max_tokens": config.max_tokens,
                "context_window": config.context_window,
                "prompt_version": engine.prompt_builder.version,
            },
            indent=2,
        )
    )


def interactive_session(
    engine: RAGEngine,
    verbose: bool = False,
) -> int:
    """Run an interactive question-answering session."""

    print("\nEnterprise AI Knowledge Assistant")
    print("=" * 50)
    print(f"Model: {engine.llm.model}")

    print(f"Top K: {engine.config.top_k} | Min score: {engine.config.min_score}")

    print("Type /help for commands or /exit to quit.")

    while True:
        try:
            query = input("\nYou: ").strip()

            if not query:
                continue

            command = query.lower()

            if command in {"/exit", "/quit"}:
                print("Goodbye!")
                return 0

            if command == "/help":
                print(HELP_MESSAGE)
                continue

            if command == "/config":
                display_config(engine)
                continue

            if query.startswith("/"):
                print("Unknown command. Type /help.")
                continue

            response = engine.ask(query)

            display_response(
                response,
                verbose=verbose,
            )

        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            return 0

        except Exception:
            logging.getLogger(__name__).exception("Question processing failed.")

            print(
                "Unable to process the question. Please try again.",
                file=sys.stderr,
            )


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""

    parser = build_parser()
    args = parser.parse_args(argv)

    if args.json_output and args.query is None:
        parser.error("--json requires a question.")

    if args.query is not None and not args.query.strip():
        parser.error("Question cannot be empty.")

    logging.basicConfig(
        level=(logging.INFO if args.verbose else logging.WARNING),
        format="%(levelname)s | %(name)s | %(message)s",
        stream=sys.stderr,
    )

    try:
        config = RAGConfig(
            top_k=args.top_k,
            min_score=args.min_score,
            max_context_chars=args.max_context_chars,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            context_window=args.context_window,
        )

    except ValueError as exc:
        parser.error(str(exc))

    try:
        engine = RAGEngine.from_artifacts(
            artifact_dir=args.artifacts,
            config=config,
            embedding_model=args.embedding_model,
            llm_model=args.model,
        )

    except Exception:
        logging.getLogger(__name__).exception("RAG engine initialization failed.")

        print(
            "Unable to initialize the knowledge assistant.",
            file=sys.stderr,
        )

        return 1

    if args.query is None:
        return interactive_session(
            engine,
            verbose=args.verbose,
        )

    try:
        response = engine.ask(args.query)

        display_response(
            response,
            json_output=args.json_output,
            verbose=args.verbose,
        )

    except Exception:
        logging.getLogger(__name__).exception("Question processing failed.")

        return 1

    if response.status in {
        AnswerStatus.GENERATION_ERROR,
        AnswerStatus.INVALID_RESPONSE,
    }:
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
