import json
from unittest.mock import Mock

import pytest

from knowledge_assistant.config import RAGConfig
from knowledge_assistant.models import (
    AnswerStatus,
    Citation,
    RAGResponse,
)
from knowledge_assistant.qa_cli import (
    build_parser,
    display_config,
    display_response,
    format_response,
    interactive_session,
    main,
)
from knowledge_assistant.rag import RAGEngine


@pytest.fixture
def response():
    return RAGResponse(
        query="Remote work policy?",
        answer="Up to three days per week. [1]",
        status=AnswerStatus.ANSWERED,
        citations=[
            Citation(
                citation_id=1,
                chunk_id="hr_chunk_0000",
                document_id="hr_policy",
                source="hr_policy.txt",
                category="HR",
                score=0.85,
            )
        ],
        model="test-model",
        retrieved_count=1,
        context_chunk_count=1,
        retrieval_latency_ms=5.0,
        generation_latency_ms=100.0,
        total_latency_ms=106.0,
        prompt_tokens=300,
        completion_tokens=25,
        prompt_version="1.0",
        context_characters=100,
    )


@pytest.fixture
def engine(response):
    engine = Mock()
    engine.llm.model = "test-model"
    engine.config = RAGConfig()
    engine.prompt_builder.version = "1.0"
    engine.ask.return_value = response

    return engine


@pytest.fixture
def factory(monkeypatch, engine):
    factory = Mock(return_value=engine)

    monkeypatch.setattr(
        RAGEngine,
        "from_artifacts",
        factory,
    )

    return factory


def test_default_arguments():
    args = build_parser().parse_args([])

    assert args.query is None
    assert args.top_k == RAGConfig().top_k
    assert args.min_score == RAGConfig().min_score


def test_custom_arguments():
    args = build_parser().parse_args(
        [
            "Remote work?",
            "--top-k",
            "3",
            "--min-score",
            "0.4",
            "--model",
            "test-model",
        ]
    )

    assert args.query == "Remote work?"
    assert args.top_k == 3
    assert args.min_score == 0.4
    assert args.model == "test-model"


def test_format_answer(response):
    output = format_response(response)

    assert "Assistant:" in output
    assert "three days" in output


def test_format_sources(response):
    output = format_response(response)

    assert "Sources:" in output
    assert "[1] hr_policy.txt" in output
    assert "0.8500" in output


def test_pdf_page_display(response):
    response.citations[0].source = "finance.pdf"
    response.citations[0].page_number = 1

    output = format_response(response)

    assert "Page 1" in output


def test_format_metrics(response):
    output = format_response(response)

    assert "106.00 ms" in output
    assert "Tokens: 325" in output


def test_verbose_output(response):
    response.warnings = ["answer_level_citations_appended"]

    output = format_response(
        response,
        verbose=True,
    )

    assert "Execution details:" in output
    assert "Prompt tokens: 300" in output
    assert "answer_level_citations_appended" in output


def test_non_verbose_output(response):
    output = format_response(response)

    assert "Execution details:" not in output


def test_json_output(response, capsys):
    display_response(
        response,
        json_output=True,
    )

    output = json.loads(capsys.readouterr().out)

    assert output["status"] == "answered"
    assert output["total_tokens"] == 325


def test_abstention_output(response):
    response.status = AnswerStatus.INSUFFICIENT_EVIDENCE
    response.answer = "Insufficient evidence."
    response.citations = []

    output = format_response(response)

    assert "insufficient_evidence" in output
    assert "Sources:" not in output


def test_missing_tokens(response):
    response.prompt_tokens = None
    response.completion_tokens = None

    output = format_response(response)

    assert "Tokens:" not in output


def test_display_config(engine, capsys):
    display_config(engine)

    payload = json.loads(capsys.readouterr().out)

    assert payload["model"] == "test-model"
    assert payload["prompt_version"] == "1.0"


def test_interactive_question(
    engine,
    monkeypatch,
    capsys,
):
    inputs = iter(
        [
            "Remote work?",
            "/exit",
        ]
    )

    monkeypatch.setattr(
        "builtins.input",
        lambda _: next(inputs),
    )

    result = interactive_session(engine)

    engine.ask.assert_called_once_with("Remote work?")

    assert result == 0
    assert "Goodbye!" in capsys.readouterr().out


def test_interactive_commands(
    engine,
    monkeypatch,
    capsys,
):
    inputs = iter(
        [
            "",
            "/help",
            "/config",
            "/unknown",
            "/quit",
        ]
    )

    monkeypatch.setattr(
        "builtins.input",
        lambda _: next(inputs),
    )

    assert interactive_session(engine) == 0

    engine.ask.assert_not_called()

    output = capsys.readouterr().out

    assert "Available commands" in output
    assert "Unknown command" in output


@pytest.mark.parametrize(
    "exception",
    [KeyboardInterrupt, EOFError],
)
def test_interactive_graceful_exit(
    engine,
    monkeypatch,
    exception,
):
    monkeypatch.setattr(
        "builtins.input",
        Mock(side_effect=exception),
    )

    assert interactive_session(engine) == 0


def test_single_question(
    factory,
    engine,
    capsys,
):
    result = main(["Remote work?"])

    assert result == 0

    engine.ask.assert_called_once_with("Remote work?")

    assert "three days" in capsys.readouterr().out


def test_single_question_json(factory, capsys):
    result = main(
        [
            "Remote work?",
            "--json",
        ]
    )

    assert result == 0

    output = json.loads(capsys.readouterr().out)

    assert output["answered"] is True


def test_configuration_forwarding(factory):
    main(
        [
            "Remote work?",
            "--top-k",
            "3",
            "--min-score",
            "0.4",
            "--model",
            "test-model",
        ]
    )

    kwargs = factory.call_args.kwargs

    assert kwargs["config"].top_k == 3
    assert kwargs["config"].min_score == 0.4
    assert kwargs["llm_model"] == "test-model"


@pytest.mark.parametrize(
    "arguments",
    [
        ["--json"],
        ["   "],
        ["Question", "--top-k", "0"],
        ["Question", "--min-score", "2.0"],
    ],
)
def test_invalid_arguments(arguments):
    with pytest.raises(SystemExit) as exc:
        main(arguments)

    assert exc.value.code == 2


def test_initialization_failure(factory):
    factory.side_effect = RuntimeError("Index unavailable")

    assert main(["Question"]) == 1


@pytest.mark.parametrize(
    "status",
    [
        AnswerStatus.GENERATION_ERROR,
        AnswerStatus.INVALID_RESPONSE,
    ],
)
def test_error_exit_code(
    factory,
    engine,
    status,
):
    engine.ask.return_value.status = status

    assert main(["Question"]) == 1


def test_abstention_exit_code(factory, engine):
    engine.ask.return_value.status = AnswerStatus.INSUFFICIENT_EVIDENCE

    assert main(["Question"]) == 0


def test_processing_failure(factory, engine):
    engine.ask.side_effect = RuntimeError("Retrieval failed")

    assert main(["Question"]) == 1
