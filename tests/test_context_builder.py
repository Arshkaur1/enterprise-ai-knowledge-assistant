import pytest

from knowledge_assistant.context_builder import ContextBuilder
from knowledge_assistant.models import DocumentChunk
from knowledge_assistant.retrieval import SearchResult


def make_result(
    chunk_id="hr_chunk_0000",
    text="Employees may work remotely three days per week.",
    rank=1,
    score=0.85,
    source="hr_policy.txt",
    category="HR",
    page_number=None,
):
    chunk = DocumentChunk(
        chunk_id=chunk_id,
        document_id="test_document",
        source=source,
        file_type=source.rsplit(".", 1)[-1],
        category=category,
        text=text,
        page_number=page_number,
    )

    return SearchResult(
        rank=rank,
        score=score,
        chunk=chunk,
    )


def test_single_chunk():
    builder = ContextBuilder()

    result = builder.build([make_result()])

    assert result.included_count == 1
    assert result.retrieved_count == 1
    assert "[1]" in result.text
    assert "hr_policy.txt" in result.text
    assert "Employees may work remotely" in result.text
    assert not result.truncated


def test_multiple_chunks():
    builder = ContextBuilder()

    results = [
        make_result(),
        make_result(
            chunk_id="it_chunk_0000",
            text="Passwords must contain 12 characters.",
            rank=2,
            source="it_security.docx",
            category="IT",
        ),
    ]

    bundle = builder.build(results)

    assert bundle.included_count == 2
    assert "[1]" in bundle.text
    assert "[2]" in bundle.text
    assert len(bundle.citations) == 2


def test_empty_results():
    bundle = ContextBuilder().build([])

    assert bundle.text == ""
    assert bundle.citations == []
    assert bundle.retrieved_count == 0
    assert bundle.included_count == 0
    assert bundle.character_count == 0
    assert not bundle.truncated


def test_citation_metadata():
    bundle = ContextBuilder().build([make_result()])

    citation = bundle.citations[0]

    assert citation.citation_id == 1
    assert citation.chunk_id == "hr_chunk_0000"
    assert citation.source == "hr_policy.txt"
    assert citation.category == "HR"
    assert citation.score == 0.85


def test_pdf_page_number():
    result = make_result(
        source="finance_policy.pdf",
        category="Finance",
        page_number=1,
    )

    bundle = ContextBuilder().build([result])

    assert "Page: 1" in bundle.text
    assert bundle.citations[0].page_number == 1


def test_non_pdf_has_no_page_number():
    bundle = ContextBuilder().build([make_result()])

    assert "Page:" not in bundle.text
    assert bundle.citations[0].page_number is None


def test_duplicate_chunks():
    result = make_result()

    bundle = ContextBuilder().build([result, result])

    assert bundle.retrieved_count == 2
    assert bundle.included_count == 1
    assert len(bundle.citations) == 1


def test_empty_chunk_text():
    result = make_result(text="   ")

    bundle = ContextBuilder().build([result])

    assert bundle.included_count == 0
    assert bundle.text == ""
    assert bundle.citations == []


def test_character_count():
    bundle = ContextBuilder().build([make_result()])

    assert bundle.character_count == len(bundle.text)


def test_context_budget():
    builder = ContextBuilder(max_context_chars=200)

    results = [
        make_result(text="A" * 100),
        make_result(
            chunk_id="chunk_2",
            text="B" * 100,
            rank=2,
        ),
    ]

    bundle = builder.build(results)

    assert bundle.character_count <= 200
    assert bundle.included_count == 1
    assert bundle.truncated


def test_chunk_exceeds_budget():
    builder = ContextBuilder(max_context_chars=50)

    result = make_result(text="A" * 500)

    bundle = builder.build([result])

    assert bundle.included_count == 0
    assert bundle.truncated
    assert bundle.text == ""
    assert bundle.citations == []


def test_exact_budget_boundary():
    result = make_result()

    original = ContextBuilder().build([result])

    builder = ContextBuilder(max_context_chars=original.character_count)

    bundle = builder.build([result])

    assert bundle.included_count == 1
    assert not bundle.truncated


def test_ranking_order():
    results = [
        make_result(
            chunk_id="second",
            rank=2,
        ),
        make_result(
            chunk_id="first",
            rank=1,
        ),
    ]

    bundle = ContextBuilder().build(results)

    assert bundle.citations[0].chunk_id == "first"
    assert bundle.citations[1].chunk_id == "second"


def test_contiguous_citation_ids():
    results = [
        make_result(
            chunk_id="large_chunk",
            text="A" * 1000,
            rank=1,
        ),
        make_result(
            chunk_id="small_chunk",
            text="Short evidence.",
            rank=2,
        ),
    ]

    builder = ContextBuilder(max_context_chars=200)

    bundle = builder.build(results)

    assert bundle.truncated
    assert bundle.included_count == 1
    assert bundle.citations[0].citation_id == 1
    assert bundle.citations[0].chunk_id == "small_chunk"


@pytest.mark.parametrize("budget", [0, -100])
def test_invalid_context_budget(budget):
    with pytest.raises(ValueError):
        ContextBuilder(max_context_chars=budget)
