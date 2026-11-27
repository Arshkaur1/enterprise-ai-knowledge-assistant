import pytest

from knowledge_assistant.models import (
    Citation,
    ContextBundle,
    EvidenceQuote,
)
from knowledge_assistant.quotation_validator import (
    QuotationValidationError,
    QuotationValidator,
    normalize_whitespace,
)


@pytest.fixture
def validator():
    return QuotationValidator()


@pytest.fixture
def context():

    sources = {
        1: ("Employees may work remotely up to three days per week."),
        2: ("Passwords must contain at least 12 characters."),
    }

    text = "\n\n".join(sources.values())

    return ContextBundle(
        text=text,
        citations=[
            Citation(
                citation_id=1,
                chunk_id="hr_chunk_0",
                document_id="hr",
                source="hr.txt",
                category="HR",
                score=0.85,
            ),
            Citation(
                citation_id=2,
                chunk_id="it_chunk_0",
                document_id="it",
                source="it.txt",
                category="IT",
                score=0.80,
            ),
        ],
        retrieved_count=2,
        included_count=2,
        character_count=len(text),
        source_texts=sources,
    )


def test_whitespace_normalization():
    assert normalize_whitespace("  Employees \n may\t work  ") == "Employees may work"


def test_exact_quotation(validator, context):

    evidence = EvidenceQuote(
        citation_id=1,
        quote=("Employees may work remotely up to three days per week."),
    )

    validator.validate_quote(evidence, context)


def test_partial_exact_quotation(validator, context):

    evidence = EvidenceQuote(
        citation_id=1,
        quote="three days per week",
    )

    validator.validate_quote(evidence, context)


def test_whitespace_differences(validator, context):

    evidence = EvidenceQuote(
        citation_id=1,
        quote="Employees may work\nremotely up to   three days",
    )

    validator.validate_quote(evidence, context)


def test_fabricated_quotation(validator, context):

    evidence = EvidenceQuote(
        citation_id=1,
        quote="Employees may work remotely five days per week.",
    )

    with pytest.raises(
        QuotationValidationError,
        match="Quotation not found",
    ):
        validator.validate_quote(evidence, context)


def test_wrong_source_attribution(validator, context):

    evidence = EvidenceQuote(
        citation_id=2,
        quote="three days per week",
    )

    with pytest.raises(
        QuotationValidationError,
        match="Quotation not found",
    ):
        validator.validate_quote(evidence, context)


def test_unknown_citation_id(validator, context):

    evidence = EvidenceQuote(
        citation_id=3,
        quote="three days per week",
    )

    with pytest.raises(
        QuotationValidationError,
        match="Unknown citation ID",
    ):
        validator.validate_quote(evidence, context)


@pytest.mark.parametrize(
    "citation_id",
    [0, -1, True, "1"],
)
def test_invalid_citation_id(
    validator,
    context,
    citation_id,
):

    evidence = EvidenceQuote(
        citation_id=citation_id,
        quote="three days per week",
    )

    with pytest.raises(QuotationValidationError):
        validator.validate_quote(evidence, context)


@pytest.mark.parametrize(
    "quote",
    ["", " ", "\n\t"],
)
def test_empty_quotation(
    validator,
    context,
    quote,
):

    evidence = EvidenceQuote(
        citation_id=1,
        quote=quote,
    )

    with pytest.raises(
        QuotationValidationError,
        match="Quotation cannot be empty",
    ):
        validator.validate_quote(evidence, context)


def test_missing_source_text(validator, context):

    del context.source_texts[1]

    evidence = EvidenceQuote(
        citation_id=1,
        quote="three days per week",
    )

    with pytest.raises(
        QuotationValidationError,
        match="Source text unavailable",
    ):
        validator.validate_quote(evidence, context)


def test_paraphrased_quotation(validator, context):

    evidence = EvidenceQuote(
        citation_id=1,
        quote="Remote work is permitted three days weekly.",
    )

    with pytest.raises(QuotationValidationError):
        validator.validate_quote(evidence, context)


def test_changed_numerical_value(validator, context):

    evidence = EvidenceQuote(
        citation_id=2,
        quote="Passwords must contain at least 16 characters.",
    )

    with pytest.raises(QuotationValidationError):
        validator.validate_quote(evidence, context)


def test_multiple_valid_quotations(validator, context):

    evidence = [
        EvidenceQuote(
            citation_id=1,
            quote="three days per week",
        ),
        EvidenceQuote(
            citation_id=2,
            quote="at least 12 characters",
        ),
    ]

    validator.validate(evidence, context)


def test_multiple_quotes_with_invalid_item(
    validator,
    context,
):

    evidence = [
        EvidenceQuote(
            citation_id=1,
            quote="three days per week",
        ),
        EvidenceQuote(
            citation_id=2,
            quote="Passwords expire every 90 days.",
        ),
    ]

    with pytest.raises(QuotationValidationError):
        validator.validate(evidence, context)


def test_empty_evidence_list(validator, context):
    validator.validate([], context)


def test_non_string_quotation(validator, context):

    evidence = EvidenceQuote(
        citation_id=1,
        quote=None,  # type: ignore
    )

    with pytest.raises(
        QuotationValidationError,
        match="Quotation must be a string",
    ):
        validator.validate_quote(evidence, context)
