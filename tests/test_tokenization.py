import pytest

from knowledge_assistant.tokenization import tokenize


def test_lowercase_normalization():
    assert tokenize("REMOTE Work") == [
        "remote",
        "work",
    ]


def test_punctuation():
    assert tokenize("Hello, world!") == [
        "hello",
        "world",
    ]


def test_numeric_values():
    assert tokenize("12 characters and 500 units") == [
        "12",
        "characters",
        "and",
        "500",
        "units",
    ]


def test_hyphenated_identifier():
    assert tokenize("POLICY-001") == [
        "policy-001",
    ]


def test_underscore_identifier():
    assert tokenize("employee_id") == [
        "employee_id",
    ]


def test_dotted_identifier():
    assert tokenize("version 2.5") == [
        "version",
        "2.5",
    ]


def test_unicode_normalization():
    assert tokenize("ＦＩＮＡＮＣＥ") == [
        "finance",
    ]


def test_empty_text():
    assert tokenize("") == []


def test_whitespace_only():
    assert tokenize(" \n\t ") == []


def test_punctuation_only():
    assert tokenize("!!! ...") == []


def test_repeated_terms_preserved():
    assert tokenize("password password") == [
        "password",
        "password",
    ]


@pytest.mark.parametrize(
    "value",
    [None, 123, [], {}],
)
def test_invalid_input(value):
    with pytest.raises(TypeError):
        tokenize(value)
