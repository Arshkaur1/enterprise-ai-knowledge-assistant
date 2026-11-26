from knowledge_assistant.cleaning import clean_text


def test_empty_text():
    assert clean_text("") == ""


def test_remove_extra_whitespace():
    text = "Hello     World"

    assert clean_text(text) == "Hello World"


def test_normalize_newlines():
    text = "Hello\r\n\r\nWorld"

    assert clean_text(text) == "Hello\n\nWorld"


def test_remove_null_bytes():
    text = "Hello\x00World"

    assert clean_text(text) == "HelloWorld"


def test_preserve_punctuation_and_numbers():
    text = "Section 4.2: Maximum expense is 500."

    assert clean_text(text) == text
