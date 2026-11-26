from knowledge_assistant.generation_schema import (
    build_answer_schema,
)


def test_schema_type():
    schema = build_answer_schema([1])

    assert schema["type"] == "object"


def test_required_fields():
    schema = build_answer_schema([1])

    assert set(schema["required"]) == {
        "status",
        "answer",
        "citations",
    }


def test_allowed_statuses():
    schema = build_answer_schema([1])

    assert schema["properties"]["status"]["enum"] == [
        "answered",
        "insufficient_evidence",
    ]


def test_allowed_citations():
    schema = build_answer_schema([1, 2])

    citations = schema["properties"]["citations"]

    assert citations["items"]["enum"] == [1, 2]
    assert citations["maxItems"] == 2


def test_duplicate_citation_ids():
    schema = build_answer_schema([2, 1, 2])

    citations = schema["properties"]["citations"]

    assert citations["items"]["enum"] == [1, 2]
    assert citations["maxItems"] == 2


def test_empty_citations():
    schema = build_answer_schema([])

    citations = schema["properties"]["citations"]

    assert citations["maxItems"] == 0


def test_additional_properties_disabled():
    schema = build_answer_schema([1])

    assert schema["additionalProperties"] is False


def test_unique_citations():
    schema = build_answer_schema([1, 2])

    assert schema["properties"]["citations"]["uniqueItems"]
