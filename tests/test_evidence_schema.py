import json

import pytest

from knowledge_assistant.evidence_schema import (
    build_evidence_schema,
)


def test_schema_is_object():
    schema = build_evidence_schema([1, 2])

    assert schema["type"] == "object"


def test_required_fields():
    schema = build_evidence_schema([1])

    assert set(schema["required"]) == {
        "status",
        "reason",
        "evidence",
        "missing_information",
    }


def test_supported_statuses():
    schema = build_evidence_schema([1])

    assert schema["properties"]["status"]["enum"] == [
        "sufficient",
        "insufficient",
    ]


def test_additional_properties_disallowed():
    schema = build_evidence_schema([1])

    assert schema["additionalProperties"] is False


def test_allowed_citation_ids():
    schema = build_evidence_schema([1, 2, 3])

    citation_schema = schema["properties"]["evidence"]["items"]["properties"][
        "citation_id"
    ]

    assert citation_schema["enum"] == [1, 2, 3]


def test_duplicate_citation_ids_removed():
    schema = build_evidence_schema([3, 1, 3, 2])

    citation_schema = schema["properties"]["evidence"]["items"]["properties"][
        "citation_id"
    ]

    assert citation_schema["enum"] == [1, 2, 3]


@pytest.mark.parametrize(
    "citation_ids",
    [
        [0],
        [-1],
        [True],
        ["1"],
        [1.5],
    ],
)
def test_invalid_citation_ids(citation_ids):
    with pytest.raises(ValueError):
        build_evidence_schema(citation_ids)


def test_empty_citations():
    schema = build_evidence_schema([])

    assert schema["properties"]["evidence"]["maxItems"] == 0


def test_empty_citations_only_allow_insufficient():
    schema = build_evidence_schema([])

    assert schema["properties"]["status"]["enum"] == ["insufficient"]


def test_evidence_required_fields():
    schema = build_evidence_schema([1])

    evidence_item = schema["properties"]["evidence"]["items"]

    assert set(evidence_item["required"]) == {
        "citation_id",
        "quote",
    }


def test_evidence_disallows_extra_fields():
    schema = build_evidence_schema([1])

    evidence_item = schema["properties"]["evidence"]["items"]

    assert evidence_item["additionalProperties"] is False


def test_quote_must_be_nonempty():
    schema = build_evidence_schema([1])

    quote_schema = schema["properties"]["evidence"]["items"]["properties"]["quote"]

    assert quote_schema["type"] == "string"
    assert quote_schema["minLength"] == 1


def test_reason_must_be_nonempty():
    schema = build_evidence_schema([1])

    reason = schema["properties"]["reason"]

    assert reason["type"] == "string"
    assert reason["minLength"] == 1


def test_missing_information_schema():
    schema = build_evidence_schema([1])

    missing = schema["properties"]["missing_information"]

    assert missing["type"] == "array"
    assert missing["items"]["type"] == "string"
    assert missing["items"]["minLength"] == 1


def test_duplicate_evidence_disallowed():
    schema = build_evidence_schema([1])

    assert schema["properties"]["evidence"]["uniqueItems"] is True


def test_multiple_quotes_per_source_allowed():
    schema = build_evidence_schema([1])

    assert "maxItems" not in (schema["properties"]["evidence"])


def test_schema_is_json_serializable():
    schema = build_evidence_schema([1, 2])

    serialized = json.dumps(schema)

    assert json.loads(serialized) == schema
