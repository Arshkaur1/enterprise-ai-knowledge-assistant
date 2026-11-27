def build_evidence_schema(
    citation_ids: list[int],
) -> dict:
    """Build a JSON schema for evidence verification."""

    for citation_id in citation_ids:
        if type(citation_id) is not int or citation_id < 1:
            raise ValueError("Citation IDs must be positive integers.")

    allowed_ids = sorted(set(citation_ids))

    citation_schema = {
        "type": "integer",
        "minimum": 1,
    }

    if allowed_ids:
        citation_schema["enum"] = allowed_ids

    evidence_schema = {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "citation_id": citation_schema,
                "quote": {
                    "type": "string",
                    "minLength": 1,
                },
            },
            "required": [
                "citation_id",
                "quote",
            ],
            "additionalProperties": False,
        },
        "uniqueItems": True,
    }

    if not allowed_ids:
        evidence_schema["maxItems"] = 0

    return {
        "type": "object",
        "properties": {
            "status": {
                "type": "string",
                "enum": (
                    ["sufficient", "insufficient"] if allowed_ids else ["insufficient"]
                ),
            },
            "reason": {
                "type": "string",
                "minLength": 1,
            },
            "evidence": evidence_schema,
            "missing_information": {
                "type": "array",
                "items": {
                    "type": "string",
                    "minLength": 1,
                },
                "uniqueItems": True,
            },
        },
        "required": [
            "status",
            "reason",
            "evidence",
            "missing_information",
        ],
        "additionalProperties": False,
    }
