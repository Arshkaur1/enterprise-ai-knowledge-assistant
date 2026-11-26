def build_answer_schema(
    citation_ids: list[int],
) -> dict:
    """Build a JSON schema for grounded answer generation."""

    allowed_ids = sorted(set(citation_ids))

    citation_items = {
        "type": "integer",
        "minimum": 1,
    }

    if allowed_ids:
        citation_items["enum"] = allowed_ids

    return {
        "type": "object",
        "properties": {
            "status": {
                "type": "string",
                "enum": [
                    "answered",
                    "insufficient_evidence",
                ],
            },
            "answer": {
                "type": "string",
                "minLength": 1,
            },
            "citations": {
                "type": "array",
                "items": citation_items,
                "uniqueItems": True,
                "maxItems": len(allowed_ids),
            },
        },
        "required": [
            "status",
            "answer",
            "citations",
        ],
        "additionalProperties": False,
    }
