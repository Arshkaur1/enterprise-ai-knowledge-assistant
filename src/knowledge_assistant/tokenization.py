import re
import unicodedata

TOKEN_PATTERN = re.compile(
    r"[^\W_]+(?:[-_.][^\W_]+)*",
    flags=re.UNICODE,
)


def tokenize(text: str) -> list[str]:
    """Tokenize text for lexical document retrieval."""

    if not isinstance(text, str):
        raise TypeError("Text must be a string.")

    normalized = unicodedata.normalize(
        "NFKC",
        text,
    ).casefold()

    return TOKEN_PATTERN.findall(normalized)
