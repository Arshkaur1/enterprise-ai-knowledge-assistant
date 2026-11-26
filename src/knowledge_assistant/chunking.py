
def chunk_text(
    text: str,
    chunk_size: int = 800,
    overlap: int = 120,
) -> list[str]:
    """
    Split text into overlapping chunks.

    Chunk size and overlap are measured in characters.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and < chunk_size")

    normalized_text = " ".join(text.split())

    if not normalized_text:
        return []

    chunks = []
    start = 0
    text_length = len(normalized_text)

    while start < text_length:

        end = min(start + chunk_size, text_length)

        if end < text_length:
            boundary = normalized_text.rfind(" ", start, end)

            if boundary > start:
                end = boundary

        chunk = normalized_text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end == text_length:
            break

        next_start = end - overlap

        # Avoid starting in the middle of a word.
        if next_start > 0 and normalized_text[next_start] != " ":
            next_space = normalized_text.find(" ", next_start, end)

            if next_space != -1:
                next_start = next_space + 1

        start = next_start

    return chunks
