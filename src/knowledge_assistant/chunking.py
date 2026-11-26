
from knowledge_assistant.models import (
    DocumentChunk,
    DocumentUnit,
)


def chunk_text(
    text: str,
    chunk_size: int = 800,
    overlap: int = 120,
) -> list[str]:
    """
    Split normalized text into overlapping character chunks.

    Chunk boundaries and overlap are deterministic.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            "overlap must be >= 0 and < chunk_size"
        )

    text = " ".join(text.split())

    if not text:
        return []

    chunks = []
    start = 0

    while start < len(text):

        end = min(start + chunk_size, len(text))

        chunk = text[start:end]

        if chunk.strip():
            chunks.append(chunk)

        if end == len(text):
            break

        start = end - overlap

    return chunks


def create_document_chunks(
    unit: DocumentUnit,
    chunk_size: int = 800,
    overlap: int = 120,
) -> list[DocumentChunk]:
    """
    Convert one extracted document unit into structured chunks.
    """

    text_chunks = chunk_text(
        unit.text,
        chunk_size=chunk_size,
        overlap=overlap,
    )

    document_chunks = []

    for index, text in enumerate(text_chunks):

        page_suffix = (
            f"_p{unit.page_number}"
            if unit.page_number is not None
            else ""
        )

        chunk_id = (
            f"{unit.document_id}"
            f"{page_suffix}"
            f"_chunk_{index:04d}"
        )

        document_chunks.append(
            DocumentChunk(
                chunk_id=chunk_id,
                document_id=unit.document_id,
                source=unit.source,
                file_type=unit.file_type,
                category=unit.category,
                text=text,
                page_number=unit.page_number,
                chunk_index=index,
            )
        )

    return document_chunks
