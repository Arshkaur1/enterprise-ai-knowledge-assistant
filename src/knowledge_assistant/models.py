
from dataclasses import asdict, dataclass


@dataclass
class DocumentUnit:
    """
    Represents one extracted unit of a source document.

    For PDF files, a unit represents one page.
    For TXT and DOCX files, a unit represents the document.
    """

    document_id: str
    source: str
    file_type: str
    category: str
    text: str

    page_number: int | None = None


@dataclass
class DocumentChunk:
    """
    Represents a processed chunk ready for embedding.
    """

    chunk_id: str
    document_id: str
    source: str
    file_type: str
    category: str
    text: str

    page_number: int | None = None
    chunk_index: int = 0

    def to_dict(self) -> dict:
        return asdict(self)
