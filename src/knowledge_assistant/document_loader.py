
from pathlib import Path
from typing import cast

import pymupdf
from docx import Document

from knowledge_assistant.models import DocumentUnit

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}


def infer_category(filename: str) -> str:
    """
    Infer document category from filename.

    In production, category should ideally come from
    an explicit document registry or metadata source.
    """

    name = filename.lower()

    if "hr" in name or "employee" in name:
        return "HR"

    if "it" in name or "security" in name:
        return "IT"

    if "finance" in name or "expense" in name:
        return "Finance"

    return "General"


def load_txt(path: Path) -> list[DocumentUnit]:

    text = path.read_text(encoding="utf-8")

    return [
        DocumentUnit(
            document_id=path.stem,
            source=path.name,
            file_type="txt",
            category=infer_category(path.name),
            text=text,
        )
    ]


def load_docx(path: Path) -> list[DocumentUnit]:

    doc = Document(str(path))

    content = []

    # Extract paragraphs.
    for paragraph in doc.paragraphs:
        if paragraph.text.strip():
            content.append(paragraph.text)

    # Extract tables.
    for table in doc.tables:
        for row in table.rows:
            cells = [
                cell.text.strip()
                for cell in row.cells
            ]

            content.append(" | ".join(cells))

    text = "\n".join(content)

    return [
        DocumentUnit(
            document_id=path.stem,
            source=path.name,
            file_type="docx",
            category=infer_category(path.name),
            text=text,
        )
    ]


def load_pdf(path: Path) -> list[DocumentUnit]:

    units = []

    with pymupdf.open(path) as pdf:

        for page_number in range(1, pdf.page_count + 1):

            page = pdf.load_page(page_number - 1)
            text = cast(str, page.get_text("text"))

            units.append(
                DocumentUnit(
                    document_id=path.stem,
                    source=path.name,
                    file_type="pdf",
                    category=infer_category(path.name),
                    text=text,
                    page_number=page_number,
                )
            )

    return units


def load_document(path: str | Path) -> list[DocumentUnit]:

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Document not found: {path}"
        )

    if not path.is_file():
        raise ValueError(
            f"Expected a file, received: {path}"
        )

    extension = path.suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported document format: {extension}"
        )

    if extension == ".txt":
        return load_txt(path)

    if extension == ".docx":
        return load_docx(path)

    if extension == ".pdf":
        return load_pdf(path)

    raise ValueError(
        f"Unsupported document format: {extension}"
    )
