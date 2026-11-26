

import pymupdf
import pytest
from docx import Document

from knowledge_assistant.document_loader import load_document


def test_load_txt(tmp_path):

    file_path = tmp_path / "hr_policy.txt"

    file_path.write_text(
        "Remote work requires manager approval.",
        encoding="utf-8",
    )

    units = load_document(file_path)

    assert len(units) == 1
    assert units[0].category == "HR"
    assert units[0].file_type == "txt"
    assert "manager approval" in units[0].text


def test_load_docx(tmp_path):

    file_path = tmp_path / "it_security.docx"

    doc = Document()
    doc.add_paragraph("Multi-factor authentication is mandatory.")
    doc.save(file_path)

    units = load_document(file_path)

    assert len(units) == 1
    assert units[0].category == "IT"
    assert units[0].file_type == "docx"
    assert "authentication" in units[0].text


def test_load_pdf(tmp_path):

    file_path = tmp_path / "finance_policy.pdf"

    pdf = pymupdf.open()
    page = pdf.new_page()

    page.insert_text(
        (50, 50),
        "Expenses require manager approval.",
    )

    pdf.save(file_path)
    pdf.close()

    units = load_document(file_path)

    assert len(units) == 1
    assert units[0].category == "Finance"
    assert units[0].page_number == 1
    assert "manager approval" in units[0].text


def test_missing_file():

    with pytest.raises(FileNotFoundError):
        load_document("missing_file.pdf")


def test_unsupported_extension(tmp_path):

    file_path = tmp_path / "data.csv"
    file_path.write_text("a,b\n1,2")

    with pytest.raises(ValueError):
        load_document(file_path)
