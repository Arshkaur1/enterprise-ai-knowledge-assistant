
import json
from pathlib import Path

from knowledge_assistant.chunking import create_document_chunks
from knowledge_assistant.cleaning import clean_text
from knowledge_assistant.document_loader import load_document
from knowledge_assistant.models import DocumentChunk

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}


def process_directory(
    input_dir: str | Path,
    output_file: str | Path,
    chunk_size: int = 800,
    overlap: int = 120,
) -> list[DocumentChunk]:
    """
    Process all supported documents in a directory.

    Returns structured document chunks and writes them
    to a JSONL file.
    """

    input_dir = Path(input_dir)
    output_file = Path(output_file)

    if not input_dir.exists():
        raise FileNotFoundError(
            f"Input directory does not exist: {input_dir}"
        )

    if not input_dir.is_dir():
        raise ValueError(
            f"Expected a directory: {input_dir}"
        )

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    document_paths = sorted(
        path
        for path in input_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    all_chunks = []

    for document_path in document_paths:

        print(f"Processing: {document_path.name}")

        units = load_document(document_path)

        for unit in units:

            cleaned_text = clean_text(unit.text)

            if not cleaned_text:
                continue

            unit.text = cleaned_text

            chunks = create_document_chunks(
                unit,
                chunk_size=chunk_size,
                overlap=overlap,
            )

            all_chunks.extend(chunks)

    # Write one JSON object per line.
    with output_file.open(
        "w",
        encoding="utf-8",
    ) as file:

        for chunk in all_chunks:

            file.write(
                json.dumps(
                    chunk.to_dict(),
                    ensure_ascii=False,
                )
                + "\n"
            )

    print("\nIngestion completed.")
    print(f"Documents discovered: {len(document_paths)}")
    print(f"Chunks generated: {len(all_chunks)}")
    print(f"Output: {output_file}")

    return all_chunks
