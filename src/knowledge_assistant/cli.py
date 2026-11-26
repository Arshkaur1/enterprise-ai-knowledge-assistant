import argparse
from pathlib import Path

from knowledge_assistant.pipeline import process_directory

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main():

    parser = argparse.ArgumentParser(
        description="Enterprise AI Document Ingestion Pipeline"
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw",
        help="Input document directory",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=(PROJECT_ROOT / "data" / "processed" / "document_chunks.jsonl"),
        help="Output JSONL file",
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=800,
    )

    parser.add_argument(
        "--overlap",
        type=int,
        default=120,
    )

    args = parser.parse_args()

    process_directory(
        input_dir=args.input,
        output_file=args.output,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
    )


if __name__ == "__main__":
    main()
