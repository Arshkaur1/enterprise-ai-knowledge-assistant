
# Document Ingestion Architecture

## Objective

Convert enterprise documents into structured,
metadata-rich text chunks for downstream embedding
generation and semantic retrieval.

## Supported Formats

- PDF
- DOCX
- TXT

## Pipeline Components

1. Document discovery
2. Format-specific text extraction
3. Text normalization
4. Metadata extraction
5. Character-based chunking
6. JSONL serialization

## Metadata Schema

| Field | Description |
|---|---|
| chunk_id | Unique chunk identifier |
| document_id | Source document identifier |
| source | Original filename |
| file_type | Original document format |
| category | Document category |
| page_number | Original PDF page, when applicable |
| chunk_index | Chunk position within the unit |
| text | Extracted and cleaned content |

## Initial Chunking Configuration

- Chunk size: 800 characters
- Overlap: 120 characters
- Step size: 680 characters

## Output

data/processed/document_chunks.jsonl

## Current Limitations

- No OCR for scanned PDFs.
- No semantic chunking.
- No token-based chunking.
- Basic filename-based category inference.
- No advanced table structure preservation.
- No duplicate document detection.

## Next Milestone

Embedding generation and vector indexing.
