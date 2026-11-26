
import json

from knowledge_assistant.pipeline import process_directory


def test_end_to_end_pipeline(tmp_path):

    input_dir = tmp_path / "raw"
    output_dir = tmp_path / "processed"

    input_dir.mkdir()

    document = input_dir / "hr_policy.txt"

    document.write_text(
        "Employees need approval for remote work. "
        * 30,
        encoding="utf-8",
    )

    output_file = output_dir / "chunks.jsonl"

    chunks = process_directory(
        input_dir=input_dir,
        output_file=output_file,
        chunk_size=100,
        overlap=20,
    )

    assert len(chunks) > 1
    assert output_file.exists()

    with output_file.open(
        encoding="utf-8"
    ) as file:

        records = [
            json.loads(line)
            for line in file
        ]

    assert len(records) == len(chunks)

    assert records[0]["category"] == "HR"
    assert records[0]["source"] == "hr_policy.txt"
    assert records[0]["chunk_id"]
    assert records[0]["text"]
