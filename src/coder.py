"""Generate local lexical ICD-11 predictions from JSONL notes."""

import argparse
import json
from pathlib import Path
import sys

from src.retrieval.lexical import LexicalIndex


def run(
    input_path: Path,
    output_path: Path,
    catalogue_path: Path,
) -> None:
    paths = [input_path.resolve(), output_path.resolve(), catalogue_path.resolve()]
    if len(set(paths)) != len(paths):
        raise ValueError("Input, output, and catalogue paths must differ")

    index = LexicalIndex.from_file(catalogue_path)
    seen_ids: set[object] = set()

    with input_path.open(encoding="utf-8") as source, output_path.open(
        "w", encoding="utf-8"
    ) as output:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                note = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{input_path}:{line_number}: invalid JSON") from error
            if not isinstance(note, dict):
                raise ValueError(f"{input_path}:{line_number}: expected a JSON object")
            note_id = note.get("id")
            text = note.get("text")
            if "id" not in note or not isinstance(note_id, (str, int, float)):
                raise ValueError(f"{input_path}:{line_number}: id must be a string or number")
            if not isinstance(text, str):
                raise ValueError(f"{input_path}:{line_number}: text must be a string")
            if note_id in seen_ids:
                raise ValueError(f"{input_path}:{line_number}: duplicate input id {note_id!r}")
            seen_ids.add(note_id)

            results = index.search(text, limit=3)
            if results[0].score == 0:
                print(
                    f"Note at line {line_number}: no catalogue terms matched; "
                    "using deterministic zero-score results",
                    file=sys.stderr,
                )

            prediction = {"id": note_id, "codes": [item.code for item in results]}
            output.write(json.dumps(prediction, ensure_ascii=False) + "\n")
            output.flush()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Input JSONL notes", default=Path(__file__).resolve().parent.parent / "data" / "dev_notes.jsonl")
    parser.add_argument("output", type=Path, help="Output JSONL predictions")
    parser.add_argument(
        "--mms",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data" / "mms-2026-01.json",
        help="MMS catalogue JSON file",
    )
    args = parser.parse_args()
    try:
        run(args.input, args.output, args.mms)
    except (OSError, ValueError) as error:
        parser.exit(1, f"error: {error}\n")


if __name__ == "__main__":
    main()
