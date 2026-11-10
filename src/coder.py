"""Generate ICD-11 predictions from JSONL notes using semantic or lexical retrieval."""

import argparse
import json
from pathlib import Path
import sys

try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False

from src.retrieval.lexical import LexicalIndex
from src.retrieval.semantic import SemanticIndex


def run(
    input_path: Path,
    output_path: Path,
    catalogue_path: Path,
    use_lexical: bool = False,
    model_name: str = "all-MiniLM-L6-v2",
) -> None:
    paths = [input_path.resolve(), output_path.resolve(), catalogue_path.resolve()]
    if len(set(paths)) != len(paths):
        raise ValueError("Input, output, and catalogue paths must differ")

    # Choose retriever
    if use_lexical:
        print("Using BM25 Lexical Retriever (baseline)", file=sys.stderr)
        index = LexicalIndex.from_file(catalogue_path)
    else:
        # Initialize embedder for semantic retrieval
        if not HAS_SENTENCE_TRANSFORMERS:
            raise ImportError("sentence-transformers is required for semantic retrieval. Install with: pip install sentence-transformers")
        
        print(f"Using Semantic Retriever with model: {model_name}", file=sys.stderr)
        embedder = SentenceTransformer(model_name)
        
        # Setup cache path for embeddings
        cache_dir = catalogue_path.parent / ".embeddings_cache"
        cache_file = cache_dir / f"{catalogue_path.stem}_{model_name.replace('/', '_')}.pkl"
        
        print("Building semantic index...", file=sys.stderr)
        index = SemanticIndex.from_file(
            catalogue_path,
            embedder=embedder,
            embeddings_cache_path=cache_file
        )
    
    seen_ids: set[object] = set()

    print(f"Processing notes from {input_path}", file=sys.stderr)
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

            prediction = {"id": note_id, "codes": [item.code for item in results]}
            output.write(json.dumps(prediction, ensure_ascii=False) + "\n")
            output.flush()
    
    print(f"Predictions written to {output_path}", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input",
        type=Path,
        nargs="?",
        help="Input JSONL notes",
        default=Path(__file__).resolve().parent.parent / "data" / "dev_notes.jsonl"
    )
    parser.add_argument(
        "output",
        type=Path,
        nargs="?",
        help="Output JSONL predictions",
        default=Path(__file__).resolve().parent.parent / "predictions" / "predictions.jsonl"
    )
    parser.add_argument(
        "--mms",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data" / "mms-2026-01.json",
        help="MMS catalogue JSON file",
    )
    parser.add_argument(
        "--use-lexical",
        action="store_true",
        help="Use BM25 lexical baseline instead of semantic retrieval",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="all-MiniLM-L6-v2",
        help="Sentence-Transformers model name (for semantic retrieval only)",
    )
    args = parser.parse_args()
    try:
        run(args.input, args.output, args.mms, args.use_lexical, args.model)
    except (OSError, ValueError, ImportError) as error:
        parser.exit(1, f"error: {error}\n")


if __name__ == "__main__":
    main()



