"""A small, local BM25 index over the supplied ICD-11 catalogue."""

from collections import Counter, defaultdict
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
import unicodedata


_TOKEN = re.compile(r"[^\W_]+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    """Normalize text without removing words, numbers, or negation."""
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return _TOKEN.findall(normalized)


@dataclass(frozen=True)
class Candidate:
    code: str
    title: str
    score: float


class LexicalIndex:
    """Rank catalogue titles using BM25 with deterministic tie-breaking."""

    def __init__(self, records: list[dict[str, object]]) -> None:
        if not records:
            raise ValueError("The MMS catalogue is empty")

        self._documents: list[tuple[str, str, int]] = []
        self._postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        seen_codes: set[str] = set()

        for record in records:
            code = record.get("code")
            title = record.get("title")
            if not isinstance(code, str) or not code.strip():
                raise ValueError("Every catalogue entry must have a nonempty code")
            if not isinstance(title, str) or not title.strip():
                raise ValueError(f"Catalogue entry {code!r} has no title")
            if code in seen_codes:
                raise ValueError(f"Duplicate catalogue code: {code}")
            seen_codes.add(code)

            frequencies = Counter(tokenize(title))
            document_id = len(self._documents)
            self._documents.append((code, title, sum(frequencies.values())))
            for term, frequency in frequencies.items():
                self._postings[term].append((document_id, frequency))

        self._average_length = (
            sum(length for _, _, length in self._documents) / len(self._documents)
        )
        if self._average_length == 0:
            raise ValueError("Catalogue titles contain no searchable terms")

    @classmethod
    def from_file(cls, path: Path) -> "LexicalIndex":
        with path.open(encoding="utf-8") as file:
            records = json.load(file)
        if not isinstance(records, list) or not all(
            isinstance(record, dict) for record in records
        ):
            raise ValueError("The MMS catalogue must be a JSON array of objects")
        return cls(records)

    def search(self, text: str, limit: int = 50) -> list[Candidate]:
        if limit < 1:
            raise ValueError("The candidate limit must be at least 1")

        scores: dict[int, float] = defaultdict(float)
        document_count = len(self._documents)
        k1 = 1.5
        b = 0.75

        for term in sorted(set(tokenize(text))):
            postings = self._postings.get(term, [])
            if not postings:
                continue
            inverse_document_frequency = math.log1p(
                (document_count - len(postings) + 0.5) / (len(postings) + 0.5)
            )
            for document_id, frequency in postings:
                length = self._documents[document_id][2]
                denominator = frequency + k1 * (
                    1 - b + b * length / self._average_length
                )
                scores[document_id] += (
                    inverse_document_frequency * frequency * (k1 + 1) / denominator
                )

        # Include zero-score entries so a note with no matches still has output.
        ranked = sorted(
            range(document_count),
            key=lambda index: (
                -scores.get(index, 0.0),
                self._documents[index][0],
            ),
        )
        return [
            Candidate(
                code=self._documents[index][0],
                title=self._documents[index][1],
                score=scores.get(index, 0.0),
            )
            for index in ranked[:limit]
        ]
