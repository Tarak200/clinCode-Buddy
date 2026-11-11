"""Hybrid retriever: BM25 + dense retrieval fused with reciprocal rank fusion (RRF)."""

from collections.abc import Mapping
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Optional

from src.retrieval.extract import extract_assessment
from src.retrieval.fusion import DEFAULT_RRF_K, rank_by_score, reciprocal_rank_fusion
from src.retrieval.lexical import LexicalIndex
from src.retrieval.priors import (
    DEFAULT_CHAPTER_FACTORS,
    DEFAULT_GUARD_RANK,
    apply_chapter_prior,
)
from src.retrieval.semantic import SemanticIndex as SemanticIndexBase


@dataclass(frozen=True)
class Candidate:
    code: str
    title: str
    score: float


class HybridIndex:
    """Fuse lexical and semantic rankings of the whole note and of its assessment.

    Each (query, retriever) pair yields one ranked list. The lists are combined
    with RRF, so BM25 and cosine scores never have to share a scale, and the
    chapter prior is applied to the fused score.
    """

    def __init__(
        self,
        records: list[dict[str, object]],
        embedder=None,
        embeddings_cache_path: Optional[Path] = None,
        semantic_weight: float = 1.0,
        lexical_weight: float = 1.0,
        assessment_weight: float = 2.0,
        rrf_k: int = DEFAULT_RRF_K,
        depth: int = 100,
        use_assessment: bool = True,
        chapter_factors: Optional[Mapping[str, float]] = DEFAULT_CHAPTER_FACTORS,
        guard_rank: int = DEFAULT_GUARD_RANK,
        show_progress_bar: bool = False,
    ) -> None:
        """
        Args:
            records: ICD catalogue records
            embedder: SentenceTransformer embedder
            embeddings_cache_path: Path to cache embeddings
            semantic_weight: RRF weight of the dense lists (relative to lexical)
            lexical_weight: RRF weight of the BM25 lists
            assessment_weight: multiplier on the lists built from the extracted
                assessment; the full-note lists always keep weight 1
            rrf_k: RRF constant (larger flattens the benefit of top ranks)
            depth: how many results each list contributes before fusion
            use_assessment: also query with the extracted assessment text
            chapter_factors: per-chapter score multipliers, or None to disable
            guard_rank: apply the chapter prior only if a disease code is in
                the top `guard_rank` of the fused ranking
        """
        if depth < 1:
            raise ValueError("The fusion depth must be at least 1")
        if semantic_weight < 0 or lexical_weight < 0 or assessment_weight < 0:
            raise ValueError("Fusion weights must be non-negative")
        if semantic_weight + lexical_weight == 0:
            raise ValueError("At least one retriever must have a positive weight")

        self._semantic_index = SemanticIndexBase(
            records,
            embedder=embedder,
            embeddings_cache_path=embeddings_cache_path,
            show_progress_bar=show_progress_bar,
        )
        self._lexical_index = LexicalIndex(records)
        self._chapters = {
            str(record["code"]): str(record.get("chapter", "")) for record in records
        }
        self._semantic_weight = semantic_weight
        self._lexical_weight = lexical_weight
        self._assessment_weight = assessment_weight
        self._rrf_k = rrf_k
        self._depth = depth
        self._use_assessment = use_assessment
        self._chapter_factors = chapter_factors
        self._guard_rank = guard_rank

    @classmethod
    def from_file(
        cls,
        path: Path,
        embedder=None,
        embeddings_cache_path: Optional[Path] = None,
        **options,
    ) -> "HybridIndex":
        with path.open(encoding="utf-8") as file:
            records = json.load(file)
        if not isinstance(records, list) or not all(
            isinstance(record, dict) for record in records
        ):
            raise ValueError("The MMS catalogue must be a JSON array of objects")
        return cls(
            records,
            embedder=embedder,
            embeddings_cache_path=embeddings_cache_path,
            **options,
        )

    def _queries(self, text: str) -> list[tuple[str, float]]:
        """Return (query text, weight) pairs; the full note is always included."""
        queries = [(text, 1.0)]
        if self._use_assessment and self._assessment_weight > 0:
            assessment = extract_assessment(text)
            if assessment and assessment.split() != text.split():
                queries.append((assessment, self._assessment_weight))
        return queries

    def search(self, text: str, limit: int = 50) -> list[Candidate]:
        """Search using reciprocal rank fusion over all query/retriever lists."""
        if limit < 1:
            raise ValueError("The candidate limit must be at least 1")

        depth = max(self._depth, limit)
        rankings: list[list[str]] = []
        weights: list[float] = []
        titles: dict[str, str] = {}

        for query, query_weight in self._queries(text):
            semantic = self._semantic_index.search(query, limit=depth)
            # BM25 pads its output with zero-score entries in code order;
            # those carry no evidence and must not take part in rank fusion.
            lexical = [
                r for r in self._lexical_index.search(query, limit=depth) if r.score > 0
            ]
            for results, weight in (
                (semantic, self._semantic_weight),
                (lexical, self._lexical_weight),
            ):
                rankings.append([r.code for r in results])
                weights.append(weight * query_weight)
                titles.update((r.code, r.title) for r in results)

        scores = reciprocal_rank_fusion(rankings, weights, k=self._rrf_k)
        if self._chapter_factors:
            scores = apply_chapter_prior(
                scores,
                self._chapters,
                factors=self._chapter_factors,
                guard_rank=self._guard_rank,
            )

        return [
            Candidate(code=code, title=titles[code], score=score)
            for code, score in rank_by_score(scores)[:limit]
        ]
