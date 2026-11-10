"""Hybrid retriever combining semantic and BM25 approaches."""

from dataclasses import dataclass
from pathlib import Path
import numpy as np
from typing import Optional
import json

from src.retrieval.semantic import SemanticIndex as SemanticIndexBase
from src.retrieval.lexical import LexicalIndex


@dataclass(frozen=True)
class Candidate:
    code: str
    title: str
    score: float


class HybridIndex:
    """Combines semantic and lexical matching for better retrieval."""

    def __init__(
        self,
        records: list[dict[str, object]],
        embedder=None,
        embeddings_cache_path: Optional[Path] = None,
        semantic_weight: float = 0.7,
        lexical_weight: float = 0.3,
    ) -> None:
        """
        Initialize hybrid index.
        
        Args:
            records: ICD catalogue records
            embedder: SentenceTransformer embedder
            embeddings_cache_path: Path to cache embeddings
            semantic_weight: Weight for semantic scores (0-1)
            lexical_weight: Weight for lexical scores (0-1)
        """
        # Build both indices
        self._semantic_index = SemanticIndexBase(
            records,
            embedder=embedder,
            embeddings_cache_path=embeddings_cache_path,
        )
        self._lexical_index = LexicalIndex(records)
        
        # Normalize weights
        total = semantic_weight + lexical_weight
        self._semantic_weight = semantic_weight / total
        self._lexical_weight = lexical_weight / total

    @classmethod
    def from_file(
        cls,
        path: Path,
        embedder=None,
        embeddings_cache_path: Optional[Path] = None,
        semantic_weight: float = 0.7,
        lexical_weight: float = 0.3,
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
            semantic_weight=semantic_weight,
            lexical_weight=lexical_weight,
        )

    def search(self, text: str, limit: int = 50) -> list[Candidate]:
        """Search using hybrid approach."""
        if limit < 1:
            raise ValueError("The candidate limit must be at least 1")

        # Get results from both retrievers
        semantic_results = self._semantic_index.search(text, limit=limit)
        lexical_results = self._lexical_index.search(text, limit=limit)
        
        # Create score maps
        # Normalize semantic scores to 0-1 range (they're already close to this)
        semantic_scores = {
            r.code: (r.score + 1.0) / 2.0  # Map from [-1, 1] to [0, 1]
            for r in semantic_results
        }
        
        # Lexical scores need normalization too
        max_lexical_score = max((r.score for r in lexical_results), default=1.0)
        lexical_scores = {
            r.code: (r.score / max(max_lexical_score, 1e-8))
            for r in lexical_results
        }
        
        # Combine all codes seen
        all_codes = set(semantic_scores.keys()) | set(lexical_scores.keys())
        
        # Compute hybrid scores
        hybrid_scores = {}
        code_titles = {}
        
        for code in all_codes:
            sem_score = semantic_scores.get(code, 0.0)
            lex_score = lexical_scores.get(code, 0.0)
            hybrid_score = (
                self._semantic_weight * sem_score +
                self._lexical_weight * lex_score
            )
            hybrid_scores[code] = hybrid_score
            
            # Store title from whichever result has it
            for r in semantic_results + lexical_results:
                if r.code == code:
                    code_titles[code] = r.title
                    break
        
        # Sort by hybrid score
        sorted_codes = sorted(hybrid_scores.items(), key=lambda x: (-x[1], x[0]))
        
        candidates = [
            Candidate(
                code=code,
                title=code_titles.get(code, ""),
                score=score,
            )
            for code, score in sorted_codes[:limit]
        ]
        
        return candidates
