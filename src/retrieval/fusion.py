"""Reciprocal rank fusion (Cormack, Clarke & Buettcher, SIGIR 2009)."""

from collections.abc import Sequence

DEFAULT_RRF_K = 60


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[str]],
    weights: Sequence[float] | None = None,
    k: int = DEFAULT_RRF_K,
) -> dict[str, float]:
    """Fuse ranked lists of codes into `{code: sum(weight / (k + rank))}`.

    Ranks are 1-based and only positions matter, so lists scored on different
    scales (cosine vs. BM25) combine without normalisation. A code missing from
    a list contributes nothing for that list. Duplicates within one list count
    once, at their best rank.
    """
    if k < 0:
        raise ValueError("The RRF constant k must be non-negative")
    if weights is None:
        weights = [1.0] * len(rankings)
    if len(weights) != len(rankings):
        raise ValueError("Provide exactly one weight per ranking")
    if any(weight < 0 for weight in weights):
        raise ValueError("RRF weights must be non-negative")

    fused: dict[str, float] = {}
    for ranking, weight in zip(rankings, weights):
        seen: set[str] = set()
        rank = 0
        for code in ranking:
            if code in seen:
                continue
            seen.add(code)
            rank += 1
            fused[code] = fused.get(code, 0.0) + weight / (k + rank)
    return fused


def rank_by_score(scores: dict[str, float]) -> list[tuple[str, float]]:
    """Sort `(code, score)` best first, breaking ties by code for determinism."""
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))
