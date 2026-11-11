"""Chapter prior: demote symptom, injury and context chapters when a disease fits.

Symptom (ME/MD/MF/MG ...), injury (N*), external-cause (P*), health-factor (Q*),
special-purpose (R*) and traditional-medicine (S*) codes crowd the top of the list
for notes whose gold code is a specific disease. The prior is a soft multiplier on
the fused score, never a filter, so a note that really is an injury or a
supervision-of-pregnancy encounter can still win.
"""

from collections.abc import Mapping

# Multipliers on the fused RRF score (1.0 = no change). With the default RRF
# constant of 60, 0.9 moves a rank-1 hit to about rank 8.
DEFAULT_CHAPTER_FACTORS: Mapping[str, float] = {
    "21": 0.70,  # symptoms, signs and clinical findings
    "22": 0.75,  # injury, poisoning and external-cause consequences
    "23": 0.70,  # external causes of morbidity or mortality
    "24": 0.80,  # factors influencing health status
    "25": 0.70,  # codes for special purposes
    "26": 0.70,  # traditional medicine conditions
}

DEFAULT_GUARD_RANK = 10


def apply_chapter_prior(
    scores: Mapping[str, float],
    chapters: Mapping[str, str],
    factors: Mapping[str, float] | None = None,
    guard_rank: int = DEFAULT_GUARD_RANK,
) -> dict[str, float]:
    """Return `scores` with the chapter multipliers applied.

    The prior only applies when at least one disease code (a chapter without a
    factor, i.e. 01-20 by default) is in the top `guard_rank` of the unadjusted
    ranking; otherwise the note has no disease candidate and the scores are
    returned unchanged. Codes whose chapter is unknown count as disease codes.
    """
    if factors is None:
        factors = DEFAULT_CHAPTER_FACTORS
    for factor in factors.values():
        if not 0.0 < factor <= 1.0:
            raise ValueError("Chapter factors must be in (0, 1]")

    top = sorted(scores, key=lambda code: (-scores[code], code))[:guard_rank]
    if not any(chapters.get(code, "") not in factors for code in top):
        return dict(scores)
    return {
        code: score * factors.get(chapters.get(code, ""), 1.0)
        for code, score in scores.items()
    }
