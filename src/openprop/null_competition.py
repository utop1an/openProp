from __future__ import annotations

import math
from typing import Mapping

from .visual_evaluation import NULL_ENTITY


def rescale_null_distribution(
    probabilities: Mapping[str, float],
    *,
    candidate_count: int,
    null_scale: float,
    candidate_count_power: float,
) -> tuple[dict[str, float], str | None]:
    """Rescale an existing null prior without reconstructing model affinities."""

    if candidate_count <= 0:
        raise ValueError("candidate_count must be positive")
    if not math.isfinite(null_scale) or null_scale <= 0.0:
        raise ValueError("null_scale must be finite and positive")
    if not math.isfinite(candidate_count_power):
        raise ValueError("candidate_count_power must be finite")
    if NULL_ENTITY not in probabilities:
        raise ValueError("probability distribution is missing null")
    null_probability = probabilities[NULL_ENTITY]
    factor = null_scale * candidate_count**candidate_count_power
    denominator = 1.0 - null_probability + factor * null_probability
    if not math.isfinite(denominator) or denominator <= 0.0:
        raise ValueError("rescaled null distribution is invalid")
    adjusted = {
        key: (value * factor if key == NULL_ENTITY else value) / denominator
        for key, value in probabilities.items()
    }
    decision_key = min(adjusted, key=lambda key: (-adjusted[key], key))
    return adjusted, None if decision_key == NULL_ENTITY else decision_key
