from __future__ import annotations

import math
import random
import statistics
from collections.abc import Mapping, Sequence
from typing import Any


def paired_bootstrap_simultaneous_intervals(
    family: Mapping[str, Sequence[float]],
    *,
    samples: int = 20_000,
    seed: int = 20260828,
    confidence: float = 0.95,
) -> dict[str, Any]:
    """Return paired max-standardized-deviation bootstrap intervals.

    One resampled seed-index vector is shared by every comparison in the family.
    The critical value is the requested quantile of the maximum absolute,
    standard-error-scaled deviation across comparisons. Constant comparisons
    receive their exact degenerate interval and do not destabilize the family.
    """

    if samples <= 0:
        raise ValueError("bootstrap samples must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie strictly between zero and one")
    if not family:
        raise ValueError("at least one comparison is required")
    names = tuple(family)
    values = {name: tuple(float(value) for value in family[name]) for name in names}
    lengths = {len(row) for row in values.values()}
    if len(lengths) != 1 or not lengths or next(iter(lengths)) == 0:
        raise ValueError("all comparisons must contain the same nonzero paired sample")
    if any(not math.isfinite(value) for row in values.values() for value in row):
        raise ValueError("paired comparison values must be finite")

    count = next(iter(lengths))
    means = {name: statistics.mean(values[name]) for name in names}
    standard_errors = {
        name: statistics.stdev(values[name]) / math.sqrt(count) if count > 1 else 0.0
        for name in names
    }
    rng = random.Random(seed)
    maxima: list[float] = []
    for _ in range(samples):
        indices = tuple(rng.randrange(count) for _ in range(count))
        maximum = 0.0
        for name in names:
            standard_error = standard_errors[name]
            if standard_error == 0.0:
                continue
            bootstrap_mean = statistics.mean(values[name][index] for index in indices)
            maximum = max(maximum, abs(bootstrap_mean - means[name]) / standard_error)
        maxima.append(maximum)
    maxima.sort()
    critical_index = min(samples - 1, math.ceil(confidence * samples) - 1)
    critical_value = maxima[critical_index]
    intervals = {
        name: [
            means[name] - critical_value * standard_errors[name],
            means[name] + critical_value * standard_errors[name],
        ]
        for name in names
    }
    return {
        "method": "paired bootstrap max standardized mean deviation",
        "family": list(names),
        "family_size": len(names),
        "confidence": confidence,
        "bootstrap_samples": samples,
        "bootstrap_seed": seed,
        "shared_resample_indices": True,
        "critical_value": critical_value,
        "intervals": intervals,
    }
