from __future__ import annotations

import math
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from .candidate_evaluation import CandidateTrackingEvaluation


@dataclass(frozen=True, slots=True)
class _RateMetric:
    numerator: Callable[[CandidateTrackingEvaluation], int]
    denominator: Callable[[CandidateTrackingEvaluation], int]
    scale: float
    orientation: str


_METRICS = {
    "candidate_recall": _RateMetric(
        lambda row: row.matched, lambda row: row.truth_objects, 1.0, "higher_is_better"
    ),
    "candidate_precision": _RateMetric(
        lambda row: row.matched, lambda row: row.candidates, 1.0, "higher_is_better"
    ),
    "query_target_recall": _RateMetric(
        lambda row: row.query_target_hits,
        lambda row: row.query_target_trials,
        1.0,
        "higher_is_better",
    ),
    "track_purity": _RateMetric(
        lambda row: row.purity_correct,
        lambda row: row.purity_total,
        1.0,
        "higher_is_better",
    ),
    "identity_switches_per_100_matches": _RateMetric(
        lambda row: row.identity_switches,
        lambda row: row.matched,
        100.0,
        "lower_is_better",
    ),
    "fragmentations_per_100_truth_observations": _RateMetric(
        lambda row: row.fragmentations,
        lambda row: row.truth_objects,
        100.0,
        "lower_is_better",
    ),
    "capacity_exceeded_frame_rate": _RateMetric(
        lambda row: sum(frame.capacity_exceeded for frame in row.frames),
        lambda row: len(row.frames),
        1.0,
        "lower_is_better",
    ),
    "rejected_proposals_per_frame": _RateMetric(
        lambda row: sum(frame.rejected_proposals for frame in row.frames),
        lambda row: len(row.frames),
        1.0,
        "lower_is_better",
    ),
    "candidates_per_frame": _RateMetric(
        lambda row: row.candidates,
        lambda row: len(row.frames),
        1.0,
        "descriptive",
    ),
}


def paired_candidate_system_comparison(
    evaluations: Sequence[CandidateTrackingEvaluation],
    *,
    baseline: str,
    system: str,
    split: str,
    bootstrap_replicates: int = 10_000,
    seed: int = 20260901,
) -> dict[str, object]:
    """Compare candidate systems on exactly paired episodes and truth populations."""

    if not baseline.strip() or not system.strip() or baseline == system:
        raise ValueError("baseline and system must be distinct non-empty names")
    if split not in {"development", "calibration", "test"}:
        raise ValueError("candidate comparison split is invalid")
    if bootstrap_replicates < 100:
        raise ValueError("at least 100 bootstrap replicates are required")
    pairs = _paired_rows(evaluations, baseline, system, split)
    metrics = {
        name: _metric_comparison(
            pairs, metric, bootstrap_replicates, seed + offset
        )
        for offset, (name, metric) in enumerate(_METRICS.items())
    }
    return {
        "schema_version": 1,
        "split": split,
        "baseline": baseline,
        "system": system,
        "population": {
            "episodes": len(pairs),
            "clusters": len({left.cluster_id for left, _ in pairs}),
            "truth_population_hashes": sorted(
                {left.truth_population_sha256 for left, _ in pairs}
            ),
            "exactly_paired_by": ["cluster_id", "record_id"],
            "truth_and_query_definition_matched": True,
        },
        "bootstrap": {
            "unit": "cluster_id",
            "paired": True,
            "aggregation": "resample clusters then pool metric numerators and denominators",
            "replicates": bootstrap_replicates,
            "seed": seed,
            "interval": "percentile-95",
        },
        "metrics": metrics,
    }


def _paired_rows(
    evaluations: Sequence[CandidateTrackingEvaluation],
    baseline: str,
    system: str,
    split: str,
) -> tuple[tuple[CandidateTrackingEvaluation, CandidateTrackingEvaluation], ...]:
    selected: dict[str, dict[tuple[str, str], CandidateTrackingEvaluation]] = {
        baseline: {},
        system: {},
    }
    for row in evaluations:
        if row.split != split or row.system not in selected:
            continue
        key = (row.cluster_id, row.record_id)
        if key in selected[row.system]:
            raise ValueError(f"duplicate candidate row for {row.system}: {key}")
        selected[row.system][key] = row
    if not selected[baseline] or not selected[system]:
        raise ValueError("both candidate systems require non-empty rows")
    if set(selected[baseline]) != set(selected[system]):
        raise ValueError("candidate populations are not exactly paired")
    pairs = []
    for key in sorted(selected[baseline]):
        left = selected[baseline][key]
        right = selected[system][key]
        if _pairing_signature(left) != _pairing_signature(right):
            raise ValueError(f"paired candidate truth/query fields drifted: {key}")
        pairs.append((left, right))
    return tuple(pairs)


def _pairing_signature(row: CandidateTrackingEvaluation) -> tuple[object, ...]:
    return (
        row.source,
        row.truth_population_sha256,
        row.query_frame_id,
        row.query_target_entity_id,
        row.query_target_trials,
        row.iou_threshold,
        tuple((frame.frame_id, frame.truth_objects) for frame in row.frames),
    )


def _metric_comparison(
    pairs: Sequence[tuple[CandidateTrackingEvaluation, CandidateTrackingEvaluation]],
    metric: _RateMetric,
    replicates: int,
    seed: int,
) -> dict[str, object]:
    baseline = _pooled_rate(tuple(left for left, _ in pairs), metric)
    system = _pooled_rate(tuple(right for _, right in pairs), metric)
    if baseline is None or system is None:
        return {
            "status": "unavailable",
            "reason": "at least one system has a zero pooled denominator",
            "orientation": metric.orientation,
        }
    interval = _cluster_bootstrap_interval(pairs, metric, replicates, seed)
    episode_differences = []
    for left, right in pairs:
        left_rate = _pooled_rate((left,), metric)
        right_rate = _pooled_rate((right,), metric)
        if left_rate is not None and right_rate is not None:
            episode_differences.append(right_rate - left_rate)
    return {
        "status": "available",
        "orientation": metric.orientation,
        "baseline": baseline,
        "system": system,
        "delta_system_minus_baseline": system - baseline,
        "cluster_bootstrap_95_ci": [interval[0], interval[1]],
        "paired_episode_sign_exact_p": _paired_sign_exact(episode_differences),
        "paired_episode_sign_population": len(episode_differences),
        "baseline_numerator": sum(metric.numerator(left) for left, _ in pairs),
        "baseline_denominator": sum(metric.denominator(left) for left, _ in pairs),
        "system_numerator": sum(metric.numerator(right) for _, right in pairs),
        "system_denominator": sum(metric.denominator(right) for _, right in pairs),
    }


def _cluster_bootstrap_interval(
    pairs: Sequence[tuple[CandidateTrackingEvaluation, CandidateTrackingEvaluation]],
    metric: _RateMetric,
    replicates: int,
    seed: int,
) -> tuple[float, float]:
    grouped: dict[str, list[int]] = {}
    for index, (left, _) in enumerate(pairs):
        grouped.setdefault(left.cluster_id, []).append(index)
    names = sorted(grouped)
    generator = random.Random(seed)
    deltas: list[float] = []
    attempts = 0
    maximum_attempts = replicates * 100
    while len(deltas) < replicates and attempts < maximum_attempts:
        attempts += 1
        indices: list[int] = []
        for _ in names:
            indices.extend(grouped[generator.choice(names)])
        baseline = _pooled_rate(tuple(pairs[index][0] for index in indices), metric)
        system = _pooled_rate(tuple(pairs[index][1] for index in indices), metric)
        if baseline is not None and system is not None:
            deltas.append(system - baseline)
    if len(deltas) != replicates:
        raise ValueError("candidate cluster bootstrap could not draw valid denominators")
    deltas.sort()
    return _percentile(deltas, 0.025), _percentile(deltas, 0.975)


def _pooled_rate(
    rows: Sequence[CandidateTrackingEvaluation], metric: _RateMetric
) -> float | None:
    denominator = sum(metric.denominator(row) for row in rows)
    if denominator == 0:
        return None
    return metric.scale * sum(metric.numerator(row) for row in rows) / denominator


def _paired_sign_exact(differences: Sequence[float]) -> float:
    positive = sum(value > 0.0 for value in differences)
    negative = sum(value < 0.0 for value in differences)
    total = positive + negative
    if total == 0:
        return 1.0
    smaller = min(positive, negative)
    tail = sum(math.comb(total, index) for index in range(smaller + 1)) / (2**total)
    return min(1.0, 2.0 * tail)


def _percentile(values: Sequence[float], probability: float) -> float:
    position = probability * (len(values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return values[lower]
    fraction = position - lower
    return values[lower] * (1.0 - fraction) + values[upper] * fraction
