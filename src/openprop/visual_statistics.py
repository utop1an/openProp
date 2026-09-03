from __future__ import annotations

import math
import random
from collections.abc import Callable, Mapping, Sequence

from .visual_evaluation import (
    VisualAssociationResult,
    VisualEvaluationDataset,
    VisualQueryResult,
)


def paired_visual_system_comparison(
    dataset: VisualEvaluationDataset,
    *,
    baseline: str,
    system: str,
    split: str,
    bootstrap_replicates: int = 10_000,
    seed: int = 20260901,
    include_association: bool = True,
) -> dict[str, object]:
    """Paired cluster-bootstrap comparison on frozen query/update populations."""

    if not baseline.strip() or not system.strip() or baseline == system:
        raise ValueError("baseline and system must be distinct non-empty names")
    if split not in {"development", "calibration", "test"}:
        raise ValueError("comparison split is invalid")
    if bootstrap_replicates < 100:
        raise ValueError("at least 100 bootstrap replicates are required")
    query_pairs = _pair_rows(
        tuple(row for row in dataset.queries if row.split == split),
        baseline,
        system,
        _query_signature,
        "query",
    )
    association_pairs = (
        _pair_rows(
            tuple(row for row in dataset.associations if row.split == split),
            baseline,
            system,
            _association_signature,
            "association",
        )
        if include_association
        else ()
    )
    return {
        "schema_version": 1,
        "split": split,
        "baseline": baseline,
        "system": system,
        "bootstrap": {
            "unit": "cluster_id",
            "paired": True,
            "replicates": bootstrap_replicates,
            "seed": seed,
            "interval": "percentile-95",
        },
        "query": _query_comparison(query_pairs, bootstrap_replicates, seed),
        "association": (
            _association_comparison(association_pairs, bootstrap_replicates, seed + 1)
            if include_association
            else {
                "status": "not_requested",
                "reason": "association inference requires identical detection populations",
            }
        ),
    }


def _pair_rows(
    rows: Sequence[VisualQueryResult | VisualAssociationResult],
    baseline: str,
    system: str,
    signature: Callable[[object], tuple[object, ...]],
    unit: str,
) -> tuple[tuple[object, object], ...]:
    selected = {baseline: {}, system: {}}
    for row in rows:
        if row.system not in selected:
            continue
        key = (row.cluster_id, row.record_id)
        bucket = selected[row.system]
        if key in bucket:
            raise ValueError(f"duplicate {unit} row for {row.system}: {key}")
        bucket[key] = row
    if not selected[baseline] or not selected[system]:
        raise ValueError(f"both systems require non-empty {unit} rows")
    if set(selected[baseline]) != set(selected[system]):
        raise ValueError(f"{unit} populations are not exactly paired")
    pairs = []
    for key in sorted(selected[baseline]):
        left = selected[baseline][key]
        right = selected[system][key]
        if signature(left) != signature(right):
            raise ValueError(f"paired {unit} truth/input fields drifted: {key}")
        pairs.append((left, right))
    return tuple(pairs)


def _query_signature(row: object) -> tuple[object, ...]:
    assert isinstance(row, VisualQueryResult)
    return (
        row.source,
        row.property_name,
        row.target_entity_id,
        row.candidate_entity_ids,
        row.horizon_seconds,
        row.distractor_count,
        row.condition,
        row.eligible,
    )


def _association_signature(row: object) -> tuple[object, ...]:
    assert isinstance(row, VisualAssociationResult)
    return (
        row.source,
        row.property_name,
        row.detection_id,
        row.frame_id,
        row.candidate_entity_ids,
        row.target_entity_id,
        row.condition,
        row.distractor_count,
        row.eligible,
    )


def _query_comparison(
    pairs: Sequence[tuple[object, object]], replicates: int, seed: int
) -> dict[str, object]:
    typed = tuple((left, right) for left, right in pairs)
    metrics = {
        "top1": lambda row: float(row.top1_correct),
        "mrr": lambda row: row.reciprocal_rank,
        "coverage": lambda row: float(row.accepted),
    }
    output = _metric_comparisons(typed, metrics, replicates, seed)
    output["top1"]["mcnemar_exact_p"] = _mcnemar_exact(
        tuple((left.top1_correct, right.top1_correct) for left, right in typed)
    )
    output["mrr"]["paired_sign_exact_p"] = _paired_sign_exact(
        tuple(right.reciprocal_rank - left.reciprocal_rank for left, right in typed)
    )
    return {
        "population": len(typed),
        "clusters": len({left.cluster_id for left, _ in typed}),
        "metrics": output,
    }


def _association_comparison(
    pairs: Sequence[tuple[object, object]], replicates: int, seed: int
) -> dict[str, object]:
    typed = tuple((left, right) for left, right in pairs)
    metrics = {
        "false_update_rate": lambda row: float(row.false_update),
        "correct_update_rate": lambda row: float(row.correct_update),
        "coverage": lambda row: float(row.accepted),
    }
    output = _metric_comparisons(typed, metrics, replicates, seed)
    for name, attribute in (
        ("false_update_rate", "false_update"),
        ("correct_update_rate", "correct_update"),
    ):
        output[name]["mcnemar_exact_p"] = _mcnemar_exact(
            tuple(
                (bool(getattr(left, attribute)), bool(getattr(right, attribute)))
                for left, right in typed
            )
        )
    return {
        "population": len(typed),
        "clusters": len({left.cluster_id for left, _ in typed}),
        "metrics": output,
    }


def _metric_comparisons(
    pairs: Sequence[tuple[object, object]],
    metrics: Mapping[str, Callable[[object], float]],
    replicates: int,
    seed: int,
) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for offset, (name, metric) in enumerate(metrics.items()):
        baseline_values = tuple(metric(left) for left, _ in pairs)
        system_values = tuple(metric(right) for _, right in pairs)
        clusters = tuple(left.cluster_id for left, _ in pairs)
        delta = _mean(system_values) - _mean(baseline_values)
        interval = _cluster_bootstrap_interval(
            baseline_values,
            system_values,
            clusters,
            replicates,
            seed + offset,
        )
        result[name] = {
            "baseline": _mean(baseline_values),
            "system": _mean(system_values),
            "delta_system_minus_baseline": delta,
            "cluster_bootstrap_95_ci": [interval[0], interval[1]],
        }
    return result


def _cluster_bootstrap_interval(
    baseline: Sequence[float],
    system: Sequence[float],
    clusters: Sequence[str],
    replicates: int,
    seed: int,
) -> tuple[float, float]:
    grouped: dict[str, list[int]] = {}
    for index, cluster in enumerate(clusters):
        grouped.setdefault(cluster, []).append(index)
    names = sorted(grouped)
    if not names:
        raise ValueError("cannot bootstrap an empty paired population")
    generator = random.Random(seed)
    deltas: list[float] = []
    for _ in range(replicates):
        indices: list[int] = []
        for _ in names:
            indices.extend(grouped[generator.choice(names)])
        deltas.append(
            _mean(tuple(system[index] for index in indices))
            - _mean(tuple(baseline[index] for index in indices))
        )
    deltas.sort()
    return _percentile(deltas, 0.025), _percentile(deltas, 0.975)


def _mcnemar_exact(pairs: Sequence[tuple[bool, bool]]) -> float:
    left_only = sum(left and not right for left, right in pairs)
    right_only = sum(right and not left for left, right in pairs)
    return _two_sided_binomial(left_only, right_only)


def _paired_sign_exact(differences: Sequence[float]) -> float:
    positive = sum(value > 0.0 for value in differences)
    negative = sum(value < 0.0 for value in differences)
    return _two_sided_binomial(positive, negative)


def _two_sided_binomial(left: int, right: int) -> float:
    total = left + right
    if total == 0:
        return 1.0
    smaller = min(left, right)
    tail = sum(math.comb(total, index) for index in range(smaller + 1)) / (2**total)
    return min(1.0, 2.0 * tail)


def _percentile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ValueError("percentile requires values")
    position = probability * (len(values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return values[lower]
    fraction = position - lower
    return values[lower] * (1.0 - fraction) + values[upper] * fraction


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("mean requires values")
    return sum(values) / len(values)
