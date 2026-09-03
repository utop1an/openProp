from __future__ import annotations

import math
import random
from collections.abc import Sequence

from .visual_evaluation import VisualEvaluationDataset, VisualQueryResult


def primary_visual_query_comparisons(
    dataset: VisualEvaluationDataset,
    *,
    main_system: str,
    baselines: Sequence[str],
    split: str,
    bootstrap_replicates: int = 10_000,
    seed: int = 20260901,
) -> dict[str, object]:
    """Shared-resample, family-wise inference for preregistered query baselines."""

    names = (main_system, *baselines)
    if any(not name.strip() for name in names) or len(names) != len(set(names)):
        raise ValueError("primary visual systems must be distinct and non-empty")
    if not baselines:
        raise ValueError("primary visual comparison requires baselines")
    if split not in {"development", "calibration", "test"}:
        raise ValueError("primary visual comparison split is invalid")
    if bootstrap_replicates < 100:
        raise ValueError("at least 100 bootstrap replicates are required")
    aligned = _align_query_rows(dataset.queries, names, split)
    keys = tuple(sorted(aligned[main_system]))
    clusters = tuple(key[0] for key in keys)
    main_top1 = tuple(float(aligned[main_system][key].top1_correct) for key in keys)
    main_mrr = tuple(aligned[main_system][key].reciprocal_rank for key in keys)
    top1_deltas = {
        baseline: tuple(
            main_top1[index] - float(aligned[baseline][key].top1_correct)
            for index, key in enumerate(keys)
        )
        for baseline in baselines
    }
    mrr_deltas = {
        baseline: tuple(
            main_mrr[index] - aligned[baseline][key].reciprocal_rank
            for index, key in enumerate(keys)
        )
        for baseline in baselines
    }
    sampled_top1 = _shared_cluster_bootstrap(
        top1_deltas, clusters, bootstrap_replicates, seed
    )
    simultaneous = _simultaneous_intervals(top1_deltas, sampled_top1)
    raw_p = {
        baseline: _mcnemar_exact(
            tuple(
                (
                    aligned[baseline][key].top1_correct,
                    aligned[main_system][key].top1_correct,
                )
                for key in keys
            )
        )
        for baseline in baselines
    }
    adjusted_p = _holm_adjust(raw_p)
    comparisons = {}
    for baseline in baselines:
        mrr_interval = _cluster_percentile_interval(
            mrr_deltas[baseline], clusters, bootstrap_replicates, seed + 100 + len(comparisons)
        )
        comparisons[baseline] = {
            "top1": {
                "baseline": _mean(tuple(float(aligned[baseline][key].top1_correct) for key in keys)),
                "main_system": _mean(main_top1),
                "delta_main_minus_baseline": _mean(top1_deltas[baseline]),
                "familywise_simultaneous_95_ci": list(simultaneous[baseline]),
                "mcnemar_exact_p": raw_p[baseline],
                "holm_adjusted_p": adjusted_p[baseline],
            },
            "mrr": {
                "baseline": _mean(tuple(aligned[baseline][key].reciprocal_rank for key in keys)),
                "main_system": _mean(main_mrr),
                "delta_main_minus_baseline": _mean(mrr_deltas[baseline]),
                "cluster_bootstrap_95_ci": list(mrr_interval),
                "paired_sign_exact_p": _sign_exact(mrr_deltas[baseline]),
            },
        }
    return {
        "schema_version": 1,
        "split": split,
        "main_system": main_system,
        "baselines": list(baselines),
        "population": len(keys),
        "clusters": len(set(clusters)),
        "pairing": ["cluster_id", "record_id"],
        "bootstrap": {
            "unit": "cluster_id", "replicates": bootstrap_replicates, "seed": seed,
            "shared_resamples_across_primary_family": True,
            "simultaneous_method": "max studentized centered bootstrap deviation",
        },
        "primary_family": "query_top1_all_cases",
        "comparisons": comparisons,
    }


def _align_query_rows(rows, systems, split):
    selected = {system: {} for system in systems}
    for row in rows:
        if row.split != split or row.system not in selected:
            continue
        key = (row.cluster_id, row.record_id)
        if key in selected[row.system]:
            raise ValueError(f"duplicate primary query row for {row.system}: {key}")
        selected[row.system][key] = row
    reference = set(selected[systems[0]])
    if not reference or any(set(selected[name]) != reference for name in systems[1:]):
        raise ValueError("primary query populations are not exactly paired")
    for key in sorted(reference):
        signature = _signature(selected[systems[0]][key])
        if any(_signature(selected[name][key]) != signature for name in systems[1:]):
            raise ValueError(f"primary query truth/input fields drifted: {key}")
    return selected


def _signature(row: VisualQueryResult):
    return (
        row.source, row.property_name, row.target_entity_id, row.candidate_entity_ids,
        row.horizon_seconds, row.distractor_count, row.condition, row.eligible,
    )


def _shared_cluster_bootstrap(differences, clusters, replicates, seed):
    grouped = {}
    for index, cluster in enumerate(clusters):
        grouped.setdefault(cluster, []).append(index)
    names = sorted(grouped)
    generator = random.Random(seed)
    output = {name: [] for name in differences}
    for _ in range(replicates):
        indices = []
        for _ in names:
            indices.extend(grouped[generator.choice(names)])
        for name, values in differences.items():
            output[name].append(_mean(tuple(values[index] for index in indices)))
    return output


def _simultaneous_intervals(differences, sampled):
    points = {name: _mean(values) for name, values in differences.items()}
    deviations = {
        name: _sample_sd(tuple(value - points[name] for value in values))
        for name, values in sampled.items()
    }
    maxima = []
    replicates = len(next(iter(sampled.values())))
    for index in range(replicates):
        standardized = [
            abs(sampled[name][index] - points[name]) / deviations[name]
            for name in sampled if deviations[name] > 0.0
        ]
        maxima.append(max(standardized, default=0.0))
    critical = _percentile(sorted(maxima), 0.95)
    return {
        name: (points[name] - critical * deviations[name], points[name] + critical * deviations[name])
        for name in differences
    }


def _cluster_percentile_interval(values, clusters, replicates, seed):
    sampled = _shared_cluster_bootstrap({"metric": values}, clusters, replicates, seed)["metric"]
    sampled.sort()
    return _percentile(sampled, 0.025), _percentile(sampled, 0.975)


def _holm_adjust(p_values):
    ordered = sorted(p_values, key=lambda name: (p_values[name], name))
    adjusted = {}
    running = 0.0
    count = len(ordered)
    for rank, name in enumerate(ordered):
        running = max(running, min(1.0, (count - rank) * p_values[name]))
        adjusted[name] = running
    return adjusted


def _mcnemar_exact(pairs):
    left_only = sum(left and not right for left, right in pairs)
    right_only = sum(right and not left for left, right in pairs)
    return _binomial_exact(left_only, right_only)


def _sign_exact(values):
    return _binomial_exact(sum(value < 0 for value in values), sum(value > 0 for value in values))


def _binomial_exact(left, right):
    total = left + right
    if total == 0:
        return 1.0
    tail = sum(math.comb(total, index) for index in range(min(left, right) + 1)) / (2**total)
    return min(1.0, 2.0 * tail)


def _sample_sd(values):
    if len(values) < 2:
        return 0.0
    center = _mean(values)
    return math.sqrt(sum((value - center) ** 2 for value in values) / (len(values) - 1))


def _percentile(values, probability):
    position = probability * (len(values) - 1)
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return values[lower]
    fraction = position - lower
    return values[lower] * (1 - fraction) + values[upper] * fraction


def _mean(values):
    return sum(values) / len(values)
