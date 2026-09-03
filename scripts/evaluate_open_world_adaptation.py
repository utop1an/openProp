from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from pathlib import Path

from openprop.advanced_survival_evaluation import evaluate_survival_advanced
from openprop.open_world_adaptation import (
    OPEN_WORLD_ADAPTATION_CONDITIONS,
    OPEN_WORLD_PAIRWISE_PARTITIONS,
    OPEN_WORLD_TRIPLE_PARTITIONS,
    open_world_adaptation_data,
)
from openprop.statistical_persistence import (
    FactorizedExponentialPersistenceModel,
    PerContextExponentialPersistenceModel,
)
from openprop.target_adaptation import (
    build_target_calibration_protocol,
    fit_log_risk_affine_adapter,
)
from openprop.target_adaptation_stress import corrupt_calibration_event_labels
from openprop.target_interaction_adaptation import (
    fit_hierarchical_typed_interaction_gate,
)


METHODS = (
    "deployed_source",
    "unrestricted_global_affine",
    "pairwise_hierarchy",
    "three_way_hierarchy",
    "three_way_no_complexity_screen",
    "declared_triple_ablation",
    "target_per_context",
    "target_hazard_oracle",
)
METRICS = (
    "negative_log_likelihood",
    "concordance_index",
    "integrated_brier_score",
)


def _metrics(model, rows, horizons):
    result = evaluate_survival_advanced(model, rows, horizons_hours=horizons)
    return {
        "negative_log_likelihood": result.negative_log_likelihood,
        "concordance_index": result.concordance_index,
        "integrated_brier_score": result.integrated_brier_score,
    }


def _partition_name(partition):
    if partition is None:
        return "inactive"
    return "global" if not partition else "x".join(f"f{index}" for index in partition)


def _group_name(group):
    return "global" if not group else "|".join(group)


def _summary(values):
    return {
        "mean": statistics.fmean(values),
        "standard_deviation": statistics.stdev(values) if len(values) > 1 else 0.0,
        "minimum": min(values),
        "maximum": max(values),
    }


def _bootstrap_ci(values, *, seed, repetitions=20_000):
    rng = random.Random(seed)
    means = sorted(
        statistics.fmean(rng.choice(values) for _ in values)
        for _ in range(repetitions)
    )
    return means[round(0.025 * (repetitions - 1))], means[round(0.975 * (repetitions - 1))]


def _sign_p(wins, losses):
    count = wins + losses
    if count == 0:
        return 1.0
    tail = sum(math.comb(count, index) for index in range(0, min(wins, losses) + 1))
    return min(1.0, 2.0 * tail / (2**count))


def _paired_summary(values, *, seed):
    lower, upper = _bootstrap_ci(values, seed=seed)
    wins = sum(value > 1e-12 for value in values)
    losses = sum(value < -1e-12 for value in values)
    return {
        **_summary(values),
        "bootstrap_95_ci_lower": lower,
        "bootstrap_95_ci_upper": upper,
        "wins": wins,
        "losses": losses,
        "ties": len(values) - wins - losses,
        "two_sided_exact_sign_p": _sign_p(wins, losses),
    }


def _metric_delta(method_value, baseline_value, metric):
    if metric == "concordance_index":
        return method_value - baseline_value
    return baseline_value - method_value


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate typed adaptation with novel values and three-way shifts."
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[31, 41, 53, 67, 79, 97, 109, 127, 149, 173],
    )
    parser.add_argument("--samples-per-context", type=int, default=96)
    parser.add_argument(
        "--calibration-sizes-per-context",
        type=int,
        nargs="+",
        default=[12, 24, 48],
    )
    parser.add_argument("--source-epochs", type=int, default=500)
    parser.add_argument("--adapter-epochs", type=int, default=250)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/open_world_adaptation_results.json"),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    sizes = tuple(sorted(set(args.calibration_sizes_per_context)))
    if (
        not args.seeds
        or not sizes
        or sizes[0] <= 1
        or args.samples_per_context <= sizes[-1]
        or args.source_epochs <= 0
        or args.adapter_epochs <= 0
    ):
        raise ValueError("invalid seeds, sample sizes, or optimizer epochs")

    horizons = (1.0, 4.0, 8.0, 12.0)
    runs = []
    for seed in args.seeds:
        dataset = open_world_adaptation_data(
            samples_per_context=args.samples_per_context,
            seed=seed,
        )
        source = FactorizedExponentialPersistenceModel.fit(
            dataset.train,
            epochs=args.source_epochs,
        )
        source.calibrate(dataset.validation)
        protocol = build_target_calibration_protocol(
            dataset,  # type: ignore[arg-type]
            max_calibration_per_context=sizes[-1],
            split_seed=seed + 37_000_003,
            calibration_contexts=dataset.calibration_contexts,
        )
        for condition in OPEN_WORLD_ADAPTATION_CONDITIONS:
            test = protocol.tests[condition]
            support_rows = {
                support: tuple(
                    row
                    for row in test
                    if dataset.support_by_context[row.features()] == support
                )
                for support in sorted(set(dataset.support_by_context.values()))
            }
            changed = dataset.changed_contexts[condition]
            changed_rows = tuple(row for row in test if row.features() in changed)
            stable_rows = tuple(row for row in test if row.features() not in changed)
            oracle_hazards = dataset.test_hazards[condition]
            oracle = PerContextExponentialPersistenceModel(
                oracle_hazards,
                statistics.fmean(oracle_hazards.values()),
                frozenset({"location"}),
            )
            for size in sizes:
                clean = protocol.calibration_subset(condition, size)
                noise_levels = (0.0, 0.2) if size == sizes[-1] else (0.0,)
                for noise in noise_levels:
                    calibration = (
                        clean
                        if noise == 0.0
                        else corrupt_calibration_event_labels(
                            clean,
                            fraction=noise,
                            seed=seed + size * 1009 + 43_000_003,
                        )
                    )
                    unrestricted = fit_log_risk_affine_adapter(
                        source,
                        calibration,
                        fit_slope=True,
                        epochs=args.adapter_epochs,
                    )
                    pairwise = fit_hierarchical_typed_interaction_gate(
                        source,
                        calibration,
                        split_seed=seed + 41_000_003,
                        candidate_partitions=OPEN_WORLD_PAIRWISE_PARTITIONS,
                        activation_scope="any_predictive_gain",
                        discovery_complexity="bic",
                        epochs=args.adapter_epochs,
                    )
                    three_way = fit_hierarchical_typed_interaction_gate(
                        source,
                        calibration,
                        split_seed=seed + 41_000_003,
                        candidate_partitions=OPEN_WORLD_TRIPLE_PARTITIONS,
                        activation_scope="any_predictive_gain",
                        discovery_complexity="bic",
                        epochs=args.adapter_epochs,
                    )
                    three_way_no_screen = fit_hierarchical_typed_interaction_gate(
                        source,
                        calibration,
                        split_seed=seed + 41_000_003,
                        candidate_partitions=OPEN_WORLD_TRIPLE_PARTITIONS,
                        activation_scope="any_predictive_gain",
                        discovery_complexity="none",
                        epochs=args.adapter_epochs,
                    )
                    declared_triple = fit_hierarchical_typed_interaction_gate(
                        source,
                        calibration,
                        split_seed=seed + 41_000_003,
                        candidate_partitions=((), (1, 3, 4)),
                        activation_scope="any_predictive_gain",
                        discovery_complexity="bic",
                        epochs=args.adapter_epochs,
                    )
                    target_only = PerContextExponentialPersistenceModel.fit(calibration)
                    models = {
                        "deployed_source": source,
                        "unrestricted_global_affine": unrestricted,
                        "pairwise_hierarchy": pairwise,
                        "three_way_hierarchy": three_way,
                        "three_way_no_complexity_screen": three_way_no_screen,
                        "declared_triple_ablation": declared_triple,
                        "target_per_context": target_only,
                        "target_hazard_oracle": oracle,
                    }
                    metrics = {
                        name: _metrics(model, test, horizons)
                        for name, model in models.items()
                    }
                    metrics_by_support = {
                        support: {
                            name: _metrics(model, rows, horizons)
                            for name, model in models.items()
                        }
                        for support, rows in support_rows.items()
                    }
                    metrics_by_change = {
                        "changed": (
                            {
                                name: _metrics(model, changed_rows, horizons)
                                for name, model in models.items()
                            }
                            if changed_rows
                            else None
                        ),
                        "stable": {
                            name: _metrics(model, stable_rows, horizons)
                            for name, model in models.items()
                        },
                    }
                    runs.append(
                        {
                            "seed": seed,
                            "condition": condition,
                            "calibration_samples_per_context": size,
                            "calibration_examples": len(calibration),
                            "noise_fraction": noise,
                            "test_examples": len(test),
                            "test_only_contexts": len(protocol.test_only_contexts),
                            "metrics": metrics,
                            "metrics_by_support": metrics_by_support,
                            "metrics_by_change": metrics_by_change,
                            "activation": {
                                "pairwise_partition": _partition_name(pairwise.selected_partition),
                                "pairwise_groups": sorted(_group_name(group) for group in pairwise.significant_groups),
                                "three_way_partition": _partition_name(three_way.selected_partition),
                                "three_way_groups": sorted(_group_name(group) for group in three_way.significant_groups),
                                "three_way_no_screen_partition": _partition_name(three_way_no_screen.selected_partition),
                                "three_way_no_screen_groups": sorted(_group_name(group) for group in three_way_no_screen.significant_groups),
                                "declared_triple_partition": _partition_name(declared_triple.selected_partition),
                                "declared_triple_groups": sorted(_group_name(group) for group in declared_triple.significant_groups),
                                "three_way_candidate_groups": three_way.candidate_group_count,
                                "three_way_bonferroni_threshold": three_way.bonferroni_threshold,
                                "three_way_discovery_bic_active": three_way.discovery_bic_active,
                                "three_way_group_heterogeneity_veto": three_way.group_heterogeneity_veto,
                                "three_way_partition_heterogeneity_veto": three_way.partition_heterogeneity_veto,
                                "three_way_partition_source_p_values": three_way.partition_source_p_values,
                                "three_way_predictive_veto_p_values": three_way.partition_predictive_veto_p_values,
                            },
                        }
                    )
                    print(
                        f"seed={seed} condition={condition:44s} k={size:2d} "
                        f"noise={noise:.1f} pair={_partition_name(pairwise.selected_partition):8s} "
                        f"triple={_partition_name(three_way.selected_partition):10s} "
                        f"declared={_partition_name(declared_triple.selected_partition):10s}"
                    )

    aggregate = {}
    paired = {baseline: {} for baseline in ("deployed_source", "pairwise_hierarchy", "target_per_context")}
    for condition in OPEN_WORLD_ADAPTATION_CONDITIONS:
        aggregate[condition] = {}
        for baseline in paired:
            paired[baseline][condition] = {}
        for size in sizes:
            aggregate[condition][str(size)] = {}
            for baseline in paired:
                paired[baseline][condition][str(size)] = {}
            for noise in ((0.0, 0.2) if size == sizes[-1] else (0.0,)):
                key = str(noise)
                selected = [
                    run for run in runs
                    if run["condition"] == condition
                    and run["calibration_samples_per_context"] == size
                    and run["noise_fraction"] == noise
                ]
                aggregate[condition][str(size)][key] = {
                    method: {
                        metric: _summary([run["metrics"][method][metric] for run in selected])
                        for metric in METRICS
                    }
                    for method in METHODS
                }
                for baseline_index, baseline in enumerate(paired):
                    paired[baseline][condition][str(size)][key] = {}
                    for method_index, method in enumerate(METHODS):
                        paired[baseline][condition][str(size)][key][method] = {}
                        for metric_index, metric in enumerate(METRICS):
                            deltas = [
                                _metric_delta(
                                    run["metrics"][method][metric],
                                    run["metrics"][baseline][metric],
                                    metric,
                                )
                                for run in selected
                            ]
                            paired[baseline][condition][str(size)][key][method][metric] = _paired_summary(
                                deltas,
                                seed=(
                                    53_000_003
                                    + baseline_index * 1_000_003
                                    + list(OPEN_WORLD_ADAPTATION_CONDITIONS).index(condition) * 100_003
                                    + size * 1009
                                    + round(noise * 100) * 101
                                    + method_index * 17
                                    + metric_index
                                ),
                            )

    payload = {
        "protocol": {
            "claim_scope": "synthetic open-world support and three-way adaptation audit",
            "seeds": args.seeds,
            "samples_per_context": args.samples_per_context,
            "calibration_sizes_per_context": list(sizes),
            "calibration_contexts": len(dataset.calibration_contexts),
            "test_only_contexts": len(dataset.test_only_contexts),
            "noise_policy": "clean at every size; 20% deterministic calibration-only flips at maximum size",
            "candidate_partitions_pairwise": [list(partition) for partition in OPEN_WORLD_PAIRWISE_PARTITIONS],
            "candidate_partitions_three_way": [list(partition) for partition in OPEN_WORLD_TRIPLE_PARTITIONS],
            "discovery_complexity": "BIC/MDL screen; unscreened three-way hierarchy reported as an ablation",
            "discovery_fraction": "one third within each calibration context",
            "confirmation_fraction": "two thirds within each calibration context",
            "test_selection": "same group-disjoint test within seed; test-only plate contexts never enter calibration",
            "inference": "20,000 deterministic seed-cluster bootstrap resamples and exact sign tests",
        },
        "aggregate": aggregate,
        "paired_delta": paired,
        "runs": runs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"report: {args.output}")


if __name__ == "__main__":
    main()
