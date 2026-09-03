from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
from pathlib import Path

from openprop.advanced_survival_evaluation import evaluate_survival_advanced
from openprop.statistical_persistence import (
    FactorizedExponentialPersistenceModel,
    PerContextExponentialPersistenceModel,
)
from openprop.target_adaptation import (
    build_target_calibration_protocol,
    fit_log_risk_affine_adapter,
    select_sign_gated_model,
)
from openprop.target_adaptation_stress import (
    STRESS_CONDITIONS,
    corrupt_calibration_event_labels,
    fit_feature_grouped_sign_gate,
    target_adaptation_stress_data,
)
from openprop.target_interaction_adaptation import (
    DEFAULT_TYPED_PARTITIONS,
    fit_hierarchical_typed_interaction_gate,
)


METHODS = (
    "source",
    "global_sign_gate",
    "subject_group_gate",
    "scene_group_gate",
    "hierarchical_interaction_gate",
    "target_per_context",
)
METRICS = (
    "negative_log_likelihood",
    "concordance_index",
    "integrated_brier_score",
)
SUBSETS = ("overall", "changed_contexts", "stable_contexts")


def _metrics(report) -> dict[str, float]:
    return {
        "negative_log_likelihood": report.negative_log_likelihood,
        "concordance_index": report.concordance_index,
        "integrated_brier_score": report.integrated_brier_score,
    }


def _evaluate_subsets(model, rows, changed, horizons):
    partitions = {
        "overall": rows,
        "changed_contexts": tuple(row for row in rows if row.features() in changed),
        "stable_contexts": tuple(row for row in rows if row.features() not in changed),
    }
    return {
        name: (
            _metrics(evaluate_survival_advanced(model, subset, horizons_hours=horizons))
            if subset
            else None
        )
        for name, subset in partitions.items()
    }


def _summary(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "standard_deviation": statistics.pstdev(values),
        "minimum": min(values),
        "maximum": max(values),
    }


def _paired_summary(values: list[float], *, key: str) -> dict[str, float | int]:
    rng = random.Random(
        int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "big")
    )
    means = sorted(
        statistics.fmean(rng.choices(values, k=len(values))) for _ in range(20_000)
    )
    wins = sum(value > 1e-12 for value in values)
    losses = sum(value < -1e-12 for value in values)
    ties = len(values) - wins - losses
    non_ties = wins + losses
    if non_ties:
        tail = min(wins, losses)
        sign_p = min(
            1.0,
            2.0
            * sum(math.comb(non_ties, index) for index in range(tail + 1))
            / 2.0**non_ties,
        )
    else:
        sign_p = 1.0
    return {
        **_summary(values),
        "bootstrap_95_ci_lower": means[499],
        "bootstrap_95_ci_upper": means[19_499],
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "two_sided_exact_sign_p": sign_p,
    }


def _partition_name(partition: tuple[int, ...] | None) -> str:
    if partition is None:
        return "inactive"
    return "global" if not partition else "x".join(f"f{index}" for index in partition)


def _group_name(group: tuple[str, ...]) -> str:
    return "global" if not group else "|".join(group)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate multiplicity-controlled typed interaction adaptation."
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[31, 41, 53, 67, 79, 97, 109, 127, 149, 173],
    )
    parser.add_argument("--samples-per-context", type=int, default=32)
    parser.add_argument(
        "--calibration-sizes-per-context",
        type=int,
        nargs="+",
        default=[2, 4, 8, 16],
    )
    parser.add_argument(
        "--noise-fractions", type=float, nargs="+", default=[0.0, 0.2]
    )
    parser.add_argument(
        "--evaluation-horizons",
        type=float,
        nargs="+",
        default=[1.0, 4.0, 8.0, 12.0],
    )
    parser.add_argument("--source-epochs", type=int, default=1200)
    parser.add_argument("--adapter-epochs", type=int, default=800)
    parser.add_argument("--familywise-alpha", type=float, default=0.05)
    parser.add_argument("--target-prior-exposure-hours", type=float, default=1.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/target_interaction_adaptation_results.json"),
    )
    args = parser.parse_args()
    sizes = args.calibration_sizes_per_context
    noises = args.noise_fractions
    horizons = tuple(args.evaluation_horizons)
    if (
        not args.seeds
        or len(args.seeds) != len(set(args.seeds))
        or not sizes
        or sizes != sorted(set(sizes))
        or sizes[0] < 2
        or args.samples_per_context <= sizes[-1]
        or noises != sorted(set(noises))
        or not noises
        or any(not 0.0 <= value < 0.5 for value in noises)
        or list(horizons) != sorted(set(horizons))
        or any(value <= 0.0 or value >= 16.0 for value in horizons)
        or args.source_epochs <= 0
        or args.adapter_epochs <= 0
        or not 0.0 < args.familywise_alpha < 1.0
    ):
        parser.error("invalid seeds, samples, noise, horizons, or optimizer settings")

    runs: list[dict[str, object]] = []
    for seed in args.seeds:
        dataset = target_adaptation_stress_data(
            samples_per_context=args.samples_per_context,
            seed=seed,
        )
        source = FactorizedExponentialPersistenceModel.fit(
            dataset.train, epochs=args.source_epochs
        )
        source.calibrate(dataset.validation)
        protocol = build_target_calibration_protocol(
            dataset,  # type: ignore[arg-type]
            max_calibration_per_context=sizes[-1],
            split_seed=seed + 9_000_003,
        )
        source_cache = {
            condition: _evaluate_subsets(
                source,
                protocol.tests[condition],
                dataset.changed_contexts[condition],
                horizons,
            )
            for condition in STRESS_CONDITIONS
        }
        for condition in STRESS_CONDITIONS:
            test_rows = protocol.tests[condition]
            changed = dataset.changed_contexts[condition]
            for size in sizes:
                clean = protocol.calibration_subset(condition, size)
                for noise in noises:
                    calibration = corrupt_calibration_event_labels(
                        clean,
                        fraction=noise,
                        seed=seed + 10_000_019,
                    )
                    global_affine = fit_log_risk_affine_adapter(
                        source,
                        calibration,
                        fit_slope=True,
                        epochs=args.adapter_epochs,
                    )
                    global_gate = select_sign_gated_model(source, global_affine)
                    subject_gate = fit_feature_grouped_sign_gate(
                        source,
                        calibration,
                        feature_index=1,
                        epochs=args.adapter_epochs,
                    )
                    scene_gate = fit_feature_grouped_sign_gate(
                        source,
                        calibration,
                        feature_index=4,
                        epochs=args.adapter_epochs,
                    )
                    hierarchy = fit_hierarchical_typed_interaction_gate(
                        source,
                        calibration,
                        split_seed=seed + 12_000_037,
                        familywise_alpha=args.familywise_alpha,
                        epochs=args.adapter_epochs,
                    )
                    target_only = PerContextExponentialPersistenceModel.fit(
                        calibration,
                        prior_exposure_hours=args.target_prior_exposure_hours,
                    )
                    models = {
                        "global_sign_gate": global_gate,
                        "subject_group_gate": subject_gate,
                        "scene_group_gate": scene_gate,
                        "hierarchical_interaction_gate": hierarchy,
                        "target_per_context": target_only,
                    }
                    metrics = {"source": source_cache[condition]}
                    metrics.update(
                        {
                            name: _evaluate_subsets(
                                model, test_rows, changed, horizons
                            )
                            for name, model in models.items()
                        }
                    )
                    runs.append(
                        {
                            "seed": seed,
                            "condition": condition,
                            "calibration_samples_per_context": size,
                            "calibration_examples": len(calibration),
                            "noise_fraction": noise,
                            "test_examples": len(test_rows),
                            "changed_context_count": len(changed),
                            "metrics": metrics,
                            "activation": {
                                "global_activated": global_gate is not source,
                                "subject_activated_groups": sorted(
                                    subject_gate.activated_groups
                                ),
                                "scene_activated_groups": sorted(
                                    scene_gate.activated_groups
                                ),
                                "hierarchical_partition": _partition_name(
                                    hierarchy.selected_partition
                                ),
                                "hierarchical_groups": sorted(
                                    _group_name(group)
                                    for group in hierarchy.significant_groups
                                ),
                                "hierarchical_bonferroni_threshold": (
                                    hierarchy.bonferroni_threshold
                                ),
                                "hierarchical_candidate_groups": (
                                    hierarchy.candidate_group_count
                                ),
                                "hierarchical_partition_gains": (
                                    hierarchy.partition_mean_gains
                                ),
                                "hierarchical_heterogeneity_veto": (
                                    hierarchy.partition_heterogeneity_veto
                                ),
                                "hierarchical_p_bounds": (
                                    hierarchy.confirmation_p_values
                                ),
                            },
                        }
                    )
                    print(
                        f"seed={seed} condition={condition:28s} k={size:2d} "
                        f"noise={noise:.1f} hierarchy={_partition_name(hierarchy.selected_partition):8s} "
                        f"groups={','.join(sorted(_group_name(group) for group in hierarchy.significant_groups)) or '-'}"
                    )

    def values(condition, size, noise, method, subset, metric):
        return [
            float(run["metrics"][method][subset][metric])  # type: ignore[index]
            for run in runs
            if run["condition"] == condition
            and run["calibration_samples_per_context"] == size
            and run["noise_fraction"] == noise
            and run["metrics"][method][subset] is not None  # type: ignore[index]
        ]

    aggregate: dict[str, object] = {}
    paired: dict[str, object] = {}
    for condition in STRESS_CONDITIONS:
        aggregate[condition] = {}
        paired[condition] = {}
        for size in sizes:
            aggregate[condition][str(size)] = {}
            paired[condition][str(size)] = {}
            for noise in noises:
                noise_key = f"{noise:.1f}"
                aggregate[condition][str(size)][noise_key] = {}
                paired[condition][str(size)][noise_key] = {}
                for method in METHODS:
                    aggregate[condition][str(size)][noise_key][method] = {}
                    for subset in SUBSETS:
                        rows_by_metric = {
                            metric: values(
                                condition, size, noise, method, subset, metric
                            )
                            for metric in METRICS
                        }
                        aggregate[condition][str(size)][noise_key][method][subset] = (
                            {
                                metric: _summary(metric_values)
                                for metric, metric_values in rows_by_metric.items()
                            }
                            if rows_by_metric[METRICS[0]]
                            else None
                        )
                for method in METHODS[1:]:
                    paired[condition][str(size)][noise_key][method] = {}
                    for subset in SUBSETS:
                        baseline_nll = values(
                            condition, size, noise, "source", subset, METRICS[0]
                        )
                        if not baseline_nll:
                            paired[condition][str(size)][noise_key][method][subset] = None
                            continue
                        paired[condition][str(size)][noise_key][method][subset] = {}
                        for metric in METRICS:
                            baseline = values(
                                condition, size, noise, "source", subset, metric
                            )
                            adapted = values(
                                condition, size, noise, method, subset, metric
                            )
                            deltas = [
                                new - old
                                if metric == "concordance_index"
                                else old - new
                                for old, new in zip(baseline, adapted, strict=True)
                            ]
                            paired[condition][str(size)][noise_key][method][subset][metric] = _paired_summary(
                                deltas,
                                key=f"{condition}|{size}|{noise}|{method}|{subset}|{metric}",
                            )

    payload = {
        "protocol": {
            "seeds": args.seeds,
            "conditions": STRESS_CONDITIONS,
            "samples_per_context": args.samples_per_context,
            "target_context_count": len(dataset.contexts),
            "calibration_sizes_per_context": sizes,
            "noise_fractions": noises,
            "fixed_test_examples_per_condition": len(
                protocol.tests["in_distribution"]
            ),
            "candidate_partitions": [list(value) for value in DEFAULT_TYPED_PARTITIONS],
            "familywise_alpha": args.familywise_alpha,
            "multiplicity_control": (
                "identity-disjoint discovery-third fit; predictive likelihood-ratio "
                "e-value on confirmation two-thirds; Bonferroni across all 12 "
                "global, main-effect, and pairwise candidate groups"
            ),
            "hierarchy": (
                "a pooled parent group is vetoed when finer typed cells have mixed "
                "discovery slope signs"
            ),
            "split": (
                "outcome-independent group-id hash; nested calibration subsets; "
                "one fixed group-disjoint test"
            ),
            "noise": (
                "deterministic event-status flips on calibration only; clean test"
            ),
            "subgroups": (
                "generator truth partitions changed/stable test metrics only and "
                "is never available to fitting or model selection"
            ),
            "evaluation_horizons_hours": args.evaluation_horizons,
            "inference": (
                "paired seed-cluster 20000-resample bootstrap and exact sign test"
            ),
            "claim_scope": (
                "synthetic multiplicity-controlled sparse typed interaction adaptation"
            ),
        },
        "aggregate": aggregate,
        "paired_delta_vs_source": paired,
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
