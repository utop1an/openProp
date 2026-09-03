from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
from pathlib import Path
from typing import Any, Iterable

from openprop.advanced_survival_evaluation import evaluate_survival_advanced
from openprop.non_affine_misspecification import (
    NON_AFFINE_MISSPECIFICATION_CONDITIONS,
    affected_contexts,
    non_affine_misspecification_models,
)
from openprop.persistence_data import PersistenceTrainingExample
from openprop.statistical_persistence import (
    FactorizedExponentialPersistenceModel,
    PerContextExponentialPersistenceModel,
)
from openprop.target_adaptation import (
    RiskModel,
    build_target_calibration_protocol,
    fit_log_risk_affine_adapter,
)
from openprop.target_adaptation_stress import (
    corrupt_calibration_event_labels,
    target_adaptation_stress_data,
)
from openprop.target_interaction_adaptation import (
    DEFAULT_TYPED_PARTITIONS,
    fit_hierarchical_typed_interaction_gate,
)


METHODS = (
    "correct_source_reference",
    "deployed_source",
    "unrestricted_global_affine",
    "controlled_typed",
    "controlled_typed_bic",
    "target_per_context",
)
SUBSETS = ("all", "affected", "stable")
METRICS = (
    "negative_log_likelihood",
    "concordance_index",
    "integrated_brier_score",
)


def _metrics(
    model: RiskModel,
    rows: tuple[PersistenceTrainingExample, ...],
    horizons: tuple[float, ...],
) -> dict[str, float]:
    report = evaluate_survival_advanced(model, rows, horizons_hours=horizons)
    return {
        "negative_log_likelihood": report.negative_log_likelihood,
        "concordance_index": report.concordance_index,
        "integrated_brier_score": report.integrated_brier_score,
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


def _split_rows(
    rows: Iterable[PersistenceTrainingExample],
    affected: frozenset[tuple[str, ...]],
) -> dict[str, tuple[PersistenceTrainingExample, ...]]:
    all_rows = tuple(rows)
    return {
        "all": all_rows,
        "affected": tuple(row for row in all_rows if row.features() in affected),
        "stable": tuple(row for row in all_rows if row.features() not in affected),
    }


def _stable_hazard_delta(
    model: RiskModel,
    source: RiskModel,
    stable_contexts: Iterable[tuple[str, ...]],
) -> float | None:
    deltas = [
        abs(model.hazard_per_hour(features) - source.hazard_per_hour(features))
        for features in stable_contexts
    ]
    return max(deltas) if deltas else None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stress typed target adaptation under local non-affine source errors."
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[31, 41, 53, 67, 79, 97, 109, 127, 149, 173],
    )
    parser.add_argument("--samples-per-context", type=int, default=32)
    parser.add_argument(
        "--calibration-sizes-per-context", type=int, nargs="+", default=[4, 8, 16]
    )
    parser.add_argument("--noise-fractions", type=float, nargs="+", default=[0.0, 0.2])
    parser.add_argument(
        "--evaluation-horizons", type=float, nargs="+", default=[1.0, 4.0, 8.0, 12.0]
    )
    parser.add_argument("--source-epochs", type=int, default=1200)
    parser.add_argument("--adapter-epochs", type=int, default=800)
    parser.add_argument("--familywise-alpha", type=float, default=0.05)
    parser.add_argument("--target-prior-exposure-hours", type=float, default=1.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/non_affine_adaptation_stress_results.json"),
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
        or not noises
        or noises != sorted(set(noises))
        or any(not 0.0 <= value < 0.5 for value in noises)
        or list(horizons) != sorted(set(horizons))
        or any(value <= 0.0 or value >= 16.0 for value in horizons)
        or args.source_epochs <= 0
        or args.adapter_epochs <= 0
        or not 0.0 < args.familywise_alpha < 1.0
        or args.target_prior_exposure_hours < 0.0
    ):
        parser.error("invalid seeds, samples, noise, horizons, or optimizer settings")

    runs: list[dict[str, Any]] = []
    for seed in args.seeds:
        dataset = target_adaptation_stress_data(
            samples_per_context=args.samples_per_context,
            seed=seed,
        )
        source = FactorizedExponentialPersistenceModel.fit(
            dataset.train, epochs=args.source_epochs
        )
        source.calibrate(dataset.validation)
        models = non_affine_misspecification_models(source)
        if set(models) != set(NON_AFFINE_MISSPECIFICATION_CONDITIONS):
            raise RuntimeError("non-affine condition registry does not match models")
        protocol = build_target_calibration_protocol(
            dataset,  # type: ignore[arg-type]
            max_calibration_per_context=sizes[-1],
            split_seed=seed + 13_000_003,
        )
        test_rows = protocol.tests["in_distribution"]
        context_features = tuple(context.features() for context in dataset.contexts)
        source_metrics = _metrics(source, test_rows, horizons)
        for size in sizes:
            clean = protocol.calibration_subset("in_distribution", size)
            for noise in noises:
                calibration = corrupt_calibration_event_labels(
                    clean,
                    fraction=noise,
                    seed=seed + 14_000_019,
                )
                if {row.group_id for row in calibration} & {
                    row.group_id for row in test_rows
                }:
                    raise RuntimeError("target calibration and test identities overlap")
                target_only = PerContextExponentialPersistenceModel.fit(
                    calibration,
                    prior_exposure_hours=args.target_prior_exposure_hours,
                )
                for condition, deployed in models.items():
                    affected = affected_contexts(deployed, source, context_features)
                    row_subsets = _split_rows(test_rows, affected)
                    unrestricted = fit_log_risk_affine_adapter(
                        deployed,
                        calibration,
                        fit_slope=True,
                        epochs=args.adapter_epochs,
                    )
                    controlled = fit_hierarchical_typed_interaction_gate(
                        deployed,
                        calibration,
                        split_seed=seed + 15_000_037,
                        familywise_alpha=args.familywise_alpha,
                        activation_scope="any_predictive_gain",
                        epochs=args.adapter_epochs,
                    )
                    controlled_bic = fit_hierarchical_typed_interaction_gate(
                        deployed,
                        calibration,
                        split_seed=seed + 15_000_037,
                        familywise_alpha=args.familywise_alpha,
                        activation_scope="any_predictive_gain",
                        discovery_complexity="bic",
                        epochs=args.adapter_epochs,
                    )
                    fitted: dict[str, RiskModel] = {
                        "correct_source_reference": source,
                        "deployed_source": deployed,
                        "unrestricted_global_affine": unrestricted,
                        "controlled_typed": controlled,
                        "controlled_typed_bic": controlled_bic,
                        "target_per_context": target_only,
                    }
                    metrics: dict[str, dict[str, dict[str, float] | None]] = {}
                    for method, model in fitted.items():
                        metrics[method] = {}
                        for subset, rows in row_subsets.items():
                            metrics[method][subset] = (
                                _metrics(model, rows, horizons) if rows else None
                            )
                    stable_contexts = tuple(
                        features for features in context_features if features not in affected
                    )
                    runs.append(
                        {
                            "seed": seed,
                            "condition": condition,
                            "calibration_samples_per_context": size,
                            "calibration_examples": len(calibration),
                            "noise_fraction": noise,
                            "test_examples": len(test_rows),
                            "affected_contexts": [list(value) for value in sorted(affected)],
                            "affected_test_examples": len(row_subsets["affected"]),
                            "stable_test_examples": len(row_subsets["stable"]),
                            "metrics": metrics,
                            "max_abs_hazard_delta_vs_correct_on_stable": {
                                method: _stable_hazard_delta(
                                    model, source, stable_contexts
                                )
                                for method, model in fitted.items()
                            },
                            "activation": {
                                "unrestricted_slope": unrestricted.slope,
                                "controlled_partition": _partition_name(
                                    controlled.selected_partition
                                ),
                                "controlled_groups": sorted(
                                    _group_name(group)
                                    for group in controlled.significant_groups
                                ),
                                "controlled_bic_partition": _partition_name(
                                    controlled_bic.selected_partition
                                ),
                                "controlled_bic_groups": sorted(
                                    _group_name(group)
                                    for group in controlled_bic.significant_groups
                                ),
                                "candidate_groups": controlled.candidate_group_count,
                                "bonferroni_threshold": controlled.bonferroni_threshold,
                            },
                        }
                    )
                    print(
                        f"seed={seed} condition={condition:29s} k={size:2d} "
                        f"noise={noise:.1f} affected={len(affected):2d} "
                        f"global={unrestricted.slope:+.2f} "
                        f"typed={_partition_name(controlled.selected_partition):8s} "
                        f"bic={_partition_name(controlled_bic.selected_partition):8s}"
                    )

    def values(
        condition: str,
        size: int,
        noise: float,
        method: str,
        subset: str,
        metric: str,
    ) -> list[float]:
        result: list[float] = []
        for run in runs:
            if (
                run["condition"] != condition
                or run["calibration_samples_per_context"] != size
                or run["noise_fraction"] != noise
            ):
                continue
            row = run["metrics"][method][subset]
            if row is not None:
                result.append(float(row[metric]))
        return result

    aggregate: dict[str, Any] = {}
    paired: dict[str, Any] = {}
    for condition in NON_AFFINE_MISSPECIFICATION_CONDITIONS:
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
                        metric_values = {
                            metric: values(
                                condition, size, noise, method, subset, metric
                            )
                            for metric in METRICS
                        }
                        aggregate[condition][str(size)][noise_key][method][subset] = (
                            {
                                metric: _summary(items)
                                for metric, items in metric_values.items()
                            }
                            if all(metric_values.values())
                            else None
                        )
                for method in METHODS:
                    if method == "deployed_source":
                        continue
                    paired[condition][str(size)][noise_key][method] = {}
                    for subset in SUBSETS:
                        paired[condition][str(size)][noise_key][method][subset] = {}
                        for metric in METRICS:
                            baseline = values(
                                condition,
                                size,
                                noise,
                                "deployed_source",
                                subset,
                                metric,
                            )
                            adapted = values(
                                condition, size, noise, method, subset, metric
                            )
                            if not baseline or not adapted:
                                paired[condition][str(size)][noise_key][method][subset] = None
                                break
                            deltas = [
                                new - old
                                if metric == "concordance_index"
                                else old - new
                                for old, new in zip(baseline, adapted, strict=True)
                            ]
                            paired[condition][str(size)][noise_key][method][subset][metric] = (
                                _paired_summary(
                                    deltas,
                                    key=(
                                        f"{condition}|{size}|{noise}|{method}|"
                                        f"{subset}|{metric}"
                                    ),
                                )
                            )

    payload = {
        "protocol": {
            "seeds": args.seeds,
            "conditions": NON_AFFINE_MISSPECIFICATION_CONDITIONS,
            "samples_per_context": args.samples_per_context,
            "target_mechanism": (
                "in-distribution target rows shared by all frozen deployed-source warps"
            ),
            "target_context_count": len(dataset.contexts),
            "calibration_sizes_per_context": sizes,
            "noise_fractions": noises,
            "fixed_test_examples_per_condition": len(test_rows),
            "candidate_partitions": [list(value) for value in DEFAULT_TYPED_PARTITIONS],
            "familywise_alpha": args.familywise_alpha,
            "split": (
                "outcome-independent nested calibration and group-disjoint fixed test; "
                "typed gates use identity-disjoint discovery and confirmation"
            ),
            "pairing": (
                "all deployment warps share source training, calibration rows, test rows, "
                "event draws, censoring, and optimizer settings"
            ),
            "affected_subset": (
                "computed from deployed-versus-source predictions before target outcomes; "
                "never selected from evaluation loss"
            ),
            "noise": "deterministic calibration-only event-status flips; clean target test",
            "inference": "paired seed-cluster 20000-resample bootstrap and exact sign test",
            "claim_scope": "synthetic local non-affine source-misspecification stress test",
        },
        "aggregate": aggregate,
        "paired_delta_vs_deployed_source": paired,
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
