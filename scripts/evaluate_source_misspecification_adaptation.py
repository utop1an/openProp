from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
from pathlib import Path

from openprop.advanced_survival_evaluation import evaluate_survival_advanced
from openprop.source_misspecification import (
    SOURCE_MISSPECIFICATION_CONDITIONS,
    source_misspecification_models,
)
from openprop.statistical_persistence import (
    FactorizedExponentialPersistenceModel,
    PerContextExponentialPersistenceModel,
)
from openprop.target_adaptation import (
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
    "reversal_only_gate",
    "controlled_general_gate",
    "target_per_context",
)
METRICS = (
    "negative_log_likelihood",
    "concordance_index",
    "integrated_brier_score",
)


def _metrics(model, rows, horizons) -> dict[str, float]:
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit target adaptation under deployed-source misspecification."
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
        default=Path("artifacts/source_misspecification_adaptation_results.json"),
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

    runs: list[dict[str, object]] = []
    for seed in args.seeds:
        dataset = target_adaptation_stress_data(
            samples_per_context=args.samples_per_context,
            seed=seed,
        )
        correct_source = FactorizedExponentialPersistenceModel.fit(
            dataset.train, epochs=args.source_epochs
        )
        correct_source.calibrate(dataset.validation)
        deployed_models = source_misspecification_models(correct_source)
        protocol = build_target_calibration_protocol(
            dataset,  # type: ignore[arg-type]
            max_calibration_per_context=sizes[-1],
            split_seed=seed + 9_000_003,
        )
        test_rows = protocol.tests["in_distribution"]
        correct_metrics = _metrics(correct_source, test_rows, horizons)
        for size in sizes:
            clean = protocol.calibration_subset("in_distribution", size)
            for noise in noises:
                calibration = corrupt_calibration_event_labels(
                    clean,
                    fraction=noise,
                    seed=seed + 10_000_019,
                )
                target_only = PerContextExponentialPersistenceModel.fit(
                    calibration,
                    prior_exposure_hours=args.target_prior_exposure_hours,
                )
                target_metrics = _metrics(target_only, test_rows, horizons)
                for condition, deployed_source in deployed_models.items():
                    unrestricted = fit_log_risk_affine_adapter(
                        deployed_source,
                        calibration,
                        fit_slope=True,
                        epochs=args.adapter_epochs,
                    )
                    general_gate = fit_hierarchical_typed_interaction_gate(
                        deployed_source,
                        calibration,
                        split_seed=seed + 12_000_037,
                        familywise_alpha=args.familywise_alpha,
                        activation_scope="any_predictive_gain",
                        epochs=args.adapter_epochs,
                    )
                    reversal_gate = fit_hierarchical_typed_interaction_gate(
                        deployed_source,
                        calibration,
                        split_seed=seed + 12_000_037,
                        familywise_alpha=args.familywise_alpha,
                        epochs=args.adapter_epochs,
                    )
                    metrics = {
                        "correct_source_reference": correct_metrics,
                        "deployed_source": _metrics(
                            deployed_source, test_rows, horizons
                        ),
                        "unrestricted_global_affine": _metrics(
                            unrestricted, test_rows, horizons
                        ),
                        "reversal_only_gate": _metrics(reversal_gate, test_rows, horizons),
                        "controlled_general_gate": _metrics(
                            general_gate, test_rows, horizons
                        ),
                        "target_per_context": target_metrics,
                    }
                    runs.append(
                        {
                            "seed": seed,
                            "condition": condition,
                            "calibration_samples_per_context": size,
                            "calibration_examples": len(calibration),
                            "noise_fraction": noise,
                            "test_examples": len(test_rows),
                            "metrics": metrics,
                            "activation": {
                                "unrestricted_slope": unrestricted.slope,
                                "reversal_partition": _partition_name(
                                    reversal_gate.selected_partition
                                ),
                                "reversal_groups": sorted(
                                    _group_name(group)
                                    for group in reversal_gate.significant_groups
                                ),
                                "controlled_general_partition": _partition_name(
                                    general_gate.selected_partition
                                ),
                                "controlled_general_groups": sorted(
                                    _group_name(group)
                                    for group in general_gate.significant_groups
                                ),
                                "candidate_groups": general_gate.candidate_group_count,
                                "bonferroni_threshold": general_gate.bonferroni_threshold,
                                "controlled_general_heterogeneity_veto": (
                                    general_gate.partition_heterogeneity_veto
                                ),
                                "controlled_general_predictive_veto_p_values": (
                                    general_gate.partition_predictive_veto_p_values
                                ),
                            },
                        }
                    )
                    print(
                        f"seed={seed} condition={condition:27s} k={size:2d} "
                        f"noise={noise:.1f} affine={unrestricted.slope:+.2f} "
                        f"reversal={_partition_name(reversal_gate.selected_partition):8s} "
                        f"general={_partition_name(general_gate.selected_partition):8s} "
                        f"groups={','.join(sorted(_group_name(group) for group in general_gate.significant_groups)) or '-'}"
                    )

    def values(condition, size, noise, method, metric):
        return [
            float(run["metrics"][method][metric])  # type: ignore[index]
            for run in runs
            if run["condition"] == condition
            and run["calibration_samples_per_context"] == size
            and run["noise_fraction"] == noise
        ]

    aggregate: dict[str, object] = {}
    paired: dict[str, object] = {}
    for condition in SOURCE_MISSPECIFICATION_CONDITIONS:
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
                    aggregate[condition][str(size)][noise_key][method] = {
                        metric: _summary(
                            values(condition, size, noise, method, metric)
                        )
                        for metric in METRICS
                    }
                for method in METHODS:
                    if method == "deployed_source":
                        continue
                    paired[condition][str(size)][noise_key][method] = {}
                    for metric in METRICS:
                        baseline = values(
                            condition, size, noise, "deployed_source", metric
                        )
                        adapted = values(condition, size, noise, method, metric)
                        deltas = [
                            new - old
                            if metric == "concordance_index"
                            else old - new
                            for old, new in zip(baseline, adapted, strict=True)
                        ]
                        paired[condition][str(size)][noise_key][method][metric] = _paired_summary(
                            deltas,
                            key=f"{condition}|{size}|{noise}|{method}|{metric}",
                        )

    payload = {
        "protocol": {
            "seeds": args.seeds,
            "conditions": SOURCE_MISSPECIFICATION_CONDITIONS,
            "samples_per_context": args.samples_per_context,
            "target_mechanism": (
                "in-distribution target rows shared by every deployed-source variant"
            ),
            "target_context_count": len(dataset.contexts),
            "calibration_sizes_per_context": sizes,
            "noise_fractions": noises,
            "fixed_test_examples_per_condition": len(test_rows),
            "candidate_partitions": [list(value) for value in DEFAULT_TYPED_PARTITIONS],
            "familywise_alpha": args.familywise_alpha,
            "split": (
                "outcome-independent nested target calibration and fixed group-disjoint "
                "test; both controlled gates use the same additional identity-disjoint "
                "discovery/confirmation split"
            ),
            "pairing": (
                "all source variants share source training, target calibration rows, "
                "target test rows, event draws, censoring, and optimizer settings"
            ),
            "noise": (
                "deterministic event-status flips on calibration only; clean target test"
            ),
            "inference": (
                "paired seed-cluster 20000-resample bootstrap and exact sign test"
            ),
            "claim_scope": "synthetic deployed-source misspecification audit",
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
