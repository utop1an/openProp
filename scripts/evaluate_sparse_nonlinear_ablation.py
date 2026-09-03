from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
from pathlib import Path
from typing import Any

from openprop.advanced_survival_evaluation import evaluate_survival_advanced
from openprop.non_affine_misspecification import (
    NON_AFFINE_MISSPECIFICATION_CONDITIONS,
    affected_contexts,
    non_affine_misspecification_models,
)
from openprop.persistence_data import PersistenceTrainingExample
from openprop.sparse_nonlinear_adaptation import fit_sparse_nonlinear_typed_gate
from openprop.statistical_persistence import FactorizedExponentialPersistenceModel
from openprop.target_adaptation import RiskModel, build_target_calibration_protocol
from openprop.target_adaptation_stress import (
    corrupt_calibration_event_labels,
    target_adaptation_stress_data,
)


METRICS = (
    "negative_log_likelihood",
    "concordance_index",
    "integrated_brier_score",
)
SUBSETS = ("all", "affected", "stable")
BASELINE_METHODS = (
    "deployed_source",
    "controlled_typed_bic",
    "target_per_context",
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


def _key(row: dict[str, Any]) -> tuple[int, str, int, float]:
    return (
        int(row["seed"]),
        str(row["condition"]),
        int(row["calibration_samples_per_context"]),
        float(row["noise_fraction"]),
    )


def _split_rows(
    rows: tuple[PersistenceTrainingExample, ...],
    affected: frozenset[tuple[str, ...]],
) -> dict[str, tuple[PersistenceTrainingExample, ...]]:
    return {
        "all": rows,
        "affected": tuple(row for row in rows if row.features() in affected),
        "stable": tuple(row for row in rows if row.features() not in affected),
    }


def _partition_name(partition: tuple[int, ...] | None) -> str:
    if partition is None:
        return "inactive"
    return "x".join(f"f{index}" for index in partition)


def _group_name(group: tuple[str, ...]) -> str:
    return "|".join(group)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ablate sparse nonlinear typed repair on the frozen stress audit."
    )
    parser.add_argument(
        "--base-artifact",
        type=Path,
        default=Path("artifacts/non_affine_adaptation_stress_results.json"),
    )
    parser.add_argument("--source-epochs", type=int, default=1200)
    parser.add_argument("--adapter-epochs", type=int, default=800)
    parser.add_argument(
        "--evaluation-horizons",
        type=float,
        nargs="+",
        default=[1.0, 4.0, 8.0, 12.0],
    )
    parser.add_argument("--familywise-alpha", type=float, default=0.05)
    parser.add_argument("--max-adapted-context-fraction", type=float, default=0.5)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/sparse_nonlinear_adaptation_results.json"),
    )
    args = parser.parse_args()
    horizons = tuple(args.evaluation_horizons)
    if (
        args.source_epochs <= 0
        or args.adapter_epochs <= 0
        or list(horizons) != sorted(set(horizons))
        or any(value <= 0.0 or value >= 16.0 for value in horizons)
        or not 0.0 < args.familywise_alpha < 1.0
        or not 0.0 < args.max_adapted_context_fraction < 1.0
    ):
        parser.error("invalid optimizer, horizons, alpha, or sparsity settings")

    base_bytes = args.base_artifact.read_bytes()
    base_hash = hashlib.sha256(base_bytes).hexdigest()
    base = json.loads(base_bytes)
    protocol = base.get("protocol", {})
    seeds = [int(value) for value in protocol.get("seeds", [])]
    sizes = [int(value) for value in protocol.get("calibration_sizes_per_context", [])]
    noises = [float(value) for value in protocol.get("noise_fractions", [])]
    samples_per_context = int(protocol.get("samples_per_context", 0))
    expected_conditions = set(NON_AFFINE_MISSPECIFICATION_CONDITIONS)
    if (
        not seeds
        or not sizes
        or not noises
        or samples_per_context <= sizes[-1]
        or set(protocol.get("conditions", {})) != expected_conditions
        or protocol.get("fixed_test_examples_per_condition") != 288
    ):
        raise ValueError("base artifact does not match the frozen non-affine protocol")
    base_by_key = {_key(row): row for row in base.get("runs", [])}
    expected_count = len(seeds) * len(sizes) * len(noises) * len(expected_conditions)
    if len(base_by_key) != expected_count:
        raise ValueError("base artifact run matrix is incomplete or duplicated")

    runs: list[dict[str, Any]] = []
    for seed in seeds:
        dataset = target_adaptation_stress_data(
            samples_per_context=samples_per_context, seed=seed
        )
        source = FactorizedExponentialPersistenceModel.fit(
            dataset.train, epochs=args.source_epochs
        )
        source.calibrate(dataset.validation)
        deployed_models = non_affine_misspecification_models(source)
        target_protocol = build_target_calibration_protocol(
            dataset,  # type: ignore[arg-type]
            max_calibration_per_context=sizes[-1],
            split_seed=seed + 13_000_003,
        )
        test_rows = target_protocol.tests["in_distribution"]
        context_features = tuple(context.features() for context in dataset.contexts)
        for size in sizes:
            clean = target_protocol.calibration_subset("in_distribution", size)
            for noise in noises:
                calibration = corrupt_calibration_event_labels(
                    clean, fraction=noise, seed=seed + 14_000_019
                )
                for condition, deployed in deployed_models.items():
                    run_key = (seed, condition, size, noise)
                    base_run = base_by_key.get(run_key)
                    if base_run is None:
                        raise RuntimeError(f"missing frozen baseline run {run_key}")
                    affected = affected_contexts(
                        deployed, source, context_features
                    )
                    subsets = _split_rows(test_rows, affected)
                    gate = fit_sparse_nonlinear_typed_gate(
                        deployed,
                        calibration,
                        split_seed=seed + 15_000_037,
                        familywise_alpha=args.familywise_alpha,
                        max_adapted_context_fraction=(
                            args.max_adapted_context_fraction
                        ),
                        epochs=args.adapter_epochs,
                    )
                    sparse_metrics = {
                        subset: _metrics(gate, rows, horizons) if rows else None
                        for subset, rows in subsets.items()
                    }
                    baseline_metrics = {
                        method: base_run["metrics"][method]
                        for method in BASELINE_METHODS
                    }
                    significant_labels = {
                        _group_name(group): gate.selected_families.get(
                            (
                                "|".join(
                                    f"f{index}={value}"
                                    for index, value in zip(
                                        gate.selected_partition or (),
                                        group,
                                        strict=True,
                                    )
                                )
                            ),
                            "unknown",
                        )
                        for group in gate.significant_groups
                    }
                    runs.append(
                        {
                            "seed": seed,
                            "condition": condition,
                            "calibration_samples_per_context": size,
                            "calibration_examples": len(calibration),
                            "noise_fraction": noise,
                            "test_examples": len(test_rows),
                            "affected_test_examples": len(subsets["affected"]),
                            "stable_test_examples": len(subsets["stable"]),
                            "metrics": {
                                **baseline_metrics,
                                "sparse_nonlinear_typed": sparse_metrics,
                            },
                            "activation": {
                                "partition": _partition_name(
                                    gate.selected_partition
                                ),
                                "groups": sorted(
                                    _group_name(group)
                                    for group in gate.significant_groups
                                ),
                                "families": significant_labels,
                                "bonferroni_threshold": gate.bonferroni_threshold,
                                "partition_context_fractions": (
                                    gate.partition_context_fractions
                                ),
                            },
                        }
                    )
                    print(
                        f"seed={seed} condition={condition:29s} k={size:2d} "
                        f"noise={noise:.1f} sparse={_partition_name(gate.selected_partition):8s} "
                        f"groups={','.join(sorted(_group_name(group) for group in gate.significant_groups)) or '-'}"
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
            if _key(run) != (
                int(run["seed"]), condition, size, noise
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
                for subset in SUBSETS:
                    metric_values = {
                        metric: values(
                            condition,
                            size,
                            noise,
                            "sparse_nonlinear_typed",
                            subset,
                            metric,
                        )
                        for metric in METRICS
                    }
                    aggregate[condition][str(size)][noise_key][subset] = (
                        {
                            metric: _summary(items)
                            for metric, items in metric_values.items()
                        }
                        if all(metric_values.values())
                        else None
                    )
                for baseline in ("deployed_source", "controlled_typed_bic"):
                    paired[condition][str(size)][noise_key][baseline] = {}
                    for subset in SUBSETS:
                        paired[condition][str(size)][noise_key][baseline][subset] = {}
                        for metric in METRICS:
                            old = values(
                                condition, size, noise, baseline, subset, metric
                            )
                            new = values(
                                condition,
                                size,
                                noise,
                                "sparse_nonlinear_typed",
                                subset,
                                metric,
                            )
                            if not old or not new:
                                paired[condition][str(size)][noise_key][baseline][subset] = None
                                break
                            deltas = [
                                right - left
                                if metric == "concordance_index"
                                else left - right
                                for left, right in zip(old, new, strict=True)
                            ]
                            paired[condition][str(size)][noise_key][baseline][subset][metric] = _paired_summary(
                                deltas,
                                key=(
                                    f"{base_hash}|{condition}|{size}|{noise}|"
                                    f"{baseline}|{subset}|{metric}"
                                ),
                            )

    payload = {
        "protocol": {
            "base_artifact": str(args.base_artifact),
            "base_artifact_sha256": base_hash,
            "seeds": seeds,
            "conditions": NON_AFFINE_MISSPECIFICATION_CONDITIONS,
            "samples_per_context": samples_per_context,
            "calibration_sizes_per_context": sizes,
            "noise_fractions": noises,
            "evaluation_horizons_hours": list(horizons),
            "source_epochs": args.source_epochs,
            "adapter_epochs": args.adapter_epochs,
            "familywise_alpha": args.familywise_alpha,
            "max_adapted_context_fraction": args.max_adapted_context_fraction,
            "selection": (
                "discovery-only BIC among affine, quadratic, and hinge per typed "
                "group; identity-disjoint confirmation with Bonferroni group e-values; "
                "global partition excluded and adapted context fraction capped"
            ),
            "claim_scope": "synthetic sparse nonlinear method ablation",
        },
        "aggregate": aggregate,
        "paired_delta_sparse_vs_baseline": paired,
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
