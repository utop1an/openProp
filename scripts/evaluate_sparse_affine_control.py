from __future__ import annotations

import hashlib
import json
import math
import random
import statistics
from pathlib import Path
from typing import Any

from openprop.advanced_survival_evaluation import evaluate_survival_advanced
from openprop.non_affine_misspecification import (
    affected_contexts,
    non_affine_misspecification_models,
)
from openprop.statistical_persistence import FactorizedExponentialPersistenceModel
from openprop.target_adaptation import build_target_calibration_protocol
from openprop.target_adaptation_stress import (
    corrupt_calibration_event_labels,
    target_adaptation_stress_data,
)
from openprop.target_interaction_adaptation import (
    HierarchicalTypedInteractionGate,
    fit_hierarchical_typed_interaction_gate,
)


BASE_PATH = Path("artifacts/non_affine_adaptation_stress_results.json")
NONLINEAR_PATH = Path("artifacts/sparse_nonlinear_adaptation_results.json")
OUTPUT_PATH = Path("artifacts/sparse_affine_control_results.json")
PARTITIONS = ((1,), (4,), (1, 4))
HORIZONS = (1.0, 4.0, 8.0, 12.0)
METRICS = (
    "negative_log_likelihood",
    "concordance_index",
    "integrated_brier_score",
)
SUBSETS = ("all", "affected", "stable")


def _key(row: dict[str, Any]) -> tuple[int, str, int, float]:
    return (
        int(row["seed"]),
        str(row["condition"]),
        int(row["calibration_samples_per_context"]),
        float(row["noise_fraction"]),
    )


def _metrics(model, rows) -> dict[str, float]:
    report = evaluate_survival_advanced(model, rows, horizons_hours=HORIZONS)
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


def _paired(values: list[float], key: str) -> dict[str, float | int]:
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


def _coverage(
    gate: HierarchicalTypedInteractionGate,
    contexts: tuple[tuple[str, ...], ...],
) -> float:
    if gate.selected_partition is None:
        return 0.0
    adapted = sum(
        tuple(features[index] for index in gate.selected_partition)
        in gate.significant_groups
        for features in contexts
    )
    return adapted / len(contexts)


def main() -> None:
    base_bytes = BASE_PATH.read_bytes()
    nonlinear_bytes = NONLINEAR_PATH.read_bytes()
    base = json.loads(base_bytes)
    nonlinear = json.loads(nonlinear_bytes)
    base_hash = hashlib.sha256(base_bytes).hexdigest()
    nonlinear_hash = hashlib.sha256(nonlinear_bytes).hexdigest()
    if nonlinear["protocol"]["base_artifact_sha256"] != base_hash:
        raise ValueError("nonlinear artifact is not bound to the selected base")
    base_by_key = {_key(row): row for row in base["runs"]}
    nonlinear_by_key = {_key(row): row for row in nonlinear["runs"]}
    if set(base_by_key) != set(nonlinear_by_key) or len(base_by_key) != 240:
        raise ValueError("paired artifacts do not contain the same 240 run keys")
    protocol = nonlinear["protocol"]
    seeds = protocol["seeds"]
    sizes = protocol["calibration_sizes_per_context"]
    noises = protocol["noise_fractions"]

    runs: list[dict[str, Any]] = []
    for seed in seeds:
        dataset = target_adaptation_stress_data(
            samples_per_context=protocol["samples_per_context"], seed=seed
        )
        source = FactorizedExponentialPersistenceModel.fit(
            dataset.train, epochs=protocol["source_epochs"]
        )
        source.calibrate(dataset.validation)
        models = non_affine_misspecification_models(source)
        target_protocol = build_target_calibration_protocol(
            dataset,  # type: ignore[arg-type]
            max_calibration_per_context=sizes[-1],
            split_seed=seed + 13_000_003,
        )
        tests = target_protocol.tests["in_distribution"]
        contexts = tuple(context.features() for context in dataset.contexts)
        for size in sizes:
            clean = target_protocol.calibration_subset("in_distribution", size)
            for noise in noises:
                calibration = corrupt_calibration_event_labels(
                    clean, fraction=noise, seed=seed + 14_000_019
                )
                for condition, deployed in models.items():
                    key = (seed, condition, size, noise)
                    affected = affected_contexts(deployed, source, contexts)
                    subsets = {
                        "all": tests,
                        "affected": tuple(
                            row for row in tests if row.features() in affected
                        ),
                        "stable": tuple(
                            row for row in tests if row.features() not in affected
                        ),
                    }
                    raw_gate = fit_hierarchical_typed_interaction_gate(
                        deployed,
                        calibration,
                        split_seed=seed + 15_000_037,
                        candidate_partitions=PARTITIONS,
                        familywise_alpha=protocol["familywise_alpha"],
                        activation_scope="any_predictive_gain",
                        discovery_complexity="bic",
                        epochs=protocol["adapter_epochs"],
                    )
                    coverage = _coverage(raw_gate, contexts)
                    accepted = (
                        raw_gate.activated
                        and coverage <= protocol["max_adapted_context_fraction"]
                    )
                    selected_model = raw_gate if accepted else deployed
                    affine_metrics = {
                        subset: _metrics(selected_model, rows) if rows else None
                        for subset, rows in subsets.items()
                    }
                    nonlinear_metrics = nonlinear_by_key[key]["metrics"][
                        "sparse_nonlinear_typed"
                    ]
                    runs.append(
                        {
                            "seed": seed,
                            "condition": condition,
                            "calibration_samples_per_context": size,
                            "noise_fraction": noise,
                            "metrics": {
                                "deployed_source": base_by_key[key]["metrics"][
                                    "deployed_source"
                                ],
                                "sparse_affine_control": affine_metrics,
                                "sparse_nonlinear_typed": nonlinear_metrics,
                            },
                            "activation": {
                                "raw_partition": (
                                    "inactive"
                                    if raw_gate.selected_partition is None
                                    else "x".join(
                                        f"f{index}"
                                        for index in raw_gate.selected_partition
                                    )
                                ),
                                "adapted_context_fraction": coverage,
                                "accepted": accepted,
                            },
                        }
                    )
                    print(
                        f"seed={seed} condition={condition:29s} k={size:2d} "
                        f"noise={noise:.1f} affine={'active' if accepted else 'inactive':8s} "
                        f"coverage={coverage:.3f}"
                    )

    def values(condition, size, noise, method, subset, metric):
        result = []
        for run in runs:
            if (
                run["condition"] == condition
                and run["calibration_samples_per_context"] == size
                and run["noise_fraction"] == noise
            ):
                row = run["metrics"][method][subset]
                if row is not None:
                    result.append(float(row[metric]))
        return result

    paired: dict[str, Any] = {}
    for condition in protocol["conditions"]:
        paired[condition] = {}
        for size in sizes:
            paired[condition][str(size)] = {}
            for noise in noises:
                noise_key = f"{noise:.1f}"
                paired[condition][str(size)][noise_key] = {}
                for baseline in ("deployed_source", "sparse_affine_control"):
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
                            paired[condition][str(size)][noise_key][baseline][subset][metric] = _paired(
                                deltas,
                                f"{nonlinear_hash}|{condition}|{size}|{noise}|{baseline}|{subset}|{metric}",
                            )

    payload = {
        "protocol": {
            "base_artifact_sha256": base_hash,
            "sparse_nonlinear_artifact_sha256": nonlinear_hash,
            "run_keys": len(runs),
            "candidate_partitions": [list(value) for value in PARTITIONS],
            "discovery_complexity": "bic",
            "activation_scope": "any_predictive_gain",
            "max_adapted_context_fraction": protocol[
                "max_adapted_context_fraction"
            ],
            "comparison": (
                "affine-only controlled gate with global partition excluded and "
                "the same post-selection sparse coverage cap"
            ),
            "claim_scope": "synthetic nonlinear-basis necessity ablation",
        },
        "paired_delta_nonlinear_vs_baseline": paired,
        "runs": runs,
    }
    OUTPUT_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"report: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
