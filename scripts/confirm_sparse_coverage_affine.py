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
    NON_AFFINE_MISSPECIFICATION_CONDITIONS,
    affected_contexts,
    non_affine_misspecification_models,
)
from openprop.sparse_affine_adaptation import fit_sparse_coverage_affine_gate
from openprop.statistical_persistence import FactorizedExponentialPersistenceModel
from openprop.target_adaptation import build_target_calibration_protocol
from openprop.target_adaptation_stress import (
    corrupt_calibration_event_labels,
    target_adaptation_stress_data,
)
from openprop.target_interaction_adaptation import (
    fit_hierarchical_typed_interaction_gate,
)


CONFIRMATION_SEEDS = (181, 191, 211, 223, 239, 251, 263, 277, 293, 307)
NOISE_FRACTIONS = (0.0, 0.2)
CALIBRATION_PER_CONTEXT = 16
SAMPLES_PER_CONTEXT = 32
HORIZONS = (1.0, 4.0, 8.0, 12.0)
SOURCE_EPOCHS = 1200
ADAPTER_EPOCHS = 800
METRICS = (
    "negative_log_likelihood",
    "concordance_index",
    "integrated_brier_score",
)
SUBSETS = ("all", "affected", "stable")
OUTPUT = Path("artifacts/sparse_coverage_affine_confirmation.json")


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


def _name(partition: tuple[int, ...] | None) -> str:
    return (
        "inactive"
        if partition is None
        else "x".join(f"f{index}" for index in partition)
    )


def main() -> None:
    runs: list[dict[str, Any]] = []
    for seed in CONFIRMATION_SEEDS:
        dataset = target_adaptation_stress_data(
            samples_per_context=SAMPLES_PER_CONTEXT, seed=seed
        )
        source = FactorizedExponentialPersistenceModel.fit(
            dataset.train, epochs=SOURCE_EPOCHS
        )
        source.calibrate(dataset.validation)
        models = non_affine_misspecification_models(source)
        protocol = build_target_calibration_protocol(
            dataset,  # type: ignore[arg-type]
            max_calibration_per_context=CALIBRATION_PER_CONTEXT,
            split_seed=seed + 13_000_003,
        )
        test_rows = protocol.tests["in_distribution"]
        contexts = tuple(context.features() for context in dataset.contexts)
        clean = protocol.calibration_subset(
            "in_distribution", CALIBRATION_PER_CONTEXT
        )
        for noise in NOISE_FRACTIONS:
            calibration = corrupt_calibration_event_labels(
                clean, fraction=noise, seed=seed + 14_000_019
            )
            for condition, deployed in models.items():
                affected = affected_contexts(deployed, source, contexts)
                subsets = {
                    "all": test_rows,
                    "affected": tuple(
                        row for row in test_rows if row.features() in affected
                    ),
                    "stable": tuple(
                        row for row in test_rows if row.features() not in affected
                    ),
                }
                previous = fit_hierarchical_typed_interaction_gate(
                    deployed,
                    calibration,
                    split_seed=seed + 15_000_037,
                    familywise_alpha=0.05,
                    activation_scope="any_predictive_gain",
                    discovery_complexity="bic",
                    epochs=ADAPTER_EPOCHS,
                )
                sparse = fit_sparse_coverage_affine_gate(
                    deployed,
                    calibration,
                    split_seed=seed + 15_000_037,
                    familywise_alpha=0.05,
                    max_adapted_context_fraction=0.5,
                    epochs=ADAPTER_EPOCHS,
                )
                fitted = {
                    "deployed_source": deployed,
                    "previous_controlled_bic": previous,
                    "sparse_coverage_affine": sparse,
                }
                metrics = {
                    method: {
                        subset: _metrics(model, rows) if rows else None
                        for subset, rows in subsets.items()
                    }
                    for method, model in fitted.items()
                }
                runs.append(
                    {
                        "seed": seed,
                        "condition": condition,
                        "noise_fraction": noise,
                        "calibration_examples": len(calibration),
                        "test_examples": len(test_rows),
                        "affected_test_examples": len(subsets["affected"]),
                        "stable_test_examples": len(subsets["stable"]),
                        "metrics": metrics,
                        "activation": {
                            "previous_partition": _name(
                                previous.selected_partition
                            ),
                            "sparse_partition": _name(sparse.selected_partition),
                            "sparse_candidate_partition": _name(
                                sparse.candidate_gate.selected_partition
                            ),
                            "sparse_coverage_rejected": sparse.coverage_rejected,
                            "sparse_adapted_context_fraction": (
                                sparse.adapted_context_fraction
                            ),
                        },
                    }
                )
                print(
                    f"seed={seed} condition={condition:29s} noise={noise:.1f} "
                    f"previous={_name(previous.selected_partition):8s} "
                    f"sparse={_name(sparse.selected_partition):8s} "
                    f"coverage={sparse.adapted_context_fraction:.3f}"
                )

    def values(condition, noise, method, subset, metric):
        result = []
        for run in runs:
            if run["condition"] == condition and run["noise_fraction"] == noise:
                row = run["metrics"][method][subset]
                if row is not None:
                    result.append(float(row[metric]))
        return result

    paired: dict[str, Any] = {}
    for condition in NON_AFFINE_MISSPECIFICATION_CONDITIONS:
        paired[condition] = {}
        for noise in NOISE_FRACTIONS:
            noise_key = f"{noise:.1f}"
            paired[condition][noise_key] = {}
            for baseline in ("deployed_source", "previous_controlled_bic"):
                paired[condition][noise_key][baseline] = {}
                for subset in SUBSETS:
                    paired[condition][noise_key][baseline][subset] = {}
                    for metric in METRICS:
                        old = values(
                            condition, noise, baseline, subset, metric
                        )
                        new = values(
                            condition,
                            noise,
                            "sparse_coverage_affine",
                            subset,
                            metric,
                        )
                        if not old or not new:
                            paired[condition][noise_key][baseline][subset] = None
                            break
                        deltas = [
                            right - left
                            if metric == "concordance_index"
                            else left - right
                            for left, right in zip(old, new, strict=True)
                        ]
                        paired[condition][noise_key][baseline][subset][metric] = _paired(
                            deltas,
                            f"confirmation|{condition}|{noise}|{baseline}|{subset}|{metric}",
                        )

    def activations(method: str, condition: str, noise: float) -> int:
        key = (
            "previous_partition"
            if method == "previous"
            else "sparse_partition"
        )
        return sum(
            run["activation"][key] != "inactive"
            for run in runs
            if run["condition"] == condition
            and run["noise_fraction"] == noise
        )

    fold = paired["local_scene_fold"]["0.0"][
        "previous_controlled_bic"
    ]["affected"]["concordance_index"]
    bump = paired["local_subject_scene_bump"]["0.0"][
        "previous_controlled_bic"
    ]["all"]["negative_log_likelihood"]
    clean_control = activations("sparse", "correct_source_control", 0.0)
    noisy_control = activations("sparse", "correct_source_control", 0.2)
    previous_noisy = activations("previous", "correct_source_control", 0.2)
    criteria = {
        "clean_control_zero_activations": clean_control == 0,
        "noisy_control_at_most_two_activations": noisy_control <= 2,
        "noisy_control_strictly_better_than_previous": noisy_control < previous_noisy,
        "fold_affected_c_index_mean_gain_vs_previous_at_least_0.02": (
            fold["mean"] >= 0.02
        ),
        "fold_affected_c_index_ci_not_negative": (
            fold["bootstrap_95_ci_lower"] >= -1e-12
        ),
        "bump_all_nll_not_materially_worse_than_previous": (
            bump["bootstrap_95_ci_lower"] >= -0.01
        ),
    }
    payload = {
        "protocol": {
            "confirmation_seeds": list(CONFIRMATION_SEEDS),
            "seed_status": "fresh; absent from all adaptation development artifacts",
            "conditions": NON_AFFINE_MISSPECIFICATION_CONDITIONS,
            "samples_per_context": SAMPLES_PER_CONTEXT,
            "calibration_samples_per_context": CALIBRATION_PER_CONTEXT,
            "noise_fractions": list(NOISE_FRACTIONS),
            "evaluation_horizons_hours": list(HORIZONS),
            "source_epochs": SOURCE_EPOCHS,
            "adapter_epochs": ADAPTER_EPOCHS,
            "acceptance_criteria": (
                "frozen in this runner before inspecting confirmation outcomes"
            ),
            "claim_scope": "fresh-seed synthetic confirmation of sparse coverage closure",
        },
        "activation_counts": {
            "clean_control": {
                "previous": activations(
                    "previous", "correct_source_control", 0.0
                ),
                "sparse": clean_control,
            },
            "noisy_control": {
                "previous": previous_noisy,
                "sparse": noisy_control,
            },
        },
        "acceptance": {
            "criteria": criteria,
            "accepted": all(criteria.values()),
        },
        "paired_delta_sparse_vs_baseline": paired,
        "runs": runs,
    }
    OUTPUT.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"accepted={payload['acceptance']['accepted']}")
    print(f"report: {OUTPUT}")


if __name__ == "__main__":
    main()
