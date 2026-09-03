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
from openprop.concordance_safeguard import apply_concordance_safeguard
from openprop.non_affine_misspecification import (
    NON_AFFINE_MISSPECIFICATION_CONDITIONS,
    affected_contexts,
    non_affine_misspecification_models,
)
from openprop.persistence_data import PersistenceTrainingExample
from openprop.repeated_event_evidence import (
    ConsensusCalibration,
    decode_repeated_event_consensus,
    simulate_repeated_event_evidence,
    single_annotation_examples,
)
from openprop.statistical_persistence import FactorizedExponentialPersistenceModel
from openprop.target_adaptation import RiskModel, build_target_calibration_protocol
from openprop.target_adaptation_stress import target_adaptation_stress_data
from openprop.target_interaction_adaptation import (
    HierarchicalTypedInteractionGate,
    fit_hierarchical_typed_interaction_gate,
)


DEVELOPMENT_SEEDS = (31, 41, 53, 67, 79, 97, 109, 127, 149, 173)
METHODS = (
    "single_noisy_15",
    "repeat3_cases5_equal_budget",
    "repeat5_cases3_equal_budget",
    "repeat5_cases15_same_cases",
    "repeat5_cases15_confident",
    "repeat5_cases15_confident_rank_safe",
    "clean_oracle_15",
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


def _partition_name(partition: tuple[int, ...] | None) -> str:
    if partition is None:
        return "inactive"
    return "global" if not partition else "x".join(f"f{index}" for index in partition)


def _row_error_rate(
    decoded: tuple[PersistenceTrainingExample, ...],
    clean: tuple[PersistenceTrainingExample, ...],
) -> float:
    truth = {row.group_id: row.event_observed for row in clean}
    return statistics.fmean(
        row.event_observed != truth[row.group_id] for row in decoded
    )


def _consensus_audit(consensus: ConsensusCalibration) -> dict[str, Any]:
    return {
        "evidence_records": consensus.evidence_records,
        "annotations_per_record": consensus.annotations_per_record,
        "annotation_budget": consensus.annotation_budget,
        "retained_records": consensus.retained_records,
        "abstained_records": consensus.abstained_records,
        "estimated_flip_probability": consensus.noise_estimate.flip_probability,
        "pairwise_disagreement_rate": (
            consensus.noise_estimate.pairwise_disagreement_rate
        ),
        "compared_annotation_pairs": (
            consensus.noise_estimate.compared_annotation_pairs
        ),
        "mean_retained_posterior_confidence": (
            consensus.mean_retained_posterior_confidence
        ),
    }


def _fit_gate(
    deployed: RiskModel,
    rows: tuple[PersistenceTrainingExample, ...],
    *,
    split_seed: int,
    epochs: int,
) -> HierarchicalTypedInteractionGate:
    return fit_hierarchical_typed_interaction_gate(
        deployed,
        rows,
        split_seed=split_seed,
        familywise_alpha=0.05,
        activation_scope="any_predictive_gain",
        discovery_complexity="bic",
        epochs=epochs,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Develop repeated-evidence calibration under symmetric label noise."
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEVELOPMENT_SEEDS))
    parser.add_argument("--samples-per-context", type=int, default=32)
    parser.add_argument("--single-cases-per-context", type=int, default=15)
    parser.add_argument("--repeat3-cases-per-context", type=int, default=5)
    parser.add_argument("--repeat5-cases-per-context", type=int, default=3)
    parser.add_argument("--noise-fractions", type=float, nargs="+", default=[0.0, 0.2])
    parser.add_argument(
        "--evaluation-horizons", type=float, nargs="+", default=[1.0, 4.0, 8.0, 12.0]
    )
    parser.add_argument("--source-epochs", type=int, default=1000)
    parser.add_argument("--adapter-epochs", type=int, default=500)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/repeated_evidence_adaptation_development.json"),
    )
    args = parser.parse_args()
    horizons = tuple(args.evaluation_horizons)
    sizes = (
        args.single_cases_per_context,
        args.repeat3_cases_per_context,
        args.repeat5_cases_per_context,
    )
    if (
        not args.seeds
        or len(args.seeds) != len(set(args.seeds))
        or any(value < 3 for value in sizes)
        or args.single_cases_per_context * 1
        != args.repeat3_cases_per_context * 3
        or args.single_cases_per_context * 1
        != args.repeat5_cases_per_context * 5
        or args.samples_per_context <= args.single_cases_per_context
        or not args.noise_fractions
        or args.noise_fractions != sorted(set(args.noise_fractions))
        or any(not 0.0 <= value < 0.5 for value in args.noise_fractions)
        or list(horizons) != sorted(set(horizons))
        or any(value <= 0.0 for value in horizons)
        or args.source_epochs <= 0
        or args.adapter_epochs <= 0
    ):
        parser.error("invalid seeds, equal-budget sizes, noise, horizons, or epochs")

    runs: list[dict[str, Any]] = []
    calibration_audits: list[dict[str, Any]] = []
    for seed in args.seeds:
        dataset = target_adaptation_stress_data(
            samples_per_context=args.samples_per_context,
            seed=seed,
        )
        source = FactorizedExponentialPersistenceModel.fit(
            dataset.train, epochs=args.source_epochs
        )
        source.calibrate(dataset.validation)
        deployed_models = non_affine_misspecification_models(source)
        protocol = build_target_calibration_protocol(
            dataset,  # type: ignore[arg-type]
            max_calibration_per_context=args.single_cases_per_context,
            split_seed=seed + 13_000_003,
        )
        clean15 = protocol.calibration_subset(
            "in_distribution", args.single_cases_per_context
        )
        clean5 = protocol.calibration_subset(
            "in_distribution", args.repeat3_cases_per_context
        )
        clean3 = protocol.calibration_subset(
            "in_distribution", args.repeat5_cases_per_context
        )
        test_rows = protocol.tests["in_distribution"]
        contexts = tuple(context.features() for context in dataset.contexts)
        if {row.group_id for row in clean15} & {row.group_id for row in test_rows}:
            raise RuntimeError("calibration and test identities overlap")
        for noise in args.noise_fractions:
            evidence5_same = simulate_repeated_event_evidence(
                clean15,
                annotator_count=5,
                flip_fraction=noise,
                seed=seed + 14_000_019,
            )
            evidence3_equal = simulate_repeated_event_evidence(
                clean5,
                annotator_count=3,
                flip_fraction=noise,
                seed=seed + 14_100_043,
            )
            evidence5_equal = simulate_repeated_event_evidence(
                clean3,
                annotator_count=5,
                flip_fraction=noise,
                seed=seed + 14_200_057,
            )
            consensus3_equal = decode_repeated_event_consensus(evidence3_equal)
            consensus5_equal = decode_repeated_event_consensus(evidence5_equal)
            consensus5_same = decode_repeated_event_consensus(evidence5_same)
            consensus5_confident = decode_repeated_event_consensus(
                evidence5_same,
                minimum_posterior_confidence=0.9,
            )
            calibrations = {
                "single_noisy_15": single_annotation_examples(evidence5_same),
                "repeat3_cases5_equal_budget": consensus3_equal.examples,
                "repeat5_cases3_equal_budget": consensus5_equal.examples,
                "repeat5_cases15_same_cases": consensus5_same.examples,
                "repeat5_cases15_confident": consensus5_confident.examples,
                "clean_oracle_15": clean15,
            }
            consensus_by_method = {
                "repeat3_cases5_equal_budget": consensus3_equal,
                "repeat5_cases3_equal_budget": consensus5_equal,
                "repeat5_cases15_same_cases": consensus5_same,
                "repeat5_cases15_confident": consensus5_confident,
            }
            audit = {
                "seed": seed,
                "noise_fraction": noise,
                "single_noisy_15": {
                    "evidence_records": len(clean15),
                    "annotations_per_record_used": 1,
                    "annotation_budget": len(clean15),
                    "decoded_error_rate_evaluation_only": _row_error_rate(
                        calibrations["single_noisy_15"], clean15
                    ),
                },
                "clean_oracle_15": {
                    "evidence_records": len(clean15),
                    "annotation_budget": len(clean15),
                    "decoded_error_rate_evaluation_only": 0.0,
                },
            }
            for method, consensus in consensus_by_method.items():
                reference = clean15 if "cases15" in method else (
                    clean5 if "repeat3" in method else clean3
                )
                audit[method] = {
                    **_consensus_audit(consensus),
                    "decoded_error_rate_evaluation_only": _row_error_rate(
                        consensus.examples, reference
                    ),
                }
            audit["repeat5_cases15_confident_rank_safe"] = {
                **audit["repeat5_cases15_confident"],
                "derived_from": "repeat5_cases15_confident",
                "additional_boundary": "calibration concordance non-inferiority",
            }
            calibration_audits.append(audit)

            for condition, deployed in deployed_models.items():
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
                fitted = {
                    method: _fit_gate(
                        deployed,
                        rows,
                        split_seed=seed + 15_000_037,
                        epochs=args.adapter_epochs,
                    )
                    for method, rows in calibrations.items()
                }
                guarded = apply_concordance_safeguard(
                    deployed,
                    fitted["repeat5_cases15_confident"],
                    calibrations["repeat5_cases15_confident"],
                )
                fitted["repeat5_cases15_confident_rank_safe"] = guarded
                evaluated: dict[str, RiskModel] = {
                    "deployed_source": deployed,
                    **fitted,
                }
                runs.append(
                    {
                        "seed": seed,
                        "condition": condition,
                        "noise_fraction": noise,
                        "test_examples": len(test_rows),
                        "affected_test_examples": len(subsets["affected"]),
                        "stable_test_examples": len(subsets["stable"]),
                        "metrics": {
                            method: {
                                subset: _metrics(model, rows, horizons) if rows else None
                                for subset, rows in subsets.items()
                            }
                            for method, model in evaluated.items()
                        },
                        "activation": {
                            method: {
                                "partition": _partition_name(model.selected_partition),
                                "significant_group_count": len(model.significant_groups),
                            }
                            for method, model in fitted.items()
                        },
                        "ranking_safeguard": {
                            "accepted": guarded.accepted,
                            "source_calibration_concordance": guarded.source_calibration_concordance,
                            "candidate_calibration_concordance": guarded.candidate_calibration_concordance,
                            "concordance_delta": guarded.concordance_delta,
                            "minimum_concordance_delta": guarded.minimum_concordance_delta,
                        },
                    }
                )
                print(
                    f"seed={seed} noise={noise:.1f} condition={condition:29s} "
                    + " ".join(
                        f"{method[:7]}={_partition_name(model.selected_partition):8s}"
                        for method, model in fitted.items()
                    )
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
        for noise in args.noise_fractions:
            noise_key = f"{noise:.1f}"
            paired[condition][noise_key] = {}
            for method in METHODS[1:]:
                paired[condition][noise_key][method] = {}
                for subset in SUBSETS:
                    paired[condition][noise_key][method][subset] = {}
                    for metric in METRICS:
                        baseline = values(
                            condition, noise, METHODS[0], subset, metric
                        )
                        candidate = values(
                            condition, noise, method, subset, metric
                        )
                        if not baseline or not candidate:
                            paired[condition][noise_key][method][subset] = None
                            break
                        deltas = [
                            right - left
                            if metric == "concordance_index"
                            else left - right
                            for left, right in zip(baseline, candidate, strict=True)
                        ]
                        paired[condition][noise_key][method][subset][metric] = _paired(
                            deltas,
                            f"repeated-dev|{condition}|{noise}|{method}|{subset}|{metric}",
                        )

    paired_vs_deployed: dict[str, Any] = {}
    for condition in NON_AFFINE_MISSPECIFICATION_CONDITIONS:
        paired_vs_deployed[condition] = {}
        for noise in args.noise_fractions:
            noise_key = f"{noise:.1f}"
            paired_vs_deployed[condition][noise_key] = {}
            for method in METHODS:
                paired_vs_deployed[condition][noise_key][method] = {}
                for subset in SUBSETS:
                    paired_vs_deployed[condition][noise_key][method][subset] = {}
                    for metric in METRICS:
                        baseline = values(
                            condition, noise, "deployed_source", subset, metric
                        )
                        candidate = values(
                            condition, noise, method, subset, metric
                        )
                        if not baseline or not candidate:
                            paired_vs_deployed[condition][noise_key][method][subset] = None
                            break
                        deltas = [
                            right - left
                            if metric == "concordance_index"
                            else left - right
                            for left, right in zip(baseline, candidate, strict=True)
                        ]
                        paired_vs_deployed[condition][noise_key][method][subset][metric] = _paired(
                            deltas,
                            f"repeated-dev-source|{condition}|{noise}|{method}|{subset}|{metric}",
                        )

    def activation_count(method: str, condition: str, noise: float) -> int:
        return sum(
            run["activation"][method]["partition"] != "inactive"
            for run in runs
            if run["condition"] == condition and run["noise_fraction"] == noise
        )

    activation_counts = {
        condition: {
            f"{noise:.1f}": {
                method: activation_count(method, condition, noise)
                for method in METHODS
            }
            for noise in args.noise_fractions
        }
        for condition in NON_AFFINE_MISSPECIFICATION_CONDITIONS
    }
    payload = {
        "protocol": {
            "stage": "development; not independent confirmation",
            "seeds": args.seeds,
            "conditions": NON_AFFINE_MISSPECIFICATION_CONDITIONS,
            "samples_per_context": args.samples_per_context,
            "single_cases_per_context": args.single_cases_per_context,
            "equal_annotation_budget_per_context": args.single_cases_per_context,
            "repeat3_cases_per_context": args.repeat3_cases_per_context,
            "repeat5_cases_per_context": args.repeat5_cases_per_context,
            "same_case_repeat5_annotation_budget_per_context": (
                5 * args.single_cases_per_context
            ),
            "noise_fractions": args.noise_fractions,
            "noise_assumption": (
                "homogeneous symmetric conditionally independent annotator flips; "
                "rate estimated from calibration pairwise disagreement"
            ),
            "split": (
                "outcome-independent nested calibration and identity-disjoint fixed test; "
                "gate discovery and confirmation are identity-disjoint"
            ),
            "hidden_truth_boundary": (
                "clean calibration labels are used only for the named oracle and decoder "
                "error audit; candidate decoding receives repeated labels only"
            ),
            "claim_scope": "synthetic repeated-calibration-evidence mechanism development",
            "evaluation_horizons_hours": list(horizons),
            "source_epochs": args.source_epochs,
            "adapter_epochs": args.adapter_epochs,
        },
        "activation_counts": activation_counts,
        "paired_delta_vs_single_noisy_15": paired,
        "paired_delta_vs_deployed_source": paired_vs_deployed,
        "calibration_audits": calibration_audits,
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
