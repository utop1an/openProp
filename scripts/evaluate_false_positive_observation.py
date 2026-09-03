from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from openprop.advanced_survival_evaluation import evaluate_survival_advanced
from openprop.informative_observation import (
    ObservationAwareExponentialModel,
    informative_observation_data,
)
from openprop.observation_em import fit_observation_process_em
from openprop.simultaneous_inference import paired_bootstrap_simultaneous_intervals
from openprop.statistical_persistence import GlobalExponentialPersistenceModel


FALSE_POSITIVE_RATES = (0.0, 0.02, 0.05, 0.10)
MODELS = (
    "interval_only",
    "perfect_specificity_em",
    "estimated_specificity_em",
    "logged_observation_process",
)


def _summary(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.mean(values),
        "standard_deviation": statistics.pstdev(values),
        "minimum": min(values),
        "maximum": max(values),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Stress the hidden observation model with false-positive detections "
            "and estimate specificity from training sequences only."
        )
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[101, 211, 307, 401, 503]
    )
    parser.add_argument("--train-samples", type=int, default=1200)
    parser.add_argument("--test-samples", type=int, default=1000)
    parser.add_argument(
        "--report-output",
        type=Path,
        default=Path("artifacts/false_positive_observation_results.json"),
    )
    args = parser.parse_args()
    if not args.seeds or len(args.seeds) != len(set(args.seeds)):
        parser.error("--seeds must contain distinct values")
    if args.train_samples <= 0 or args.test_samples <= 0:
        parser.error("sample counts must be positive")

    runs: list[dict[str, object]] = []
    for seed in args.seeds:
        paired_test = None
        for false_positive_rate in FALSE_POSITIVE_RATES:
            dataset = informative_observation_data(
                train_samples=args.train_samples,
                test_samples=args.test_samples,
                false_positive_rate=false_positive_rate,
                seed=seed,
            )
            if paired_test is None:
                paired_test = dataset.exact_test
            elif dataset.exact_test != paired_test:
                raise RuntimeError("false-positive conditions must share exact test rows")

            fixed = fit_observation_process_em(dataset.episodes)
            estimated = fit_observation_process_em(
                dataset.episodes,
                estimate_false_positive_rate=True,
            )
            models = {
                "interval_only": GlobalExponentialPersistenceModel.fit(
                    dataset.interval_train
                ),
                "perfect_specificity_em": fixed.as_persistence_model(),
                "estimated_specificity_em": estimated.as_persistence_model(),
                "logged_observation_process": ObservationAwareExponentialModel.fit(
                    dataset.episodes,
                    pre_transition_inspection_probability=(
                        dataset.pre_transition_inspection_probability
                    ),
                    post_transition_inspection_probability=(
                        dataset.post_transition_inspection_probability
                    ),
                    detection_sensitivity=dataset.detection_sensitivity,
                    false_positive_rate=dataset.false_positive_rate,
                ),
            }
            model_results: dict[str, dict[str, float]] = {}
            for name, model in models.items():
                evaluation = evaluate_survival_advanced(
                    model,
                    dataset.exact_test,
                    horizons_hours=(1.0, 4.0, 8.0, 12.0),
                )
                hazard = model.hazard_per_hour(dataset.exact_test[0].features())
                model_results[name] = {
                    "hazard_per_hour": hazard,
                    "absolute_hazard_error": abs(
                        hazard - dataset.true_hazard_per_hour
                    ),
                    "exact_test_nll": evaluation.negative_log_likelihood,
                    "integrated_brier_score": evaluation.integrated_brier_score,
                }
            runs.append(
                {
                    "seed": seed,
                    "false_positive_rate": false_positive_rate,
                    "truth": {
                        "hazard_per_hour": dataset.true_hazard_per_hour,
                        "pre_transition_inspection_probability": (
                            dataset.pre_transition_inspection_probability
                        ),
                        "post_transition_inspection_probability": (
                            dataset.post_transition_inspection_probability
                        ),
                        "detection_sensitivity": dataset.detection_sensitivity,
                        "false_positive_rate": dataset.false_positive_rate,
                    },
                    "perfect_specificity_estimate": {
                        "hazard_per_hour": fixed.hazard_per_hour,
                        "false_positive_rate": fixed.false_positive_rate,
                        "converged": fixed.converged,
                    },
                    "estimated_observation_process": {
                        "hazard_per_hour": estimated.hazard_per_hour,
                        "pre_transition_inspection_probability": (
                            estimated.pre_transition_inspection_probability
                        ),
                        "post_transition_inspection_probability": (
                            estimated.post_transition_inspection_probability
                        ),
                        "detection_sensitivity": estimated.detection_sensitivity,
                        "false_positive_rate": estimated.false_positive_rate,
                        "absolute_false_positive_error": abs(
                            estimated.false_positive_rate - false_positive_rate
                        ),
                        "iterations": estimated.iterations,
                        "converged": estimated.converged,
                        "initial_training_observation_nll": (
                            estimated.average_negative_log_likelihood_history[0]
                        ),
                        "final_training_observation_nll": (
                            estimated.average_negative_log_likelihood_history[-1]
                        ),
                    },
                    "models": model_results,
                }
            )
            print(
                f"seed={seed} fpr={false_positive_rate:.2f} "
                f"fixed_hazard={fixed.hazard_per_hour:.4f} "
                f"estimated=({estimated.hazard_per_hour:.4f},"
                f"{estimated.false_positive_rate:.4f})"
            )

    def values(rate: float, model: str, metric: str) -> list[float]:
        return [
            float(run["models"][model][metric])  # type: ignore[index]
            for run in runs
            if run["false_positive_rate"] == rate
        ]

    aggregate = {
        f"{rate:.2f}": {
            model: {
                metric: _summary(values(rate, model, metric))
                for metric in (
                    "hazard_per_hour",
                    "absolute_hazard_error",
                    "exact_test_nll",
                    "integrated_brier_score",
                )
            }
            for model in MODELS
        }
        for rate in FALSE_POSITIVE_RATES
    }
    for rate in FALSE_POSITIVE_RATES:
        selected = [
            run["estimated_observation_process"]
            for run in runs
            if run["false_positive_rate"] == rate
        ]
        aggregate[f"{rate:.2f}"]["estimated_parameters"] = {  # type: ignore[index]
            metric: _summary(
                [float(row[metric]) for row in selected]  # type: ignore[index]
            )
            for metric in (
                "hazard_per_hour",
                "false_positive_rate",
                "absolute_false_positive_error",
                "iterations",
                "initial_training_observation_nll",
                "final_training_observation_nll",
            )
        }
    paired_nll_advantage = {
        f"{rate:.2f}": [
            fixed - estimated
            for fixed, estimated in zip(
                values(rate, "perfect_specificity_em", "exact_test_nll"),
                values(rate, "estimated_specificity_em", "exact_test_nll"),
                strict=True,
            )
        ]
        for rate in FALSE_POSITIVE_RATES[1:]
    }
    simultaneous_report = paired_bootstrap_simultaneous_intervals(
        paired_nll_advantage,
        confidence=0.95,
        samples=20_000,
        seed=202_608_26,
    )
    simultaneous = simultaneous_report["intervals"]
    primary = {
        rate: {
            "mean_fixed_minus_estimated_nll": statistics.mean(differences),
            "simultaneous_bootstrap_95_ci": list(simultaneous[rate]),
            "wins_ties_losses": [
                sum(value > 0 for value in differences),
                sum(value == 0 for value in differences),
                sum(value < 0 for value in differences),
            ],
        }
        for rate, differences in paired_nll_advantage.items()
    }
    payload = {
        "protocol": {
            "seeds": args.seeds,
            "train_samples": args.train_samples,
            "test_samples": args.test_samples,
            "true_hazard_per_hour": 0.25,
            "pre_transition_inspection_probability": 0.15,
            "post_transition_inspection_probability": 0.75,
            "detection_sensitivity": 0.65,
            "false_positive_rates": list(FALSE_POSITIVE_RATES),
            "paired_training_transition_draws": True,
            "paired_exact_test_partitions": True,
            "estimation_data": "training observation sequences only",
            "validation_or_test_tuning": False,
            "primary_family": "fixed-specificity minus estimated-specificity exact-test NLL at the three nonzero false-positive rates",
            "simultaneous_inference": "paired max-standardized-deviation bootstrap, 20000 shared seed resamples",
            "claim_scope": "synthetic false-positive observation-process mechanism validation",
        },
        "aggregate": aggregate,
        "primary_paired_inference": primary,
        "runs": runs,
    }
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"report: {args.report_output}")


if __name__ == "__main__":
    main()
