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
from openprop.statistical_persistence import GlobalExponentialPersistenceModel


CONDITIONS = {
    "noninformative_perfect": (0.35, 0.35, 1.0),
    "informative_perfect": (0.15, 0.75, 1.0),
    "noninformative_missed": (0.35, 0.35, 0.65),
    "informative_missed": (0.15, 0.75, 0.65),
}
MODELS = (
    "interval_only",
    "observation_aware_logged",
    "observation_aware_estimated",
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
            "Estimate observation propensities and sensitivity from training "
            "logs, then compare with logged-parameter and interval baselines."
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
        default=Path("artifacts/observation_parameter_estimation_results.json"),
    )
    args = parser.parse_args()
    if not args.seeds or len(args.seeds) != len(set(args.seeds)):
        parser.error("--seeds must contain distinct values")
    if args.train_samples <= 0 or args.test_samples <= 0:
        parser.error("sample counts must be positive")

    runs: list[dict[str, object]] = []
    for seed in args.seeds:
        paired_test = None
        for condition, (
            pre_probability,
            post_probability,
            sensitivity,
        ) in CONDITIONS.items():
            dataset = informative_observation_data(
                train_samples=args.train_samples,
                test_samples=args.test_samples,
                pre_transition_inspection_probability=pre_probability,
                post_transition_inspection_probability=post_probability,
                detection_sensitivity=sensitivity,
                seed=seed,
            )
            if paired_test is None:
                paired_test = dataset.exact_test
            elif dataset.exact_test != paired_test:
                raise RuntimeError("factorial conditions must share exact test rows")

            estimate = fit_observation_process_em(dataset.episodes)
            models = {
                "interval_only": GlobalExponentialPersistenceModel.fit(
                    dataset.interval_train
                ),
                "observation_aware_logged": ObservationAwareExponentialModel.fit(
                    dataset.episodes,
                    pre_transition_inspection_probability=pre_probability,
                    post_transition_inspection_probability=post_probability,
                    detection_sensitivity=sensitivity,
                ),
                "observation_aware_estimated": estimate.as_persistence_model(),
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

            estimation = {
                "hazard_per_hour": estimate.hazard_per_hour,
                "pre_transition_inspection_probability": (
                    estimate.pre_transition_inspection_probability
                ),
                "post_transition_inspection_probability": (
                    estimate.post_transition_inspection_probability
                ),
                "detection_sensitivity": estimate.detection_sensitivity,
                "absolute_hazard_error": abs(
                    estimate.hazard_per_hour - dataset.true_hazard_per_hour
                ),
                "absolute_pre_inspection_error": abs(
                    estimate.pre_transition_inspection_probability - pre_probability
                ),
                "absolute_post_inspection_error": abs(
                    estimate.post_transition_inspection_probability - post_probability
                ),
                "absolute_sensitivity_error": abs(
                    estimate.detection_sensitivity - sensitivity
                ),
                "iterations": estimate.iterations,
                "converged": estimate.converged,
                "initial_training_observation_nll": (
                    estimate.average_negative_log_likelihood_history[0]
                ),
                "final_training_observation_nll": (
                    estimate.average_negative_log_likelihood_history[-1]
                ),
            }
            runs.append(
                {
                    "seed": seed,
                    "condition": condition,
                    "truth": {
                        "hazard_per_hour": dataset.true_hazard_per_hour,
                        "pre_transition_inspection_probability": pre_probability,
                        "post_transition_inspection_probability": post_probability,
                        "detection_sensitivity": sensitivity,
                    },
                    "estimated_observation_process": estimation,
                    "models": model_results,
                }
            )
            print(
                f"seed={seed} condition={condition} "
                f"estimated=({estimate.hazard_per_hour:.4f},"
                f"{estimate.pre_transition_inspection_probability:.4f},"
                f"{estimate.post_transition_inspection_probability:.4f},"
                f"{estimate.detection_sensitivity:.4f})"
            )

    def values(
        condition: str,
        section: str,
        metric: str,
        model: str | None = None,
    ) -> list[float]:
        selected: list[float] = []
        for run in runs:
            if run["condition"] != condition:
                continue
            container = run[section]  # type: ignore[index]
            if model is not None:
                container = container[model]
            selected.append(float(container[metric]))
        return selected

    aggregate = {
        condition: {
            "models": {
                model: {
                    metric: _summary(values(condition, "models", metric, model))
                    for metric in (
                        "hazard_per_hour",
                        "absolute_hazard_error",
                        "exact_test_nll",
                        "integrated_brier_score",
                    )
                }
                for model in MODELS
            },
            "parameter_estimation": {
                metric: _summary(
                    values(condition, "estimated_observation_process", metric)
                )
                for metric in (
                    "hazard_per_hour",
                    "pre_transition_inspection_probability",
                    "post_transition_inspection_probability",
                    "detection_sensitivity",
                    "absolute_hazard_error",
                    "absolute_pre_inspection_error",
                    "absolute_post_inspection_error",
                    "absolute_sensitivity_error",
                    "iterations",
                    "initial_training_observation_nll",
                    "final_training_observation_nll",
                )
            },
            "convergence_rate": statistics.mean(
                float(run["estimated_observation_process"]["converged"])  # type: ignore[index]
                for run in runs
                if run["condition"] == condition
            ),
        }
        for condition in CONDITIONS
    }
    oracle_gap = {
        condition: {
            metric: _summary(
                [
                    estimated - logged
                    for estimated, logged in zip(
                        values(
                            condition,
                            "models",
                            metric,
                            "observation_aware_estimated",
                        ),
                        values(
                            condition,
                            "models",
                            metric,
                            "observation_aware_logged",
                        ),
                        strict=True,
                    )
                ]
            )
            for metric in (
                "absolute_hazard_error",
                "exact_test_nll",
                "integrated_brier_score",
            )
        }
        for condition in CONDITIONS
    }
    payload = {
        "protocol": {
            "seeds": args.seeds,
            "train_samples": args.train_samples,
            "test_samples": args.test_samples,
            "true_hazard_per_hour": 0.25,
            "followup_hours": 12.0,
            "opportunity_interval_hours": 0.5,
            "conditions": {
                name: {
                    "pre_transition_inspection_probability": values_[0],
                    "post_transition_inspection_probability": values_[1],
                    "detection_sensitivity": values_[2],
                }
                for name, values_ in CONDITIONS.items()
            },
            "paired_exact_test_partitions": True,
            "paired_training_latent_trajectories": True,
            "estimation_data": "training observation sequences only",
            "validation_or_test_tuning": False,
            "claim_scope": "synthetic observation-parameter identifiability validation",
        },
        "aggregate": aggregate,
        "estimated_minus_logged_oracle_gap": oracle_gap,
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
