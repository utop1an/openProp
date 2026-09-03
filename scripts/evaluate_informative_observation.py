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
from openprop.statistical_persistence import GlobalExponentialPersistenceModel


CONDITIONS = {
    "noninformative_perfect": (0.35, 0.35, 1.0),
    "informative_perfect": (0.15, 0.75, 1.0),
    "noninformative_missed": (0.35, 0.35, 0.65),
    "informative_missed": (0.15, 0.75, 0.65),
}
MODELS = ("naive_detection", "interval_only", "observation_aware")


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
            "Evaluate state-dependent inspection and missed detection with a "
            "joint hidden-state observation likelihood."
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
        default=Path("artifacts/informative_observation_results.json"),
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

            models = {
                "naive_detection": GlobalExponentialPersistenceModel.fit(
                    dataset.naive_train
                ),
                "interval_only": GlobalExponentialPersistenceModel.fit(
                    dataset.interval_train
                ),
                "observation_aware": ObservationAwareExponentialModel.fit(
                    dataset.episodes,
                    pre_transition_inspection_probability=pre_probability,
                    post_transition_inspection_probability=post_probability,
                    detection_sensitivity=sensitivity,
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
                    "condition": condition,
                    "inspection": {
                        "pre_transition_probability": pre_probability,
                        "post_transition_probability": post_probability,
                        "detection_sensitivity": sensitivity,
                    },
                    "models": model_results,
                }
            )
            print(
                f"seed={seed} condition={condition} "
                + " ".join(
                    f"{name}={result['hazard_per_hour']:.4f}"
                    for name, result in model_results.items()
                )
            )

    def metric_values(condition: str, model: str, metric: str) -> list[float]:
        return [
            float(run["models"][model][metric])  # type: ignore[index]
            for run in runs
            if run["condition"] == condition
        ]

    aggregate = {
        condition: {
            model: {
                metric: _summary(metric_values(condition, model, metric))
                for metric in (
                    "hazard_per_hour",
                    "absolute_hazard_error",
                    "exact_test_nll",
                    "integrated_brier_score",
                )
            }
            for model in MODELS
        }
        for condition in CONDITIONS
    }
    paired_gains = {
        condition: {
            baseline: {
                metric: _summary(
                    [
                        baseline_value - aware_value
                        for baseline_value, aware_value in zip(
                            metric_values(condition, baseline, metric),
                            metric_values(condition, "observation_aware", metric),
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
            for baseline in ("naive_detection", "interval_only")
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
                    "pre_transition_inspection_probability": values[0],
                    "post_transition_inspection_probability": values[1],
                    "detection_sensitivity": values[2],
                }
                for name, values in CONDITIONS.items()
            },
            "paired_exact_test_partitions": True,
            "paired_training_latent_trajectories": True,
            "independent_random_streams": (
                "latent transitions, observation noise, ordering, and exact test"
            ),
            "observation_parameters": (
                "logged policy and sensor-calibration inputs; not estimated "
                "from test outcomes"
            ),
            "claim_scope": "synthetic joint observation-state mechanism validation",
        },
        "aggregate": aggregate,
        "paired_observation_aware_gains": paired_gains,
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
