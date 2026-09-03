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
            "Stress observation-process identifiability across training size "
            "and detector sensitivity."
        )
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[101, 211, 307, 401, 503]
    )
    parser.add_argument(
        "--train-sizes", type=int, nargs="+", default=[50, 100, 300, 1200]
    )
    parser.add_argument(
        "--sensitivities", type=float, nargs="+", default=[0.3, 0.65]
    )
    parser.add_argument("--test-samples", type=int, default=1000)
    parser.add_argument(
        "--report-output",
        type=Path,
        default=Path("artifacts/observation_identifiability_results.json"),
    )
    args = parser.parse_args()
    if not args.seeds or len(args.seeds) != len(set(args.seeds)):
        parser.error("--seeds must contain distinct values")
    if (
        not args.train_sizes
        or len(args.train_sizes) != len(set(args.train_sizes))
        or any(size <= 0 for size in args.train_sizes)
    ):
        parser.error("--train-sizes must contain distinct positive values")
    if (
        not args.sensitivities
        or len(args.sensitivities) != len(set(args.sensitivities))
        or any(not 0.0 < value <= 1.0 for value in args.sensitivities)
    ):
        parser.error("--sensitivities must contain distinct values in (0, 1]")
    if args.test_samples <= 0:
        parser.error("--test-samples must be positive")

    true_hazard = 0.25
    pre_probability = 0.15
    post_probability = 0.4
    runs: list[dict[str, object]] = []
    exact_test_by_seed: dict[int, object] = {}
    for sensitivity in args.sensitivities:
        for train_size in args.train_sizes:
            for seed in args.seeds:
                dataset = informative_observation_data(
                    train_samples=train_size,
                    test_samples=args.test_samples,
                    true_hazard_per_hour=true_hazard,
                    pre_transition_inspection_probability=pre_probability,
                    post_transition_inspection_probability=post_probability,
                    detection_sensitivity=sensitivity,
                    seed=seed,
                )
                if seed not in exact_test_by_seed:
                    exact_test_by_seed[seed] = dataset.exact_test
                elif dataset.exact_test != exact_test_by_seed[seed]:
                    raise RuntimeError("stress conditions must share exact test rows")

                estimate = fit_observation_process_em(dataset.episodes)
                estimated_model = estimate.as_persistence_model()
                logged_model = ObservationAwareExponentialModel.fit(
                    dataset.episodes,
                    pre_transition_inspection_probability=pre_probability,
                    post_transition_inspection_probability=post_probability,
                    detection_sensitivity=sensitivity,
                )
                estimated_evaluation = evaluate_survival_advanced(
                    estimated_model,
                    dataset.exact_test,
                    horizons_hours=(1.0, 4.0, 8.0, 12.0),
                )
                logged_evaluation = evaluate_survival_advanced(
                    logged_model,
                    dataset.exact_test,
                    horizons_hours=(1.0, 4.0, 8.0, 12.0),
                )
                positive_episodes = sum(
                    "positive" in episode.results for episode in dataset.episodes
                )
                runs.append(
                    {
                        "seed": seed,
                        "train_size": train_size,
                        "detection_sensitivity": sensitivity,
                        "positive_episodes": positive_episodes,
                        "positive_episode_fraction": positive_episodes / train_size,
                        "converged": estimate.converged,
                        "iterations": estimate.iterations,
                        "estimated": {
                            "hazard_per_hour": estimate.hazard_per_hour,
                            "pre_transition_inspection_probability": (
                                estimate.pre_transition_inspection_probability
                            ),
                            "post_transition_inspection_probability": (
                                estimate.post_transition_inspection_probability
                            ),
                            "detection_sensitivity": estimate.detection_sensitivity,
                            "absolute_hazard_error": abs(
                                estimate.hazard_per_hour - true_hazard
                            ),
                            "absolute_pre_inspection_error": abs(
                                estimate.pre_transition_inspection_probability
                                - pre_probability
                            ),
                            "absolute_post_inspection_error": abs(
                                estimate.post_transition_inspection_probability
                                - post_probability
                            ),
                            "absolute_sensitivity_error": abs(
                                estimate.detection_sensitivity - sensitivity
                            ),
                            "exact_test_nll": (
                                estimated_evaluation.negative_log_likelihood
                            ),
                            "integrated_brier_score": (
                                estimated_evaluation.integrated_brier_score
                            ),
                        },
                        "logged_oracle": {
                            "hazard_per_hour": logged_model.hazard,
                            "absolute_hazard_error": abs(
                                logged_model.hazard - true_hazard
                            ),
                            "exact_test_nll": logged_evaluation.negative_log_likelihood,
                            "integrated_brier_score": (
                                logged_evaluation.integrated_brier_score
                            ),
                        },
                    }
                )
                print(
                    f"n={train_size} sensitivity={sensitivity:g} seed={seed} "
                    f"positive={positive_episodes} hazard={estimate.hazard_per_hour:.4f}"
                )

    def selected(
        train_size: int,
        sensitivity: float,
        section: str | None,
        metric: str,
    ) -> list[float]:
        values: list[float] = []
        for run in runs:
            if (
                run["train_size"] != train_size
                or run["detection_sensitivity"] != sensitivity
            ):
                continue
            container = run if section is None else run[section]  # type: ignore[index]
            values.append(float(container[metric]))
        return values

    aggregate = {
        str(sensitivity): {
            str(train_size): {
                "positive_episodes": _summary(
                    selected(train_size, sensitivity, None, "positive_episodes")
                ),
                "positive_episode_fraction": _summary(
                    selected(
                        train_size,
                        sensitivity,
                        None,
                        "positive_episode_fraction",
                    )
                ),
                "convergence_rate": statistics.mean(
                    selected(train_size, sensitivity, None, "converged")
                ),
                "estimated": {
                    metric: _summary(
                        selected(train_size, sensitivity, "estimated", metric)
                    )
                    for metric in (
                        "absolute_hazard_error",
                        "absolute_pre_inspection_error",
                        "absolute_post_inspection_error",
                        "absolute_sensitivity_error",
                        "exact_test_nll",
                        "integrated_brier_score",
                    )
                },
                "logged_oracle": {
                    metric: _summary(
                        selected(train_size, sensitivity, "logged_oracle", metric)
                    )
                    for metric in (
                        "absolute_hazard_error",
                        "exact_test_nll",
                        "integrated_brier_score",
                    )
                },
            }
            for train_size in args.train_sizes
        }
        for sensitivity in args.sensitivities
    }
    payload = {
        "protocol": {
            "seeds": args.seeds,
            "train_sizes": args.train_sizes,
            "test_samples": args.test_samples,
            "true_hazard_per_hour": true_hazard,
            "pre_transition_inspection_probability": pre_probability,
            "post_transition_inspection_probability": post_probability,
            "detection_sensitivities": args.sensitivities,
            "paired_nested_training_trajectories": True,
            "paired_exact_test_partitions": True,
            "estimation_data": "training observation sequences only",
            "claim_scope": "synthetic sample-identifiability stress test",
        },
        "aggregate": aggregate,
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
