from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from openprop.irregular_recurrent_observation import (
    collapse_to_mean_grid,
    fit_irregular_recurrent_observation_em,
    irregular_observation_negative_log_likelihood,
    irregular_recurrent_observation_data,
)
from openprop.recurrent_observation import (
    EstimatedRecurrentObservationProcess,
    fit_recurrent_observation_em,
    recurrent_exact_brier_score,
    recurrent_exact_negative_log_likelihood,
    recurrent_exact_test_rows,
)
from openprop.simultaneous_inference import paired_bootstrap_simultaneous_intervals


GAP_CONTRASTS = (0.0, 0.50, 0.75, 0.90)
MODELS = ("mean_grid_em", "exact_interval_em", "logged_process")
HORIZONS_HOURS = (1.0, 2.0, 4.0, 8.0, 12.0)


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
            "Compare exact elapsed-interval CTMC estimation with a deliberately "
            "misspecified common-mean-grid fit under bursty observation timing."
        )
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[101, 211, 307, 401, 503]
    )
    parser.add_argument("--train-episodes", type=int, default=600)
    parser.add_argument("--test-rows", type=int, default=20_000)
    parser.add_argument(
        "--report-output",
        type=Path,
        default=Path("artifacts/irregular_observation_results.json"),
    )
    args = parser.parse_args()
    if not args.seeds or len(args.seeds) != len(set(args.seeds)):
        parser.error("--seeds must contain distinct values")
    if args.train_episodes <= 0 or args.test_rows <= 0:
        parser.error("sample counts must be positive")

    forward_rate = 0.30
    return_rate = 0.45
    mean_interval = 0.75
    observation_count = 16
    q0 = 0.70
    q1 = 0.75
    sensitivity = 0.90
    false_positive_rate = 0.04
    oracle = EstimatedRecurrentObservationProcess(
        forward_rate,
        return_rate,
        q0,
        q1,
        sensitivity,
        false_positive_rate,
        0,
        True,
        (),
        0,
    )
    runs: list[dict[str, object]] = []
    for seed in args.seeds:
        paired_test = None
        paired_long_gap_positions = None
        for contrast in GAP_CONTRASTS:
            dataset = irregular_recurrent_observation_data(
                seed=seed,
                episode_count=args.train_episodes,
                observation_count=observation_count,
                forward_rate_per_hour=forward_rate,
                return_rate_per_hour=return_rate,
                mean_interval_hours=mean_interval,
                gap_contrast=contrast,
                inspection_probability_state_0=q0,
                inspection_probability_state_1=q1,
                detection_sensitivity=sensitivity,
                false_positive_rate=false_positive_rate,
            )
            exact_test = recurrent_exact_test_rows(
                seed=seed + 4_000_003,
                row_count=args.test_rows,
                forward_rate_per_hour=forward_rate,
                return_rate_per_hour=return_rate,
                horizons_hours=HORIZONS_HOURS,
            )
            if paired_test is None:
                paired_test = exact_test
            elif exact_test != paired_test:
                raise RuntimeError("gap conditions must share exact-state test rows")
            if contrast > 0.0:
                long_gap_positions = tuple(
                    tuple(
                        index
                        for index, interval in enumerate(episode.intervals_hours)
                        if interval > mean_interval
                    )
                    for episode in dataset.episodes
                )
                if paired_long_gap_positions is None:
                    paired_long_gap_positions = long_gap_positions
                elif long_gap_positions != paired_long_gap_positions:
                    raise RuntimeError("gap conditions must share burst positions")

            mean_grid = fit_recurrent_observation_em(
                collapse_to_mean_grid(dataset.episodes),
                max_iterations=120,
                tolerance=1e-6,
            )
            exact_interval = fit_irregular_recurrent_observation_em(
                dataset.episodes,
                max_iterations=120,
                tolerance=1e-6,
            )
            processes = {
                "mean_grid_em": mean_grid,
                "exact_interval_em": exact_interval,
                "logged_process": oracle,
            }
            model_results = {
                name: {
                    "forward_rate_per_hour": process.forward_rate_per_hour,
                    "return_rate_per_hour": process.return_rate_per_hour,
                    "absolute_forward_rate_error": abs(
                        process.forward_rate_per_hour - forward_rate
                    ),
                    "absolute_return_rate_error": abs(
                        process.return_rate_per_hour - return_rate
                    ),
                    "actual_interval_train_nll": (
                        irregular_observation_negative_log_likelihood(
                            dataset.episodes, process
                        )
                    ),
                    "exact_state_test_nll": recurrent_exact_negative_log_likelihood(
                        exact_test,
                        forward_rate_per_hour=process.forward_rate_per_hour,
                        return_rate_per_hour=process.return_rate_per_hour,
                    ),
                    "exact_state_test_brier": recurrent_exact_brier_score(
                        exact_test,
                        forward_rate_per_hour=process.forward_rate_per_hour,
                        return_rate_per_hour=process.return_rate_per_hour,
                    ),
                }
                for name, process in processes.items()
            }
            intervals = dataset.episodes[0].intervals_hours
            runs.append(
                {
                    "seed": seed,
                    "gap_contrast": contrast,
                    "schedule": {
                        "minimum_interval_hours": min(intervals),
                        "maximum_interval_hours": max(intervals),
                        "mean_interval_hours": statistics.mean(intervals),
                        "total_followup_hours": sum(intervals),
                        "long_gaps_per_episode": sum(
                            interval > mean_interval for interval in intervals
                        ),
                    },
                    "mean_grid_estimate": {
                        "forward_rate_per_hour": mean_grid.forward_rate_per_hour,
                        "return_rate_per_hour": mean_grid.return_rate_per_hour,
                        "iterations": mean_grid.iterations,
                        "converged": mean_grid.converged,
                    },
                    "exact_interval_estimate": {
                        "forward_rate_per_hour": exact_interval.forward_rate_per_hour,
                        "return_rate_per_hour": exact_interval.return_rate_per_hour,
                        "inspection_probability_state_0": (
                            exact_interval.inspection_probability_state_0
                        ),
                        "inspection_probability_state_1": (
                            exact_interval.inspection_probability_state_1
                        ),
                        "detection_sensitivity": (
                            exact_interval.detection_sensitivity
                        ),
                        "false_positive_rate": exact_interval.false_positive_rate,
                        "iterations": exact_interval.iterations,
                        "converged": exact_interval.converged,
                        "initial_training_observation_nll": (
                            exact_interval.average_negative_log_likelihood_history[0]
                        ),
                        "final_training_observation_nll": (
                            exact_interval.average_negative_log_likelihood_history[-1]
                        ),
                    },
                    "models": model_results,
                }
            )
            print(
                f"seed={seed} contrast={contrast:.2f} "
                f"mean_grid=({mean_grid.forward_rate_per_hour:.4f},"
                f"{mean_grid.return_rate_per_hour:.4f}) "
                f"exact=({exact_interval.forward_rate_per_hour:.4f},"
                f"{exact_interval.return_rate_per_hour:.4f})"
            )

    def values(contrast: float, model: str, metric: str) -> list[float]:
        return [
            float(run["models"][model][metric])  # type: ignore[index]
            for run in runs
            if run["gap_contrast"] == contrast
        ]

    metrics = (
        "forward_rate_per_hour",
        "return_rate_per_hour",
        "absolute_forward_rate_error",
        "absolute_return_rate_error",
        "actual_interval_train_nll",
        "exact_state_test_nll",
        "exact_state_test_brier",
    )
    aggregate = {
        f"{contrast:.2f}": {
            model: {
                metric: _summary(values(contrast, model, metric))
                for metric in metrics
            }
            for model in MODELS
        }
        for contrast in GAP_CONTRASTS
    }
    paired_advantage = {
        f"{contrast:.2f}": [
            mean_grid - exact_interval
            for mean_grid, exact_interval in zip(
                values(contrast, "mean_grid_em", "exact_state_test_nll"),
                values(contrast, "exact_interval_em", "exact_state_test_nll"),
                strict=True,
            )
        ]
        for contrast in GAP_CONTRASTS[1:]
    }
    simultaneous_report = paired_bootstrap_simultaneous_intervals(
        paired_advantage,
        confidence=0.95,
        samples=20_000,
        seed=202_608_27,
    )
    simultaneous = simultaneous_report["intervals"]
    primary = {
        contrast: {
            "mean_grid_minus_exact_interval_nll": statistics.mean(differences),
            "simultaneous_bootstrap_95_ci": list(simultaneous[contrast]),
            "wins_ties_losses": [
                sum(value > 0.0 for value in differences),
                sum(value == 0.0 for value in differences),
                sum(value < 0.0 for value in differences),
            ],
        }
        for contrast, differences in paired_advantage.items()
    }
    payload = {
        "protocol": {
            "seeds": args.seeds,
            "train_episodes": args.train_episodes,
            "test_rows": args.test_rows,
            "observation_count": observation_count,
            "mean_interval_hours": mean_interval,
            "total_followup_hours": mean_interval * observation_count,
            "gap_contrasts": list(GAP_CONTRASTS),
            "long_gaps_per_episode": max(1, observation_count // 8),
            "forward_rate_per_hour": forward_rate,
            "return_rate_per_hour": return_rate,
            "inspection_probability_state_0": q0,
            "inspection_probability_state_1": q1,
            "detection_sensitivity": sensitivity,
            "false_positive_rate": false_positive_rate,
            "rate_search_bounds_per_hour": [1e-6, 5.0],
            "paired_training_random_streams": True,
            "paired_burst_positions": True,
            "paired_exact_state_test_rows": True,
            "estimation_data": "training elapsed intervals plus missing/negative/positive outcomes only; latent paths discarded",
            "validation_or_test_tuning": False,
            "primary_family": "mean-grid minus exact-interval exact-state test NLL at three nonzero gap contrasts",
            "simultaneous_inference": "paired max-standardized-deviation bootstrap, 20000 shared seed resamples",
            "zero_contrast_role": "compatibility control excluded from the primary family",
            "logged_process_role": "oracle-style data-generating reference, not a learned comparator",
            "claim_scope": "synthetic bursty irregular-observation mechanism validation",
        },
        "aggregate": aggregate,
        "primary_paired_inference": primary,
        "simultaneous_inference_details": simultaneous_report,
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
