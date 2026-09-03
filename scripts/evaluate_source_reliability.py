from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import statistics
from pathlib import Path

from openprop.recurrent_observation import (
    recurrent_exact_brier_score,
    recurrent_exact_negative_log_likelihood,
    recurrent_exact_test_rows,
)
from openprop.source_reliability_evaluation import (
    source_grounding_brier_score,
    source_grounding_negative_log_likelihood,
    sourced_grounding_test_rows,
)
from openprop.simultaneous_inference import paired_bootstrap_simultaneous_intervals
from openprop.source_reliability_observation import (
    EstimatedSourceReliabilityProcess,
    SourceEmissionParameters,
    fit_source_reliability_em,
    sourced_recurrent_observation_data,
)


SEVERITIES = (0.0, 0.33, 0.67, 1.0)
HORIZONS_HOURS = (1.0, 2.0, 4.0, 8.0, 12.0)
MODELS = ("pooled_sources_em", "source_aware_em", "logged_process")


def _source_parameters(severity: float) -> tuple[SourceEmissionParameters, ...]:
    return (
        SourceEmissionParameters(
            "source_a",
            0.65 + 0.25 * severity,
            0.65 - 0.25 * severity,
            0.80 + 0.15 * severity,
            0.10 - 0.08 * severity,
        ),
        SourceEmissionParameters(
            "source_b",
            0.65 - 0.25 * severity,
            0.65 + 0.25 * severity,
            0.80 - 0.15 * severity,
            0.10 + 0.08 * severity,
        ),
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
        description="Audit source-specific reliability under conflicting source evidence."
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[101, 211, 307, 401, 503])
    parser.add_argument("--train-episodes", type=int, default=400)
    parser.add_argument("--test-rows", type=int, default=20_000)
    parser.add_argument("--max-iterations", type=int, default=120)
    parser.add_argument(
        "--report-output",
        type=Path,
        default=Path("artifacts/source_reliability_results.json"),
    )
    args = parser.parse_args()
    if not args.seeds or len(args.seeds) != len(set(args.seeds)):
        parser.error("--seeds must contain distinct values")
    if min(args.train_episodes, args.test_rows, args.max_iterations) <= 0:
        parser.error("sample and iteration counts must be positive")

    forward_rate = 0.30
    return_rate = 0.45
    runs: list[dict[str, object]] = []
    for seed in args.seeds:
        paired_test = recurrent_exact_test_rows(
            seed=seed + 4_000_003,
            row_count=args.test_rows,
            forward_rate_per_hour=forward_rate,
            return_rate_per_hour=return_rate,
            horizons_hours=HORIZONS_HOURS,
        )
        for severity in SEVERITIES:
            truth = _source_parameters(severity)
            dataset = sourced_recurrent_observation_data(
                seed=seed,
                episode_count=args.train_episodes,
                forward_rate_per_hour=forward_rate,
                return_rate_per_hour=return_rate,
                source_parameters=truth,
                opportunity_interval_hours=0.5,
                followup_hours=12.0,
            )
            aware = fit_source_reliability_em(
                dataset.episodes,
                max_iterations=args.max_iterations,
                tolerance=1e-6,
            )
            grounding_test = sourced_grounding_test_rows(
                seed=seed + 5_000_003,
                row_count=args.test_rows,
                forward_rate_per_hour=forward_rate,
                return_rate_per_hour=return_rate,
                source_parameters=truth,
                opportunity_interval_hours=0.5,
            )
            pooled = fit_source_reliability_em(
                dataset.episodes,
                pooled_sources=True,
                max_iterations=args.max_iterations,
                tolerance=1e-6,
            )
            oracle = EstimatedSourceReliabilityProcess(
                forward_rate, return_rate, truth, False, 0, True, (), 0
            )
            processes = {
                "pooled_sources_em": pooled,
                "source_aware_em": aware,
                "logged_process": oracle,
            }
            model_values = {
                name: (process.forward_rate_per_hour, process.return_rate_per_hour)
                for name, process in processes.items()
            }
            models = {
                name: {
                    "forward_rate_per_hour": rates[0],
                    "return_rate_per_hour": rates[1],
                    "absolute_forward_rate_error": abs(rates[0] - forward_rate),
                    "absolute_return_rate_error": abs(rates[1] - return_rate),
                    "exact_state_test_nll": recurrent_exact_negative_log_likelihood(
                        paired_test,
                        forward_rate_per_hour=rates[0],
                        return_rate_per_hour=rates[1],
                    ),
                    "exact_state_test_brier": recurrent_exact_brier_score(
                        paired_test,
                        forward_rate_per_hour=rates[0],
                        return_rate_per_hour=rates[1],
                    ),
                    "source_grounding_test_nll": source_grounding_negative_log_likelihood(
                        grounding_test, processes[name]
                    ),
                    "source_grounding_test_brier": source_grounding_brier_score(
                        grounding_test, processes[name]
                    ),
                }
                for name, rates in model_values.items()
            }
            disagreements = comparable = 0
            for episode in dataset.episodes:
                for step in episode.results_by_step:
                    if all(item.result != "missing" for item in step):
                        comparable += 1
                        disagreements += int(step[0].result != step[1].result)
            runs.append(
                {
                    "seed": seed,
                    "severity": severity,
                    "source_truth": [asdict(value) for value in truth],
                    "comparable_source_opportunities": comparable,
                    "source_disagreement_rate": disagreements / comparable,
                    "source_aware_estimate": {
                        "iterations": aware.iterations,
                        "converged": aware.converged,
                        "initial_training_observation_nll": aware.average_negative_log_likelihood_history[0],
                        "final_training_observation_nll": aware.average_negative_log_likelihood_history[-1],
                        "source_parameters": [asdict(value) for value in aware.source_parameters],
                    },
                    "pooled_estimate": {
                        "iterations": pooled.iterations,
                        "converged": pooled.converged,
                        "initial_training_observation_nll": pooled.average_negative_log_likelihood_history[0],
                        "final_training_observation_nll": pooled.average_negative_log_likelihood_history[-1],
                        "shared_parameters": asdict(pooled.source_parameters[0]),
                    },
                    "models": models,
                }
            )
            print(
                f"seed={seed} severity={severity:.2f} "
                f"pooled=({pooled.forward_rate_per_hour:.4f},{pooled.return_rate_per_hour:.4f}) "
                f"aware=({aware.forward_rate_per_hour:.4f},{aware.return_rate_per_hour:.4f})"
            )

    def values(severity: float, model: str, metric: str) -> list[float]:
        return [
            float(run["models"][model][metric])  # type: ignore[index]
            for run in runs
            if run["severity"] == severity
        ]

    metrics = (
        "forward_rate_per_hour",
        "return_rate_per_hour",
        "absolute_forward_rate_error",
        "absolute_return_rate_error",
        "exact_state_test_nll",
        "source_grounding_test_nll",
        "source_grounding_test_brier",
        "exact_state_test_brier",
    )
    aggregate = {
        f"{severity:.2f}": {
            model: {metric: _summary(values(severity, model, metric)) for metric in metrics}
            for model in MODELS
        }
        for severity in SEVERITIES
    }
    paired_advantage = {
        f"{severity:.2f}": [
            pooled - aware
            for pooled, aware in zip(
                values(severity, "pooled_sources_em", "source_grounding_test_nll"),
                values(severity, "source_aware_em", "source_grounding_test_nll"),
                strict=True,
            )
        ]
        for severity in SEVERITIES[1:]
    }
    simultaneous_report = paired_bootstrap_simultaneous_intervals(
        paired_advantage, confidence=0.95, samples=20_000, seed=73_109
    )
    primary = {
        severity: {
            "paired_differences": paired_advantage[severity],
            "mean_pooled_minus_source_aware_grounding_nll": statistics.mean(paired_advantage[severity]),
            "simultaneous_bootstrap_95_ci": list(simultaneous_report["intervals"][severity]),
            "wins_ties_losses": [
                sum(value > 1e-12 for value in paired_advantage[severity]),
                sum(abs(value) <= 1e-12 for value in paired_advantage[severity]),
                sum(value < -1e-12 for value in paired_advantage[severity]),
            ],
        }
        for severity in paired_advantage
    }
    report = {
        "benchmark": "source-specific reliability under conflicting recurrent observations",
        "design": {
            "seeds": args.seeds,
            "train_episodes_per_condition": args.train_episodes,
            "exact_test_rows_per_seed": args.test_rows,
            "severity_levels": list(SEVERITIES),
            "fixed_source_average_parameters": {
                "inspection_probability_state_0": 0.65,
                "inspection_probability_state_1": 0.65,
                "detection_sensitivity": 0.80,
                "false_positive_rate": 0.10,
            },
            "paired_training_random_streams": True,
            "paired_exact_state_test_rows_within_seed": True,
            "paired_source_grounding_state_paths_within_seed": True,
            "latent_training_states_retained": False,
            "primary_family": "pooled minus source-aware filtered current-state NLL at three nonzero source-conflict severities",
            "zero_severity_role": "nested compatibility control excluded from the primary family",
            "simultaneous_inference": "paired max-standardized-deviation bootstrap, 20000 shared seed resamples",
        },
        "runs": runs,
        "aggregate": aggregate,
        "primary_paired_inference": primary,
        "simultaneous_inference_details": simultaneous_report,
    }
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.report_output}")


if __name__ == "__main__":
    main()
