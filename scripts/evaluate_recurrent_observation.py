from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from openprop.observation_em import fit_observation_process_em
from openprop.recurrent_observation import (
    fit_recurrent_observation_em,
    recurrent_exact_brier_score,
    recurrent_exact_negative_log_likelihood,
    recurrent_exact_test_rows,
    recurrent_observation_data,
)
from openprop.simultaneous_inference import paired_bootstrap_simultaneous_intervals


RETURN_RATES = (0.0, 0.15, 0.30, 0.45)
MODELS = ("irreversible_em", "reversible_em", "logged_reversible")
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
            "Evaluate a reversible binary CTMC observation model against the "
            "irreversible observation-process baseline."
        )
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[101, 211, 307, 401, 503]
    )
    parser.add_argument("--train-episodes", type=int, default=600)
    parser.add_argument("--test-rows", type=int, default=2000)
    parser.add_argument("--followup-hours", type=float, default=8.0)
    parser.add_argument("--opportunity-interval-hours", type=float, default=0.5)
    parser.add_argument(
        "--report-output",
        type=Path,
        default=Path("artifacts/recurrent_observation_results.json"),
    )
    args = parser.parse_args()
    if not args.seeds or len(args.seeds) != len(set(args.seeds)):
        parser.error("--seeds must contain distinct values")
    if args.train_episodes <= 0 or args.test_rows <= 0:
        parser.error("sample counts must be positive")
    if args.followup_hours <= 0 or args.opportunity_interval_hours <= 0:
        parser.error("time values must be positive")

    forward_rate = 0.30
    q0 = 0.70
    q1 = 0.75
    sensitivity = 0.90
    false_positive_rate = 0.04
    runs: list[dict[str, object]] = []
    for seed in args.seeds:
        shared_test_structure: tuple[tuple[str, float], ...] | None = None
        for return_rate in RETURN_RATES:
            dataset = recurrent_observation_data(
                seed=seed,
                episode_count=args.train_episodes,
                forward_rate_per_hour=forward_rate,
                return_rate_per_hour=return_rate,
                opportunity_interval_hours=args.opportunity_interval_hours,
                followup_hours=args.followup_hours,
                inspection_probability_state_0=q0,
                inspection_probability_state_1=q1,
                detection_sensitivity=sensitivity,
                false_positive_rate=false_positive_rate,
            )
            exact_test = recurrent_exact_test_rows(
                seed=seed + 3_000_003,
                row_count=args.test_rows,
                forward_rate_per_hour=forward_rate,
                return_rate_per_hour=return_rate,
                horizons_hours=HORIZONS_HOURS,
            )
            test_structure = tuple(
                (row.row_id, row.horizon_hours) for row in exact_test
            )
            if shared_test_structure is None:
                shared_test_structure = test_structure
            elif test_structure != shared_test_structure:
                raise RuntimeError("return-rate conditions must share test row structure")

            irreversible = fit_observation_process_em(
                dataset.episodes,
                estimate_false_positive_rate=True,
                max_iterations=120,
                tolerance=1e-6,
            )
            reversible = fit_recurrent_observation_em(
                dataset.episodes,
                max_iterations=120,
                tolerance=1e-6,
            )
            rates = {
                "irreversible_em": (irreversible.hazard_per_hour, 0.0),
                "reversible_em": (
                    reversible.forward_rate_per_hour,
                    reversible.return_rate_per_hour,
                ),
                "logged_reversible": (forward_rate, return_rate),
            }
            model_results = {
                name: {
                    "forward_rate_per_hour": model_forward,
                    "return_rate_per_hour": model_return,
                    "absolute_forward_rate_error": abs(model_forward - forward_rate),
                    "absolute_return_rate_error": abs(model_return - return_rate),
                    "exact_test_nll": recurrent_exact_negative_log_likelihood(
                        exact_test,
                        forward_rate_per_hour=model_forward,
                        return_rate_per_hour=model_return,
                    ),
                    "exact_test_brier_score": recurrent_exact_brier_score(
                        exact_test,
                        forward_rate_per_hour=model_forward,
                        return_rate_per_hour=model_return,
                    ),
                }
                for name, (model_forward, model_return) in rates.items()
            }
            runs.append(
                {
                    "seed": seed,
                    "return_rate_per_hour": return_rate,
                    "truth": {
                        "forward_rate_per_hour": forward_rate,
                        "return_rate_per_hour": return_rate,
                        "inspection_probability_state_0": q0,
                        "inspection_probability_state_1": q1,
                        "detection_sensitivity": sensitivity,
                        "false_positive_rate": false_positive_rate,
                    },
                    "irreversible_estimate": {
                        "hazard_per_hour": irreversible.hazard_per_hour,
                        "false_positive_rate": irreversible.false_positive_rate,
                        "iterations": irreversible.iterations,
                        "converged": irreversible.converged,
                    },
                    "reversible_estimate": {
                        "forward_rate_per_hour": reversible.forward_rate_per_hour,
                        "return_rate_per_hour": reversible.return_rate_per_hour,
                        "inspection_probability_state_0": (
                            reversible.inspection_probability_state_0
                        ),
                        "inspection_probability_state_1": (
                            reversible.inspection_probability_state_1
                        ),
                        "detection_sensitivity": reversible.detection_sensitivity,
                        "false_positive_rate": reversible.false_positive_rate,
                        "iterations": reversible.iterations,
                        "converged": reversible.converged,
                        "initializations_tried": reversible.initializations_tried,
                        "initial_training_observation_nll": (
                            reversible.average_negative_log_likelihood_history[0]
                        ),
                        "final_training_observation_nll": (
                            reversible.average_negative_log_likelihood_history[-1]
                        ),
                    },
                    "models": model_results,
                }
            )
            print(
                f"seed={seed} return={return_rate:.2f} "
                f"irreversible={irreversible.hazard_per_hour:.4f} "
                f"reversible=({reversible.forward_rate_per_hour:.4f},"
                f"{reversible.return_rate_per_hour:.4f})"
            )

    def values(rate: float, model: str, metric: str) -> list[float]:
        return [
            float(run["models"][model][metric])  # type: ignore[index]
            for run in runs
            if run["return_rate_per_hour"] == rate
        ]

    metrics = (
        "forward_rate_per_hour",
        "return_rate_per_hour",
        "absolute_forward_rate_error",
        "absolute_return_rate_error",
        "exact_test_nll",
        "exact_test_brier_score",
    )
    aggregate = {
        f"{rate:.2f}": {
            model: {metric: _summary(values(rate, model, metric)) for metric in metrics}
            for model in MODELS
        }
        for rate in RETURN_RATES
    }
    for rate in RETURN_RATES:
        selected = [
            run["reversible_estimate"]
            for run in runs
            if run["return_rate_per_hour"] == rate
        ]
        aggregate[f"{rate:.2f}"]["reversible_estimated_parameters"] = {  # type: ignore[index]
            metric: _summary(
                [float(row[metric]) for row in selected]  # type: ignore[index]
            )
            for metric in (
                "inspection_probability_state_0",
                "inspection_probability_state_1",
                "detection_sensitivity",
                "false_positive_rate",
                "iterations",
                "initial_training_observation_nll",
                "final_training_observation_nll",
            )
        }

    paired_nll_advantage = {
        f"{rate:.2f}": [
            irreversible - reversible
            for irreversible, reversible in zip(
                values(rate, "irreversible_em", "exact_test_nll"),
                values(rate, "reversible_em", "exact_test_nll"),
                strict=True,
            )
        ]
        for rate in RETURN_RATES[1:]
    }
    simultaneous_report = paired_bootstrap_simultaneous_intervals(
        paired_nll_advantage,
        confidence=0.95,
        samples=20_000,
        seed=202_608_27,
    )
    simultaneous = simultaneous_report["intervals"]
    primary = {
        rate: {
            "mean_irreversible_minus_reversible_nll": statistics.mean(differences),
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
            "train_episodes": args.train_episodes,
            "test_rows": args.test_rows,
            "followup_hours": args.followup_hours,
            "opportunity_interval_hours": args.opportunity_interval_hours,
            "horizons_hours": list(HORIZONS_HOURS),
            "forward_rate_per_hour": forward_rate,
            "return_rates_per_hour": list(RETURN_RATES),
            "inspection_probability_state_0": q0,
            "inspection_probability_state_1": q1,
            "detection_sensitivity": sensitivity,
            "false_positive_rate": false_positive_rate,
            "paired_training_random_streams": True,
            "paired_exact_test_horizons_and_outcome_uniforms": True,
            "estimation_data": "training missing/negative/positive observation sequences only; latent state paths discarded",
            "validation_or_test_tuning": False,
            "primary_family": "irreversible minus reversible exact-test NLL at the three nonzero return rates",
            "simultaneous_inference": "paired max-standardized-deviation bootstrap, 20000 shared seed resamples",
            "zero_return_condition_role": "compatibility and added-complexity control; excluded from primary family",
            "logged_reversible_role": "oracle-style data-generating-rate reference; not a learned comparator",
            "claim_scope": "synthetic recurrent binary-state mechanism validation",
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
