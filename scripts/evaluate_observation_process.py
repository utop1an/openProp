from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from openprop.observation_process import observation_process_data
from openprop.statistical_persistence import PerContextExponentialPersistenceModel
from openprop.survival_evaluation import survival_negative_log_likelihood


def _summary(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.mean(values),
        "standard_deviation": statistics.pstdev(values),
        "minimum": min(values),
        "maximum": max(values),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure observation-frequency bias with interval-censored training."
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[101, 211, 307, 401, 503]
    )
    parser.add_argument("--samples-per-schedule", type=int, default=600)
    parser.add_argument("--test-samples-per-schedule", type=int, default=400)
    parser.add_argument(
        "--report-output",
        type=Path,
        default=Path("artifacts/observation_process_results.json"),
    )
    args = parser.parse_args()
    if not args.seeds or len(args.seeds) != len(set(args.seeds)):
        parser.error("--seeds must contain distinct values")

    runs: list[dict[str, object]] = []
    for seed in args.seeds:
        dataset = observation_process_data(
            samples_per_schedule=args.samples_per_schedule,
            test_samples_per_schedule=args.test_samples_per_schedule,
            seed=seed,
        )
        naive = PerContextExponentialPersistenceModel.fit(
            dataset.naive_train, prior_exposure_hours=0
        )
        interval = PerContextExponentialPersistenceModel.fit(
            dataset.interval_train, prior_exposure_hours=0
        )
        schedules = sorted({row.features() for row in dataset.interval_train})
        naive_hazards = {
            features[-1]: naive.hazard_per_hour(features) for features in schedules
        }
        interval_hazards = {
            features[-1]: interval.hazard_per_hour(features) for features in schedules
        }
        true_hazard = dataset.true_hazard_per_hour
        naive_error = statistics.mean(
            abs(value - true_hazard) for value in naive_hazards.values()
        )
        interval_error = statistics.mean(
            abs(value - true_hazard) for value in interval_hazards.values()
        )
        run = {
            "seed": seed,
            "true_hazard_per_hour": true_hazard,
            "naive": {
                "hazards": naive_hazards,
                "mean_absolute_hazard_error": naive_error,
                "exact_test_nll": survival_negative_log_likelihood(
                    naive, dataset.exact_test
                ),
                "schedule_gap": max(naive_hazards.values())
                - min(naive_hazards.values()),
            },
            "interval_aware": {
                "hazards": interval_hazards,
                "mean_absolute_hazard_error": interval_error,
                "exact_test_nll": survival_negative_log_likelihood(
                    interval, dataset.exact_test
                ),
                "schedule_gap": max(interval_hazards.values())
                - min(interval_hazards.values()),
            },
        }
        runs.append(run)
        print(
            f"seed={seed} hazard error naive={naive_error:.4f} "
            f"interval={interval_error:.4f}"
        )

    def values(model: str, metric: str) -> list[float]:
        return [float(run[model][metric]) for run in runs]  # type: ignore[index]

    payload = {
        "protocol": {
            "seeds": args.seeds,
            "samples_per_schedule": args.samples_per_schedule,
            "test_samples_per_schedule": args.test_samples_per_schedule,
            "inspection_intervals_hours": [0.5, 4.0],
            "latent_hazard_per_hour": 0.25,
            "claim_scope": "synthetic mechanism validation",
        },
        "aggregate": {
            model: {
                metric: _summary(values(model, metric))
                for metric in (
                    "mean_absolute_hazard_error",
                    "exact_test_nll",
                    "schedule_gap",
                )
            }
            for model in ("naive", "interval_aware")
        },
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
