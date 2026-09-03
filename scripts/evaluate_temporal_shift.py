from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from openprop.advanced_survival_evaluation import evaluate_survival_advanced
from openprop.compositional_persistence import compositional_location_data
from openprop.statistical_persistence import (
    FactorizedExponentialPersistenceModel,
    FactorizedPiecewiseExponentialPersistenceModel,
    FactorizedWeibullPersistenceModel,
)


def _summary(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.mean(values),
        "standard_deviation": statistics.pstdev(values),
        "minimum": min(values),
        "maximum": max(values),
    }


def _metrics(report) -> dict[str, float]:
    return {
        "negative_log_likelihood": report.negative_log_likelihood,
        "concordance_index": report.concordance_index,
        "integrated_brier_score": report.integrated_brier_score,
    }


def _fit_models(dataset, args):
    exponential = FactorizedExponentialPersistenceModel.fit(
        dataset.train, epochs=args.exponential_epochs
    )
    weibull = FactorizedWeibullPersistenceModel.fit(
        dataset.train, epochs=args.weibull_epochs
    )
    piecewise = FactorizedPiecewiseExponentialPersistenceModel.fit(
        dataset.train,
        bin_edges_hours=tuple(args.bin_edges),
        epochs=args.piecewise_epochs,
    )
    exponential.calibrate(dataset.validation)
    weibull.calibrate(dataset.validation)
    piecewise.calibrate(dataset.validation)
    return {
        "factorized_exponential": exponential,
        "factorized_weibull": weibull,
        "factorized_piecewise": piecewise,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate survival extrapolation from short to long follow-up."
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[31, 41, 53, 67, 79])
    parser.add_argument("--shapes", type=float, nargs="+", default=[0.6, 1.6])
    parser.add_argument("--samples-per-context", type=int, default=80)
    parser.add_argument("--train-horizon", type=float, default=6.0)
    parser.add_argument("--validation-horizon", type=float, default=12.0)
    parser.add_argument("--test-horizon", type=float, default=24.0)
    parser.add_argument(
        "--evaluation-horizons",
        type=float,
        nargs="+",
        default=[1.0, 4.0, 8.0, 12.0, 18.0],
    )
    parser.add_argument("--bin-edges", type=float, nargs="+", default=[2.0, 4.0])
    parser.add_argument("--exponential-epochs", type=int, default=1200)
    parser.add_argument("--weibull-epochs", type=int, default=1600)
    parser.add_argument("--piecewise-epochs", type=int, default=1600)
    parser.add_argument(
        "--report-output",
        type=Path,
        default=Path("artifacts/temporal_shift_results.json"),
    )
    args = parser.parse_args()
    if (
        not args.seeds
        or len(args.seeds) != len(set(args.seeds))
        or not args.shapes
        or any(shape <= 0 for shape in args.shapes)
        or not (
            0.5 <= args.train_horizon
            <= args.validation_horizon
            <= args.test_horizon
        )
        or not args.bin_edges
        or args.bin_edges != sorted(set(args.bin_edges))
        or args.evaluation_horizons != sorted(set(args.evaluation_horizons))
        or any(
            horizon <= 0 or horizon >= args.test_horizon
            for horizon in args.evaluation_horizons
        )
    ):
        parser.error("invalid seeds, shapes, horizons, or bin edges")

    conditions = {
        "in_distribution": {
            "train": args.test_horizon,
            "validation": args.test_horizon,
            "test": args.test_horizon,
        },
        "duration_shift": {
            "train": args.train_horizon,
            "validation": args.validation_horizon,
            "test": args.test_horizon,
        },
    }
    runs: list[dict[str, object]] = []
    for true_shape in args.shapes:
        for seed in args.seeds:
            datasets = {
                name: compositional_location_data(
                    samples_per_context=args.samples_per_context,
                    weibull_shape=true_shape,
                    censor_after_hours_by_split=horizons,
                    seed=seed,
                )
                for name, horizons in conditions.items()
            }
            if datasets["in_distribution"].test != datasets["duration_shift"].test:
                raise RuntimeError("paired temporal-shift test partitions differ")
            for condition, dataset in datasets.items():
                models = _fit_models(dataset, args)
                reports = {
                    name: evaluate_survival_advanced(
                        model,
                        dataset.test,
                        horizons_hours=tuple(args.evaluation_horizons),
                    )
                    for name, model in models.items()
                }
                run = {
                    "seed": seed,
                    "true_shape": true_shape,
                    "condition": condition,
                    "horizons_hours": conditions[condition],
                    "learned_shape": models["factorized_weibull"].shape,
                    "piecewise_log_multipliers": list(
                        models["factorized_piecewise"].log_multipliers
                    ),
                    "models": {
                        name: _metrics(report) for name, report in reports.items()
                    },
                }
                runs.append(run)
                print(
                    f"shape={true_shape:.2f} seed={seed:>3d} {condition:15s} "
                    f"NLL exp={reports['factorized_exponential'].negative_log_likelihood:.4f} "
                    f"weib={reports['factorized_weibull'].negative_log_likelihood:.4f} "
                    f"piece={reports['factorized_piecewise'].negative_log_likelihood:.4f}"
                )

    model_names = (
        "factorized_exponential",
        "factorized_weibull",
        "factorized_piecewise",
    )
    aggregate: dict[str, object] = {}
    shift_penalties: dict[str, object] = {}
    for shape in args.shapes:
        shape_key = f"{shape:g}"
        aggregate[shape_key] = {}
        for condition in conditions:
            selected = [
                run
                for run in runs
                if run["true_shape"] == shape and run["condition"] == condition
            ]
            aggregate[shape_key][condition] = {
                model: {
                    metric: _summary(
                        [
                            float(run["models"][model][metric])  # type: ignore[index]
                            for run in selected
                        ]
                    )
                    for metric in (
                        "negative_log_likelihood",
                        "concordance_index",
                        "integrated_brier_score",
                    )
                }
                for model in model_names
            }
        shift_penalties[shape_key] = {}
        for model in model_names:
            penalties: dict[str, list[float]] = {
                "negative_log_likelihood": [],
                "integrated_brier_score": [],
            }
            for seed in args.seeds:
                by_condition = {
                    run["condition"]: run
                    for run in runs
                    if run["true_shape"] == shape and run["seed"] == seed
                }
                for metric in penalties:
                    shifted = float(
                        by_condition["duration_shift"]["models"][model][metric]  # type: ignore[index]
                    )
                    control = float(
                        by_condition["in_distribution"]["models"][model][metric]  # type: ignore[index]
                    )
                    penalties[metric].append(shifted - control)
            shift_penalties[shape_key][model] = {
                metric: _summary(values) for metric, values in penalties.items()
            }

    payload = {
        "protocol": {
            "seeds": args.seeds,
            "true_shapes": args.shapes,
            "samples_per_context": args.samples_per_context,
            "conditions": conditions,
            "piecewise_bin_edges_hours": args.bin_edges,
            "evaluation_horizons_hours": args.evaluation_horizons,
            "paired_test_partitions": True,
            "calibration": "condition-specific validation only; never test",
            "claim_scope": "synthetic duration-shift mechanism validation",
        },
        "aggregate": aggregate,
        "shift_penalties": shift_penalties,
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
