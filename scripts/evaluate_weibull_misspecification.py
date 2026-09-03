from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from openprop.advanced_survival_evaluation import evaluate_survival_advanced
from openprop.compositional_persistence import compositional_location_data
from openprop.statistical_persistence import (
    FactorizedExponentialPersistenceModel,
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test exponential model misspecification under Weibull dynamics."
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[31, 41, 53, 67, 79])
    parser.add_argument("--shapes", type=float, nargs="+", default=[0.6, 1.0, 1.6])
    parser.add_argument("--samples-per-context", type=int, default=80)
    parser.add_argument("--exponential-epochs", type=int, default=1200)
    parser.add_argument("--weibull-epochs", type=int, default=1600)
    parser.add_argument(
        "--report-output",
        type=Path,
        default=Path("artifacts/weibull_misspecification_results.json"),
    )
    args = parser.parse_args()
    if (
        not args.seeds
        or len(args.seeds) != len(set(args.seeds))
        or not args.shapes
        or any(shape <= 0 for shape in args.shapes)
    ):
        parser.error("seeds must be distinct and shapes must be positive")

    runs: list[dict[str, object]] = []
    for true_shape in args.shapes:
        for seed in args.seeds:
            dataset = compositional_location_data(
                samples_per_context=args.samples_per_context,
                weibull_shape=true_shape,
                seed=seed,
            )
            exponential = FactorizedExponentialPersistenceModel.fit(
                dataset.train, epochs=args.exponential_epochs
            )
            exponential_scale = exponential.calibrate(dataset.validation)
            weibull = FactorizedWeibullPersistenceModel.fit(
                dataset.train, epochs=args.weibull_epochs
            )
            weibull_scale = weibull.calibrate(dataset.validation)
            exponential_report = evaluate_survival_advanced(exponential, dataset.test)
            weibull_report = evaluate_survival_advanced(weibull, dataset.test)
            run = {
                "seed": seed,
                "true_shape": true_shape,
                "learned_shape": weibull.shape,
                "exponential_validation_scale": exponential_scale,
                "weibull_validation_scale": weibull_scale,
                "factorized_exponential": _metrics(exponential_report),
                "factorized_weibull": _metrics(weibull_report),
                "weibull_nll_improvement": (
                    exponential_report.negative_log_likelihood
                    - weibull_report.negative_log_likelihood
                ),
            }
            runs.append(run)
            print(
                f"shape={true_shape:.2f} seed={seed:>3d} "
                f"learned={weibull.shape:.3f} "
                f"NLL exp={exponential_report.negative_log_likelihood:.4f} "
                f"weibull={weibull_report.negative_log_likelihood:.4f}"
            )

    aggregates: dict[str, object] = {}
    for shape in args.shapes:
        selected = [run for run in runs if run["true_shape"] == shape]
        model_rows: dict[str, object] = {}
        for model in ("factorized_exponential", "factorized_weibull"):
            model_rows[model] = {
                metric: _summary(
                    [float(run[model][metric]) for run in selected]  # type: ignore[index]
                )
                for metric in (
                    "negative_log_likelihood",
                    "concordance_index",
                    "integrated_brier_score",
                )
            }
        aggregates[f"{shape:g}"] = {
            "learned_shape": _summary(
                [float(run["learned_shape"]) for run in selected]
            ),
            "weibull_nll_improvement": _summary(
                [float(run["weibull_nll_improvement"]) for run in selected]
            ),
            "weibull_nll_win_rate": sum(
                float(run["weibull_nll_improvement"]) > 0 for run in selected
            )
            / len(selected),
            "models": model_rows,
        }

    payload = {
        "protocol": {
            "seeds": args.seeds,
            "true_shapes": args.shapes,
            "samples_per_context": args.samples_per_context,
            "exponential_epochs": args.exponential_epochs,
            "weibull_epochs": args.weibull_epochs,
            "split": "feature values seen; complete validation/test contexts held out",
            "calibration": "one global rate scale fit on validation only",
            "claim_scope": "synthetic model-misspecification validation",
        },
        "aggregate_by_true_shape": aggregates,
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
