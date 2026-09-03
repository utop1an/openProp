from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from openprop.advanced_survival_evaluation import evaluate_survival_advanced
from openprop.cox_persistence import FactorizedCoxPersistenceModel
from openprop.latent_mechanism_shift import (
    MECHANISM_CONDITIONS,
    latent_mechanism_shift_data,
)
from openprop.semiparametric_evaluation import evaluate_semiparametric_survival
from openprop.statistical_persistence import (
    FactorizedExponentialPersistenceModel,
    FactorizedPiecewiseExponentialPersistenceModel,
    FactorizedWeibullPersistenceModel,
)
from openprop.survival_evaluation import model_risk_score
from openprop.synthetic_survival_oracle import SyntheticWeibullOracle


PARAMETRIC_MODELS = (
    "factorized_exponential",
    "factorized_weibull",
    "factorized_piecewise",
)
ALL_MODELS = PARAMETRIC_MODELS + ("factorized_cox",)


def _summary(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.mean(values),
        "standard_deviation": statistics.pstdev(values),
        "minimum": min(values),
        "maximum": max(values),
    }


def _fit_models(dataset, args):
    exponential = FactorizedExponentialPersistenceModel.fit(
        dataset.train,
        epochs=args.exponential_epochs,
    )
    weibull = FactorizedWeibullPersistenceModel.fit(
        dataset.train,
        epochs=args.weibull_epochs,
    )
    piecewise = FactorizedPiecewiseExponentialPersistenceModel.fit(
        dataset.train,
        bin_edges_hours=tuple(args.bin_edges),
        epochs=args.piecewise_epochs,
    )
    cox = FactorizedCoxPersistenceModel.fit(
        dataset.train,
        epochs=args.cox_epochs,
    )
    horizons = tuple(args.evaluation_horizons)
    exponential.calibrate(dataset.validation)
    weibull.calibrate(dataset.validation)
    piecewise.calibrate(dataset.validation)
    cox.calibrate_baseline(dataset.validation, horizons_hours=horizons)
    return {
        "factorized_exponential": exponential,
        "factorized_weibull": weibull,
        "factorized_piecewise": piecewise,
        "factorized_cox": cox,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate typed survival models under paired latent-mechanism shifts."
        )
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[31, 41, 53, 67, 79])
    parser.add_argument("--samples-per-context", type=int, default=80)
    parser.add_argument(
        "--evaluation-horizons",
        type=float,
        nargs="+",
        default=[1.0, 4.0, 8.0, 12.0],
    )
    parser.add_argument("--bin-edges", type=float, nargs="+", default=[2.0, 6.0])
    parser.add_argument("--exponential-epochs", type=int, default=1200)
    parser.add_argument("--weibull-epochs", type=int, default=1600)
    parser.add_argument("--piecewise-epochs", type=int, default=1600)
    parser.add_argument("--cox-epochs", type=int, default=800)
    parser.add_argument(
        "--report-output",
        type=Path,
        default=Path("artifacts/latent_mechanism_shift_results.json"),
    )
    args = parser.parse_args()
    if (
        not args.seeds
        or len(args.seeds) != len(set(args.seeds))
        or args.samples_per_context <= 0
        or args.evaluation_horizons
        != sorted(set(args.evaluation_horizons))
        or any(horizon <= 0 or horizon >= 16.0 for horizon in args.evaluation_horizons)
        or args.bin_edges != sorted(set(args.bin_edges))
        or any(edge <= 0 for edge in args.bin_edges)
        or min(
            args.exponential_epochs,
            args.weibull_epochs,
            args.piecewise_epochs,
            args.cox_epochs,
        )
        <= 0
    ):
        parser.error("invalid seeds, samples, horizons, bin edges, or epochs")

    runs: list[dict[str, object]] = []
    horizons = tuple(args.evaluation_horizons)
    for seed in args.seeds:
        dataset = latent_mechanism_shift_data(
            samples_per_context=args.samples_per_context,
            seed=seed,
        )
        models = _fit_models(dataset, args)
        for condition, test_rows in dataset.tests.items():
            model_results: dict[str, dict[str, float | None]] = {}
            for name, model in models.items():
                if name == "factorized_cox":
                    report = evaluate_semiparametric_survival(
                        model,
                        test_rows,
                        horizons_hours=horizons,
                    )
                    model_results[name] = {
                        "negative_log_likelihood": None,
                        "concordance_index": report.concordance_index,
                        "integrated_brier_score": report.integrated_brier_score,
                    }
                else:
                    report = evaluate_survival_advanced(
                        model,
                        test_rows,
                        horizons_hours=horizons,
                    )
                    model_results[name] = {
                        "negative_log_likelihood": report.negative_log_likelihood,
                        "concordance_index": report.concordance_index,
                        "integrated_brier_score": report.integrated_brier_score,
                    }
            oracle = SyntheticWeibullOracle(
                dataset.test_hazards[condition],
                float(MECHANISM_CONDITIONS[condition]["weibull_shape"]),
            )
            oracle_report = evaluate_survival_advanced(
                oracle,
                test_rows,
                horizons_hours=horizons,
            )
            oracle_result = {
                "negative_log_likelihood": oracle_report.negative_log_likelihood,
                "concordance_index": oracle_report.concordance_index,
                "integrated_brier_score": oracle_report.integrated_brier_score,
            }
            context_risks = {
                "|".join(features): {
                    "true_hazard_per_hour": true_hazard,
                    "model_risk_scores": {
                        name: model_risk_score(model, features)
                        for name, model in models.items()
                    },
                }
                for features, true_hazard in dataset.test_hazards[condition].items()
            }
            runs.append(
                {
                    "seed": seed,
                    "condition": condition,
                    "models": model_results,
                    "oracle": oracle_result,
                    "context_risks": context_risks,
                    "fit": {
                        "weibull_shape": models["factorized_weibull"].shape,
                        "piecewise_log_multipliers": list(
                            models["factorized_piecewise"].log_multipliers
                        ),
                        "cox_initial_partial_nll": (
                            models["factorized_cox"].initial_partial_nll
                        ),
                        "cox_final_partial_nll": (
                            models["factorized_cox"].final_partial_nll
                        ),
                        "cox_validation_baseline_scale": (
                            models["factorized_cox"].baseline_scale
                        ),
                    },
                }
            )
            print(
                f"seed={seed} condition={condition:26s} "
                + " ".join(
                    f"{name} C={metrics['concordance_index']:.3f} "
                    f"IBS={metrics['integrated_brier_score']:.3f}"
                    for name, metrics in model_results.items()
                )
            )

    def metric_values(condition: str, model: str, metric: str) -> list[float]:
        return [
            float(run["models"][model][metric])  # type: ignore[index]
            for run in runs
            if run["condition"] == condition
            and run["models"][model][metric] is not None  # type: ignore[index]
        ]
    def oracle_metric_values(condition: str, metric: str) -> list[float]:
        return [
            float(run["oracle"][metric])  # type: ignore[index]
            for run in runs
            if run["condition"] == condition
        ]

    aggregate = {
        condition: {
            model: {
                metric: _summary(metric_values(condition, model, metric))
                for metric in (
                    ("concordance_index", "integrated_brier_score")
                    if model == "factorized_cox"
                    else (
                        "negative_log_likelihood",
                        "concordance_index",
                        "integrated_brier_score",
                    )
                )
            }
            for model in ALL_MODELS
        }
        for condition in MECHANISM_CONDITIONS
    }
    oracle_aggregate = {
        condition: {
            metric: _summary(oracle_metric_values(condition, metric))
            for metric in (
                "negative_log_likelihood",
                "concordance_index",
                "integrated_brier_score",
            )
        }
        for condition in MECHANISM_CONDITIONS
    }
    oracle_regret: dict[str, object] = {}
    for condition in MECHANISM_CONDITIONS:
        oracle_regret[condition] = {}
        for model in ALL_MODELS:
            metrics = (
                ("concordance_index", "integrated_brier_score")
                if model == "factorized_cox"
                else (
                    "negative_log_likelihood",
                    "concordance_index",
                    "integrated_brier_score",
                )
            )
            oracle_regret[condition][model] = {}
            for metric in metrics:
                model_values = metric_values(condition, model, metric)
                oracle_values = oracle_metric_values(condition, metric)
                if metric == "concordance_index":
                    regrets = [
                        oracle - fitted
                        for fitted, oracle in zip(
                            model_values,
                            oracle_values,
                            strict=True,
                        )
                    ]
                else:
                    regrets = [
                        fitted - oracle
                        for fitted, oracle in zip(
                            model_values,
                            oracle_values,
                            strict=True,
                        )
                    ]
                oracle_regret[condition][model][metric] = _summary(regrets)

    payload = {
        "protocol": {
            "seeds": args.seeds,
            "samples_per_context": args.samples_per_context,
            "conditions": MECHANISM_CONDITIONS,
            "evaluation_horizons_hours": args.evaluation_horizons,
            "piecewise_bin_edges_hours": args.bin_edges,
            "paired_test_latent_and_censor_draws": True,
            "shared_source_train_and_validation": True,
            "calibration": "source validation only; never shifted test",
            "cox_metrics": (
                "C-index and IBS only; continuous event NLL is undefined for "
                "the Breslow step baseline"
            ),
            "oracle": (
                "generator truth is evaluation-only and never exposed to fitted models"
            ),
            "regret_direction": "positive values mean worse than same-test oracle",
            "claim_scope": "synthetic latent-mechanism shift failure analysis",
        },
        "aggregate": aggregate,
        "oracle_aggregate": oracle_aggregate,
        "oracle_regret": oracle_regret,
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
