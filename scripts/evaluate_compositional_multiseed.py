from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from openprop.advanced_survival_evaluation import evaluate_survival_advanced
from openprop.compositional_persistence import (
    compositional_grounding_benchmark,
    compositional_grounding_registry,
    compositional_location_data,
    evaluate_grounding_model,
)
from openprop.neural_persistence import NeuralPersistenceModel
from openprop.statistical_persistence import (
    FactorizedExponentialPersistenceModel,
    GlobalExponentialPersistenceModel,
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
        description="Run the compositional OOD experiment across random seeds."
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[31, 41, 53, 67, 79])
    parser.add_argument("--samples-per-context", type=int, default=80)
    parser.add_argument("--grounding-repetitions", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=350)
    parser.add_argument(
        "--report-output",
        type=Path,
        default=Path("artifacts/compositional_persistence_multiseed_results.json"),
    )
    args = parser.parse_args()
    if not args.seeds or len(set(args.seeds)) != len(args.seeds):
        parser.error("--seeds must contain distinct values")

    grounding_cases = compositional_grounding_benchmark(
        repetitions=args.grounding_repetitions
    )
    registry = compositional_grounding_registry()
    runs: list[dict[str, object]] = []
    for seed in args.seeds:
        dataset = compositional_location_data(
            samples_per_context=args.samples_per_context,
            seed=seed,
        )
        global_model = GlobalExponentialPersistenceModel.fit(dataset.train)
        global_survival = evaluate_survival_advanced(global_model, dataset.test)
        factorized_model = FactorizedExponentialPersistenceModel.fit(dataset.train)
        factorized_scale = factorized_model.calibrate(dataset.validation)
        factorized_survival = evaluate_survival_advanced(
            factorized_model, dataset.test
        )
        training = NeuralPersistenceModel.fit(
            dataset.train,
            epochs=args.epochs,
            learning_rate=0.015,
            embedding_dim=8,
            hidden_dim=32,
            depth=2,
            seed=seed,
        )
        scale = training.model.calibrate(dataset.validation)
        neural_survival = evaluate_survival_advanced(training.model, dataset.test)
        global_grounding = evaluate_grounding_model(
            "global-exponential", global_model, grounding_cases, registry
        )
        factorized_grounding = evaluate_grounding_model(
            "factorized-exponential", factorized_model, grounding_cases, registry
        )
        neural_grounding = evaluate_grounding_model(
            "neural-compositional", training.model, grounding_cases, registry
        )
        run = {
            "seed": seed,
            "validation_hazard_scale": scale,
            "factorized_validation_hazard_scale": factorized_scale,
            "global": {
                "negative_log_likelihood": global_survival.negative_log_likelihood,
                "concordance_index": global_survival.concordance_index,
                "integrated_brier_score": global_survival.integrated_brier_score,
                "grounding_top1": global_grounding.top1_accuracy,
            },
            "factorized": {
                "negative_log_likelihood": factorized_survival.negative_log_likelihood,
                "concordance_index": factorized_survival.concordance_index,
                "integrated_brier_score": factorized_survival.integrated_brier_score,
                "grounding_top1": factorized_grounding.top1_accuracy,
                "grounding_by_tag": dict(factorized_grounding.accuracy_by_tag),
            },
            "neural": {
                "negative_log_likelihood": neural_survival.negative_log_likelihood,
                "concordance_index": neural_survival.concordance_index,
                "integrated_brier_score": neural_survival.integrated_brier_score,
                "grounding_top1": neural_grounding.top1_accuracy,
                "grounding_by_tag": dict(neural_grounding.accuracy_by_tag),
            },
        }
        runs.append(run)
        print(
            f"seed={seed:>3d} factorized NLL="
            f"{factorized_survival.negative_log_likelihood:.4f} "
            f"Top-1={factorized_grounding.top1_accuracy:.3f}; "
            f"neural NLL={neural_survival.negative_log_likelihood:.4f} "
            f"Top-1={neural_grounding.top1_accuracy:.3f}"
        )

    def values(model: str, metric: str) -> list[float]:
        return [float(run[model][metric]) for run in runs]  # type: ignore[index]

    payload = {
        "seeds": args.seeds,
        "protocol": {
            "samples_per_context": args.samples_per_context,
            "grounding_repetitions": args.grounding_repetitions,
            "epochs": args.epochs,
            "split": "feature values seen; complete validation/test contexts held out",
        },
        "aggregate": {
            model: {
                metric: _summary(values(model, metric))
                for metric in (
                    "negative_log_likelihood",
                    "concordance_index",
                    "integrated_brier_score",
                    "grounding_top1",
                )
            }
            for model in ("global", "factorized", "neural")
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
