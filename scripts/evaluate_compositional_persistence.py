from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from openprop.advanced_survival_evaluation import evaluate_survival_advanced
from openprop.compositional_persistence import (
    compositional_grounding_benchmark,
    compositional_grounding_registry,
    compositional_location_data,
    evaluate_grounding_model,
)
from openprop.neural_persistence import NeuralPersistenceModel
from openprop.persistence import ExponentialPersistenceModel
from openprop.statistical_persistence import (
    FactorizedExponentialPersistenceModel,
    GlobalExponentialPersistenceModel,
    PerContextExponentialPersistenceModel,
)
from openprop.temporal_grounding import NoDecayPersistenceModel


def _survival_payload(name, evaluation):
    return {
        "name": name,
        "examples": evaluation.examples,
        "negative_log_likelihood": evaluation.negative_log_likelihood,
        "concordance_index": evaluation.concordance_index,
        "integrated_brier_score": evaluation.integrated_brier_score,
        "horizons": [asdict(item) for item in evaluation.horizons],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate compositional persistence under held-out context combinations."
    )
    parser.add_argument("--samples-per-context", type=int, default=80)
    parser.add_argument("--grounding-repetitions", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=350)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument(
        "--report-output",
        type=Path,
        default=Path("artifacts/compositional_persistence_results.json"),
    )
    args = parser.parse_args()
    if args.samples_per_context <= 0 or args.grounding_repetitions <= 0 or args.epochs <= 0:
        parser.error("sample counts, repetitions, and epochs must be positive")

    dataset = compositional_location_data(
        samples_per_context=args.samples_per_context,
        seed=args.seed,
    )
    global_model = GlobalExponentialPersistenceModel.fit(dataset.train)
    context_model = PerContextExponentialPersistenceModel.fit(dataset.train)
    factorized_model = FactorizedExponentialPersistenceModel.fit(dataset.train)
    factorized_scale = factorized_model.calibrate(dataset.validation)
    training = NeuralPersistenceModel.fit(
        dataset.train,
        epochs=args.epochs,
        learning_rate=0.015,
        embedding_dim=8,
        hidden_dim=32,
        depth=2,
        seed=args.seed,
    )
    validation_scale = training.model.calibrate(dataset.validation)

    survival_reports = [
        _survival_payload("global-exponential", evaluate_survival_advanced(global_model, dataset.test)),
        _survival_payload("per-context-exponential", evaluate_survival_advanced(context_model, dataset.test)),
        _survival_payload("neural-compositional", evaluate_survival_advanced(training.model, dataset.test)),
        _survival_payload(
            "factorized-exponential",
            evaluate_survival_advanced(factorized_model, dataset.test),
        ),
    ]

    grounding_cases = compositional_grounding_benchmark(
        repetitions=args.grounding_repetitions
    )
    registry = compositional_grounding_registry()
    grounding_reports = [
        evaluate_grounding_model("no-decay", NoDecayPersistenceModel(), grounding_cases, registry),
        evaluate_grounding_model("fixed-four-hour", ExponentialPersistenceModel(), grounding_cases, registry),
        evaluate_grounding_model("global-exponential", global_model, grounding_cases, registry),
        evaluate_grounding_model("per-context-exponential", context_model, grounding_cases, registry),
        evaluate_grounding_model(
            "factorized-exponential",
            factorized_model,
            grounding_cases,
            registry,
        ),
        evaluate_grounding_model("neural-compositional", training.model, grounding_cases, registry),
    ]

    payload = {
        "dataset": {
            "train_examples": len(dataset.train),
            "validation_examples": len(dataset.validation),
            "test_examples": len(dataset.test),
            "train_contexts": sum(context.split == "train" for context in dataset.contexts),
            "validation_contexts": sum(context.split == "validation" for context in dataset.contexts),
            "test_contexts": sum(context.split == "test" for context in dataset.contexts),
            "split_contract": "all feature values seen in train; validation/test tuples held out",
        },
        "training": {
            "initial_loss": training.initial_loss,
            "final_loss": training.final_loss,
            "validation_hazard_scale": validation_scale,
            "factorized_validation_hazard_scale": factorized_scale,
            "epochs": training.epochs,
            "seed": args.seed,
        },
        "survival": survival_reports,
        "grounding": [asdict(report) for report in grounding_reports],
    }
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(
        f"split: train={len(dataset.train)} validation={len(dataset.validation)} "
        f"test={len(dataset.test)}"
    )
    print(
        f"neural loss: {training.initial_loss:.4f} -> {training.final_loss:.4f}; "
        f"validation scale={validation_scale:.4f}"
    )
    print("survival on held-out context combinations:")
    for report in survival_reports:
        print(
            f"  {report['name']:25s} NLL={report['negative_log_likelihood']:.4f} "
            f"C-index={report['concordance_index']:.3f} "
            f"IBS={report['integrated_brier_score']:.4f}"
        )
    print("grounding on compositional OOD cases:")
    for report in grounding_reports:
        print(
            f"  {report.name:25s} Top-1={report.top1_accuracy:.3f} "
            f"MRR={report.mean_reciprocal_rank:.3f}"
        )
    print(f"report: {args.report_output}")


if __name__ == "__main__":
    main()
