from __future__ import annotations

import argparse
import json
from pathlib import Path

from openprop.compositional_persistence import (
    compositional_grounding_benchmark,
    compositional_grounding_registry,
    compositional_location_data,
)
from openprop.typed_context_ablation import (
    TYPED_CONTEXT_CONDITIONS,
    aggregate_typed_context_runs,
    evaluate_typed_context_seed,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the frozen typed-context component ablation."
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[31, 41, 53, 67, 79, 83, 97, 109, 127, 149],
    )
    parser.add_argument("--samples-per-context", type=int, default=80)
    parser.add_argument("--grounding-repetitions", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=1200)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument(
        "--report-output",
        type=Path,
        default=Path("artifacts/typed_context_component_ablation.json"),
    )
    args = parser.parse_args()
    if not args.seeds or len(set(args.seeds)) != len(args.seeds):
        parser.error("--seeds must contain distinct values")
    if args.samples_per_context <= 0 or args.grounding_repetitions <= 0:
        parser.error("sample and grounding counts must be positive")

    cases = compositional_grounding_benchmark(
        repetitions=args.grounding_repetitions
    )
    registry = compositional_grounding_registry()
    runs = []
    for seed in args.seeds:
        dataset = compositional_location_data(
            samples_per_context=args.samples_per_context,
            seed=seed,
        )
        run = evaluate_typed_context_seed(
            seed=seed,
            dataset=dataset,
            grounding_cases=cases,
            registry=registry,
            epochs=args.epochs,
        )
        runs.append(run)
        full = run["conditions"]["full_context"]
        print(
            f"seed={seed:>3d} full NLL={full['negative_log_likelihood']:.4f} "
            f"C-index={full['concordance_index']:.3f} "
            f"Top-1={full['grounding_top1']:.3f}"
        )

    summary = aggregate_typed_context_runs(
        runs,
        bootstrap_samples=args.bootstrap_samples,
    )
    payload = {
        "protocol": {
            "dataset": "compositional_location_data",
            "split": "all feature values seen; validation/test tuples held out",
            "selection": "condition matrix and metrics fixed before execution",
            "training": "train only; one hazard scale calibrated on validation",
            "test_policy": "same paired test rows for every condition; no test selection",
            "samples_per_context": args.samples_per_context,
            "grounding_repetitions": args.grounding_repetitions,
            "epochs": args.epochs,
            "bootstrap_samples": args.bootstrap_samples,
            "delta_orientation": "positive paired delta means full_context is better",
        },
        "feature_columns": {
            "0": "property name (constant and excluded from all ablations)",
            "1": "subject type",
            "2": "relation predicate",
            "3": "relation object",
            "4": "scene",
        },
        "conditions": [
            {
                "name": condition.name,
                "active_feature_indices": list(condition.active_feature_indices),
                "active_groups": list(condition.active_groups),
            }
            for condition in TYPED_CONTEXT_CONDITIONS
        ],
        **summary,
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
