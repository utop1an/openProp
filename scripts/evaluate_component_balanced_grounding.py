from __future__ import annotations

import argparse
import json
from pathlib import Path

from openprop.component_balanced_grounding import (
    CONFIRMATION_SEEDS,
    DEVELOPMENT_SEEDS,
    GROUNDING_MODEL_CONDITIONS,
    GROUNDING_PROBES,
    aggregate_component_balanced_runs,
    component_balanced_grounding_benchmark,
    evaluate_component_balanced_seed,
)
from openprop.compositional_persistence import compositional_location_data


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate component-balanced typed grounding on split seeds."
    )
    parser.add_argument(
        "--phase", choices=("development", "confirmation"), default="confirmation"
    )
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--samples-per-context", type=int, default=80)
    parser.add_argument("--epochs", type=int, default=1200)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--report-output", type=Path)
    args = parser.parse_args()
    seeds = tuple(args.seeds or (
        DEVELOPMENT_SEEDS if args.phase == "development" else CONFIRMATION_SEEDS
    ))
    if not seeds or len(set(seeds)) != len(seeds):
        parser.error("seeds must contain distinct values")
    if args.samples_per_context <= 0 or args.epochs <= 0:
        parser.error("sample count and epochs must be positive")
    output = args.report_output or Path(
        f"artifacts/component_balanced_grounding_{args.phase}.json"
    )

    cases = component_balanced_grounding_benchmark()
    runs = []
    for seed in seeds:
        dataset = compositional_location_data(
            samples_per_context=args.samples_per_context,
            seed=seed,
        )
        run = evaluate_component_balanced_seed(
            seed=seed,
            dataset=dataset,
            cases=cases,
            epochs=args.epochs,
        )
        runs.append(run)
        full = run["conditions"]["full_context"]
        print(
            f"seed={seed:>3d} full={full['top1']:.3f} "
            + " ".join(
                f"{name}={full['top1_by_probe'][name]:.3f}"
                for name in ("subject", "relation", "scene")
            ),
            flush=True,
        )

    summary = aggregate_component_balanced_runs(
        runs, bootstrap_samples=args.bootstrap_samples
    )
    payload = {
        "phase": args.phase,
        "protocol": {
            "case_design": "analytic confidence-age crossover from true generator hazards",
            "truth_boundary": "current truth labels never enter matcher entities",
            "axis_isolation": "within each case both plausible candidates share all typed context values",
            "development_seeds": list(DEVELOPMENT_SEEDS),
            "confirmation_seeds": list(CONFIRMATION_SEEDS),
            "samples_per_context": args.samples_per_context,
            "epochs": args.epochs,
            "bootstrap_samples": args.bootstrap_samples,
            "confirmation_policy": "confirmation seeds untouched until analytic cases pass development audit",
        },
        "cases": {
            "total": len(cases),
            "probes": [probe.name for probe in GROUNDING_PROBES],
            "target_old": sum("target-old" in case.tags for case in cases),
            "target_new": sum("target-new" in case.tags for case in cases),
        },
        "conditions": {
            name: list(indices) for name, indices in GROUNDING_MODEL_CONDITIONS.items()
        },
        **summary,
        "runs": runs,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"report: {output}", flush=True)


if __name__ == "__main__":
    main()
