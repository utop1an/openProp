from __future__ import annotations

import argparse
import json
from pathlib import Path

from openprop.observation_grounding import (
    CONFIRMATION_SEEDS,
    DEVELOPMENT_SEEDS,
    SCENE_SCHEDULES,
    aggregate_observation_grounding_runs,
    evaluate_observation_grounding_seed,
    observation_grounding_benchmark,
    scene_conditioned_observation_data,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate downstream grounding under inspection-frequency confounding."
    )
    parser.add_argument(
        "--phase", choices=("development", "confirmation"), default="confirmation"
    )
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--samples-per-scene", type=int, default=600)
    parser.add_argument("--test-samples-per-scene", type=int, default=20)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--report-output", type=Path)
    args = parser.parse_args()
    seeds = tuple(
        args.seeds
        or (DEVELOPMENT_SEEDS if args.phase == "development" else CONFIRMATION_SEEDS)
    )
    if not seeds or len(set(seeds)) != len(seeds):
        parser.error("seeds must contain distinct values")
    if args.samples_per_scene <= 0 or args.test_samples_per_scene <= 0:
        parser.error("sample counts must be positive")
    output = args.report_output or Path(
        f"artifacts/observation_grounding_{args.phase}.json"
    )

    cases = observation_grounding_benchmark()
    runs = []
    for seed in seeds:
        dataset = scene_conditioned_observation_data(
            samples_per_scene=args.samples_per_scene,
            test_samples_per_scene=args.test_samples_per_scene,
            seed=seed,
        )
        run = evaluate_observation_grounding_seed(
            seed=seed,
            dataset=dataset,
            cases=cases,
        )
        runs.append(run)
        print(
            f"seed={seed:>4d} "
            + " ".join(
                f"{name}={run['conditions'][name]['top1']:.3f}"
                for name in ("naive", "interval_aware", "oracle")
            ),
            flush=True,
        )

    summary = aggregate_observation_grounding_runs(
        runs,
        bootstrap_samples=args.bootstrap_samples,
    )
    payload = {
        "phase": args.phase,
        "protocol": {
            "claim_scope": "synthetic controlled decision evidence",
            "latent_hazard_per_hour": 0.25,
            "scene_inspection_intervals_hours": dict(SCENE_SCHEDULES),
            "development_seeds": list(DEVELOPMENT_SEEDS),
            "confirmation_seeds": list(CONFIRMATION_SEEDS),
            "samples_per_scene": args.samples_per_scene,
            "test_samples_per_scene": args.test_samples_per_scene,
            "cases_fixed_before_confirmation": True,
            "case_design": "target has a 3h observation and distractor a 4h observation under equal latent hazards",
            "target_scene_balance": "20 frequent-scene and 20 sparse-scene targets",
            "truth_boundary": "current truth labels never enter matcher entities",
            "candidate_order_policy": "analytic cases alternate candidate order; reversal is tested",
            "query_policy": "scene is persistence context but is not a query constraint",
            "primary_metric": "overall top1",
            "secondary_metrics": [
                "worst target-scene top1",
                "absolute target-scene top1 gap",
            ],
            "bootstrap_samples": args.bootstrap_samples,
        },
        "cases": {
            "total": len(cases),
            "target_frequent_scene": sum(
                "target-frequent-scene" in case.tags for case in cases
            ),
            "target_sparse_scene": sum(
                "target-sparse-scene" in case.tags for case in cases
            ),
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
