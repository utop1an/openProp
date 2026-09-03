from __future__ import annotations

import argparse
import json
from pathlib import Path

from openprop.association_benchmark import (
    ASSOCIATION_CONDITIONS,
    association_benchmark_registry,
    association_benchmark_split,
    calibrate_association_policy,
    default_association_benchmark_associator,
    evaluate_association,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate and evaluate robust multi-entity visual association."
    )
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--calibration-per-condition", type=int, default=20)
    parser.add_argument("--test-per-condition", type=int, default=40)
    parser.add_argument("--max-false-update-rate", type=float, default=0.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/association_benchmark.json"),
    )
    args = parser.parse_args()
    if args.calibration_per_condition <= 0 or args.test_per_condition <= 0:
        parser.error("case counts must be positive")
    if not 0.0 <= args.max_false_update_rate <= 1.0:
        parser.error("max false-update rate must be in [0, 1]")

    split = association_benchmark_split(
        seed=args.seed,
        calibration_per_condition=args.calibration_per_condition,
        test_per_condition=args.test_per_condition,
    )
    registry = association_benchmark_registry()
    base = default_association_benchmark_associator(registry)
    calibration = calibrate_association_policy(
        base,
        split.calibration,
        max_false_update_rate=args.max_false_update_rate,
    )
    associator = default_association_benchmark_associator(
        registry,
        policy=calibration.policy,
    )
    test = evaluate_association(associator, split.test)
    policy = calibration.policy
    payload = {
        "protocol": {
            "scope": "synthetic mechanism validation",
            "seed": args.seed,
            "conditions": list(ASSOCIATION_CONDITIONS),
            "calibration_cases": len(split.calibration),
            "test_cases": len(split.test),
            "group_disjoint": True,
            "threshold_selection": "calibration split only",
            "test_outcomes_used_for_selection": False,
            "truth_boundary": (
                "target_entity_id is evaluation-only and never passed to the associator"
            ),
            "abstention_policy": (
                "ambiguous, misleading, null, and conflicting detections remain uncommitted"
            ),
            "candidate_order_audit": "reverse complete candidate order per test case",
            "query_paraphrase_audit": (
                "same typed constraints with a distinct surface form"
            ),
            "primary_safety_metric": "false_update_rate",
            "primary_utility_metric": "correct_updates",
        },
        "selected_policy": {
            "acceptance_threshold": policy.acceptance_threshold,
            "margin_threshold": policy.margin_threshold,
            "minimum_detection_confidence": policy.minimum_detection_confidence,
            "minimum_value_confidence": policy.minimum_value_confidence,
            "null_weight": policy.null_weight,
            "query_weight": policy.query_weight,
            "visual_weight": policy.visual_weight,
            "track_weight": policy.track_weight,
        },
        "calibration": {
            "searched_policies": calibration.searched_policies,
            "feasible_policies": calibration.feasible_policies,
            "max_false_update_rate": calibration.max_false_update_rate,
            "evaluation": calibration.validation.to_dict(),
        },
        "test": test.to_dict(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"policy=accept:{policy.acceptance_threshold:.2f},"
        f"margin:{policy.margin_threshold:.2f} "
        f"test_correct={test.correct_updates}/{test.total} "
        f"false_updates={test.false_updates} "
        f"abstentions={test.abstentions} "
        f"order_invariance={test.candidate_order_invariance:.3f} "
        f"paraphrase_invariance={test.query_paraphrase_invariance:.3f}",
        flush=True,
    )
    print(f"report: {args.output}", flush=True)


if __name__ == "__main__":
    main()
