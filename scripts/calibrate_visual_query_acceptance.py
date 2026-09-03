from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from openprop.query_decision import (
    apply_query_acceptance_policy,
    calibrate_query_acceptance_policy,
)
from openprop.visual_evaluation import (
    VisualEvaluationDataset,
    read_visual_results_jsonl,
    write_visual_results_jsonl,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit final-query admission gates on calibration rows only."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--system", required=True)
    parser.add_argument(
        "--acceptance-thresholds", nargs="+", type=float,
        default=(0.50, 0.60, 0.70, 0.80, 0.90),
    )
    parser.add_argument(
        "--margin-thresholds", nargs="+", type=float,
        default=(0.00, 0.05, 0.10, 0.15, 0.20),
    )
    parser.add_argument(
        "--null-scales", nargs="+", type=float,
        default=(0.25, 0.50, 1.0, 2.0, 4.0),
    )
    parser.add_argument(
        "--candidate-count-powers", nargs="+", type=float,
        default=(0.0, 0.5, 1.0),
    )
    parser.add_argument("--max-false-answer-rate", type=float, default=0.0)
    parser.add_argument("--policy-output", type=Path, required=True)
    parser.add_argument("--results-output", type=Path, required=True)
    args = parser.parse_args()
    paths = (args.input.resolve(), args.policy_output.resolve(), args.results_output.resolve())
    if len(set(paths)) != len(paths):
        raise ValueError("input, policy output, and results output must differ")
    dataset = read_visual_results_jsonl(args.input)
    calibration = tuple(
        row for row in dataset.queries
        if row.split == "calibration" and row.system == args.system
    )
    policy = calibrate_query_acceptance_policy(
        calibration,
        acceptance_thresholds=args.acceptance_thresholds,
        margin_thresholds=args.margin_thresholds,
        null_scales=args.null_scales,
        candidate_count_powers=args.candidate_count_powers,
        max_false_answer_rate=args.max_false_answer_rate,
    )
    selected = tuple(row for row in dataset.queries if row.system == args.system)
    applied = {
        (row.record_id, row.system): row
        for row in apply_query_acceptance_policy(selected, policy)
    }
    queries = tuple(
        applied.get((row.record_id, row.system), row) for row in dataset.queries
    )
    write_visual_results_jsonl(
        args.results_output,
        VisualEvaluationDataset(dataset.properties, dataset.associations, queries),
    )
    policy_payload = {
        "schema_version": 1,
        "input": str(args.input),
        "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "calibration_only_selection": True,
        "test_truth_used_for_selection": False,
        "calibration_record_ids_sha256": hashlib.sha256(
            "\n".join(sorted(row.record_id for row in calibration)).encode("utf-8")
        ).hexdigest(),
        "policy": asdict(policy),
    }
    args.policy_output.parent.mkdir(parents=True, exist_ok=True)
    args.policy_output.write_text(
        json.dumps(policy_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"system={args.system} calibration_rows={policy.calibration_rows} "
        f"acceptance={policy.acceptance_threshold:.3f} "
        f"margin={policy.margin_threshold:.3f} "
        f"null_scale={policy.null_scale:.3f} "
        f"count_power={policy.candidate_count_power:.3f} "
        f"false_answers={policy.false_answers}"
    )


if __name__ == "__main__":
    main()
