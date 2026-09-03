from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from openprop.combined_confidence import (
    apply_combined_confidence_calibration,
    fit_combined_confidence_calibration,
)
from openprop.query_decision import (
    apply_query_acceptance_policy,
    calibrate_query_acceptance_policy,
)
from openprop.visual_calibration import (
    apply_acceptance_policy,
    calibrate_acceptance_policy,
)
from openprop.visual_evaluation import (
    VisualEvaluationDataset,
    read_visual_results_jsonl,
    write_visual_results_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze the complete calibration-only visual decision pipeline."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--system", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--policy-output", type=Path, required=True)
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
    parser.add_argument("--max-false-update-rate", type=float, default=0.0)
    parser.add_argument("--max-false-answer-rate", type=float, default=0.0)
    parser.add_argument("--minimum-source-rows", type=int, default=30)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = (args.input.resolve(), args.output.resolve(), args.policy_output.resolve())
    if len(set(paths)) != len(paths):
        raise ValueError("input, output, and policy output must differ")
    dataset = read_visual_results_jsonl(args.input)
    association_calibration = tuple(
        row for row in dataset.associations
        if row.system == args.system and row.split == "calibration"
    )
    association_policy = calibrate_acceptance_policy(
        association_calibration,
        acceptance_thresholds=args.acceptance_thresholds,
        margin_thresholds=args.margin_thresholds,
        null_scales=args.null_scales,
        candidate_count_powers=args.candidate_count_powers,
        max_false_update_rate=args.max_false_update_rate,
    )
    selected_associations = tuple(
        row for row in dataset.associations if row.system == args.system
    )
    association_applied = apply_acceptance_policy(
        selected_associations, association_policy
    )
    association_index = {
        (row.record_id, row.system): row for row in association_applied
    }
    after_association = tuple(
        association_index.get((row.record_id, row.system), row)
        for row in dataset.associations
    )

    confidence_calibration_rows = tuple(
        row for row in after_association
        if row.system == args.system and row.split == "calibration"
    )
    confidence_policy = fit_combined_confidence_calibration(
        confidence_calibration_rows,
        minimum_source_rows=args.minimum_source_rows,
    )
    confidence_applied = apply_combined_confidence_calibration(
        tuple(row for row in after_association if row.system == args.system),
        confidence_policy,
    )
    confidence_index = {
        (row.record_id, row.system): row for row in confidence_applied
    }
    final_associations = tuple(
        confidence_index.get((row.record_id, row.system), row)
        for row in after_association
    )

    query_calibration = tuple(
        row for row in dataset.queries
        if row.system == args.system and row.split == "calibration"
    )
    query_policy = calibrate_query_acceptance_policy(
        query_calibration,
        acceptance_thresholds=args.acceptance_thresholds,
        margin_thresholds=args.margin_thresholds,
        null_scales=args.null_scales,
        candidate_count_powers=args.candidate_count_powers,
        max_false_answer_rate=args.max_false_answer_rate,
    )
    query_applied = apply_query_acceptance_policy(
        tuple(row for row in dataset.queries if row.system == args.system),
        query_policy,
    )
    query_index = {(row.record_id, row.system): row for row in query_applied}
    final_queries = tuple(
        query_index.get((row.record_id, row.system), row) for row in dataset.queries
    )
    final_dataset = VisualEvaluationDataset(
        dataset.properties, final_associations, final_queries
    )
    write_visual_results_jsonl(args.output, final_dataset)
    input_bytes = args.input.read_bytes()
    output_bytes = args.output.read_bytes()
    audit = {
        "schema_version": 1,
        "system": args.system,
        "input": str(args.input),
        "input_sha256": hashlib.sha256(input_bytes).hexdigest(),
        "output": str(args.output),
        "output_sha256": hashlib.sha256(output_bytes).hexdigest(),
        "calibration_only_selection": True,
        "test_truth_used_for_selection": False,
        "execution_order": [
            "association_null_and_admission",
            "combined_update_confidence",
            "final_query_null_and_admission",
        ],
        "calibration_populations": {
            "association_sha256": _record_hash(association_calibration),
            "combined_confidence_sha256": _record_hash(confidence_calibration_rows),
            "query_sha256": _record_hash(query_calibration),
        },
        "policies": {
            "association": asdict(association_policy),
            "combined_confidence": asdict(confidence_policy),
            "query": asdict(query_policy),
        },
    }
    args.policy_output.parent.mkdir(parents=True, exist_ok=True)
    args.policy_output.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"system={args.system} association_rows={len(association_calibration)} "
        f"query_rows={len(query_calibration)}"
    )
    print(f"output: {args.output}")
    print(f"policy: {args.policy_output}")


def _record_hash(rows: tuple[object, ...]) -> str:
    identifiers = "\n".join(sorted(str(getattr(row, "record_id")) for row in rows))
    return hashlib.sha256(identifiers.encode("utf-8")).hexdigest()


if __name__ == "__main__":
    main()
