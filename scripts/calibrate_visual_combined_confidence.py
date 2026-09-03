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
from openprop.visual_evaluation import (
    VisualEvaluationDataset,
    read_visual_results_jsonl,
    write_visual_results_jsonl,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit calibration-only monotone combined-update confidence."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--system", required=True)
    parser.add_argument("--minimum-source-rows", type=int, default=30)
    parser.add_argument("--policy-output", type=Path, required=True)
    parser.add_argument("--results-output", type=Path, required=True)
    args = parser.parse_args()
    paths = (args.input.resolve(), args.policy_output.resolve(), args.results_output.resolve())
    if len(set(paths)) != len(paths):
        raise ValueError("input, policy output, and results output must differ")
    dataset = read_visual_results_jsonl(args.input)
    calibration_rows = tuple(
        row for row in dataset.associations
        if row.split == "calibration" and row.system == args.system
    )
    calibration = fit_combined_confidence_calibration(
        calibration_rows, minimum_source_rows=args.minimum_source_rows
    )
    selected = tuple(row for row in dataset.associations if row.system == args.system)
    applied = {
        (row.record_id, row.system): row
        for row in apply_combined_confidence_calibration(selected, calibration)
    }
    associations = tuple(
        applied.get((row.record_id, row.system), row) for row in dataset.associations
    )
    write_visual_results_jsonl(
        args.results_output,
        VisualEvaluationDataset(dataset.properties, associations, dataset.queries),
    )
    policy = {
        "schema_version": 1,
        "input": str(args.input),
        "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "calibration_only_selection": True,
        "test_truth_used_for_selection": False,
        "calibration_record_ids_sha256": hashlib.sha256(
            "\n".join(sorted(row.record_id for row in calibration_rows)).encode()
        ).hexdigest(),
        "policy": asdict(calibration),
    }
    args.policy_output.parent.mkdir(parents=True, exist_ok=True)
    args.policy_output.write_text(
        json.dumps(policy, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"system={args.system} calibration_rows={calibration.calibration_rows} "
        f"source_models={len(calibration.source_models)}"
    )


if __name__ == "__main__":
    main()
