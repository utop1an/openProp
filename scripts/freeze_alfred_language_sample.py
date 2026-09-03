from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from openprop.alfred_adapter import load_alfred_language_dataset
from openprop.alfred_language_evaluation import select_stratified_cases


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freeze an ALFRED language sample manifest before model requests."
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--split", default="valid_unseen")
    parser.add_argument("--trajectories-per-task", type=int, default=10)
    parser.add_argument("--trajectory-offset", type=int, required=True)
    parser.add_argument("--annotation-index", type=int, required=True)
    parser.add_argument("--reference-offset", type=int, default=0)
    parser.add_argument("--reference-annotation-index", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    dataset = load_alfred_language_dataset(args.root, splits=(args.split,))
    selected = select_stratified_cases(
        dataset.cases,
        split=args.split,
        trajectories_per_task=args.trajectories_per_task,
        trajectory_offset=args.trajectory_offset,
        annotation_index=args.annotation_index,
    )
    reference = select_stratified_cases(
        dataset.cases,
        split=args.split,
        trajectories_per_task=args.trajectories_per_task,
        trajectory_offset=args.reference_offset,
        annotation_index=args.reference_annotation_index,
    )
    selected_tasks = {case.task_id for case in selected}
    reference_tasks = {case.task_id for case in reference}
    selected_queries = {case.query for case in selected}
    reference_queries = {case.query for case in reference}
    if selected_tasks & reference_tasks:
        parser.error("selected and reference samples share task IDs")
    payload = {
        "protocol": {
            "frozen_before_model_requests": True,
            "split": args.split,
            "trajectories_per_task": args.trajectories_per_task,
            "trajectory_offset": args.trajectory_offset,
            "annotation_index": args.annotation_index,
            "ordering": "case_id lexical order within task type",
            "reference_trajectory_offset": args.reference_offset,
            "reference_annotation_index": args.reference_annotation_index,
            "task_id_overlap_with_reference": 0,
            "query_overlap_with_reference": len(selected_queries & reference_queries),
            "gold_labels_in_manifest": False,
        },
        "cases": [
            {
                "case_id": case.case_id,
                "task_id": case.task_id,
                "task_type": case.task_type,
                "annotation_index": case.annotation_index,
                "query_sha256": hashlib.sha256(case.query.encode("utf-8")).hexdigest(),
                "source_path": case.source_path,
            }
            for case in selected
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload["protocol"], indent=2, sort_keys=True))
    print(f"cases: {len(selected)}")
    print(f"report: {args.output}")


if __name__ == "__main__":
    main()
