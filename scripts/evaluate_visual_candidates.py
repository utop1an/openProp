from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from openprop.candidate_evaluation import (
    CandidateFrameTruth,
    CandidateTruthObject,
    aggregate_candidate_tracking,
    evaluate_candidate_tracking,
)
from openprop.candidate_replay import track_candidate_input
from openprop.candidate_calibration import candidate_tracking_policy_from_frozen_payload
from openprop.candidate_tracking import CandidateTrackingPolicy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate frozen candidate tracking after attaching separate truth."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--system", default="candidate-tracker")
    parser.add_argument("--audit", type=Path)
    parser.add_argument("--policy", type=Path)
    parser.add_argument("--minimum-proposal-confidence", type=float, default=0.25)
    parser.add_argument("--minimum-link-score", type=float, default=0.35)
    parser.add_argument("--max-missed-frames", type=int, default=2)
    parser.add_argument("--max-active-tracks", type=int, default=12)
    parser.add_argument("--max-proposals-per-frame", type=int, default=12)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audit_path = args.audit or Path(str(args.output) + ".audit.json")
    paths = (
        args.input.resolve(), args.truth.resolve(), args.output.resolve(), audit_path.resolve(),
        *((args.policy.resolve(),) if args.policy is not None else ()),
    )
    if len(set(paths)) != len(paths):
        raise ValueError("candidate input, truth, output, and audit paths must differ")
    input_bytes = args.input.read_bytes()
    input_payload = json.loads(input_bytes)
    if not isinstance(input_payload, dict):
        raise ValueError("candidate input must be a JSON object")
    policy_bytes = args.policy.read_bytes() if args.policy is not None else None
    policy = (
        candidate_tracking_policy_from_frozen_payload(json.loads(policy_bytes))
        if policy_bytes is not None
        else CandidateTrackingPolicy(
            minimum_proposal_confidence=args.minimum_proposal_confidence,
            minimum_link_score=args.minimum_link_score,
            max_missed_frames=args.max_missed_frames,
            max_active_tracks=args.max_active_tracks,
            max_proposals_per_frame=args.max_proposals_per_frame,
        )
    )
    # This outcome is immutable before the truth file is read.
    run = track_candidate_input(input_payload, policy=policy)

    truth_bytes = args.truth.read_bytes()
    truth = json.loads(truth_bytes)
    if not isinstance(truth, dict) or truth.get("schema_version") != 1:
        raise ValueError("candidate truth must be a schema-version-1 object")
    if truth.get("evaluation_only") is not True:
        raise ValueError("candidate truth must be marked evaluation-only")
    if truth.get("episode_id") != input_payload.get("episode_id"):
        raise ValueError("candidate truth episode does not match input")
    frame_rows = truth.get("frames")
    if not isinstance(frame_rows, list):
        raise ValueError("candidate truth frames must be an array")
    frame_truth = []
    for row in frame_rows:
        if not isinstance(row, dict) or not isinstance(row.get("objects"), list):
            raise ValueError("candidate truth frame is malformed")
        frame_truth.append(
            CandidateFrameTruth(
                _text(row.get("frame_id"), "truth frame_id"),
                tuple(
                    CandidateTruthObject(
                        _text(item.get("entity_id"), "truth entity_id"),
                        tuple(item.get("region", ())),
                    )
                    for item in row["objects"]
                    if isinstance(item, dict)
                ),
            )
        )
        if len(frame_truth[-1].objects) != len(row["objects"]):
            raise ValueError("candidate truth object must be an object")
    query = truth.get("query")
    if not isinstance(query, dict):
        raise ValueError("candidate truth query must be an object")
    target = query.get("target_entity_id")
    if target is not None:
        target = _text(target, "query target_entity_id")
    evaluation = evaluate_candidate_tracking(
        run,
        tuple(frame_truth),
        cluster_id=_text(truth.get("cluster_id"), "cluster_id"),
        record_id=_text(input_payload.get("episode_id"), "episode_id"),
        split=_text(truth.get("split"), "split"),
        system=_text(args.system, "system"),
        source=_text(truth.get("source"), "source"),
        query_frame_id=_text(query.get("frame_id"), "query frame_id"),
        query_target_entity_id=target,
        iou_threshold=args.iou_threshold,
    )
    summary = aggregate_candidate_tracking((evaluation,), split=evaluation.split)
    output_payload = {
        "schema_version": 1,
        "episode_id": input_payload["episode_id"],
        "system": args.system,
        "evaluation": asdict(evaluation),
        "summary": summary,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    output_bytes = args.output.read_bytes()
    audit = {
        "schema_version": 1,
        "input": str(args.input),
        "input_sha256": hashlib.sha256(input_bytes).hexdigest(),
        "truth": str(args.truth),
        "truth_sha256": hashlib.sha256(truth_bytes).hexdigest(),
        "output": str(args.output),
        "output_sha256": hashlib.sha256(output_bytes).hexdigest(),
        "tracking_frozen_before_truth_load": True,
        "test_truth_used_for_policy_selection": False,
        "policy": asdict(policy),
        "frozen_policy": (
            None
            if args.policy is None
            else {
                "path": str(args.policy),
                "sha256": hashlib.sha256(policy_bytes).hexdigest(),
            }
        ),
        "denominators": {
            "frames": summary["frames"],
            "truth_objects": summary["truth_objects"],
            "query_target_trials": summary["query_target_trials"],
        },
    }
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"episode={input_payload['episode_id']} frames={summary['frames']} "
        f"candidate_recall={summary['candidate_recall']}"
    )
    print(f"output: {args.output}")
    print(f"audit: {audit_path}")


def _text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


if __name__ == "__main__":
    main()
