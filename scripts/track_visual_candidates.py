from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from openprop.candidate_replay import build_tracked_vlm_input, track_candidate_input
from openprop.candidate_calibration import candidate_tracking_policy_from_frozen_payload
from openprop.candidate_tracking import CandidateTrackingPolicy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Track detector proposals into opaque VLM candidate identities."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path)
    parser.add_argument("--policy", type=Path)
    parser.add_argument("--minimum-proposal-confidence", type=float, default=0.25)
    parser.add_argument("--minimum-link-score", type=float, default=0.35)
    parser.add_argument("--max-missed-frames", type=int, default=2)
    parser.add_argument("--max-active-tracks", type=int, default=12)
    parser.add_argument("--max-proposals-per-frame", type=int, default=12)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audit_path = args.audit or Path(str(args.output) + ".audit.json")
    paths = (
        args.input.resolve(), args.output.resolve(), audit_path.resolve(),
        *((args.policy.resolve(),) if args.policy is not None else ()),
    )
    if len(set(paths)) != len(paths):
        raise ValueError("candidate input, output, and audit paths must differ")
    input_bytes = args.input.read_bytes()
    payload = json.loads(input_bytes)
    if not isinstance(payload, dict):
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
    run = track_candidate_input(payload, policy=policy)
    output = build_tracked_vlm_input(payload, run, policy=policy)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    output_bytes = args.output.read_bytes()
    audit = {
        "schema_version": 1,
        "input": str(args.input),
        "input_sha256": hashlib.sha256(input_bytes).hexdigest(),
        "output": str(args.output),
        "output_sha256": hashlib.sha256(output_bytes).hexdigest(),
        "truth_used_for_tracking": False,
        "frozen_policy": (
            None
            if args.policy is None
            else {
                "path": str(args.policy),
                "sha256": hashlib.sha256(policy_bytes).hexdigest(),
            }
        ),
        "candidate_generation": output["candidate_generation"],
        "per_frame": [
            {
                "frame_id": frame.source_frame.frame_id,
                "candidates": len(frame.candidates),
                "rejected_proposal_ids": list(frame.rejected_proposal_ids),
                "expired_track_ids": list(frame.expired_track_ids),
                "capacity_exceeded": frame.capacity_exceeded,
            }
            for frame in run.frames
        ],
    }
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"frames={len(run.frames)} tracks={len(run.track_ids)} "
        f"capacity_exceeded={output['candidate_generation']['capacity_exceeded_frames']}"
    )
    print(f"output: {args.output}")
    print(f"audit: {audit_path}")


if __name__ == "__main__":
    main()
