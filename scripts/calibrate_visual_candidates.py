from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from openprop.candidate_calibration import (
    calibrate_candidate_tracking_policy,
    candidate_calibration_case_from_payloads,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freeze candidate proposal/link policy on calibration episodes only."
    )
    parser.add_argument("--input", nargs="+", type=Path, required=True)
    parser.add_argument("--truth", nargs="+", type=Path, required=True)
    parser.add_argument("--minimum-proposal-confidences", nargs="+", type=float, default=(0.1, 0.25, 0.5, 0.75))
    parser.add_argument("--minimum-link-scores", nargs="+", type=float, default=(0.2, 0.35, 0.5, 0.7))
    parser.add_argument("--max-missed-frames", nargs="+", type=int, default=(0, 1, 2, 4))
    parser.add_argument("--minimum-candidate-recall", type=float, default=0.9)
    parser.add_argument("--maximum-identity-switch-rate", type=float, default=0.05)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.input) != len(args.truth):
        raise ValueError("candidate calibration input/truth counts must match")
    resolved = [path.resolve() for path in (*args.input, *args.truth, args.output)]
    if len(resolved) != len(set(resolved)):
        raise ValueError("candidate calibration paths must be distinct")
    cases = []
    artifacts = []
    seen_episodes = set()
    for input_path, truth_path in zip(args.input, args.truth):
        input_bytes = input_path.read_bytes()
        truth_bytes = truth_path.read_bytes()
        input_payload = json.loads(input_bytes)
        truth_payload = json.loads(truth_bytes)
        if not isinstance(input_payload, dict) or not isinstance(truth_payload, dict):
            raise ValueError("candidate calibration artifacts must be JSON objects")
        episode = input_payload.get("episode_id")
        if not isinstance(episode, str) or episode in seen_episodes:
            raise ValueError("candidate calibration episode IDs must be unique")
        seen_episodes.add(episode)
        cases.append(candidate_calibration_case_from_payloads(input_payload, truth_payload))
        artifacts.append(
            {
                "episode_id": episode,
                "input": str(input_path),
                "input_sha256": hashlib.sha256(input_bytes).hexdigest(),
                "truth": str(truth_path),
                "truth_sha256": hashlib.sha256(truth_bytes).hexdigest(),
            }
        )
    frozen = calibrate_candidate_tracking_policy(
        tuple(cases),
        minimum_proposal_confidences=args.minimum_proposal_confidences,
        minimum_link_scores=args.minimum_link_scores,
        max_missed_frames=args.max_missed_frames,
        minimum_candidate_recall=args.minimum_candidate_recall,
        maximum_identity_switch_rate=args.maximum_identity_switch_rate,
    )
    payload = {
        "schema_version": 1,
        "calibration_only_selection": True,
        "test_truth_used_for_selection": False,
        "artifacts": artifacts,
        "frozen": asdict(frozen),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"cases={frozen.calibration_cases} searched={frozen.searched_policies} "
        f"feasible={frozen.feasible_policies} output: {args.output}"
    )


if __name__ == "__main__":
    main()
