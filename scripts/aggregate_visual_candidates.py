from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from openprop.candidate_evaluation import (
    aggregate_candidate_tracking_matrix,
    candidate_evaluation_from_mapping,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate per-episode candidate evaluations by system."
    )
    parser.add_argument("--input", nargs="+", type=Path, required=True)
    parser.add_argument(
        "--split", choices=("development", "calibration", "test"), required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    resolved = [path.resolve() for path in (*args.input, args.output)]
    if len(resolved) != len(set(resolved)):
        raise ValueError("candidate aggregate inputs/output must be distinct")
    evaluations = []
    seen: set[tuple[str, str]] = set()
    source_hashes = {}
    for path in args.input:
        data = path.read_bytes()
        payload = json.loads(data)
        if not isinstance(payload, dict) or payload.get("schema_version") != 1:
            raise ValueError("candidate result must be a schema-version-1 object")
        episode = payload.get("episode_id")
        system = payload.get("system")
        if not isinstance(episode, str) or not isinstance(system, str):
            raise ValueError("candidate result requires episode_id and system")
        key = (episode, system)
        if key in seen:
            raise ValueError("candidate episode/system result is duplicated")
        seen.add(key)
        evaluation = payload.get("evaluation")
        if not isinstance(evaluation, dict):
            raise ValueError("candidate result is missing evaluation")
        evaluations.append(candidate_evaluation_from_mapping(evaluation))
        source_hashes[str(path)] = hashlib.sha256(data).hexdigest()
    report = aggregate_candidate_tracking_matrix(tuple(evaluations), split=args.split)
    report["input_sha256"] = dict(sorted(source_hashes.items()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"systems={len(report['systems'])} output: {args.output}")


if __name__ == "__main__":
    main()
