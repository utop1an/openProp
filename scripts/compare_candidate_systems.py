from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from openprop.candidate_evaluation import candidate_evaluation_from_mapping
from openprop.candidate_statistics import paired_candidate_system_comparison


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run exactly paired cluster-bootstrap inference for candidate systems."
    )
    parser.add_argument("--input", nargs="+", type=Path, required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--system", required=True)
    parser.add_argument(
        "--split", choices=("development", "calibration", "test"), required=True
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    resolved = [path.resolve() for path in (*args.input, args.output)]
    if len(resolved) != len(set(resolved)):
        raise ValueError("candidate comparison inputs/output must be distinct")
    evaluations = []
    input_hashes = {}
    for path in args.input:
        data = path.read_bytes()
        payload = json.loads(data)
        if not isinstance(payload, dict) or payload.get("schema_version") != 1:
            raise ValueError("candidate result must be a schema-version-1 object")
        evaluation = payload.get("evaluation")
        if not isinstance(evaluation, dict):
            raise ValueError("candidate result is missing evaluation")
        evaluations.append(candidate_evaluation_from_mapping(evaluation))
        input_hashes[str(path)] = hashlib.sha256(data).hexdigest()
    report = paired_candidate_system_comparison(
        tuple(evaluations),
        baseline=args.baseline,
        system=args.system,
        split=args.split,
        bootstrap_replicates=args.bootstrap_replicates,
        seed=args.seed,
    )
    report["input_sha256"] = dict(sorted(input_hashes.items()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"episodes={report['population']['episodes']} output: {args.output}")


if __name__ == "__main__":
    main()
