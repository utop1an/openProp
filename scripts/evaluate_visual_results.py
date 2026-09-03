from __future__ import annotations

import argparse
import json
from pathlib import Path

from openprop.visual_evaluation import (
    aggregate_visual_evaluation,
    read_visual_results_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate frozen OpenProp visual experiment JSONL without changing "
            "thresholds or dropping failures."
        )
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--split",
        choices=("development", "calibration", "test"),
        required=True,
    )
    parser.add_argument("--ece-bins", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail unless output already equals the deterministic aggregation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.ece_bins <= 0:
        raise ValueError("ece-bins must be positive")
    dataset = read_visual_results_jsonl(args.input)
    report = aggregate_visual_evaluation(
        dataset,
        split=args.split,
        ece_bins=args.ece_bins,
    )
    report["input"] = str(args.input)
    report["input_sha256"] = _sha256(args.input.read_bytes())
    payload = (
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    if args.check:
        if not args.output.is_file():
            raise FileNotFoundError(f"missing checked output: {args.output}")
        if args.output.read_text(encoding="utf-8") != payload:
            raise ValueError("visual evaluation report drifted")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(
        f"split={args.split} systems={len(report['systems'])} "
        f"input_sha256={report['input_sha256']}"
    )
    print(f"report: {args.output}")


def _sha256(payload: bytes) -> str:
    import hashlib

    return hashlib.sha256(payload).hexdigest()


if __name__ == "__main__":
    main()
