from __future__ import annotations

import argparse
import json
from pathlib import Path

from openprop.visual_evaluation import read_visual_results_jsonl
from openprop.visual_statistics import paired_visual_system_comparison


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run paired cluster-level inference for two visual systems."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--system", required=True)
    parser.add_argument("--split", choices=("development", "calibration", "test"), required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument(
        "--query-only",
        action="store_true",
        help="Skip association inference when systems have different detections.",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = paired_visual_system_comparison(
        read_visual_results_jsonl(args.input),
        baseline=args.baseline,
        system=args.system,
        split=args.split,
        bootstrap_replicates=args.bootstrap_replicates,
        seed=args.seed,
        include_association=not args.query_only,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"output: {args.output}")


if __name__ == "__main__":
    main()
