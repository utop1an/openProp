from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from openprop.visual_evaluation import read_visual_results_jsonl
from openprop.visual_primary_statistics import primary_visual_query_comparisons


def main() -> None:
    parser = argparse.ArgumentParser(description="Run family-wise primary visual query comparisons.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--main-system", required=True)
    parser.add_argument("--baselines", nargs="+", required=True)
    parser.add_argument("--split", choices=("development", "calibration", "test"), required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve():
        raise ValueError("primary visual input/output must differ")
    report = primary_visual_query_comparisons(
        read_visual_results_jsonl(args.input), main_system=args.main_system,
        baselines=tuple(args.baselines), split=args.split,
        bootstrap_replicates=args.bootstrap_replicates, seed=args.seed,
    )
    report["input"] = str(args.input)
    report["input_sha256"] = hashlib.sha256(args.input.read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"comparisons={len(report['comparisons'])} output: {args.output}")


if __name__ == "__main__":
    main()
