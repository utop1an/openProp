from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from openprop.alfred_adapter import load_alfred_language_dataset


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit official ALFRED lite trajectories for typed language goals."
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/alfred_language_feasibility_audit.json"),
    )
    args = parser.parse_args()
    dataset = load_alfred_language_dataset(args.root)
    payload = {
        "audit": dataset.audit,
        "sample_cases": [asdict(item) for item in dataset.cases[:12]],
        "sample_exclusions": [asdict(item) for item in dataset.exclusions[:12]],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(dataset.audit, ensure_ascii=False, indent=2, sort_keys=True))
    print(f"report: {args.output}")


if __name__ == "__main__":
    main()
