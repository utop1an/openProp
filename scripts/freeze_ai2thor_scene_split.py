from __future__ import annotations

import argparse
import json
from pathlib import Path

from openprop.ai2thor_protocol import (
    DEFAULT_AI2THOR_SPLIT_SEED,
    write_ai2thor_scene_split,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze or verify the model-output-blind iTHOR scene split."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/ai2thor_protocol/scene_split.json"),
    )
    parser.add_argument("--seed", default=DEFAULT_AI2THOR_SPLIT_SEED)
    parser.add_argument("--check", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = write_ai2thor_scene_split(args.output, seed=args.seed, check=args.check)
    counts = {split: 0 for split in ("development", "calibration", "test")}
    for row in payload["assignments"]:
        counts[row["split"]] += 1
    print(
        json.dumps(
            {
                "output": str(args.output),
                "check": args.check,
                "split_sha256": payload["split_sha256"],
                "counts": counts,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
