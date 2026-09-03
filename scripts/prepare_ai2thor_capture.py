from __future__ import annotations

import argparse
import json
from pathlib import Path

from openprop.ai2thor_capture import prepare_ai2thor_capture_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify an AI2-THOR bundle and split VLM inputs from truth."
    )
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source", default="ai2thor-rgb")
    parser.add_argument("--movement-threshold-metres", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = prepare_ai2thor_capture_manifest(
        args.manifest,
        args.output_dir,
        source=args.source,
        movement_threshold_metres=args.movement_threshold_metres,
    )
    report_path = args.output_dir / "preparation-report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
