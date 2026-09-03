from __future__ import annotations

import argparse
import json
from pathlib import Path

from openprop.ai2thor_capture import verify_ai2thor_capture_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify a content-addressed OpenProp AI2-THOR capture bundle."
    )
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--report", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = verify_ai2thor_capture_manifest(args.manifest)
    rendered = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
