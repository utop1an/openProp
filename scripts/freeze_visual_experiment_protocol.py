from __future__ import annotations

import argparse
import json
from pathlib import Path

from openprop.visual_experiment_protocol import write_visual_experiment_protocol


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freeze or byte-check the ICLR visual experiment matrix."
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("artifacts/visual_protocol/experiment_protocol.json"),
    )
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    payload = write_visual_experiment_protocol(args.output, check=args.check)
    print(json.dumps({
        "output": str(args.output), "check": args.check,
        "protocol_sha256": payload["protocol_sha256"],
        "systems": len(payload["systems"]),
        "primary_comparisons": len(payload["primary_comparisons"]),
        "evidence_tiers": len(payload["evidence_tiers"]),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
