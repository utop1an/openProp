from __future__ import annotations

import argparse
import json
from pathlib import Path

from openprop.paper_claims import verify_claim_manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify paper claims against frozen artifact hashes and metrics."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("paper/claims.json"),
    )
    parser.add_argument("--repository-root", type=Path)
    parser.add_argument("--report-output", type=Path)
    args = parser.parse_args()
    report = verify_claim_manifest(
        args.manifest,
        repository_root=args.repository_root,
    )
    if args.report_output is not None:
        args.report_output.parent.mkdir(parents=True, exist_ok=True)
        args.report_output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(
        f"verified={report['verified']} claims={report['claims']} "
        f"artifacts={report['artifacts']} metric_checks={report['metric_checks']}"
    )


if __name__ == "__main__":
    main()
