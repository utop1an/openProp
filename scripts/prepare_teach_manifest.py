from __future__ import annotations

import argparse
from pathlib import Path

from openprop.teach_manifest import (
    prepare_official_teach_manifest,
    write_teach_manifest_preparation_report,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare a strict OpenProp manifest from official TEACh archives."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="extracted TEACh root containing games/ and images/",
    )
    parser.add_argument("--games-root", type=Path)
    parser.add_argument("--states-root", type=Path)
    parser.add_argument("--splits", nargs="+")
    parser.add_argument(
        "--output-manifest",
        type=Path,
        default=Path("data/teach/manifest.jsonl"),
    )
    parser.add_argument(
        "--initial-state-directory",
        type=Path,
        default=Path("data/teach/prepared/initial_states"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("artifacts/teach_manifest_preparation.json"),
    )
    args = parser.parse_args()
    games_root = args.games_root or args.data_root / "games"
    states_root = args.states_root or args.data_root / "images"
    report = prepare_official_teach_manifest(
        games_root=games_root,
        states_root=states_root,
        output_manifest=args.output_manifest,
        initial_state_directory=args.initial_state_directory,
        splits=args.splits,
    )
    write_teach_manifest_preparation_report(args.report, report)
    print(
        f"sessions={report['sessions']} floorplans={report['floorplans']} "
        f"splits={','.join(report['splits'])}"
    )
    print(f"manifest_sha256={report['manifest_sha256']}")
    print(f"manifest: {args.output_manifest}")
    print(f"report: {args.report}")


if __name__ == "__main__":
    main()

