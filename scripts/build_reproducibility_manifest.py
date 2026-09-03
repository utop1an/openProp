from __future__ import annotations

import argparse
import json
from pathlib import Path

from openprop.reproducibility import (
    build_reproducibility_manifest,
    verify_reproducibility_manifest,
    write_reproducibility_manifest,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build or check the paper reproducibility manifest."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("paper/reproducibility_manifest.json"),
    )
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--require-runtime-match", action="store_true")
    parser.add_argument("--require-git-revision", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output if args.output.is_absolute() else root / args.output
    if args.check:
        report = verify_reproducibility_manifest(
            output,
            repository_root=root,
            require_runtime_match=args.require_runtime_match,
            require_release_revision=args.require_git_revision,
        )
    else:
        payload = write_reproducibility_manifest(root, output)
        if args.require_git_revision and not payload["release_gates"][
            "clean_git_revision_bound"
        ]:
            parser.error("release build requires a clean bound git revision")
        report = {
            "written": str(output),
            "source_files": len(payload["source_snapshot"]["files"]),
            "source_tree_sha256": payload["source_snapshot"]["tree_sha256"],
            "experiments": len(payload["experiments"]),
            "release_ready": payload["release_gates"]["submission_release_ready"],
        }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

