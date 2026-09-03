from __future__ import annotations

import argparse
import json
from pathlib import Path

from openprop.reproducibility import verify_reproducibility_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify the reproducibility manifest.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("paper/reproducibility_manifest.json"),
    )
    parser.add_argument("--require-runtime-match", action="store_true")
    parser.add_argument("--require-git-revision", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    manifest = args.manifest if args.manifest.is_absolute() else root / args.manifest
    report = verify_reproducibility_manifest(
        manifest,
        repository_root=root,
        require_runtime_match=args.require_runtime_match,
        require_release_revision=args.require_git_revision,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

