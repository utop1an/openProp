"""Download a bounded ADT ground-truth screening pool without exposing CDN URLs."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any, Sequence


DEFAULT_ACTIVITIES = ("clean", "decoration", "meal")
SEQUENCE_PATTERN = re.compile(r"^Apartment_release_([^_]+)_")


def load_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("sequences"), dict):
        raise ValueError("ADT manifest must contain a sequences object")
    return payload


def select_sequences(
    manifest: dict[str, Any], activities: Sequence[str]
) -> list[dict[str, Any]]:
    allowed = frozenset(activities)
    selected: list[dict[str, Any]] = []
    for name, artifacts in manifest["sequences"].items():
        match = SEQUENCE_PATTERN.match(name)
        if match is None or match.group(1) not in allowed or "10s_sample" in name:
            continue
        groundtruth = artifacts.get("main_groundtruth")
        if not isinstance(groundtruth, dict):
            raise ValueError(f"missing main_groundtruth metadata for {name}")
        byte_count = groundtruth.get("file_size_bytes")
        if not isinstance(byte_count, int) or byte_count <= 0:
            raise ValueError(f"invalid main_groundtruth byte count for {name}")
        selected.append(
            {
                "activity": match.group(1),
                "declared_groundtruth_bytes": byte_count,
                "sequence_name": name,
            }
        )
    return sorted(selected, key=lambda row: row["sequence_name"])


def build_plan(
    manifest_path: Path,
    manifest: dict[str, Any],
    activities: Sequence[str],
) -> dict[str, Any]:
    selected = select_sequences(manifest, activities)
    if not selected:
        raise ValueError("screening selection is empty")
    return {
        "schema_version": 1,
        "protocol_id": "openprop-adt-groundtruth-screening-v1",
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "activities": sorted(set(activities)),
        "data_types": [6],
        "declared_download_bytes": sum(
            row["declared_groundtruth_bytes"] for row in selected
        ),
        "sequences": selected,
        "contains_download_urls": False,
        "performance_evidence": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cdn-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--plan-output", type=Path, required=True)
    parser.add_argument("--downloader", default="aria_dataset_downloader")
    parser.add_argument("--activities", nargs="+", default=list(DEFAULT_ACTIVITIES))
    parser.add_argument("--max-gib", type=float, default=2.5)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = args.cdn_manifest.resolve(strict=True)
    manifest = load_manifest(manifest_path)
    plan = build_plan(manifest_path, manifest, args.activities)
    maximum_bytes = int(args.max_gib * (1024**3))
    if plan["declared_download_bytes"] > maximum_bytes:
        raise SystemExit(
            "selection exceeds --max-gib: "
            f"{plan['declared_download_bytes'] / (1024**3):.2f} GiB > {args.max_gib:.2f} GiB"
        )

    args.plan_output.parent.mkdir(parents=True, exist_ok=True)
    args.plan_output.write_text(
        json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"planned_sequences={len(plan['sequences'])} "
        f"declared_gib={plan['declared_download_bytes'] / (1024**3):.2f}"
    )
    print(f"plan={args.plan_output}")
    if args.dry_run:
        return

    args.output_root.mkdir(parents=True, exist_ok=True)
    command = [
        args.downloader,
        "-c",
        str(manifest_path),
        "-o",
        str(args.output_root),
        "-l",
        *(row["sequence_name"] for row in plan["sequences"]),
        "-d",
        "6",
    ]
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
