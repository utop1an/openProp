"""Freeze an outcome-blind ADT visual-download cohort for the OpenProp pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


SEGMENTATION_QUOTAS = {
    "clean/regular": 6,
    "decoration/regular": 4,
    "decoration/skeleton": 2,
    "meal/regular": 4,
    "meal/skeleton": 2,
}


def stable_key(seed: str, name: str) -> str:
    return hashlib.sha256(f"{seed}\0{name}".encode()).hexdigest()


def stratum(sequence_name: str, activity: str) -> str:
    condition = "skeleton" if "_skeleton_" in sequence_name else "regular"
    return f"{activity}/{condition}"


def assign_splits(rows: list[dict[str, Any]], seed: str) -> dict[str, str]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        grouped[stratum(row["sequence_name"], row["activity"])].append(
            row["sequence_name"]
        )
    assigned = {}
    cycle = ("train", "train", "train", "calibration", "test")
    for group, names in sorted(grouped.items()):
        del group
        for index, name in enumerate(sorted(names, key=lambda item: stable_key(seed, item))):
            assigned[name] = cycle[index % len(cycle)]
    return assigned


def build_selection(
    screening_plan: dict[str, Any],
    cdn_manifest: dict[str, Any],
    seed: str,
) -> dict[str, Any]:
    rows = screening_plan["sequences"]
    splits = assign_splits(rows, seed)
    grouped: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        grouped[stratum(row["sequence_name"], row["activity"])].append(
            row["sequence_name"]
        )
    segmentation_names = set()
    for group, quota in SEGMENTATION_QUOTAS.items():
        names = sorted(grouped[group], key=lambda item: stable_key(seed + "/seg", item))
        if len(names) < quota:
            raise ValueError(f"stratum {group} has {len(names)} sequences, needs {quota}")
        segmentation_names.update(names[:quota])

    selected = []
    totals = {"main_groundtruth": 0, "segmentation": 0, "video_main_rgb": 0}
    for row in sorted(rows, key=lambda item: item["sequence_name"]):
        name = row["sequence_name"]
        artifacts = cdn_manifest["sequences"].get(name)
        if not isinstance(artifacts, dict):
            raise ValueError(f"sequence missing from CDN manifest: {name}")
        sizes = {}
        for artifact_name in totals:
            metadata = artifacts.get(artifact_name)
            if not isinstance(metadata, dict) or not isinstance(
                metadata.get("file_size_bytes"), int
            ):
                raise ValueError(f"missing {artifact_name} metadata for {name}")
            sizes[artifact_name] = metadata["file_size_bytes"]
        totals["main_groundtruth"] += sizes["main_groundtruth"]
        totals["video_main_rgb"] += sizes["video_main_rgb"]
        if name in segmentation_names:
            totals["segmentation"] += sizes["segmentation"]
        selected.append(
            {
                "sequence_name": name,
                "activity": row["activity"],
                "condition": "skeleton" if "_skeleton_" in name else "regular",
                "split": splits[name],
                "download": {
                    "main_groundtruth": True,
                    "video_main_rgb": True,
                    "segmentation": name in segmentation_names,
                },
                "declared_bytes": sizes,
            }
        )
    return {
        "schema_version": 1,
        "protocol_id": "openprop-adt-pilot-selection-v1",
        "selection_seed": seed,
        "selection_uses_screening_outcomes": False,
        "split_scope": "pilot_sequence_level_not_subject_generalization",
        "performance_evidence": False,
        "sequence_count": len(selected),
        "segmentation_sequence_count": len(segmentation_names),
        "declared_download_bytes": totals,
        "sequences": selected,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screening-plan", type=Path, required=True)
    parser.add_argument("--cdn-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", default="openprop-adt-pilot-v1")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    screening = json.loads(args.screening_plan.read_text(encoding="utf-8"))
    cdn = json.loads(args.cdn_manifest.read_text(encoding="utf-8"))
    selection = build_selection(screening, cdn, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(selection, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    sizes = selection["declared_download_bytes"]
    print(
        f"sequences={selection['sequence_count']} "
        f"segmentation_sequences={selection['segmentation_sequence_count']}"
    )
    print(
        "remaining_visual_gib="
        f"{(sizes['video_main_rgb'] + sizes['segmentation']) / (1024**3):.2f}"
    )
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
