"""Rank downloaded ADT ground truth for OpenProp moved-entity experiments."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Any, Iterable


RGB_STREAM_ID = "214-1"
MOVEMENT_THRESHOLD_M = 0.10
ACTIVITY_PATTERN = re.compile(r"^Apartment_release_([^_]+)_")


def euclidean(left: tuple[float, float, float], right: tuple[float, float, float]) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right)))


def max_displacement(points: Iterable[tuple[int, tuple[float, float, float]]]) -> float:
    ordered = sorted(points)
    if len(ordered) < 2:
        return 0.0
    origin = ordered[0][1]
    return max(euclidean(origin, point) for _, point in ordered[1:])


def score_sequence(sequence_dir: Path) -> dict[str, Any]:
    instances = json.loads((sequence_dir / "instances.json").read_text(encoding="utf-8"))
    object_rows = {
        str(uid): row
        for uid, row in instances.items()
        if row.get("instance_type") == "object"
    }
    category_counts = Counter(row.get("category", "unknown") for row in object_rows.values())

    positions: dict[str, list[tuple[int, tuple[float, float, float]]]] = defaultdict(list)
    with (sequence_dir / "scene_objects.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            uid = row["object_uid"]
            timestamp = int(row["timestamp[ns]"])
            if uid not in object_rows or timestamp < 0:
                continue
            positions[uid].append(
                (
                    timestamp,
                    (
                        float(row["t_wo_x[m]"]),
                        float(row["t_wo_y[m]"]),
                        float(row["t_wo_z[m]"]),
                    ),
                )
            )

    displacement = {uid: max_displacement(points) for uid, points in positions.items()}
    moving = {uid for uid, distance in displacement.items() if distance >= MOVEMENT_THRESHOLD_M}

    visible_times: dict[str, list[int]] = defaultdict(list)
    with (sequence_dir / "2d_bounding_box.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            uid = row["object_uid"]
            if row["stream_id"] == RGB_STREAM_ID and uid in moving:
                visible_times[uid].append(int(row["timestamp[ns]"]))

    max_gap_s = 0.0
    for timestamps in visible_times.values():
        unique = sorted(set(timestamps))
        if len(unique) > 2:
            gaps = [right - left for left, right in zip(unique, unique[1:])]
            typical = median(gaps)
            max_gap_s = max(max_gap_s, max(gaps) / 1e9 - typical / 1e9)

    moving_categories = Counter(object_rows[uid].get("category", "unknown") for uid in moving)
    ambiguous_moving_objects = sum(
        count for category, count in moving_categories.items() if category_counts[category] > 1
    )
    max_candidate_multiplicity = max(
        (category_counts[category] for category in moving_categories), default=0
    )
    visible_moving = len(set(visible_times) & moving)
    max_move = max(displacement.values(), default=0.0)
    selection_score = (
        2.0 * min(visible_moving, 5)
        + min(max_move, 2.0)
        + math.log1p(ambiguous_moving_objects)
        + min(max_gap_s, 5.0) / 5.0
    )
    activity_match = ACTIVITY_PATTERN.match(sequence_dir.name)
    return {
        "sequence_name": sequence_dir.name,
        "activity": activity_match.group(1) if activity_match else "unknown",
        "has_skeleton_condition": "_skeleton_" in sequence_dir.name,
        "object_count": len(object_rows),
        "moving_object_count_10cm": len(moving),
        "visible_moving_object_count": visible_moving,
        "ambiguous_moving_object_count": ambiguous_moving_objects,
        "max_candidate_category_multiplicity": max_candidate_multiplicity,
        "max_displacement_m": round(max_move, 6),
        "max_visibility_gap_s": round(max_gap_s, 6),
        "selection_score": round(selection_score, 6),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    for sequence_dir in sorted(path for path in args.data_root.iterdir() if path.is_dir()):
        required = ("instances.json", "scene_objects.csv", "2d_bounding_box.csv")
        if all((sequence_dir / name).is_file() for name in required):
            rows.append(score_sequence(sequence_dir))
    if not rows:
        raise SystemExit("no complete ADT ground-truth sequences found")
    rows.sort(key=lambda row: (-row["selection_score"], row["sequence_name"]))
    report = {
        "schema_version": 1,
        "protocol_id": "openprop-adt-sequence-screening-v1",
        "movement_threshold_m": MOVEMENT_THRESHOLD_M,
        "rgb_stream_id": RGB_STREAM_ID,
        "sequence_count": len(rows),
        "performance_evidence": False,
        "sequences": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"ranked_sequences={len(rows)} output={args.output}")
    for row in rows[:15]:
        print(
            f"{row['selection_score']:.3f} {row['sequence_name']} "
            f"moving={row['moving_object_count_10cm']} "
            f"visible={row['visible_moving_object_count']} "
            f"ambiguous={row['ambiguous_moving_object_count']} "
            f"gap_s={row['max_visibility_gap_s']:.2f}"
        )


if __name__ == "__main__":
    main()
