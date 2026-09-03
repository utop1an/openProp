"""Build a participant-disjoint, model-output-blind VISOR media pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


SPECIAL_CONTACTS = {
    "hand-not-in-contact",
    "none-of-the-above",
    "inconclusive",
    None,
}
HAND_NAMES = {"left hand", "right hand", "left glove", "right glove", "hand", "glove"}
FRAME_NUMBER = re.compile(r"(\d+)(?=\.jpg$)")


def stable_key(seed: str, value: str) -> str:
    return hashlib.sha256(f"{seed}|{value}".encode("utf-8")).hexdigest()


def frame_number(frame: dict[str, Any]) -> int:
    name = str(frame.get("image", {}).get("name", ""))
    match = FRAME_NUMBER.search(name)
    if not match:
        raise ValueError(f"malformed VISOR frame name: {name!r}")
    return int(match.group(1))


def video_statistics(path: Path, split: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    frames = payload.get("video_annotations")
    if not isinstance(frames, list) or not frames:
        raise ValueError(f"VISOR annotation contains no frames: {path}")
    ordered = sorted(frames, key=frame_number)
    subsequences: set[str] = set()
    names: set[str] = set()
    nonhand_ids: set[str] = set()
    contact_links = 0
    ambiguous_frames = 0
    observations: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, frame in enumerate(ordered):
        image = frame.get("image", {})
        subsequence = str(image.get("subsequence", ""))
        if not subsequence:
            raise ValueError(f"VISOR frame lacks subsequence: {path}")
        subsequences.add(subsequence)
        per_name: Counter[str] = Counter()
        annotations = frame.get("annotations")
        if not isinstance(annotations, list):
            raise ValueError(f"VISOR frame lacks annotations: {path}")
        for annotation in annotations:
            entity_id = annotation.get("id")
            name = str(annotation.get("name", "")).strip().lower()
            if not isinstance(entity_id, str) or not name:
                raise ValueError(f"malformed VISOR entity annotation: {path}")
            names.add(name)
            observations[(subsequence, entity_id)].append(index)
            if name not in HAND_NAMES:
                nonhand_ids.add(entity_id)
                per_name[name] += 1
            if annotation.get("in_contact_object") not in SPECIAL_CONTACTS:
                contact_links += 1
        if any(count > 1 for count in per_name.values()):
            ambiguous_frames += 1
    reappearance_gaps = sum(
        sum(right - left > 1 for left, right in zip(indices, indices[1:]))
        for indices in observations.values()
    )
    return {
        "video_id": path.stem,
        "participant_id": path.stem.split("_", 1)[0],
        "official_split": split,
        "frame_count": len(ordered),
        "subsequence_count": len(subsequences),
        "open_vocabulary_name_count": len(names),
        "nonhand_instance_count": len(nonhand_ids),
        "contact_link_count": contact_links,
        "same_name_ambiguous_frame_count": ambiguous_frames,
        "sparse_reappearance_gap_count": reappearance_gaps,
    }


def eligible(row: dict[str, Any]) -> bool:
    signals = (
        row["contact_link_count"],
        row["same_name_ambiguous_frame_count"],
        row["sparse_reappearance_gap_count"],
    )
    return (
        row["frame_count"] >= 6
        and row["nonhand_instance_count"] >= 2
        and any(value > 0 for value in signals)
    )


def reserve_participants(
    rows: list[dict[str, Any]], seed: str, target_videos: int, minimum_participants: int
) -> set[str]:
    by_participant: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_participant[row["participant_id"]].append(row)
    selected: set[str] = set()
    count = 0
    for participant in sorted(by_participant, key=lambda value: stable_key(seed, value)):
        selected.add(participant)
        count += len(by_participant[participant])
        if count >= target_videos and len(selected) >= minimum_participants:
            break
    if count < target_videos or len(selected) < minimum_participants:
        raise RuntimeError("not enough participant-disjoint VISOR candidates")
    return selected


def sample(rows: list[dict[str, Any]], seed: str, count: int) -> list[dict[str, Any]]:
    if len(rows) < count:
        raise RuntimeError(f"VISOR role has only {len(rows)} eligible videos; need {count}")
    return sorted(rows, key=lambda row: stable_key(seed, row["video_id"]))[:count]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", default="openprop-visor-pilot-v1")
    parser.add_argument("--train-count", type=int, default=24)
    parser.add_argument("--calibration-count", type=int, default=8)
    parser.add_argument("--test-count", type=int, default=12)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    all_rows = []
    for split in ("train", "val"):
        folder = args.data_root / "GroundTruth-SparseAnnotations" / "annotations" / split
        all_rows.extend(video_statistics(path, split) for path in sorted(folder.glob("*.json")))
    candidates = [row for row in all_rows if eligible(row)]
    val_rows = [row for row in candidates if row["official_split"] == "val"]
    train_rows = [row for row in candidates if row["official_split"] == "train"]

    test_participants = reserve_participants(
        val_rows, args.seed + "|test-participants", args.test_count, 4
    )
    test = sample(
        [row for row in val_rows if row["participant_id"] in test_participants],
        args.seed + "|test-videos",
        args.test_count,
    )
    remaining_train = [
        row for row in train_rows if row["participant_id"] not in test_participants
    ]
    calibration_participants = reserve_participants(
        remaining_train,
        args.seed + "|calibration-participants",
        args.calibration_count,
        3,
    )
    calibration = sample(
        [row for row in remaining_train if row["participant_id"] in calibration_participants],
        args.seed + "|calibration-videos",
        args.calibration_count,
    )
    train = sample(
        [
            row
            for row in remaining_train
            if row["participant_id"] not in calibration_participants
        ],
        args.seed + "|train-videos",
        args.train_count,
    )

    selected = []
    for role, rows in (("train", train), ("calibration", calibration), ("test", test)):
        for row in rows:
            selected.append(
                {
                    **row,
                    "split": role,
                    "download": {"sparse_rgb_frames": True, "dense_annotations": False},
                }
            )
    participants_by_split = {
        role: sorted({row["participant_id"] for row in selected if row["split"] == role})
        for role in ("train", "calibration", "test")
    }
    if any(
        set(participants_by_split[left]) & set(participants_by_split[right])
        for left, right in (("train", "calibration"), ("train", "test"), ("calibration", "test"))
    ):
        raise AssertionError("VISOR participant leakage")
    report = {
        "schema_version": 1,
        "protocol_id": "openprop-visor-pilot-selection-v1",
        "selection_seed": args.seed,
        "minimum_eligibility": {
            "frame_count": 6,
            "nonhand_instance_count": 2,
            "requires_contact_ambiguity_or_reappearance_signal": True,
        },
        "selection_uses_model_outputs": False,
        "selection_uses_test_performance": False,
        "screening_statistics_are_not_performance_evidence": True,
        "split_is_participant_disjoint": True,
        "participants_by_split": participants_by_split,
        "inventory_count": len(all_rows),
        "eligible_count": len(candidates),
        "selected_count": len(selected),
        "performance_evidence": False,
        "videos": sorted(selected, key=lambda row: (row["split"], row["video_id"])),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"inventory={len(all_rows)} eligible={len(candidates)} selected={len(selected)} "
        f"participants={participants_by_split}"
    )


if __name__ == "__main__":
    main()
