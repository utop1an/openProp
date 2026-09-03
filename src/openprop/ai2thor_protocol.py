from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path


AI2THOR_SCENE_SPLIT_PROTOCOL = "openprop-ai2thor-scene-split-v1"
DEFAULT_AI2THOR_SPLIT_SEED = "openprop-iclr2027-v1"
_SPLIT_SIZES = {"development": 18, "calibration": 6, "test": 6}


def default_ai2thor_scene_catalog() -> dict[str, tuple[str, ...]]:
    return {
        "kitchen": tuple(f"FloorPlan{index}" for index in range(1, 31)),
        "living_room": tuple(f"FloorPlan{index}" for index in range(201, 231)),
        "bedroom": tuple(f"FloorPlan{index}" for index in range(301, 331)),
        "bathroom": tuple(f"FloorPlan{index}" for index in range(401, 431)),
    }


def build_ai2thor_scene_split(
    *,
    seed: str = DEFAULT_AI2THOR_SPLIT_SEED,
    scene_catalog: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, object]:
    if not isinstance(seed, str) or not seed.strip():
        raise ValueError("scene split seed must be a non-empty string")
    catalog = scene_catalog or default_ai2thor_scene_catalog()
    if set(catalog) != {"kitchen", "living_room", "bedroom", "bathroom"}:
        raise ValueError("scene catalog must contain the four iTHOR room categories")
    assignments: list[dict[str, str]] = []
    all_scenes: set[str] = set()
    for category in sorted(catalog):
        raw_scenes = tuple(catalog[category])
        if len(raw_scenes) != 30 or len(set(raw_scenes)) != 30:
            raise ValueError("each iTHOR room category must contain 30 unique scenes")
        if any(not isinstance(scene, str) or not scene.strip() for scene in raw_scenes):
            raise ValueError("scene names must be non-empty strings")
        overlap = all_scenes.intersection(raw_scenes)
        if overlap:
            raise ValueError("scene catalog contains cross-category duplicates")
        all_scenes.update(raw_scenes)
        ordered = sorted(
            raw_scenes,
            key=lambda scene: (
                hashlib.sha256(
                    f"{seed.strip()}|{category}|{scene}".encode("utf-8")
                ).hexdigest(),
                scene,
            ),
        )
        cursor = 0
        for split, size in _SPLIT_SIZES.items():
            for scene in ordered[cursor : cursor + size]:
                assignments.append(
                    {"scene": scene, "category": category, "split": split}
                )
            cursor += size
    assignments.sort(key=lambda item: (item["split"], item["category"], item["scene"]))
    payload: dict[str, object] = {
        "schema_version": 1,
        "protocol_id": AI2THOR_SCENE_SPLIT_PROTOCOL,
        "seed": seed.strip(),
        "selection_uses_model_outputs": False,
        "category_split_sizes": dict(_SPLIT_SIZES),
        "assignments": assignments,
    }
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    payload["split_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return payload


def write_ai2thor_scene_split(
    path: str | Path,
    *,
    seed: str = DEFAULT_AI2THOR_SPLIT_SEED,
    check: bool = False,
) -> dict[str, object]:
    destination = Path(path)
    payload = build_ai2thor_scene_split(seed=seed)
    rendered = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if check:
        if not destination.is_file():
            raise FileNotFoundError(f"missing frozen scene split: {destination}")
        if destination.read_text(encoding="utf-8") != rendered:
            raise ValueError("frozen AI2-THOR scene split drifted")
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(rendered, encoding="utf-8")
    return payload
