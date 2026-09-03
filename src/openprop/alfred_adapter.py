from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from .models import PropertyConstraint, QueryFrame, RelationValue


SUPPORTED_TASKS = {
    "pick_and_place_simple",
    "pick_clean_then_place_in_recep",
    "pick_cool_then_place_in_recep",
    "pick_heat_then_place_in_recep",
}

EXCLUDED_TASKS = {
    "look_at_obj_in_light": "multi_entity_goal",
    "pick_and_place_with_movable_recep": "nested_multi_entity_goal",
    "pick_two_obj_and_place": "multiple_target_instances",
}

INSIDE_RECEPTACLES = {
    "bathtubbasin",
    "bowl",
    "box",
    "cabinet",
    "cup",
    "drawer",
    "fridge",
    "garbagecan",
    "microwave",
    "mug",
    "pan",
    "pot",
    "safe",
    "sinkbasin",
}


@dataclass(frozen=True, slots=True)
class AlfredLanguageCase:
    case_id: str
    task_id: str
    split: str
    task_type: str
    floorplan: str
    query: str
    gold_frame: QueryFrame
    annotation_index: int
    source_path: str


@dataclass(frozen=True, slots=True)
class AlfredExclusion:
    task_id: str
    split: str
    task_type: str
    reason: str
    source_path: str


@dataclass(frozen=True, slots=True)
class AlfredLanguageDataset:
    cases: tuple[AlfredLanguageCase, ...]
    exclusions: tuple[AlfredExclusion, ...]
    audit: Mapping[str, object]


def _words(value: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", " ", value).casefold()


def _relation(parent_target: str) -> RelationValue:
    predicate = (
        "inside"
        if parent_target.casefold() in INSIDE_RECEPTACLES
        else "on"
    )
    return RelationValue(predicate, {"object": _words(parent_target)})


def _gold_frame(query: str, task_type: str, params: Mapping[str, object]) -> QueryFrame:
    object_target = str(params.get("object_target", "")).strip()
    parent_target = str(params.get("parent_target", "")).strip()
    if not object_target or not parent_target:
        raise ValueError("supported ALFRED task requires object_target and parent_target")
    constraints: list[PropertyConstraint] = [
        PropertyConstraint("type", _words(object_target), 0.35)
    ]
    if task_type == "pick_clean_then_place_in_recep":
        constraints.append(PropertyConstraint("cleanliness", "clean", 0.35))
    elif task_type == "pick_heat_then_place_in_recep":
        constraints.append(PropertyConstraint("thermal_state", "hot", 0.35))
    elif task_type == "pick_cool_then_place_in_recep":
        constraints.append(PropertyConstraint("thermal_state", "cold", 0.35))
    else:
        constraints[0] = PropertyConstraint("type", _words(object_target), 0.50)
    location_weight = 1.0 - sum(item.relevance for item in constraints)
    constraints.append(
        PropertyConstraint("location", _relation(parent_target), location_weight)
    )
    return QueryFrame(query, tuple(constraints))


def load_alfred_language_dataset(
    root: str | Path,
    *,
    splits: tuple[str, ...] = ("valid_seen", "valid_unseen"),
) -> AlfredLanguageDataset:
    """Load human ALFRED task descriptions into typed property frames.

    This adapter intentionally excludes goals that refer to multiple entities or
    multiple target instances. It uses task metadata only for gold labels and
    makes no claim about visual observation histories.
    """

    source = Path(root)
    if not source.is_dir():
        raise ValueError(f"ALFRED root is not a directory: {source}")
    if not splits or len(splits) != len(set(splits)):
        raise ValueError("splits must be non-empty and unique")
    cases: list[AlfredLanguageCase] = []
    exclusions: list[AlfredExclusion] = []
    trajectory_counts: Counter[str] = Counter()
    case_counts: Counter[str] = Counter()
    task_counts: Counter[str] = Counter()
    floorplans: dict[str, set[str]] = {split: set() for split in splits}
    seen_case_ids: set[str] = set()
    for split in splits:
        split_root = source / split
        if not split_root.is_dir():
            raise ValueError(f"missing ALFRED split directory: {split_root}")
        for path in sorted(split_root.rglob("traj_data.json")):
            row = json.loads(path.read_text(encoding="utf-8"))
            try:
                task_id = str(row["task_id"]).strip()
                task_type = str(row["task_type"]).strip()
                floorplan = str(row["scene"]["floor_plan"]).strip()
                params = row["pddl_params"]
                annotations = row["turk_annotations"]["anns"]
            except (KeyError, TypeError) as error:
                raise ValueError(f"invalid ALFRED trajectory schema: {path}") from error
            if not task_id or not task_type or not floorplan or not isinstance(params, Mapping):
                raise ValueError(f"invalid ALFRED trajectory identifiers: {path}")
            if not isinstance(annotations, list) or not annotations:
                raise ValueError(f"ALFRED trajectory has no annotations: {path}")
            trajectory_counts[split] += 1
            floorplans[split].add(floorplan)
            relative = path.relative_to(source).as_posix()
            if task_type not in SUPPORTED_TASKS:
                exclusions.append(
                    AlfredExclusion(
                        task_id,
                        split,
                        task_type,
                        EXCLUDED_TASKS.get(task_type, "unsupported_task_type"),
                        relative,
                    )
                )
                continue
            for annotation_index, annotation in enumerate(annotations):
                if not isinstance(annotation, Mapping):
                    raise ValueError(f"invalid ALFRED annotation: {path}")
                query = str(annotation.get("task_desc", "")).strip()
                if not query:
                    raise ValueError(f"empty ALFRED task description: {path}")
                case_id = f"{task_id}:ann-{annotation_index}"
                if case_id in seen_case_ids:
                    raise ValueError(f"duplicate ALFRED language case: {case_id}")
                seen_case_ids.add(case_id)
                cases.append(
                    AlfredLanguageCase(
                        case_id,
                        task_id,
                        split,
                        task_type,
                        floorplan,
                        query,
                        _gold_frame(query, task_type, params),
                        annotation_index,
                        relative,
                    )
                )
                case_counts[split] += 1
                task_counts[task_type] += 1
    if not cases:
        raise ValueError("ALFRED adapter produced no supported language cases")
    audit = {
        "protocol": {
            "source": "official ALFRED 2.1.0 lite trajectory JSONs",
            "language": "human task descriptions",
            "gold_labels": "trajectory PDDL parameters and task family",
            "observation_claim": "none; lite release has no frame-level visibility state",
            "excluded_task_policy": dict(sorted(EXCLUDED_TASKS.items())),
        },
        "trajectories_by_split": dict(sorted(trajectory_counts.items())),
        "cases_by_split": dict(sorted(case_counts.items())),
        "cases_by_task_type": dict(sorted(task_counts.items())),
        "floorplans_by_split": {
            split: len(floorplans[split]) for split in sorted(floorplans)
        },
        "excluded_trajectories_by_reason": dict(
            sorted(Counter(item.reason for item in exclusions).items())
        ),
        "supported_cases": len(cases),
        "excluded_trajectories": len(exclusions),
    }
    return AlfredLanguageDataset(tuple(cases), tuple(exclusions), audit)
