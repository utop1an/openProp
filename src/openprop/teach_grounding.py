from __future__ import annotations

import copy
import json
from collections import Counter, defaultdict
from typing import Any, Iterable, Sequence

from .models import Entity, Observation, PropertyConstraint, PropertyDefinition, QueryFrame, ValueType
from .property_registry import PropertyRegistry
from .teach_adapter import TeachReplay
from .temporal_grounding import TemporalGroundingCase


TEACH_BOOLEAN_STATE_PROPERTIES = (
    "isToggled",
    "isBroken",
    "isFilledWithLiquid",
    "isDirty",
    "isUsedUp",
    "isCooked",
    "isOpen",
    "isPickedUp",
    "simbotIsCooked",
    "simbotIsFilledWithWater",
    "simbotIsBoiled",
    "simbotIsFilledWithCoffee",
    "simbotPickedUp",
)


def teach_grounding_registry(
    property_names: Sequence[str] = TEACH_BOOLEAN_STATE_PROPERTIES,
) -> PropertyRegistry:
    """Register exact typed TEACh states without assigning unvalidated decay."""

    if not property_names or len(property_names) != len(set(property_names)):
        raise ValueError("property_names must be non-empty and unique")
    registry = PropertyRegistry()
    registry.register(
        PropertyDefinition("type", "TEACh simulator object type", ValueType.CATEGORICAL)
    )
    registry.register(
        PropertyDefinition("scene", "TEACh floorplan", ValueType.CATEGORICAL)
    )
    for name in property_names:
        registry.register(
            PropertyDefinition(name, f"TEACh state field {name}", ValueType.CATEGORICAL)
        )
    return registry


def _value_key(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _value_text(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def build_teach_gold_grounding_cases(
    episode_id: str,
    scene: str,
    replay: TeachReplay,
    *,
    property_names: Sequence[str] = TEACH_BOOLEAN_STATE_PROPERTIES,
    source: str = "teach-egocentric-replay",
) -> tuple[TemporalGroundingCase, ...]:
    """Build gold-query cases without copying final truth into matcher entities."""

    if not episode_id.strip() or not scene.strip():
        raise ValueError("episode_id and scene cannot be empty")
    if replay.final_truth is None:
        raise ValueError("TEACh gold grounding requires a final truth snapshot")
    if not property_names or len(property_names) != len(set(property_names)):
        raise ValueError("property_names must be non-empty and unique")
    if any(snapshot.is_final for snapshot in replay.observations):
        raise ValueError("final truth cannot appear among observation snapshots")

    last_observed: dict[str, dict[str, Observation]] = defaultdict(dict)
    observed_types: dict[str, str] = {}
    for snapshot in sorted(replay.observations, key=lambda item: item.timestamp):
        for raw in snapshot.objects:
            if not raw.get("visible", False) or not raw.get("objectId"):
                continue
            object_id = str(raw["objectId"])
            object_type = str(raw.get("objectType", "unknown"))
            observed_types[object_id] = object_type
            last_observed[object_id]["type"] = Observation(
                object_type,
                timestamp=snapshot.timestamp,
                source=source,
            )
            last_observed[object_id]["scene"] = Observation(
                scene,
                timestamp=snapshot.timestamp,
                source=source,
            )
            for property_name in property_names:
                if property_name in raw:
                    last_observed[object_id][property_name] = Observation(
                        copy.deepcopy(raw[property_name]),
                        timestamp=snapshot.timestamp,
                        source=source,
                    )

    final_by_id = {
        str(raw["objectId"]): raw
        for raw in replay.final_truth.objects
        if raw.get("objectId")
    }
    by_type: dict[str, list[str]] = defaultdict(list)
    for object_id, object_type in observed_types.items():
        if object_id in final_by_id:
            by_type[object_type].append(object_id)

    cases: list[TemporalGroundingCase] = []
    for object_type in sorted(by_type):
        candidate_ids = sorted(by_type[object_type])
        if len(candidate_ids) < 2:
            continue
        for property_name in property_names:
            if any(property_name not in final_by_id[item] for item in candidate_ids):
                continue
            targets_by_value: dict[str, list[str]] = defaultdict(list)
            values: dict[str, Any] = {}
            for object_id in candidate_ids:
                value = copy.deepcopy(final_by_id[object_id][property_name])
                key = _value_key(value)
                targets_by_value[key].append(object_id)
                values[key] = value
            for value_key in sorted(targets_by_value):
                targets = targets_by_value[value_key]
                if len(targets) != 1:
                    continue
                target_raw_id = targets[0]
                desired_value = values[value_key]
                query = (
                    f"the {object_type} whose {property_name} is "
                    f"{_value_text(desired_value)}"
                )
                entities = tuple(
                    Entity(
                        f"{episode_id}:{object_id}",
                        dict(last_observed[object_id]),
                    )
                    for object_id in candidate_ids
                )
                target_id = f"{episode_id}:{target_raw_id}"
                target_observation = last_observed[target_raw_id].get(property_name)
                distractor_matches = [
                    observation
                    for object_id in candidate_ids
                    if object_id != target_raw_id
                    and (
                        observation := last_observed[object_id].get(property_name)
                    )
                    is not None
                    and observation.value == desired_value
                ]
                tags = ["teach", "gold-query", property_name]
                if target_observation is None:
                    tags.append("target-property-unobserved")
                    tags.append("unobservable-target-state")
                elif target_observation.value != desired_value:
                    tags.append("target-changed-after-last-observation")
                    tags.append("unobservable-target-state")
                else:
                    tags.append("target-last-observation-matches")
                    if not distractor_matches:
                        tags.extend(("static-identifiable", "primary-evaluable"))
                    elif target_observation.timestamp is None or any(
                        item.timestamp is None for item in distractor_matches
                    ):
                        tags.append("unknown-observation-time")
                    else:
                        newest_distractor = max(
                            float(item.timestamp) for item in distractor_matches
                        )
                        target_time = float(target_observation.timestamp)
                        if target_time > newest_distractor:
                            tags.extend(
                                (
                                    "stale-distractor-match",
                                    "temporal-discriminative",
                                    "primary-evaluable",
                                )
                            )
                        elif target_time == newest_distractor:
                            tags.append("input-evidence-tie")
                        else:
                            tags.append("recency-conflict")
                if "primary-evaluable" not in tags:
                    tags.append("temporal-challenge")
                case_index = len(cases)
                cases.append(
                    TemporalGroundingCase(
                        case_id=(
                            f"teach:{episode_id}:{property_name}:{case_index:04d}"
                        ),
                        query=query,
                        entities=entities,
                        target_id=target_id,
                        gold_frame=QueryFrame(
                            query,
                            (
                                PropertyConstraint("type", object_type, 0.25),
                                PropertyConstraint(property_name, desired_value, 0.75),
                            ),
                        ),
                        as_of=replay.final_truth.timestamp,
                        current_truth={
                            f"{episode_id}:{object_id}": {
                                "type": object_type,
                                property_name: copy.deepcopy(
                                    final_by_id[object_id][property_name]
                                ),
                            }
                            for object_id in candidate_ids
                        },
                        tags=tuple(tags),
                    )
                )
    return tuple(cases)


def audit_teach_grounding_cases(
    cases: Iterable[TemporalGroundingCase],
) -> dict[str, object]:
    rows = tuple(cases)
    candidate_sizes = Counter(len(case.entities) for case in rows)
    property_counts = Counter(
        case.gold_frame.constraints[-1].property_name for case in rows
    )
    tag_counts = Counter(tag for case in rows for tag in case.tags)
    return {
        "cases": len(rows),
        "primary_evaluable_cases": tag_counts["primary-evaluable"],
        "temporal_discriminative_cases": tag_counts["temporal-discriminative"],
        "unobservable_target_cases": tag_counts["unobservable-target-state"],
        "input_evidence_tie_cases": tag_counts["input-evidence-tie"],
        "temporal_challenge_cases": tag_counts["temporal-challenge"],
        "candidate_size_min": min(candidate_sizes, default=0),
        "candidate_size_max": max(candidate_sizes, default=0),
        "candidate_size_mean": (
            sum(len(case.entities) for case in rows) / len(rows) if rows else 0.0
        ),
        "candidate_size_histogram": {
            str(size): count for size, count in sorted(candidate_sizes.items())
        },
        "cases_per_property": dict(sorted(property_counts.items())),
        "tag_counts": dict(sorted(tag_counts.items())),
        "target_ties_in_final_truth": 0,
        "matcher_input_source": "visible replay snapshots only",
        "query_and_label_source": "evaluation-only final truth",
    }
