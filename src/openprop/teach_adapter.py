from __future__ import annotations

import copy
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .observation_history import ObservationHistoryRecord


DEFAULT_TEACH_STATE_PROPERTIES = (
    "isToggled",
    "isBroken",
    "isFilledWithLiquid",
    "isDirty",
    "isUsedUp",
    "isCooked",
    "isOpen",
    "isPickedUp",
    "simbotLastParentReceptacle",
    "simbotIsCooked",
    "simbotIsFilledWithWater",
    "simbotIsBoiled",
    "simbotIsFilledWithCoffee",
    "simbotPickedUp",
)

_STATE_FILE_PATTERN = re.compile(r"^statediff\.(?P<time>.+)\.json$")


@dataclass(frozen=True, slots=True)
class TeachReplaySnapshot:
    timestamp: float
    objects: tuple[Mapping[str, Any], ...]
    is_final: bool = False


@dataclass(frozen=True, slots=True)
class TeachReplay:
    observations: tuple[TeachReplaySnapshot, ...]
    final_truth: TeachReplaySnapshot | None


def _objects_with_custom_metadata(
    state: Mapping[str, Any],
) -> list[dict[str, Any]]:
    metadata = state.get("custom_object_metadata", {})
    objects: list[dict[str, Any]] = []
    for raw_object in state.get("objects", ()):
        item = copy.deepcopy(dict(raw_object))
        item.update(copy.deepcopy(dict(metadata.get(item.get("objectId"), {}))))
        objects.append(item)
    return objects


def apply_teach_state_diff(
    initial_state: Mapping[str, Any],
    state_diff: Mapping[str, Any],
) -> dict[str, Any]:
    """Reconstruct a TEACh state diff relative to its initial episode state."""

    state = copy.deepcopy(dict(initial_state))
    objects = _objects_with_custom_metadata(state)
    by_id = {str(item["objectId"]): item for item in objects}
    replaced_base_ids: set[str] = set()
    for object_id, raw_changes in state_diff.get("objects", {}).items():
        object_id = str(object_id)
        changes = copy.deepcopy(dict(raw_changes))
        if object_id in by_id:
            by_id[object_id].update(changes)
            continue
        parts = object_id.split("|")
        base_id = "|".join(parts[:4]) if len(parts) > 4 else object_id
        if base_id not in by_id:
            raise ValueError(
                f"state diff references unknown TEACh object {object_id!r}"
            )
        created = copy.deepcopy(by_id[base_id])
        created["objectId"] = object_id
        created.update(changes)
        by_id[object_id] = created
        replaced_base_ids.add(base_id)
    state["objects"] = [
        item
        for object_id, item in by_id.items()
        if object_id not in replaced_base_ids
    ]
    state["custom_object_metadata"] = {}
    return state


def reconstruct_teach_snapshots(
    initial_state: Mapping[str, Any],
    state_diffs: Iterable[tuple[float, Mapping[str, Any]]],
) -> tuple[TeachReplaySnapshot, ...]:
    """Reconstruct chronologically ordered snapshots from initial-relative diffs."""

    rows = sorted(state_diffs, key=lambda item: item[0])
    timestamps = [timestamp for timestamp, _ in rows]
    if len(timestamps) != len(set(timestamps)):
        raise ValueError("TEACh replay timestamps must be unique")
    snapshots: list[TeachReplaySnapshot] = []
    for timestamp, state_diff in rows:
        if not math.isfinite(timestamp) or timestamp < 0:
            raise ValueError("TEACh replay timestamps must be finite and nonnegative")
        state = apply_teach_state_diff(initial_state, state_diff)
        snapshots.append(
            TeachReplaySnapshot(
                float(timestamp),
                tuple(copy.deepcopy(state["objects"])),
            )
        )
    return tuple(snapshots)


def read_teach_replay(
    initial_state: Mapping[str, Any],
    state_directory: str | Path,
    *,
    final_timestamp: float | None = None,
) -> TeachReplay:
    """Read TEACh replay outputs named statediff.<timestamp>.json."""

    numeric: list[tuple[float, Mapping[str, Any]]] = []
    final_diff: Mapping[str, Any] | None = None
    for path in Path(state_directory).glob("statediff.*.json"):
        match = _STATE_FILE_PATTERN.match(path.name)
        if match is None:
            continue
        token = match.group("time")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if token == "end":
            final_diff = payload
        else:
            try:
                numeric.append((float(token), payload))
            except ValueError as error:
                raise ValueError(
                    f"invalid TEACh state timestamp in {path.name!r}"
                ) from error
    observations = reconstruct_teach_snapshots(initial_state, numeric)
    final_truth = None
    if final_diff is not None:
        if final_timestamp is None:
            raise ValueError(
                "final_timestamp is required because statediff.end.json has no time"
            )
        if not math.isfinite(final_timestamp):
            raise ValueError("final_timestamp must be finite")
        if observations and final_timestamp < observations[-1].timestamp:
            raise ValueError("final_timestamp cannot precede observation snapshots")
        state = apply_teach_state_diff(initial_state, final_diff)
        final_truth = TeachReplaySnapshot(
            final_timestamp,
            tuple(copy.deepcopy(state["objects"])),
            is_final=True,
        )
    elif final_timestamp is not None:
        raise ValueError("final_timestamp was provided without statediff.end.json")
    return TeachReplay(observations, final_truth)


@dataclass(slots=True)
class _ActiveEpisode:
    observed_at: float
    last_confirmed_at: float
    value: Any
    subject_type: str
    episode_index: int


def _state_label(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "none"
    if isinstance(value, (str, int, float)):
        return str(value)
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def teach_visible_observation_history(
    episode_id: str,
    snapshots: Sequence[TeachReplaySnapshot],
    *,
    scene: str,
    property_names: Sequence[str] = DEFAULT_TEACH_STATE_PROPERTIES,
    source: str = "teach-egocentric-replay",
) -> tuple[ObservationHistoryRecord, ...]:
    """Extract state episodes using only objects visible at each replay snapshot."""

    if not episode_id.strip():
        raise ValueError("episode_id cannot be empty")
    ordered = sorted(snapshots, key=lambda item: item.timestamp)
    if any(item.is_final for item in ordered):
        raise ValueError("final truth cannot be exposed as an observation snapshot")
    if len({item.timestamp for item in ordered}) != len(ordered):
        raise ValueError("observation snapshot timestamps must be unique")

    active: dict[tuple[str, str], _ActiveEpisode] = {}
    completed: list[ObservationHistoryRecord] = []
    episode_counts: dict[tuple[str, str], int] = {}

    for snapshot in ordered:
        for obj in snapshot.objects:
            if not obj.get("visible", False):
                continue
            object_id = str(obj.get("objectId", ""))
            subject_type = str(obj.get("objectType", "unknown"))
            if not object_id:
                continue
            for property_name in property_names:
                if property_name not in obj:
                    continue
                key = (object_id, property_name)
                value = copy.deepcopy(obj[property_name])
                current = active.get(key)
                if current is None:
                    index = episode_counts.get(key, 0)
                    episode_counts[key] = index + 1
                    active[key] = _ActiveEpisode(
                        snapshot.timestamp,
                        snapshot.timestamp,
                        value,
                        subject_type,
                        index,
                    )
                    continue
                if current.value == value:
                    current.last_confirmed_at = snapshot.timestamp
                    continue
                completed.append(
                    ObservationHistoryRecord(
                        record_id=(
                            f"teach:{episode_id}:{object_id}:{property_name}:"
                            f"{current.episode_index}"
                        ),
                        entity_id=f"{episode_id}:{object_id}",
                        property_name=property_name,
                        subject_type=current.subject_type,
                        state_predicate=_state_label(current.value),
                        context_object="none",
                        scene=scene,
                        observed_at=current.observed_at,
                        followup_at=snapshot.timestamp,
                        state_changed=True,
                        source=source,
                        last_confirmed_at=current.last_confirmed_at,
                    )
                )
                index = episode_counts.get(key, 0)
                episode_counts[key] = index + 1
                active[key] = _ActiveEpisode(
                    snapshot.timestamp,
                    snapshot.timestamp,
                    value,
                    subject_type,
                    index,
                )

    for (object_id, property_name), current in sorted(active.items()):
        completed.append(
            ObservationHistoryRecord(
                record_id=(
                    f"teach:{episode_id}:{object_id}:{property_name}:"
                    f"{current.episode_index}"
                ),
                entity_id=f"{episode_id}:{object_id}",
                property_name=property_name,
                subject_type=current.subject_type,
                state_predicate=_state_label(current.value),
                context_object="none",
                scene=scene,
                observed_at=current.observed_at,
                followup_at=current.last_confirmed_at,
                state_changed=False,
                source=source,
            )
        )
    return tuple(completed)


def teach_hidden_current_truth(
    snapshot: TeachReplaySnapshot,
    *,
    property_names: Sequence[str] = DEFAULT_TEACH_STATE_PROPERTIES,
) -> dict[str, dict[str, Any]]:
    """Return evaluation-only current truth; never pass this mapping to a matcher."""

    if not snapshot.is_final:
        raise ValueError("hidden current truth requires a final replay snapshot")
    return {
        str(obj["objectId"]): {
            property_name: copy.deepcopy(obj[property_name])
            for property_name in property_names
            if property_name in obj
        }
        for obj in snapshot.objects
        if "objectId" in obj
    }
