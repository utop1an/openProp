from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .models import (
    PropertyDefinition,
    PropertyUpdatePolicy,
    TemporalPolicy,
    ValueType,
)
from .property_registry import PropertyRegistry
from .vlm import VisualFrame


_BOOLEAN_STATES: tuple[tuple[str, str, str, str, str], ...] = (
    ("openable", "isOpen", "open_state", "open", "closed"),
    ("toggleable", "isToggled", "toggle_state", "on", "off"),
    ("breakable", "isBroken", "broken_state", "broken", "intact"),
    ("dirtyable", "isDirty", "cleanliness", "dirty", "clean"),
    ("cookable", "isCooked", "cooking_state", "cooked", "raw"),
    ("sliceable", "isSliced", "slice_state", "sliced", "whole"),
    ("canFillWithLiquid", "isFilledWithLiquid", "fill_state", "filled", "empty"),
    ("pickupable", "isPickedUp", "held_state", "held", "not_held"),
)


@dataclass(frozen=True, slots=True)
class AI2ThorObjectTruth:
    """Evaluation-only simulator state. Never materialize this as matcher input."""

    entity_id: str
    object_type: str
    visible: bool
    values: Mapping[str, object]

    def __post_init__(self) -> None:
        if not self.entity_id.strip() or not self.object_type.strip():
            raise ValueError("AI2-THOR object identity and type cannot be empty")
        if "current_truth" in self.values or "target" in self.values:
            raise ValueError("reserved evaluation markers cannot be object properties")


@dataclass(frozen=True, slots=True)
class AI2ThorFrameBundle:
    """Trusted VLM frame and separately held simulator truth."""

    frame: VisualFrame
    scene_name: str
    current_truth: tuple[AI2ThorObjectTruth, ...]

    def __post_init__(self) -> None:
        if not self.scene_name.strip():
            raise ValueError("scene_name cannot be empty")
        ids = [item.entity_id for item in self.current_truth]
        if len(ids) != len(set(ids)):
            raise ValueError("AI2-THOR truth contains duplicate object IDs")
        visible_ids = {
            item.entity_id for item in self.current_truth if item.visible
        }
        if not set(self.frame.candidate_entity_ids).issubset(visible_ids):
            raise ValueError("frame candidates must be visible in simulator metadata")

    def truth_by_entity(self) -> dict[str, AI2ThorObjectTruth]:
        return {item.entity_id: item for item in self.current_truth}


@dataclass(frozen=True, slots=True)
class AI2ThorPropertyChange:
    property_name: str
    before: object
    after: object


@dataclass(frozen=True, slots=True)
class AI2ThorTransitionTruth:
    """Evaluation-only changes caused by one simulator action."""

    scene_name: str
    action: str
    before_frame_id: str
    after_frame_id: str
    changes: Mapping[str, tuple[AI2ThorPropertyChange, ...]]

    @property
    def changed_entity_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self.changes))


def ai2thor_property_registry(
    registry: PropertyRegistry | None = None,
) -> PropertyRegistry:
    """Register the typed state vocabulary used by the AI2-THOR experiment."""

    result = registry or PropertyRegistry()
    definitions = (
        PropertyDefinition("type", "AI2-THOR object category", ValueType.CATEGORICAL),
        PropertyDefinition(
            "open_state",
            "whether an openable object is open or closed",
            ValueType.CATEGORICAL,
            metadata={"allowed_values": ("open", "closed")},
        ),
        PropertyDefinition(
            "toggle_state",
            "whether a toggleable object is on or off",
            ValueType.CATEGORICAL,
            metadata={"allowed_values": ("on", "off")},
        ),
        PropertyDefinition(
            "broken_state",
            "whether a breakable object is broken or intact",
            ValueType.CATEGORICAL,
            metadata={"allowed_values": ("broken", "intact")},
        ),
        PropertyDefinition(
            "cleanliness",
            "whether a dirtyable object is dirty or clean",
            ValueType.CATEGORICAL,
            metadata={"allowed_values": ("dirty", "clean")},
        ),
        PropertyDefinition(
            "cooking_state",
            "whether a cookable object is cooked or raw",
            ValueType.CATEGORICAL,
            metadata={"allowed_values": ("cooked", "raw")},
        ),
        PropertyDefinition(
            "slice_state",
            "whether a sliceable object is sliced or whole",
            ValueType.CATEGORICAL,
            metadata={"allowed_values": ("sliced", "whole")},
        ),
        PropertyDefinition(
            "fill_state",
            "whether a fillable object contains liquid",
            ValueType.CATEGORICAL,
            metadata={"allowed_values": ("filled", "empty")},
        ),
        PropertyDefinition(
            "fill_liquid",
            "liquid currently contained by a fillable object",
            ValueType.CATEGORICAL,
            metadata={"allowed_values": ("water", "coffee", "wine")},
        ),
        PropertyDefinition(
            "held_state",
            "whether the agent is holding the object",
            ValueType.CATEGORICAL,
            metadata={"allowed_values": ("held", "not_held")},
        ),
        PropertyDefinition(
            "temperature",
            "AI2-THOR object temperature category",
            ValueType.CATEGORICAL,
            metadata={"allowed_values": ("Hot", "RoomTemp", "Cold")},
        ),
        PropertyDefinition(
            "position",
            "object world position in metres",
            ValueType.VECTOR,
            unit="m",
            metadata={"dimensions": 3},
            update_policy=PropertyUpdatePolicy(allow_visual_updates=False),
        ),
        PropertyDefinition(
            "parent_receptacle",
            "receptacle currently containing or supporting the object",
            ValueType.ENTITY_REFERENCE,
        ),
        PropertyDefinition(
            "motion_state",
            "whether an object changed position during the observation interval",
            ValueType.CATEGORICAL,
            metadata={"allowed_values": ("moved", "stationary")},
            temporal_policy=TemporalPolicy(half_life_seconds=300.0),
        ),
    )
    for definition in definitions:
        existing = result.get(definition.name)
        if existing is None:
            result.register(definition)
        elif existing.value_type is not definition.value_type:
            raise ValueError(
                f"existing {definition.name} has incompatible value type"
            )
    return result


def normalize_ai2thor_regions(
    instance_detections_2d: Mapping[str, Sequence[float]],
    *,
    screen_width: float,
    screen_height: float,
    candidate_entity_ids: Sequence[str],
) -> dict[str, tuple[float, float, float, float]]:
    """Normalize AI2-THOR pixel boxes for trusted VLM identity anchors."""

    width = _number(screen_width, "screen_width")
    height = _number(screen_height, "screen_height")
    if width <= 0.0 or height <= 0.0:
        raise ValueError("screen dimensions must be positive")
    result: dict[str, tuple[float, float, float, float]] = {}
    for entity_id in candidate_entity_ids:
        raw = instance_detections_2d.get(entity_id)
        if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
            raise ValueError(f"missing instance detection for {entity_id}")
        if len(raw) != 4:
            raise ValueError("AI2-THOR instance boxes must contain four coordinates")
        x1, y1, x2, y2 = (
            _number(value, f"instance_box[{index}]")
            for index, value in enumerate(raw)
        )
        if not (0.0 <= x1 < x2 <= width and 0.0 <= y1 < y2 <= height):
            raise ValueError("AI2-THOR instance box is outside the image")
        result[entity_id] = (
            x1 / width,
            y1 / height,
            x2 / width,
            y2 / height,
        )
    return result


def _valid_ai2thor_box(
    raw: object,
    *,
    screen_width: float,
    screen_height: float,
) -> bool:
    """Return whether a simulator box is a finite, non-degenerate image anchor."""

    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        return False
    if len(raw) != 4:
        return False
    try:
        x1, y1, x2, y2 = (
            _number(value, f"instance_box[{index}]")
            for index, value in enumerate(raw)
        )
    except (TypeError, ValueError):
        return False
    return (
        0.0 <= x1 < x2 <= screen_width
        and 0.0 <= y1 < y2 <= screen_height
    )



def extract_ai2thor_frame(
    metadata: Mapping[str, Any],
    *,
    frame_id: str,
    image_url: str,
    captured_at: float,
    source: str = "ai2thor-rgb",
    candidate_entity_ids: Sequence[str] | None = None,
    instance_detections_2d: Mapping[str, Sequence[float]] | None = None,
) -> AI2ThorFrameBundle:
    """Convert one AI2-THOR Event metadata payload without importing AI2-THOR."""

    scene_name = _text(metadata.get("sceneName"), "sceneName")
    objects = metadata.get("objects")
    if isinstance(objects, tuple):
        objects = list(objects)
    if not isinstance(objects, list):
        raise ValueError("AI2-THOR metadata must contain an objects array")

    truths: list[AI2ThorObjectTruth] = []
    visible_ids: list[str] = []
    for raw in objects:
        if not isinstance(raw, Mapping):
            raise ValueError("AI2-THOR object metadata rows must be objects")
        entity_id = _text(raw.get("objectId"), "objectId")
        object_type = _text(raw.get("objectType"), "objectType")
        visible = raw.get("visible")
        if not isinstance(visible, bool):
            raise ValueError("AI2-THOR visible must be boolean")
        values = _object_values(raw, object_type)
        truths.append(AI2ThorObjectTruth(entity_id, object_type, visible, values))
        if visible:
            visible_ids.append(entity_id)

    if candidate_entity_ids is None:
        # AI2-THOR's metadata visibility predicate and rendered instance
        # detections are not equivalent. A metadata-visible object can lack a
        # usable 2D box (for example, large receptacle surfaces), while the
        # segmentation pass can contain non-interactable scene geometry.
        # Only region-anchored visible objects are safe default VLM candidates;
        # the omitted objects remain in evaluation-only current_truth.
        if instance_detections_2d is None:
            candidates = tuple(sorted(visible_ids))
        else:
            screen_width = _number(metadata.get("screenWidth"), "screenWidth")
            screen_height = _number(metadata.get("screenHeight"), "screenHeight")
            if screen_width <= 0.0 or screen_height <= 0.0:
                raise ValueError("screen dimensions must be positive")
            valid_detection_ids = {
                entity_id
                for entity_id, raw in instance_detections_2d.items()
                if isinstance(entity_id, str)
                and _valid_ai2thor_box(
                    raw,
                    screen_width=screen_width,
                    screen_height=screen_height,
                )
            }
            candidates = tuple(
                sorted(set(visible_ids) & valid_detection_ids)
            )
    else:
        candidates = tuple(candidate_entity_ids)
        if len(candidates) != len(set(candidates)):
            raise ValueError("candidate_entity_ids cannot contain duplicates")
        unknown = set(candidates) - set(visible_ids)
        if unknown:
            raise ValueError(
                f"AI2-THOR candidates are not visible: {sorted(unknown)}"
            )
    regions: dict[str, tuple[float, float, float, float]] = {}
    if instance_detections_2d is not None:
        regions = normalize_ai2thor_regions(
            instance_detections_2d,
            screen_width=metadata.get("screenWidth"),
            screen_height=metadata.get("screenHeight"),
            candidate_entity_ids=candidates,
        )

    frame = VisualFrame(
        frame_id,
        image_url,
        captured_at,
        source,
        candidates,
        regions,
    )
    return AI2ThorFrameBundle(frame, scene_name, tuple(truths))


def derive_ai2thor_transition(
    before: AI2ThorFrameBundle,
    after: AI2ThorFrameBundle,
    *,
    action: str,
    movement_threshold_metres: float = 0.05,
) -> AI2ThorTransitionTruth:
    if before.scene_name != after.scene_name:
        raise ValueError("transition frames must come from the same scene")
    if not action.strip():
        raise ValueError("action cannot be empty")
    if (
        not math.isfinite(movement_threshold_metres)
        or movement_threshold_metres < 0.0
    ):
        raise ValueError("movement threshold must be finite and nonnegative")

    before_index = before.truth_by_entity()
    after_index = after.truth_by_entity()
    changes: dict[str, tuple[AI2ThorPropertyChange, ...]] = {}
    for entity_id in sorted(before_index.keys() & after_index.keys()):
        old = before_index[entity_id].values
        new = after_index[entity_id].values
        entity_changes: list[AI2ThorPropertyChange] = []
        for property_name in sorted(old.keys() & new.keys()):
            if property_name == "position":
                if _distance(old[property_name], new[property_name]) >= movement_threshold_metres:
                    entity_changes.append(
                        AI2ThorPropertyChange(
                            property_name,
                            old[property_name],
                            new[property_name],
                        )
                    )
                    entity_changes.append(
                        AI2ThorPropertyChange(
                            "motion_state",
                            "stationary",
                            "moved",
                        )
                    )
            elif old[property_name] != new[property_name]:
                entity_changes.append(
                    AI2ThorPropertyChange(
                        property_name,
                        old[property_name],
                        new[property_name],
                    )
                )
        if entity_changes:
            changes[entity_id] = tuple(entity_changes)
    return AI2ThorTransitionTruth(
        before.scene_name,
        action.strip(),
        before.frame.frame_id,
        after.frame.frame_id,
        changes,
    )


def _object_values(raw: Mapping[str, Any], object_type: str) -> dict[str, object]:
    values: dict[str, object] = {"type": object_type}
    for capability, state_key, property_name, positive, negative in _BOOLEAN_STATES:
        if raw.get(capability) is not True:
            continue
        state = raw.get(state_key)
        if not isinstance(state, bool):
            raise ValueError(f"{state_key} must be boolean when {capability}=true")
        values[property_name] = positive if state else negative

    if raw.get("canFillWithLiquid") is True and raw.get("isFilledWithLiquid") is True:
        liquid = raw.get("fillLiquid")
        if liquid is not None:
            values["fill_liquid"] = _text(liquid, "fillLiquid")

    temperature = raw.get("ObjectTemperature")
    if temperature is not None:
        values["temperature"] = _text(temperature, "ObjectTemperature")

    position = raw.get("position")
    if not isinstance(position, Mapping):
        raise ValueError("AI2-THOR object position must be an object")
    coordinates = tuple(_number(position.get(axis), f"position.{axis}") for axis in ("x", "y", "z"))
    values["position"] = coordinates

    parents = raw.get("parentReceptacles")
    if parents is not None:
        if not isinstance(parents, list) or any(
            not isinstance(item, str) or not item.strip() for item in parents
        ):
            raise ValueError("parentReceptacles must be null or an array of IDs")
        if parents:
            values["parent_receptacle"] = sorted(parents)[0]
    return values


def _distance(left: object, right: object) -> float:
    if not isinstance(left, tuple) or not isinstance(right, tuple):
        raise ValueError("position values must be tuples")
    if len(left) != 3 or len(right) != 3:
        raise ValueError("position values must have three dimensions")
    return math.sqrt(
        sum((float(a) - float(b)) ** 2 for a, b in zip(left, right))
    )


def _text(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()


def _number(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field_name} must be finite")
    return result
