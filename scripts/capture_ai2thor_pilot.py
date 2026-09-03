from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform as host_platform
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any, Mapping, Sequence


@dataclass(frozen=True, slots=True)
class Intervention:
    family: str
    object_id: str
    object_type: str
    action: str
    arguments: Mapping[str, object]
    state_field: str
    before_state: object
    expected_after_state: object


INTERVENTION_FAMILIES = (
    "move_receptacle",
    "open",
    "toggle",
    "dirty",
    "fill",
    "cook",
    "slice",
    "break",
)
_AUDITED_STATE_FIELDS = (
    "isOpen",
    "isToggled",
    "isDirty",
    "isFilledWithLiquid",
    "isCooked",
    "isSliced",
    "isBroken",
    "isMoving",
    "parentReceptacles",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Capture deterministic AI2-THOR before/after RGB, metadata, and "
            "instance boxes for the first OpenProp VLM pilot."
        )
    )
    parser.add_argument("--scene", default="FloorPlan1")
    parser.add_argument(
        "--families",
        nargs="+",
        choices=INTERVENTION_FAMILIES,
        default=INTERVENTION_FAMILIES,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/ai2thor_pilot"),
    )
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument(
        "--settling-max-steps",
        type=int,
        default=12,
        help="Maximum Pass actions used to establish a stable pre-intervention scene.",
    )
    parser.add_argument(
        "--settling-stable-steps",
        type=int,
        default=2,
        help="Consecutive stable Pass actions required before before-frame capture.",
    )
    parser.add_argument(
        "--position-tolerance-metres",
        type=float,
        default=0.005,
        help="Maximum object displacement treated as stable and used in change audits.",
    )
    parser.add_argument(
        "--platform",
        choices=("cloud", "default"),
        default="cloud",
        help="Use cloud for Ubuntu headless capture; default opens a Unity window.",
    )
    return parser.parse_args()


def choose_intervention(
    metadata: Mapping[str, Any],
    family: str,
    *,
    visible_only: bool,
) -> Intervention | None:
    objects = metadata.get("objects")
    if not isinstance(objects, list):
        raise ValueError("AI2-THOR metadata must contain an objects array")
    candidates: list[Intervention] = []
    for row in objects:
        if not isinstance(row, Mapping):
            raise ValueError("AI2-THOR object rows must be objects")
        if visible_only and row.get("visible") is not True:
            continue
        intervention = intervention_for_object(row, family)
        if intervention is not None:
            candidates.append(intervention)
    if not candidates:
        return None
    return min(candidates, key=lambda item: (item.object_type, item.object_id))


def intervention_for_object(
    row: Mapping[str, Any],
    family: str,
) -> Intervention | None:
    object_id = _text(row.get("objectId"), "objectId")
    object_type = _text(row.get("objectType"), "objectType")
    if family == "move_receptacle":
        return None
    if family == "open":
        if row.get("openable") is not True or not isinstance(row.get("isOpen"), bool):
            return None
        state = bool(row["isOpen"])
        return Intervention(
            family,
            object_id,
            object_type,
            "CloseObject" if state else "OpenObject",
            {"objectId": object_id, "forceAction": True},
            "isOpen",
            state,
            not state,
        )
    if family == "toggle":
        if row.get("toggleable") is not True or not isinstance(
            row.get("isToggled"), bool
        ):
            return None
        state = bool(row["isToggled"])
        return Intervention(
            family,
            object_id,
            object_type,
            "ToggleObjectOff" if state else "ToggleObjectOn",
            {"objectId": object_id, "forceAction": True},
            "isToggled",
            state,
            not state,
        )
    if family == "dirty":
        if row.get("dirtyable") is not True or not isinstance(row.get("isDirty"), bool):
            return None
        state = bool(row["isDirty"])
        return Intervention(
            family,
            object_id,
            object_type,
            "CleanObject" if state else "DirtyObject",
            {"objectId": object_id, "forceAction": True},
            "isDirty",
            state,
            not state,
        )
    if family == "fill":
        if row.get("canFillWithLiquid") is not True or not isinstance(
            row.get("isFilledWithLiquid"), bool
        ):
            return None
        state = bool(row["isFilledWithLiquid"])
        action = "EmptyLiquidFromObject" if state else "FillObjectWithLiquid"
        arguments: dict[str, object] = {
            "objectId": object_id,
            "forceAction": True,
        }
        if not state:
            arguments["fillLiquid"] = "water"
        return Intervention(
            family,
            object_id,
            object_type,
            action,
            arguments,
            "isFilledWithLiquid",
            state,
            not state,
        )
    irreversible = {
        "cook": ("cookable", "isCooked", "CookObject"),
        "slice": ("sliceable", "isSliced", "SliceObject"),
        "break": ("breakable", "isBroken", "BreakObject"),
    }
    if family in irreversible:
        capability, state_field, action = irreversible[family]
        if row.get(capability) is not True or not isinstance(
            row.get(state_field), bool
        ):
            return None
        if row[state_field] is True:
            return None
        return Intervention(
            family,
            object_id,
            object_type,
            action,
            {"objectId": object_id, "forceAction": True},
            state_field,
            False,
            True,
        )
    raise ValueError(f"unknown intervention family: {family}")


def action_succeeded(metadata: Mapping[str, Any]) -> tuple[bool, str]:
    success = metadata.get("lastActionSuccess")
    if not isinstance(success, bool):
        raise ValueError("lastActionSuccess must be boolean")
    message = metadata.get("errorMessage", "")
    if not isinstance(message, str):
        raise ValueError("errorMessage must be a string")
    return success, message


def capture_event(
    event: object,
    destination: Path,
    stem: str,
    *,
    bundle_root: Path,
) -> dict[str, object]:
    from PIL import Image

    metadata = getattr(event, "metadata", None)
    frame = getattr(event, "frame", None)
    boxes = getattr(event, "instance_detections2D", None)
    if not isinstance(metadata, Mapping):
        raise ValueError("AI2-THOR event metadata is missing")
    if frame is None:
        raise ValueError("AI2-THOR event RGB frame is missing")
    if not isinstance(boxes, Mapping):
        raise ValueError("AI2-THOR instance detections are missing")
    destination.mkdir(parents=True, exist_ok=True)
    image_path = destination / f"{stem}.png"
    metadata_path = destination / f"{stem}.metadata.json"
    boxes_path = destination / f"{stem}.boxes.json"
    Image.fromarray(frame).save(image_path)
    _write_json(metadata_path, metadata)
    _write_json(boxes_path, boxes)
    return {
        "image": file_artifact(image_path, bundle_root=bundle_root),
        "metadata": file_artifact(metadata_path, bundle_root=bundle_root),
        "boxes": file_artifact(boxes_path, bundle_root=bundle_root),
    }


def file_artifact(path: Path, *, bundle_root: Path) -> dict[str, object]:
    resolved = path.resolve()
    root = bundle_root.resolve()
    try:
        relative = resolved.relative_to(root)
    except ValueError as error:
        raise ValueError("capture artifact must stay inside the bundle root") from error
    payload = resolved.read_bytes()
    return {
        "path": relative.as_posix(),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def make_controller(args: argparse.Namespace) -> object:
    from ai2thor.controller import Controller

    kwargs: dict[str, object] = {
        "scene": args.scene,
        "width": args.width,
        "height": args.height,
        "renderInstanceSegmentation": True,
    }
    if args.platform == "cloud":
        from ai2thor.platform import CloudRendering

        kwargs["platform"] = CloudRendering
    return Controller(**kwargs)


def position_for_intervention(controller: object, family: str) -> Intervention | None:
    if family == "move_receptacle":
        return move_receptacle_intervention(controller)
    event = controller.last_event
    visible = choose_intervention(event.metadata, family, visible_only=True)
    if visible is not None:
        return visible
    all_candidates = event.metadata.get("objects")
    if not isinstance(all_candidates, list):
        raise ValueError("AI2-THOR metadata must contain objects")
    interventions = [
        intervention
        for row in all_candidates
        if isinstance(row, Mapping)
        and (intervention := intervention_for_object(row, family)) is not None
    ]
    for intervention in sorted(
        interventions, key=lambda item: (item.object_type, item.object_id)
    ):
        poses_event = controller.step(
            action="GetInteractablePoses",
            objectId=intervention.object_id,
        )
        success, _ = action_succeeded(poses_event.metadata)
        poses = poses_event.metadata.get("actionReturn")
        if not success or not isinstance(poses, list) or not poses:
            continue
        teleport_event = controller.step(action="TeleportFull", **poses[0])
        teleported, _ = action_succeeded(teleport_event.metadata)
        if not teleported:
            continue
        visible = choose_intervention(
            teleport_event.metadata, family, visible_only=True
        )
        if visible is not None and visible.object_id == intervention.object_id:
            return visible
    return None


def move_receptacle_intervention(controller: object) -> Intervention | None:
    """Choose a deterministic visible movable object and in-view destination."""

    objects = controller.last_event.metadata.get("objects")
    if not isinstance(objects, list):
        raise ValueError("AI2-THOR metadata must contain objects")
    targets = sorted(
        (
            row
            for row in objects
            if isinstance(row, Mapping)
            and row.get("visible") is True
            and (row.get("pickupable") is True or row.get("moveable") is True)
            and isinstance(row.get("objectId"), str)
        ),
        key=lambda row: (_text(row.get("objectType"), "objectType"), row["objectId"]),
    )
    receptacles = sorted(
        (
            row
            for row in objects
            if isinstance(row, Mapping)
            and row.get("receptacle") is True
            and isinstance(row.get("objectId"), str)
        ),
        key=lambda row: (_text(row.get("objectType"), "objectType"), row["objectId"]),
    )
    for target in targets:
        current = set(_string_sequence(target.get("parentReceptacles")))
        for receptacle in receptacles:
            receptacle_id = _text(receptacle.get("objectId"), "objectId")
            if receptacle_id in current or receptacle_id == target["objectId"]:
                continue
            positions_event = controller.step(
                action="GetSpawnCoordinatesAboveReceptacle",
                objectId=receptacle_id,
                anywhere=False,
            )
            success, _ = action_succeeded(positions_event.metadata)
            positions = positions_event.metadata.get("actionReturn")
            if not success or not isinstance(positions, list) or not positions:
                continue
            valid_positions = [
                position
                for position in positions
                if isinstance(position, Mapping) and _valid_vector(position)
            ]
            if not valid_positions:
                continue
            position = min(valid_positions, key=_vector_key)
            return Intervention(
                "move_receptacle",
                _text(target.get("objectId"), "objectId"),
                _text(target.get("objectType"), "objectType"),
                "PlaceObjectAtPoint",
                {"objectId": target["objectId"], "position": dict(position)},
                "parentReceptacles",
                sorted(current),
                receptacle_id,
            )
    return None


def wait_for_scene_settled(
    controller: object,
    *,
    max_steps: int,
    stable_steps: int,
    position_tolerance_metres: float,
) -> dict[str, object]:
    """Advance physics until object poses are stable for consecutive steps."""

    if max_steps <= 0 or stable_steps <= 0 or stable_steps > max_steps:
        raise ValueError("settling steps must be positive and stable_steps <= max_steps")
    if not math.isfinite(position_tolerance_metres) or position_tolerance_metres < 0:
        raise ValueError("position tolerance must be a finite non-negative number")
    previous = _object_pose_map(controller.last_event.metadata)
    consecutive = 0
    step_audits: list[dict[str, object]] = []
    for step_index in range(1, max_steps + 1):
        event = controller.step(action="Pass")
        success, error_message = action_succeeded(event.metadata)
        if not success:
            return {
                "settled": False,
                "steps": step_index,
                "stable_steps_required": stable_steps,
                "reason": error_message or "Pass action failed",
                "step_audits": step_audits,
            }
        current = _object_pose_map(event.metadata)
        moved = _moved_objects(previous, current, position_tolerance_metres)
        consecutive = consecutive + 1 if not moved else 0
        step_audits.append({"step": step_index, "moved_object_ids": moved})
        if consecutive >= stable_steps:
            return {
                "settled": True,
                "steps": step_index,
                "stable_steps_required": stable_steps,
                "position_tolerance_metres": position_tolerance_metres,
                "step_audits": step_audits,
            }
        previous = current
    return {
        "settled": False,
        "steps": max_steps,
        "stable_steps_required": stable_steps,
        "position_tolerance_metres": position_tolerance_metres,
        "reason": "scene did not reach the required consecutive stable steps",
        "step_audits": step_audits,
    }


def audit_intervention_changes(
    before_metadata: Mapping[str, Any],
    after_metadata: Mapping[str, Any],
    intervention: Intervention,
    *,
    position_tolerance_metres: float,
) -> dict[str, object]:
    """Separate the intended target transition from every other derived change."""

    before = _object_rows(before_metadata)
    after = _object_rows(after_metadata)
    changes: list[dict[str, object]] = []
    for object_id in sorted(set(before) | set(after)):
        left = before.get(object_id)
        right = after.get(object_id)
        if left is None or right is None:
            changes.append(
                {
                    "object_id": object_id,
                    "object_type": _object_type(left or right),
                    "field": "presence",
                    "before": left is not None,
                    "after": right is not None,
                }
            )
            continue
        for field in _AUDITED_STATE_FIELDS:
            left_value = _normalized_field(left.get(field))
            right_value = _normalized_field(right.get(field))
            if left_value != right_value:
                changes.append(
                    {
                        "object_id": object_id,
                        "object_type": _object_type(right),
                        "field": field,
                        "before": left_value,
                        "after": right_value,
                    }
                )
        displacement = _position_distance(left.get("position"), right.get("position"))
        if displacement is not None and displacement > position_tolerance_metres:
            changes.append(
                {
                    "object_id": object_id,
                    "object_type": _object_type(right),
                    "field": "position",
                    "distance_metres": displacement,
                    "before": _normalized_field(left.get("position")),
                    "after": _normalized_field(right.get("position")),
                }
            )
    target_changes = [
        change for change in changes if change["object_id"] == intervention.object_id
    ]
    non_target_changes = [
        change for change in changes if change["object_id"] != intervention.object_id
    ]
    actual_after = _observed_intervention_state(after.get(intervention.object_id), intervention)
    return {
        "state_field": intervention.state_field,
        "expected_after_state": intervention.expected_after_state,
        "actual_after_state": actual_after,
        "intended_change_observed": _expected_state_matches(actual_after, intervention),
        "target_changes": target_changes,
        "non_target_changes": non_target_changes,
        "non_target_changes_are_causal_labels": False,
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.width <= 0 or args.height <= 0:
        raise ValueError("capture dimensions must be positive")
    if len(args.families) != len(set(args.families)):
        raise ValueError("intervention families cannot contain duplicates")
    if (
        args.settling_max_steps <= 0
        or args.settling_stable_steps <= 0
        or args.settling_stable_steps > args.settling_max_steps
    ):
        raise ValueError("settling steps must be positive and stable <= maximum")
    if (
        not math.isfinite(args.position_tolerance_metres)
        or args.position_tolerance_metres < 0
    ):
        raise ValueError("position tolerance must be finite and non-negative")
    controller = make_controller(args)
    records: list[dict[str, object]] = []
    try:
        for family in args.families:
            controller.reset(args.scene)
            intervention = position_for_intervention(controller, family)
            record: dict[str, object] = {
                "family": family,
                "scene": args.scene,
                "status": "excluded",
            }
            if intervention is None:
                record["reason"] = "no reachable eligible object"
                records.append(record)
                continue
            settling = wait_for_scene_settled(
                controller,
                max_steps=args.settling_max_steps,
                stable_steps=args.settling_stable_steps,
                position_tolerance_metres=args.position_tolerance_metres,
            )
            record["settling"] = settling
            if settling["settled"] is not True:
                record["reason"] = "pre-intervention scene did not settle"
                records.append(record)
                continue
            intervention = position_for_intervention(controller, family)
            if intervention is None:
                record["reason"] = "no eligible object after settling"
                records.append(record)
                continue
            episode_dir = args.output_dir / args.scene / family
            before_metadata = controller.last_event.metadata
            before = capture_event(
                controller.last_event,
                episode_dir,
                "before",
                bundle_root=args.output_dir,
            )
            action_event = controller.step(
                action=intervention.action,
                **dict(intervention.arguments),
            )
            success, error_message = action_succeeded(action_event.metadata)
            change_audit = audit_intervention_changes(
                before_metadata,
                action_event.metadata,
                intervention,
                position_tolerance_metres=args.position_tolerance_metres,
            )
            semantic_success = success and change_audit["intended_change_observed"] is True
            after = capture_event(
                action_event,
                episode_dir,
                "after",
                bundle_root=args.output_dir,
            )
            record.update(
                {
                    "status": "captured" if semantic_success else "failed",
                    "object_id": intervention.object_id,
                    "object_type": intervention.object_type,
                    "action": intervention.action,
                    "arguments": dict(intervention.arguments),
                    "before_state": intervention.before_state,
                    "expected_after_state": intervention.expected_after_state,
                    "last_action_success": success,
                    "error_message": error_message,
                    "semantic_success": semantic_success,
                    "change_audit": change_audit,
                    "before": before,
                    "after": after,
                }
            )
            records.append(record)
    finally:
        controller.stop()
    return build_manifest(args, records=records)


def build_manifest(
    args: argparse.Namespace,
    *,
    records: Sequence[Mapping[str, object]],
    initialization_error: BaseException | None = None,
) -> dict[str, object]:
    manifest: dict[str, object] = {
        "schema_version": 2,
        "evaluation_only": True,
        "truth_boundary": (
            "raw metadata and target identity must not be passed to the VLM or matcher"
        ),
        "ai2thor_version": version("ai2thor"),
        "host_platform": host_platform.platform(),
        "render_platform": args.platform,
        "scene": args.scene,
        "width": args.width,
        "height": args.height,
        "settling_policy": {
            "max_steps": getattr(args, "settling_max_steps", 12),
            "stable_steps": getattr(args, "settling_stable_steps", 2),
            "position_tolerance_metres": getattr(
                args, "position_tolerance_metres", 0.005
            ),
        },
        "families_requested": list(args.families),
        "run_status": "initialization_failed" if initialization_error else "completed",
        "records": list(records),
    }
    if initialization_error is not None:
        manifest["initialization_error"] = {
            "type": type(initialization_error).__name__,
            "message": str(initialization_error),
        }
    return manifest


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(_jsonable(payload), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )


def _jsonable(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "tolist"):
        return _jsonable(value.tolist())
    if hasattr(value, "item"):
        return _jsonable(value.item())
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"cannot serialize AI2-THOR value: {type(value).__name__}")


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _string_sequence(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError("parentReceptacles must be null or an array of strings")
    return tuple(value)


def _valid_vector(value: Mapping[str, object]) -> bool:
    return all(
        isinstance(value.get(axis), (int, float))
        and not isinstance(value.get(axis), bool)
        and math.isfinite(float(value[axis]))
        for axis in ("x", "y", "z")
    )


def _vector_key(value: Mapping[str, object]) -> tuple[float, float, float]:
    return tuple(float(value[axis]) for axis in ("x", "y", "z"))


def _object_rows(metadata: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    rows = metadata.get("objects")
    if not isinstance(rows, list):
        raise ValueError("AI2-THOR metadata must contain an objects array")
    result: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("AI2-THOR object rows must be objects")
        object_id = _text(row.get("objectId"), "objectId")
        if object_id in result:
            raise ValueError("AI2-THOR object IDs must be unique")
        result[object_id] = row
    return result


def _object_type(row: Mapping[str, Any] | None) -> str | None:
    if row is None:
        return None
    value = row.get("objectType")
    return value if isinstance(value, str) and value else None


def _normalized_field(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _normalized_field(item) for key, item in sorted(value.items())}
    if isinstance(value, list):
        return sorted((_normalized_field(item) for item in value), key=str)
    return value


def _position_distance(left: object, right: object) -> float | None:
    if not isinstance(left, Mapping) or not isinstance(right, Mapping):
        return None
    if not _valid_vector(left) or not _valid_vector(right):
        return None
    return math.sqrt(
        sum((float(right[axis]) - float(left[axis])) ** 2 for axis in ("x", "y", "z"))
    )


def _object_pose_map(metadata: Mapping[str, Any]) -> dict[str, Mapping[str, object]]:
    return {
        object_id: {
            "position": row["position"],
            "isMoving": row.get("isMoving"),
        }
        for object_id, row in _object_rows(metadata).items()
        if isinstance(row.get("position"), Mapping) and _valid_vector(row["position"])
    }


def _moved_objects(
    before: Mapping[str, Mapping[str, object]],
    after: Mapping[str, Mapping[str, object]],
    tolerance: float,
) -> list[str]:
    moved: list[str] = []
    for object_id in sorted(set(before) | set(after)):
        if object_id not in before or object_id not in after:
            moved.append(object_id)
            continue
        distance = _position_distance(
            before[object_id].get("position"), after[object_id].get("position")
        )
        if after[object_id].get("isMoving") is True or (
            distance is not None and distance > tolerance
        ):
            moved.append(object_id)
    return moved


def _observed_intervention_state(
    row: Mapping[str, Any] | None, intervention: Intervention
) -> object:
    if row is None:
        return None
    value = row.get(intervention.state_field)
    if intervention.family == "move_receptacle":
        return list(_string_sequence(value))
    return value


def _expected_state_matches(actual: object, intervention: Intervention) -> bool:
    if intervention.family == "move_receptacle":
        return isinstance(actual, list) and intervention.expected_after_state in actual
    return actual == intervention.expected_after_state


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / f"{args.scene}.capture-manifest.json"
    try:
        manifest = run(args)
    except Exception as error:
        manifest = build_manifest(
            args,
            records=(),
            initialization_error=error,
        )
        _write_json(manifest_path, manifest)
        print("captured=0/0")
        print(f"manifest: {manifest_path}")
        raise
    _write_json(manifest_path, manifest)
    captured = sum(row["status"] == "captured" for row in manifest["records"])
    print(f"captured={captured}/{len(manifest['records'])}")
    print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main()
