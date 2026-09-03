from __future__ import annotations

import argparse
import hashlib
import json
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
    before_state: object
    expected_after_state: object


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
        choices=("open", "toggle", "dirty", "fill"),
        default=("open", "toggle", "dirty", "fill"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/ai2thor_pilot"),
    )
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
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
            state,
            not state,
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


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.width <= 0 or args.height <= 0:
        raise ValueError("capture dimensions must be positive")
    if len(args.families) != len(set(args.families)):
        raise ValueError("intervention families cannot contain duplicates")
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
            episode_dir = args.output_dir / args.scene / family
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
            after = capture_event(
                action_event,
                episode_dir,
                "after",
                bundle_root=args.output_dir,
            )
            record.update(
                {
                    "status": "captured" if success else "failed",
                    "object_id": intervention.object_id,
                    "object_type": intervention.object_type,
                    "action": intervention.action,
                    "arguments": dict(intervention.arguments),
                    "before_state": intervention.before_state,
                    "expected_after_state": intervention.expected_after_state,
                    "last_action_success": success,
                    "error_message": error_message,
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
