from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from pathlib import Path

from .ai2thor_adapter import (
    AI2ThorFrameBundle,
    AI2ThorTransitionTruth,
    derive_ai2thor_transition,
    extract_ai2thor_frame,
)


CAPTURE_TRUTH_BOUNDARY = (
    "raw metadata and target identity must not be passed to the VLM or matcher"
)
_SHA256 = re.compile(r"[0-9a-f]{64}")
_ARTIFACT_KEYS = ("image", "metadata", "boxes")


def verify_ai2thor_capture_manifest(path: str | Path) -> dict[str, object]:
    """Verify a portable AI2-THOR capture bundle without exposing its truth."""

    manifest_path = Path(path).resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("capture manifest must be a JSON object")
    if payload.get("schema_version") != 2:
        raise ValueError("capture manifest schema_version must be 2")
    if payload.get("evaluation_only") is not True:
        raise ValueError("capture manifest must mark simulator truth evaluation-only")
    if payload.get("truth_boundary") != CAPTURE_TRUTH_BOUNDARY:
        raise ValueError("capture manifest truth boundary drifted")

    scene = _nonempty(payload.get("scene"), "scene")
    run_status = payload.get("run_status")
    if run_status not in {"completed", "initialization_failed"}:
        raise ValueError("capture manifest has invalid run_status")
    families = payload.get("families_requested")
    if not isinstance(families, list) or not families:
        raise ValueError("families_requested must be a non-empty array")
    requested = tuple(_nonempty(item, "family") for item in families)
    if len(set(requested)) != len(requested):
        raise ValueError("families_requested contains duplicates")

    records = payload.get("records")
    if not isinstance(records, list):
        raise ValueError("capture manifest records must be an array")
    if run_status == "initialization_failed":
        if records:
            raise ValueError("initialization_failed manifest cannot contain records")
        error = payload.get("initialization_error")
        if not isinstance(error, Mapping):
            raise ValueError("initialization failure details are missing")
        _nonempty(error.get("type"), "initialization error type")
        _nonempty(error.get("message"), "initialization error message")
        return _report(manifest_path, scene, run_status, records, artifact_count=0)

    seen: set[str] = set()
    artifact_count = 0
    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError("capture record must be an object")
        family = _nonempty(record.get("family"), "record family")
        if family in seen:
            raise ValueError("capture record families must be unique")
        seen.add(family)
        if family not in requested:
            raise ValueError("capture record family was not requested")
        if record.get("scene") != scene:
            raise ValueError("capture record scene does not match manifest")
        status = record.get("status")
        if status == "excluded":
            _nonempty(record.get("reason"), "exclusion reason")
            continue
        if status not in {"captured", "failed"}:
            raise ValueError("capture record has invalid status")
        success = record.get("last_action_success")
        semantic_success = record.get("semantic_success", success)
        if not isinstance(success, bool) or not isinstance(semantic_success, bool):
            raise ValueError("capture success fields must be boolean")
        if semantic_success and not success:
            raise ValueError("semantic success requires action success")
        if semantic_success is not (status == "captured"):
            raise ValueError("record status contradicts semantic success")
        if "semantic_success" in record:
            settling = record.get("settling")
            if not isinstance(settling, Mapping) or settling.get("settled") is not True:
                raise ValueError("attempted capture requires a passed settling audit")
            change_audit = record.get("change_audit")
            if not isinstance(change_audit, Mapping):
                raise ValueError("semantic capture requires a change audit")
            intended = change_audit.get("intended_change_observed")
            if not isinstance(intended, bool):
                raise ValueError("change audit must report intended_change_observed")
            if semantic_success is not (success and intended):
                raise ValueError("semantic success contradicts the change audit")
            if change_audit.get("non_target_changes_are_causal_labels") is not False:
                raise ValueError("non-target changes must not be causal labels")
        _nonempty(record.get("object_id"), "object_id")
        _nonempty(record.get("object_type"), "object_type")
        _nonempty(record.get("action"), "action")
        for stage in ("before", "after"):
            artifacts = record.get(stage)
            if not isinstance(artifacts, Mapping):
                raise ValueError(f"{stage} capture artifacts are missing")
            if set(artifacts) != set(_ARTIFACT_KEYS):
                raise ValueError(f"{stage} capture artifacts must be image/metadata/boxes")
            for kind in _ARTIFACT_KEYS:
                artifact_path = _verify_artifact(
                    manifest_path.parent,
                    artifacts[kind],
                    kind=kind,
                )
                artifact_count += 1
                if kind == "image":
                    if artifact_path.read_bytes()[:8] != b"\x89PNG\r\n\x1a\n":
                        raise ValueError("capture image is not a PNG")
                else:
                    decoded = json.loads(artifact_path.read_text(encoding="utf-8"))
                    if not isinstance(decoded, Mapping):
                        raise ValueError(f"capture {kind} must contain a JSON object")
                    if kind == "metadata" and not isinstance(decoded.get("objects"), list):
                        raise ValueError("capture metadata must contain an objects array")
    if seen != set(requested):
        raise ValueError("completed capture must account for every requested family")
    return _report(manifest_path, scene, run_status, records, artifact_count)


def resolve_capture_artifact(
    manifest_path: str | Path,
    reference: object,
    *,
    kind: str,
) -> Path:
    """Resolve one already-schema-validated artifact with confinement and hash checks."""

    return _verify_artifact(Path(manifest_path).resolve().parent, reference, kind=kind)


def prepare_ai2thor_capture_manifest(
    manifest_path: str | Path,
    output_directory: str | Path,
    *,
    source: str = "ai2thor-rgb",
    movement_threshold_metres: float = 0.05,
) -> dict[str, object]:
    """Build hash-bound VLM inputs and separate truth from a verified bundle."""

    manifest = Path(manifest_path).resolve()
    verification = verify_ai2thor_capture_manifest(manifest)
    if verification["run_status"] != "completed":
        raise ValueError("cannot prepare an initialization-failed capture")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    output = Path(output_directory).resolve()
    input_root = output / "inputs"
    truth_root = output / "truth"
    input_root.mkdir(parents=True, exist_ok=True)
    truth_root.mkdir(parents=True, exist_ok=True)
    episodes: list[dict[str, object]] = []
    for record in payload["records"]:
        if record["status"] != "captured":
            continue
        scene = _nonempty(record.get("scene"), "record scene")
        family = _nonempty(record.get("family"), "record family")
        episode_id = f"{scene}.{family}"
        bundles: list[AI2ThorFrameBundle] = []
        for index, stage in enumerate(("before", "after")):
            artifacts = record[stage]
            metadata_path = resolve_capture_artifact(
                manifest, artifacts["metadata"], kind="metadata"
            )
            boxes_path = resolve_capture_artifact(
                manifest, artifacts["boxes"], kind="boxes"
            )
            image_path = resolve_capture_artifact(
                manifest, artifacts["image"], kind="image"
            )
            bundles.append(
                extract_ai2thor_frame(
                    json.loads(metadata_path.read_text(encoding="utf-8")),
                    frame_id=f"{episode_id}.{stage}",
                    image_url=str(image_path),
                    captured_at=float(index),
                    source=source,
                    instance_detections_2d=json.loads(
                        boxes_path.read_text(encoding="utf-8")
                    ),
                )
            )
        transition = derive_ai2thor_transition(
            bundles[0],
            bundles[1],
            action=_nonempty(record.get("action"), "record action"),
            movement_threshold_metres=movement_threshold_metres,
        )
        input_path = input_root / f"{episode_id}.json"
        truth_path = truth_root / f"{episode_id}.json"
        _write_json(
            input_path,
            {
                "schema_version": 1,
                "episode_id": episode_id,
                "capture_manifest_sha256": verification["manifest_sha256"],
                "frames": [_frame_input(item) for item in bundles],
            },
        )
        _write_json(
            truth_path,
            {
                "schema_version": 1,
                "evaluation_only": True,
                "episode_id": episode_id,
                "capture_manifest_sha256": verification["manifest_sha256"],
                "frames": [_truth_frame(item) for item in bundles],
                "transition": _transition_payload(transition),
            },
        )
        episodes.append(
            {
                "episode_id": episode_id,
                "family": family,
                "input": _output_artifact(input_path, output),
                "truth": _output_artifact(truth_path, output),
                "changed_entities": list(transition.changed_entity_ids),
                "candidate_coverage": {
                    stage: _candidate_coverage(bundle)
                    for stage, bundle in zip(("before", "after"), bundles)
                },
            }
        )
    return {
        "schema_version": 1,
        "evaluation_only": True,
        "capture_manifest_sha256": verification["manifest_sha256"],
        "capture_status_counts": verification["status_counts"],
        "prepared_episodes": len(episodes),
        "episodes": episodes,
        "truth_exposed_to_matcher": False,
    }


def _verify_artifact(root: Path, reference: object, *, kind: str) -> Path:
    if not isinstance(reference, Mapping) or set(reference) != {"path", "bytes", "sha256"}:
        raise ValueError(f"{kind} artifact reference must contain path/bytes/sha256")
    relative_text = _nonempty(reference.get("path"), f"{kind} artifact path")
    relative = Path(relative_text)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("capture artifact path must be relative and confined")
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError("capture artifact path escapes bundle root") from error
    if not resolved.is_file():
        raise FileNotFoundError(f"missing capture artifact: {relative_text}")
    data = resolved.read_bytes()
    expected_size = reference.get("bytes")
    if isinstance(expected_size, bool) or not isinstance(expected_size, int) or expected_size < 0:
        raise ValueError("capture artifact bytes must be a non-negative integer")
    if len(data) != expected_size:
        raise ValueError("capture artifact byte count drifted")
    expected_hash = reference.get("sha256")
    if not isinstance(expected_hash, str) or _SHA256.fullmatch(expected_hash) is None:
        raise ValueError("capture artifact sha256 must be lowercase hex")
    if hashlib.sha256(data).hexdigest() != expected_hash:
        raise ValueError("capture artifact hash drifted")
    return resolved


def _frame_input(bundle: AI2ThorFrameBundle) -> dict[str, object]:
    frame = bundle.frame
    return {
        "frame_id": frame.frame_id,
        "image_url": frame.image_url,
        "captured_at": frame.captured_at,
        "source": frame.source,
        "candidate_entity_ids": list(frame.candidate_entity_ids),
        "candidate_regions": {
            entity_id: list(region)
            for entity_id, region in frame.candidate_regions.items()
        },
    }


def _truth_frame(bundle: AI2ThorFrameBundle) -> dict[str, object]:
    return {
        "frame_id": bundle.frame.frame_id,
        "scene_name": bundle.scene_name,
        "objects": [
            {
                "entity_id": item.entity_id,
                "object_type": item.object_type,
                "visible": item.visible,
                "values": dict(item.values),
            }
            for item in bundle.current_truth
        ],
    }


def _candidate_coverage(bundle: AI2ThorFrameBundle) -> dict[str, object]:
    visible = {
        item.entity_id for item in bundle.current_truth if item.visible
    }
    candidates = set(bundle.frame.candidate_entity_ids)
    return {
        "visible_entities": len(visible),
        "anchored_candidates": len(candidates),
        "coverage": len(candidates) / len(visible) if visible else 1.0,
        "unanchored_visible_entity_ids": sorted(visible - candidates),
    }


def _transition_payload(transition: AI2ThorTransitionTruth) -> dict[str, object]:
    return {
        "scene_name": transition.scene_name,
        "action": transition.action,
        "before_frame_id": transition.before_frame_id,
        "after_frame_id": transition.after_frame_id,
        "changes": {
            entity_id: [
                {
                    "property_name": change.property_name,
                    "before": change.before,
                    "after": change.after,
                }
                for change in changes
            ]
            for entity_id, changes in transition.changes.items()
        },
    }


def _output_artifact(path: Path, root: Path) -> dict[str, object]:
    data = path.read_bytes()
    return {
        "path": path.resolve().relative_to(root.resolve()).as_posix(),
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _report(
    manifest_path: Path,
    scene: str,
    run_status: str,
    records: list[object],
    artifact_count: int,
) -> dict[str, object]:
    counts = {name: 0 for name in ("captured", "failed", "excluded")}
    for record in records:
        if isinstance(record, Mapping) and record.get("status") in counts:
            counts[str(record["status"])] += 1
    return {
        "schema_version": 1,
        "manifest": str(manifest_path),
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "scene": scene,
        "run_status": run_status,
        "records": len(records),
        "artifacts_verified": artifact_count,
        "status_counts": counts,
        "truth_exposed_to_matcher": False,
    }


def _nonempty(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()
