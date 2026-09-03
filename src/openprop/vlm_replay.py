from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from pathlib import Path


_SHA256 = re.compile(r"[0-9a-f]{64}")
_FORBIDDEN_INPUT_KEYS = {
    "objects",
    "current_truth",
    "truth",
    "truth_artifact",
    "transition",
    "changes",
    "target",
    "target_entity_id",
    "evaluation_only",
}


def write_captured_vlm_response(
    path: str | Path,
    *,
    input_artifact: str | Path,
    provider: str,
    model: str,
    system_id: str,
    response: Mapping[str, object],
    request_settings: Mapping[str, object],
) -> dict[str, object]:
    input_path = Path(input_artifact).resolve()
    input_payload = _validated_vlm_input(input_path)
    if not isinstance(response, Mapping):
        raise ValueError("captured VLM response must be a JSON object")
    payload: dict[str, object] = {
        "schema_version": 1,
        "provider": _nonempty(provider, "provider"),
        "model": _nonempty(model, "model"),
        "system_id": _nonempty(system_id, "system_id"),
        "input_artifact": input_path.name,
        "input_artifact_sha256": _sha256(input_path.read_bytes()),
        "input_episode_id": _nonempty(input_payload.get("episode_id"), "episode_id"),
        "request_settings": dict(request_settings),
        "response": dict(response),
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def read_captured_vlm_response(
    path: str | Path,
    *,
    input_artifact: str | Path,
) -> dict[str, object]:
    response_path = Path(path).resolve()
    payload = json.loads(response_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or payload.get("schema_version") != 1:
        raise ValueError("captured VLM response must use schema_version 1")
    for field in ("provider", "model", "system_id", "input_episode_id"):
        _nonempty(payload.get(field), field)
    input_path = Path(input_artifact).resolve()
    input_payload = _validated_vlm_input(input_path)
    expected_hash = payload.get("input_artifact_sha256")
    if not isinstance(expected_hash, str) or _SHA256.fullmatch(expected_hash) is None:
        raise ValueError("input_artifact_sha256 must be lowercase hex")
    if _sha256(input_path.read_bytes()) != expected_hash:
        raise ValueError("captured VLM response input hash drifted")
    if payload.get("input_episode_id") != input_payload.get("episode_id"):
        raise ValueError("captured VLM response episode does not match input")
    if not isinstance(payload.get("request_settings"), Mapping):
        raise ValueError("request_settings must be a JSON object")
    if not isinstance(payload.get("response"), Mapping):
        raise ValueError("response must be a JSON object")
    return dict(payload)


def _validated_vlm_input(path: Path) -> Mapping[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("VLM input artifact must be a JSON object")
    forbidden = sorted(_find_forbidden_keys(payload))
    if forbidden:
        raise ValueError(f"VLM input artifact contains truth fields: {forbidden}")
    frames = payload.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError("VLM input artifact must contain non-empty frames")
    return payload


def _find_forbidden_keys(value: object) -> set[str]:
    result: set[str] = set()
    if isinstance(value, Mapping):
        for key, item in value.items():
            if isinstance(key, str) and key in _FORBIDDEN_INPUT_KEYS:
                result.add(key)
            result.update(_find_forbidden_keys(item))
    elif isinstance(value, list):
        for item in value:
            result.update(_find_forbidden_keys(item))
    return result


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _nonempty(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()
