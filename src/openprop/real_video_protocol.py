from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from pathlib import Path


REAL_VIDEO_PROTOCOL = "openprop-real-video-v1"
REAL_VIDEO_TRUTH_BOUNDARY = (
    "evaluation annotations and query targets must not be passed to the VLM or matcher"
)
_SHA256 = re.compile(r"[0-9a-f]{64}")
_SPLITS = {"development", "calibration", "test"}
_SOURCE_KINDS = {"self_recorded", "public_dataset", "licensed_web"}


def verify_real_video_manifest(path: str | Path) -> dict[str, object]:
    """Fail closed on media integrity, annotation quality, and split leakage."""

    manifest_path = Path(path).resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("real-video manifest must be a JSON object")
    if payload.get("schema_version") != 1:
        raise ValueError("real-video manifest must use schema_version 1")
    if payload.get("protocol") != REAL_VIDEO_PROTOCOL:
        raise ValueError("real-video protocol identifier drifted")
    if payload.get("evaluation_only") is not True:
        raise ValueError("real-video manifest must be marked evaluation-only")
    if payload.get("truth_boundary") != REAL_VIDEO_TRUTH_BOUNDARY:
        raise ValueError("real-video truth boundary drifted")
    collection_id = _text(payload.get("collection_id"), "collection_id")
    source_policy = _source_policy(payload.get("source_policy"))
    annotation_protocol = _annotation_protocol(payload.get("annotation_protocol"))
    episodes = payload.get("episodes")
    if not isinstance(episodes, list) or not episodes:
        raise ValueError("real-video manifest must contain episodes")

    episode_ids: set[str] = set()
    cluster_splits: dict[str, str] = {}
    media_count = 0
    split_counts = {name: 0 for name in sorted(_SPLITS)}
    event_count = 0
    for episode in episodes:
        if not isinstance(episode, Mapping):
            raise ValueError("real-video episode must be an object")
        episode_id = _text(episode.get("episode_id"), "episode_id")
        if episode_id in episode_ids:
            raise ValueError("real-video episode IDs must be unique")
        episode_ids.add(episode_id)
        split = episode.get("split")
        if split not in _SPLITS:
            raise ValueError("real-video episode has invalid split")
        split_counts[str(split)] += 1
        cluster_id = _text(episode.get("cluster_id"), "cluster_id")
        previous_split = cluster_splits.setdefault(cluster_id, str(split))
        if previous_split != split:
            raise ValueError("room/person cluster leaks across splits")
        _text(episode.get("source"), "episode source")
        _text(episode.get("condition"), "condition")
        distractors = _integer(episode.get("distractor_count"), "distractor_count")
        if distractors < 0:
            raise ValueError("distractor_count must be nonnegative")
        frame_ids, candidate_ids, first_time, last_time, episode_media = _verify_frames(
            manifest_path.parent, episode.get("frames")
        )
        media_count += episode_media
        _verify_case_fields(episode, candidate_ids, first_time, last_time)
        event_count += _verify_annotations(
            episode.get("annotations"), frame_ids, candidate_ids
        )

    return {
        "schema_version": 1,
        "protocol": REAL_VIDEO_PROTOCOL,
        "manifest": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path.read_bytes()),
        "collection_id": collection_id,
        "source_kind": source_policy["source_kind"],
        "annotation_tier": annotation_protocol["tier"],
        "episodes": len(episodes),
        "clusters": len(cluster_splits),
        "media_verified": media_count,
        "events": event_count,
        "split_counts": split_counts,
        "truth_exposed_to_matcher": False,
    }


def prepare_real_video_manifest(
    manifest_path: str | Path,
    output_directory: str | Path,
) -> dict[str, object]:
    """Materialize physically separate VLM input, replay case, and truth files."""

    manifest = Path(manifest_path).resolve()
    verification = verify_real_video_manifest(manifest)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    output = Path(output_directory).resolve()
    roots = {name: output / name for name in ("inputs", "cases", "truth")}
    for root in roots.values():
        root.mkdir(parents=True, exist_ok=True)
    prepared: list[dict[str, object]] = []
    for episode in payload["episodes"]:
        episode_id = episode["episode_id"]
        input_path = roots["inputs"] / f"{episode_id}.json"
        case_path = roots["cases"] / f"{episode_id}.json"
        truth_path = roots["truth"] / f"{episode_id}.json"
        frames = []
        for row in episode["frames"]:
            image_path = _resolve_artifact(manifest.parent, row["image"])
            frames.append(
                {
                    "frame_id": row["frame_id"],
                    "image_url": str(image_path),
                    "captured_at": row["captured_at"],
                    "source": episode["source"],
                    "candidate_entity_ids": [
                        item["entity_id"] for item in row["candidates"]
                    ],
                    "candidate_regions": {
                        item["entity_id"]: item["region"]
                        for item in row["candidates"]
                    },
                }
            )
        _write_json(
            input_path,
            {
                "schema_version": 1,
                "episode_id": episode_id,
                "collection_id": payload["collection_id"],
                "manifest_sha256": verification["manifest_sha256"],
                "frames": frames,
            },
        )
        _write_json(
            case_path,
            {
                "schema_version": 1,
                "case_id": episode_id,
                "initial_entities": episode["initial_entities"],
                "query_time": episode["query_time"],
                "query": episode["query"],
                "query_candidate_entity_ids": episode["query_candidate_entity_ids"],
            },
        )
        annotations = episode["annotations"]
        _write_json(
            truth_path,
            {
                "schema_version": 1,
                "evaluation_only": True,
                "case_id": episode_id,
                "cluster_id": episode["cluster_id"],
                "split": episode["split"],
                "source": episode["source"],
                "condition": episode["condition"],
                "distractor_count": episode["distractor_count"],
                "frames": annotations["frames"],
                "query": annotations["query"],
            },
        )
        prepared.append(
            {
                "episode_id": episode_id,
                "split": episode["split"],
                "cluster_id": episode["cluster_id"],
                "input": _output_artifact(input_path, output),
                "case": _output_artifact(case_path, output),
                "truth": _output_artifact(truth_path, output),
            }
        )
    return {
        **verification,
        "schema_version": 1,
        "prepared_episodes": len(prepared),
        "episodes": prepared,
        "truth_exposed_to_matcher": False,
    }


def _source_policy(raw: object) -> Mapping[str, object]:
    if not isinstance(raw, Mapping):
        raise ValueError("source_policy must be an object")
    kind = raw.get("source_kind")
    if kind not in _SOURCE_KINDS:
        raise ValueError("source_policy has invalid source_kind")
    for field in ("source_name", "license", "redistribution"):
        _text(raw.get(field), f"source_policy {field}")
    if kind == "self_recorded":
        _text(raw.get("consent_basis"), "source_policy consent_basis")
    return raw


def _annotation_protocol(raw: object) -> Mapping[str, object]:
    if not isinstance(raw, Mapping):
        raise ValueError("annotation_protocol must be an object")
    tier = raw.get("tier")
    if tier not in {"pilot", "final"}:
        raise ValueError("annotation tier must be pilot or final")
    count = _integer(raw.get("annotator_count"), "annotator_count")
    adjudicated = raw.get("adjudicated")
    if not isinstance(adjudicated, bool):
        raise ValueError("adjudicated must be boolean")
    _text(raw.get("candidate_source"), "candidate_source")
    _text(raw.get("guideline_version"), "guideline_version")
    if tier == "final":
        agreement = _number(raw.get("agreement_value"), "agreement_value")
        minimum = _number(raw.get("minimum_agreement"), "minimum_agreement")
        _text(raw.get("agreement_metric"), "agreement_metric")
        if agreement > 1.0 or minimum > 1.0 or minimum < 0.8:
            raise ValueError("final annotation agreement gate must be within [0.8, 1]")
        if count < 3 or not adjudicated:
            raise ValueError("final annotations require three annotators and adjudication")
        if agreement < minimum:
            raise ValueError("final annotation agreement is below the frozen gate")
    if count < 1:
        raise ValueError("annotator_count must be positive")
    return raw


def _verify_frames(
    root: Path, raw: object
) -> tuple[set[str], set[str], float, float, int]:
    if not isinstance(raw, list) or not raw:
        raise ValueError("episode frames must be a non-empty array")
    frame_ids: set[str] = set()
    candidate_ids: set[str] = set()
    previous_time = -math.inf
    first_time: float | None = None
    for row in raw:
        if not isinstance(row, Mapping):
            raise ValueError("episode frame must be an object")
        frame_id = _text(row.get("frame_id"), "frame_id")
        if frame_id in frame_ids:
            raise ValueError("frame IDs must be unique within an episode")
        frame_ids.add(frame_id)
        captured_at = _number(row.get("captured_at"), "captured_at")
        if first_time is None:
            first_time = captured_at
        if captured_at <= previous_time:
            raise ValueError("frame times must be strictly increasing")
        previous_time = captured_at
        image = _resolve_artifact(root, row.get("image"))
        if not _is_supported_image(image.read_bytes()):
            raise ValueError("real-video frame is not PNG, JPEG, or WebP")
        candidates = row.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            raise ValueError("every frame must contain candidate boxes")
        local_ids: set[str] = set()
        for candidate in candidates:
            if not isinstance(candidate, Mapping):
                raise ValueError("candidate must be an object")
            entity_id = _text(candidate.get("entity_id"), "candidate entity_id")
            if entity_id in local_ids:
                raise ValueError("candidate IDs must be unique within a frame")
            local_ids.add(entity_id)
            candidate_ids.add(entity_id)
            _region(candidate.get("region"), "candidate region")
    assert first_time is not None
    return frame_ids, candidate_ids, first_time, previous_time, len(raw)


def _verify_case_fields(
    episode: Mapping[str, object],
    candidate_ids: set[str],
    first_time: float,
    last_time: float,
) -> None:
    initial = episode.get("initial_entities")
    if not isinstance(initial, list) or not initial:
        raise ValueError("initial_entities must be a non-empty array")
    if not all(isinstance(row, Mapping) for row in initial):
        raise ValueError("initial entity must be an object")
    initial_ids = {_text(row.get("entity_id"), "initial entity_id") for row in initial}
    if len(initial_ids) != len(initial) or initial_ids != candidate_ids:
        raise ValueError("initial_entities must exactly cover all candidates")
    for row in initial:
        properties = row.get("properties")
        if not isinstance(properties, Mapping):
            raise ValueError("initial entity properties must be an object")
        for property_name, observation in properties.items():
            _text(property_name, "initial property name")
            if not isinstance(observation, Mapping):
                raise ValueError("initial observation must be an object")
            confidence = _number(
                observation.get("confidence"), "initial observation confidence"
            )
            if confidence > 1.0:
                raise ValueError("initial observation confidence cannot exceed one")
            if _number(observation.get("timestamp"), "initial observation timestamp") >= first_time:
                raise ValueError("initial observations must precede every video frame")
            _text(observation.get("source"), "initial observation source")
    query_time = _number(episode.get("query_time"), "query_time")
    if query_time < last_time:
        raise ValueError("query_time cannot precede the visual history")
    query = episode.get("query")
    if not isinstance(query, Mapping) or not isinstance(query.get("constraints"), list):
        raise ValueError("query must contain constraints")
    _text(query.get("text"), "query text")
    constraints = query["constraints"]
    if not constraints:
        raise ValueError("query constraints must be non-empty")
    for constraint in constraints:
        if not isinstance(constraint, Mapping):
            raise ValueError("query constraint must be an object")
        _text(constraint.get("property_name"), "query constraint property_name")
        relevance = _number(constraint.get("relevance"), "query constraint relevance")
        if relevance > 1.0:
            raise ValueError("query constraint relevance cannot exceed one")
    query_candidates = episode.get("query_candidate_entity_ids")
    if not isinstance(query_candidates, list) or not query_candidates:
        raise ValueError("query_candidate_entity_ids must be non-empty")
    query_ids = [_text(item, "query candidate") for item in query_candidates]
    if len(query_ids) != len(set(query_ids)) or not set(query_ids).issubset(candidate_ids):
        raise ValueError("query candidates must be unique known candidates")


def _verify_annotations(
    raw: object, frame_ids: set[str], candidate_ids: set[str]
) -> int:
    if not isinstance(raw, Mapping):
        raise ValueError("annotations must be an object")
    frames = raw.get("frames")
    if not isinstance(frames, list):
        raise ValueError("annotation frames must be an array")
    seen_frames: set[str] = set()
    event_ids: set[str] = set()
    for row in frames:
        if not isinstance(row, Mapping):
            raise ValueError("annotation frame must be an object")
        frame_id = _text(row.get("frame_id"), "annotation frame_id")
        if frame_id in seen_frames:
            raise ValueError("annotation frame IDs must be unique")
        seen_frames.add(frame_id)
        events = row.get("events")
        if not isinstance(events, list):
            raise ValueError("annotation events must be an array")
        for event in events:
            if not isinstance(event, Mapping):
                raise ValueError("annotation event must be an object")
            event_id = _text(event.get("event_id"), "event_id")
            if event_id in event_ids:
                raise ValueError("event IDs must be unique within an episode")
            event_ids.add(event_id)
            _text(event.get("property_name"), "event property_name")
            target = _text(event.get("target_entity_id"), "event target_entity_id")
            if target not in candidate_ids:
                raise ValueError("event target must be a known candidate")
            _region(event.get("region"), "event region")
    if seen_frames != frame_ids:
        raise ValueError("annotations must cover every frame exactly")
    query = raw.get("query")
    if not isinstance(query, Mapping):
        raise ValueError("query annotation must be an object")
    _text(query.get("record_id"), "query record_id")
    _text(query.get("property_name"), "query property_name")
    target = query.get("target_entity_id")
    if target is not None:
        _text(target, "query target_entity_id")
    _number(query.get("horizon_seconds"), "query horizon_seconds")
    if not isinstance(query.get("eligible", True), bool):
        raise ValueError("query eligible must be boolean")
    return len(event_ids)


def _resolve_artifact(root: Path, raw: object) -> Path:
    if not isinstance(raw, Mapping) or set(raw) != {"path", "bytes", "sha256"}:
        raise ValueError("image reference must contain path/bytes/sha256")
    relative = Path(_text(raw.get("path"), "image path"))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("image path must be relative and confined")
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError("image path escapes manifest root") from error
    if not resolved.is_file():
        raise FileNotFoundError(f"missing real-video image: {relative.as_posix()}")
    data = resolved.read_bytes()
    size = raw.get("bytes")
    digest = raw.get("sha256")
    if isinstance(size, bool) or not isinstance(size, int) or size < 0:
        raise ValueError("image byte count must be a nonnegative integer")
    if len(data) != size:
        raise ValueError("real-video image byte count drifted")
    if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
        raise ValueError("image sha256 must be lowercase hex")
    if _sha256(data) != digest:
        raise ValueError("real-video image hash drifted")
    return resolved


def _region(raw: object, field: str) -> tuple[float, float, float, float]:
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)) or len(raw) != 4:
        raise ValueError(f"{field} must contain four coordinates")
    values = tuple(_number(item, field) for item in raw)
    if not (0.0 <= values[0] < values[2] <= 1.0):
        raise ValueError(f"{field} x coordinates are invalid")
    if not (0.0 <= values[1] < values[3] <= 1.0):
        raise ValueError(f"{field} y coordinates are invalid")
    return values


def _is_supported_image(data: bytes) -> bool:
    return (
        data.startswith(b"\x89PNG\r\n\x1a\n")
        or data.startswith(b"\xff\xd8\xff")
        or (len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP")
    )


def _output_artifact(path: Path, root: Path) -> dict[str, object]:
    data = path.read_bytes()
    return {
        "path": path.relative_to(root).as_posix(),
        "bytes": len(data),
        "sha256": _sha256(data),
    }


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def _number(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{field} must be finite and nonnegative")
    return result


def _integer(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")
    return value


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()
