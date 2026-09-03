from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict

from .candidate_tracking import (
    CandidateSourceFrame,
    CandidateTracker,
    CandidateTrackingPolicy,
    CandidateTrackingRun,
    RegionProposal,
)


_FORBIDDEN_KEYS = {
    "truth",
    "current_truth",
    "target",
    "target_entity_id",
    "objects",
    "annotations",
    "evaluation_only",
}


def track_candidate_input(
    payload: Mapping[str, object],
    *,
    policy: CandidateTrackingPolicy | None = None,
) -> CandidateTrackingRun:
    """Run candidate tracking from a truth-free detector proposal artifact."""

    if payload.get("schema_version") != 1:
        raise ValueError("candidate input must use schema_version 1")
    _text(payload.get("episode_id"), "episode_id")
    leaked = sorted(_find_keys(payload, _FORBIDDEN_KEYS))
    if leaked:
        raise ValueError(f"candidate input contains evaluation truth fields: {leaked}")
    frame_rows = payload.get("frames")
    if not isinstance(frame_rows, list) or not frame_rows:
        raise ValueError("candidate input must contain frames")
    frames: list[CandidateSourceFrame] = []
    proposals: list[RegionProposal] = []
    for row in frame_rows:
        if not isinstance(row, Mapping):
            raise ValueError("candidate input frame must be an object")
        frame_id = _text(row.get("frame_id"), "frame_id")
        frames.append(
            CandidateSourceFrame(
                frame_id,
                _text(row.get("image_url"), "image_url"),
                _number(row.get("captured_at"), "captured_at"),
                _text(row.get("source"), "source"),
            )
        )
        proposal_rows = row.get("proposals")
        if not isinstance(proposal_rows, list):
            raise ValueError("candidate frame proposals must be an array")
        for proposal in proposal_rows:
            if not isinstance(proposal, Mapping):
                raise ValueError("candidate proposal must be an object")
            region = proposal.get("region")
            if not isinstance(region, list):
                raise ValueError("candidate proposal region must be an array")
            external = proposal.get("external_track_id")
            if external is not None:
                external = _text(external, "external_track_id")
            proposals.append(
                RegionProposal(
                    _text(proposal.get("proposal_id"), "proposal_id"),
                    frame_id,
                    tuple(region),
                    _number(proposal.get("confidence"), "proposal confidence"),
                    _text(proposal.get("semantic_type", "unknown"), "semantic_type"),
                    external,
                )
            )
    return CandidateTracker(policy).track(tuple(frames), tuple(proposals))


def build_tracked_vlm_input(
    source_payload: Mapping[str, object],
    run: CandidateTrackingRun,
    *,
    policy: CandidateTrackingPolicy | None = None,
) -> dict[str, object]:
    """Serialize only tracked opaque IDs/boxes for the ordinary VLM boundary."""

    episode_id = _text(source_payload.get("episode_id"), "episode_id")
    selected_policy = policy or CandidateTrackingPolicy()
    frames = []
    for frame in run.visual_frames:
        frames.append(
            {
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
        )
    return {
        "schema_version": 1,
        "episode_id": episode_id,
        "candidate_generation": {
            "protocol": "openprop-candidate-tracking-v1",
            "policy": asdict(selected_policy),
            "frames": len(run.frames),
            "tracks": len(run.track_ids),
            "rejected_proposals": sum(
                len(frame.rejected_proposal_ids) for frame in run.frames
            ),
            "capacity_exceeded_frames": sum(
                frame.capacity_exceeded for frame in run.frames
            ),
        },
        "frames": frames,
    }


def _find_keys(value: object, targets: set[str]) -> set[str]:
    found: set[str] = set()
    if isinstance(value, Mapping):
        for key, item in value.items():
            if isinstance(key, str) and key in targets:
                found.add(key)
            found.update(_find_keys(item, targets))
    elif isinstance(value, list):
        for item in value:
            found.update(_find_keys(item, targets))
    return found


def _text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def _number(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result
