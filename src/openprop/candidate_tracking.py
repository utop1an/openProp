from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Mapping, Sequence

from .models import Entity, QueryFrame
from .visual_pipeline import VisualUpdateOrchestrator, VisualUpdateRun
from .vlm import VisualFrame


@dataclass(frozen=True, slots=True)
class CandidateSourceFrame:
    frame_id: str
    image_url: str
    captured_at: float
    source: str

    def __post_init__(self) -> None:
        if not self.frame_id.strip() or not self.image_url.strip() or not self.source.strip():
            raise ValueError("candidate source frame fields cannot be empty")
        if not math.isfinite(self.captured_at):
            raise ValueError("candidate source frame time must be finite")


@dataclass(frozen=True, slots=True)
class RegionProposal:
    proposal_id: str
    frame_id: str
    region: tuple[float, float, float, float]
    confidence: float
    semantic_type: str = "unknown"
    external_track_id: str | None = None

    def __post_init__(self) -> None:
        if not self.proposal_id.strip() or not self.frame_id.strip():
            raise ValueError("proposal and frame IDs cannot be empty")
        _region(self.region)
        _probability(self.confidence, "proposal confidence")
        if not self.semantic_type.strip():
            raise ValueError("proposal semantic type cannot be empty")
        if self.external_track_id is not None and not self.external_track_id.strip():
            raise ValueError("external_track_id cannot be empty")


@dataclass(frozen=True, slots=True)
class CandidateTrackingPolicy:
    minimum_proposal_confidence: float = 0.25
    minimum_link_score: float = 0.35
    iou_weight: float = 0.4
    external_id_weight: float = 0.6
    external_id_mismatch_veto: bool = True
    max_missed_frames: int = 2
    max_active_tracks: int = 12
    max_proposals_per_frame: int = 12

    def __post_init__(self) -> None:
        _probability(self.minimum_proposal_confidence, "minimum proposal confidence")
        _probability(self.minimum_link_score, "minimum link score")
        for name, value in (
            ("iou_weight", self.iou_weight),
            ("external_id_weight", self.external_id_weight),
        ):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.iou_weight + self.external_id_weight <= 0.0:
            raise ValueError("candidate tracking requires positive evidence weight")
        if self.max_missed_frames < 0:
            raise ValueError("max_missed_frames must be nonnegative")
        if self.max_active_tracks <= 0 or self.max_proposals_per_frame <= 0:
            raise ValueError("candidate tracking capacity must be positive")


@dataclass(frozen=True, slots=True)
class TrackedCandidate:
    proposal_id: str
    track_id: str
    region: tuple[float, float, float, float]
    confidence: float
    semantic_type: str
    external_track_id: str | None
    link_score: float
    status: str


@dataclass(frozen=True, slots=True)
class CandidateTrackingFrame:
    source_frame: CandidateSourceFrame
    candidates: tuple[TrackedCandidate, ...]
    rejected_proposal_ids: tuple[str, ...] = ()
    expired_track_ids: tuple[str, ...] = ()
    capacity_exceeded: bool = False

    def visual_frame(self) -> VisualFrame:
        ordered = tuple(sorted(self.candidates, key=lambda item: item.track_id))
        return VisualFrame(
            self.source_frame.frame_id,
            self.source_frame.image_url,
            self.source_frame.captured_at,
            self.source_frame.source,
            tuple(item.track_id for item in ordered),
            {item.track_id: item.region for item in ordered},
        )


@dataclass(frozen=True, slots=True)
class CandidateTrackingRun:
    frames: tuple[CandidateTrackingFrame, ...]

    @property
    def visual_frames(self) -> tuple[VisualFrame, ...]:
        return tuple(frame.visual_frame() for frame in self.frames)

    @property
    def track_ids(self) -> tuple[str, ...]:
        return tuple(
            sorted({item.track_id for frame in self.frames for item in frame.candidates})
        )


@dataclass(frozen=True, slots=True)
class CandidateAwareVisualRun:
    tracking: CandidateTrackingRun
    visual: VisualUpdateRun
    created_entity_ids: tuple[str, ...]


@dataclass(slots=True)
class _TrackState:
    track_id: str
    semantic_type: str
    region: tuple[float, float, float, float]
    external_track_id: str | None
    last_frame_index: int


class CandidateTracker:
    """Deterministic query-independent proposal linking with explicit new-track null."""

    def __init__(self, policy: CandidateTrackingPolicy | None = None) -> None:
        self.policy = policy or CandidateTrackingPolicy()

    def track(
        self,
        frames: Sequence[CandidateSourceFrame],
        proposals: Sequence[RegionProposal],
    ) -> CandidateTrackingRun:
        if not frames:
            raise ValueError("candidate tracking requires frames")
        ordered = sorted(enumerate(frames), key=lambda item: (item[1].captured_at, item[0]))
        if len({frame.frame_id for frame in frames}) != len(frames):
            raise ValueError("candidate source frame IDs must be unique")
        grouped = {frame.frame_id: [] for frame in frames}
        seen_proposals: set[str] = set()
        for proposal in proposals:
            if proposal.proposal_id in seen_proposals:
                raise ValueError("candidate proposal IDs must be globally unique")
            seen_proposals.add(proposal.proposal_id)
            if proposal.frame_id not in grouped:
                raise ValueError("candidate proposal references an unknown frame")
            grouped[proposal.frame_id].append(proposal)

        states: dict[str, _TrackState] = {}
        next_track = 1
        output: list[CandidateTrackingFrame] = []
        for frame_index, (_, frame) in enumerate(ordered):
            expired = tuple(
                sorted(
                    track_id
                    for track_id, state in states.items()
                    if frame_index - state.last_frame_index > self.policy.max_missed_frames + 1
                )
            )
            for track_id in expired:
                del states[track_id]
            frame_proposals = sorted(
                grouped[frame.frame_id], key=lambda item: (-item.confidence, item.proposal_id)
            )
            rejected = [
                item.proposal_id
                for item in frame_proposals
                if item.confidence < self.policy.minimum_proposal_confidence
            ]
            eligible = [
                item for item in frame_proposals
                if item.confidence >= self.policy.minimum_proposal_confidence
            ]
            capacity = len(eligible) > self.policy.max_proposals_per_frame
            if capacity:
                rejected.extend(
                    item.proposal_id
                    for item in eligible[self.policy.max_proposals_per_frame :]
                )
                eligible = eligible[: self.policy.max_proposals_per_frame]
            active = sorted(
                states.values(),
                key=lambda item: (-item.last_frame_index, item.track_id),
            )
            if len(active) > self.policy.max_active_tracks:
                capacity = True
                active = active[: self.policy.max_active_tracks]
            active.sort(key=lambda item: item.track_id)
            scores = tuple(
                tuple(self._link_score(proposal, state) for state in active)
                for proposal in eligible
            )
            assignment = self._assign(scores)
            tracked: list[TrackedCandidate] = []
            for proposal, assigned in zip(eligible, assignment):
                score = 0.0
                if assigned == 0:
                    track_id = f"track-{next_track:06d}"
                    next_track += 1
                    status = "new"
                else:
                    state = active[assigned - 1]
                    track_id = state.track_id
                    score = scores[len(tracked)][assigned - 1]
                    status = "linked"
                states[track_id] = _TrackState(
                    track_id,
                    proposal.semantic_type,
                    proposal.region,
                    proposal.external_track_id,
                    frame_index,
                )
                tracked.append(
                    TrackedCandidate(
                        proposal.proposal_id,
                        track_id,
                        proposal.region,
                        proposal.confidence,
                        proposal.semantic_type,
                        proposal.external_track_id,
                        score,
                        status,
                    )
                )
            output.append(
                CandidateTrackingFrame(
                    frame,
                    tuple(tracked),
                    tuple(sorted(rejected)),
                    expired,
                    capacity,
                )
            )
        return CandidateTrackingRun(tuple(output))

    def _link_score(self, proposal: RegionProposal, state: _TrackState) -> float:
        if (
            proposal.semantic_type != "unknown"
            and state.semantic_type != "unknown"
            and proposal.semantic_type != state.semantic_type
        ):
            return 0.0
        external_available = (
            proposal.external_track_id is not None
            and state.external_track_id is not None
        )
        if (
            external_available
            and proposal.external_track_id != state.external_track_id
            and self.policy.external_id_mismatch_veto
        ):
            return 0.0
        overlap = _iou(proposal.region, state.region)
        if not external_available:
            return overlap
        external = float(proposal.external_track_id == state.external_track_id)
        weight = self.policy.iou_weight + self.policy.external_id_weight
        return (
            self.policy.iou_weight * overlap
            + self.policy.external_id_weight * external
        ) / weight

    def _assign(self, scores: Sequence[Sequence[float]]) -> tuple[int, ...]:
        if not scores:
            return ()
        track_count = len(scores[0]) if scores else 0
        if any(len(row) != track_count for row in scores):
            raise ValueError("candidate link score rows must align")

        @lru_cache(maxsize=None)
        def solve(index: int, used: int) -> tuple[float, tuple[int, ...]]:
            if index == len(scores):
                return 0.0, ()
            tail_score, tail = solve(index + 1, used)
            choices = [(self.policy.minimum_link_score + tail_score, (0,) + tail)]
            for track_index, score in enumerate(scores[index], start=1):
                bit = 1 << (track_index - 1)
                if used & bit or score < self.policy.minimum_link_score:
                    continue
                tail_score, tail = solve(index + 1, used | bit)
                choices.append((score + tail_score, (track_index,) + tail))
            return max(choices, key=lambda item: (item[0], tuple(-x for x in item[1])))

        return solve(0, 0)[1]


class CandidateAwareVisualOrchestrator:
    """Place candidate generation and track birth before ordinary VLM updates."""

    def __init__(
        self,
        tracker: CandidateTracker,
        visual: VisualUpdateOrchestrator,
    ) -> None:
        self.tracker = tracker
        self.visual = visual

    def run(
        self,
        query: QueryFrame,
        frames: Sequence[CandidateSourceFrame],
        proposals: Sequence[RegionProposal],
    ) -> CandidateAwareVisualRun:
        tracking = self.tracker.track(frames, proposals)
        created = self.visual.state.ensure_entities(tracking.track_ids)
        visual_run = self.visual.run(query, tracking.visual_frames)
        return CandidateAwareVisualRun(tracking, visual_run, created)

    def replay(
        self,
        query: QueryFrame,
        frames: Sequence[CandidateSourceFrame],
        proposals: Sequence[RegionProposal],
        captured_response: Mapping[str, object],
    ) -> CandidateAwareVisualRun:
        tracking = self.tracker.track(frames, proposals)
        created = self.visual.state.ensure_entities(tracking.track_ids)
        visual_run = self.visual.replay(query, tracking.visual_frames, captured_response)
        return CandidateAwareVisualRun(tracking, visual_run, created)


def _region(region: Sequence[float]) -> None:
    if len(region) != 4 or any(not math.isfinite(value) for value in region):
        raise ValueError("candidate region must contain four finite coordinates")
    if not (0.0 <= region[0] < region[2] <= 1.0):
        raise ValueError("candidate region x coordinates are invalid")
    if not (0.0 <= region[1] < region[3] <= 1.0):
        raise ValueError("candidate region y coordinates are invalid")


def _iou(left: Sequence[float], right: Sequence[float]) -> float:
    _region(left)
    _region(right)
    width = max(0.0, min(left[2], right[2]) - max(left[0], right[0]))
    height = max(0.0, min(left[3], right[3]) - max(left[1], right[1]))
    intersection = width * height
    left_area = (left[2] - left[0]) * (left[3] - left[1])
    right_area = (right[2] - right[0]) * (right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union > 0.0 else 0.0


def _probability(value: float, name: str) -> None:
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1]")
