from __future__ import annotations

import math
import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from typing import Mapping, Sequence

from .candidate_tracking import CandidateTrackingRun, TrackedCandidate


@dataclass(frozen=True, slots=True)
class CandidateTruthObject:
    entity_id: str
    region: tuple[float, float, float, float]

    def __post_init__(self) -> None:
        if not self.entity_id.strip():
            raise ValueError("candidate truth entity_id cannot be empty")
        _region(self.region)


@dataclass(frozen=True, slots=True)
class CandidateFrameTruth:
    frame_id: str
    objects: tuple[CandidateTruthObject, ...]

    def __post_init__(self) -> None:
        if not self.frame_id.strip():
            raise ValueError("candidate truth frame_id cannot be empty")
        identifiers = [item.entity_id for item in self.objects]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("candidate truth entity IDs must be unique per frame")


@dataclass(frozen=True, slots=True)
class CandidateFrameMetrics:
    frame_id: str
    truth_objects: int
    candidates: int
    matched: int
    misses: int
    false_positives: int
    rejected_proposals: int
    capacity_exceeded: bool
    matched_truth_tracks: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        if not self.frame_id.strip():
            raise ValueError("candidate metric frame_id cannot be empty")
        counts = (
            self.truth_objects,
            self.candidates,
            self.matched,
            self.misses,
            self.false_positives,
            self.rejected_proposals,
        )
        if any(value < 0 for value in counts):
            raise ValueError("candidate frame metric counts must be nonnegative")
        if self.matched + self.misses != self.truth_objects:
            raise ValueError("candidate frame truth denominator is inconsistent")
        if self.matched + self.false_positives != self.candidates:
            raise ValueError("candidate frame prediction denominator is inconsistent")
        if len(self.matched_truth_tracks) != self.matched:
            raise ValueError("candidate frame matched identities are inconsistent")
        truths = [item[0] for item in self.matched_truth_tracks]
        tracks = [item[1] for item in self.matched_truth_tracks]
        if len(truths) != len(set(truths)) or len(tracks) != len(set(tracks)):
            raise ValueError("candidate frame matching must be one-to-one")


@dataclass(frozen=True, slots=True)
class CandidateTrackingEvaluation:
    cluster_id: str
    record_id: str
    split: str
    system: str
    source: str
    truth_population_sha256: str
    query_frame_id: str
    query_target_entity_id: str | None
    iou_threshold: float
    frames: tuple[CandidateFrameMetrics, ...]
    identity_switches: int
    fragmentations: int
    purity_correct: int
    purity_total: int
    query_target_trials: int
    query_target_hits: int

    def __post_init__(self) -> None:
        if (
            not self.cluster_id.strip()
            or not self.record_id.strip()
            or not self.system.strip()
            or not self.source.strip()
            or not self.query_frame_id.strip()
        ):
            raise ValueError("candidate evaluation identity fields cannot be empty")
        if len(self.truth_population_sha256) != 64 or any(
            value not in "0123456789abcdef" for value in self.truth_population_sha256
        ):
            raise ValueError("candidate truth population hash is invalid")
        if self.split not in {"development", "calibration", "test"}:
            raise ValueError("candidate evaluation split is invalid")
        if not 0.0 <= self.iou_threshold <= 1.0 or not math.isfinite(
            self.iou_threshold
        ):
            raise ValueError("candidate evaluation IoU threshold is invalid")
        if not self.frames or len({item.frame_id for item in self.frames}) != len(self.frames):
            raise ValueError("candidate evaluation frames must be non-empty and unique")
        if self.query_frame_id not in {item.frame_id for item in self.frames}:
            raise ValueError("candidate query frame is absent from evaluation frames")
        counts = (
            self.identity_switches,
            self.fragmentations,
            self.purity_correct,
            self.purity_total,
            self.query_target_trials,
            self.query_target_hits,
        )
        if any(value < 0 for value in counts):
            raise ValueError("candidate evaluation counts must be nonnegative")
        if self.purity_correct > self.purity_total:
            raise ValueError("candidate purity counts are inconsistent")
        if self.query_target_hits > self.query_target_trials:
            raise ValueError("candidate query-target counts are inconsistent")
        if self.query_target_trials != int(self.query_target_entity_id is not None):
            raise ValueError("candidate query-target denominator is inconsistent")

    @property
    def truth_objects(self) -> int:
        return sum(frame.truth_objects for frame in self.frames)

    @property
    def candidates(self) -> int:
        return sum(frame.candidates for frame in self.frames)

    @property
    def matched(self) -> int:
        return sum(frame.matched for frame in self.frames)


def evaluate_candidate_tracking(
    run: CandidateTrackingRun,
    truth: Sequence[CandidateFrameTruth],
    *,
    cluster_id: str,
    record_id: str,
    split: str,
    system: str,
    source: str,
    query_frame_id: str,
    query_target_entity_id: str | None,
    iou_threshold: float = 0.5,
    max_candidates_per_frame: int = 16,
) -> CandidateTrackingEvaluation:
    """Attach identity truth after tracking and retain every frame denominator."""

    if not cluster_id.strip() or not record_id.strip() or not system.strip() or not source.strip():
        raise ValueError("candidate evaluation cluster/system/source cannot be empty")
    if split not in {"development", "calibration", "test"}:
        raise ValueError("candidate evaluation split is invalid")
    if not 0.0 <= iou_threshold <= 1.0 or not math.isfinite(iou_threshold):
        raise ValueError("candidate evaluation IoU threshold is invalid")
    if max_candidates_per_frame <= 0:
        raise ValueError("max_candidates_per_frame must be positive")
    truth_index = {frame.frame_id: frame for frame in truth}
    if len(truth_index) != len(truth):
        raise ValueError("candidate truth frame IDs must be unique")
    run_ids = {frame.source_frame.frame_id for frame in run.frames}
    if set(truth_index) != run_ids:
        raise ValueError("candidate truth must cover every tracking frame exactly")
    if query_frame_id not in run_ids:
        raise ValueError("query frame is absent from candidate tracking run")
    query_truth_ids = {
        item.entity_id for item in truth_index[query_frame_id].objects
    }
    if (
        query_target_entity_id is not None
        and query_target_entity_id not in query_truth_ids
    ):
        raise ValueError("query target is absent from query-frame truth")

    frame_metrics: list[CandidateFrameMetrics] = []
    histories: dict[str, list[str | None]] = {}
    track_truth_counts: dict[str, Counter[str]] = {}
    query_hit = 0
    query_trials = int(query_target_entity_id is not None)
    for tracked_frame in run.frames:
        frame_id = tracked_frame.source_frame.frame_id
        frame_truth = truth_index[frame_id]
        candidates = tracked_frame.candidates
        if len(candidates) > max_candidates_per_frame:
            raise ValueError("candidate evaluation exact matching capacity exceeded")
        pairs = _maximum_iou_matching(
            frame_truth.objects, candidates, iou_threshold
        )
        matched_truth = {left for left, _ in pairs}
        mapping = tuple(
            sorted(
                (
                    frame_truth.objects[left].entity_id,
                    candidates[right].track_id,
                )
                for left, right in pairs
            )
        )
        mapping_index = dict(mapping)
        all_truth_ids = {item.entity_id for item in frame_truth.objects}
        for entity_id in set(histories) | all_truth_ids:
            histories.setdefault(entity_id, []).append(mapping_index.get(entity_id))
        for entity_id, track_id in mapping:
            track_truth_counts.setdefault(track_id, Counter())[entity_id] += 1
        if frame_id == query_frame_id and query_target_entity_id is not None:
            query_hit = int(query_target_entity_id in mapping_index)
        frame_metrics.append(
            CandidateFrameMetrics(
                frame_id,
                len(frame_truth.objects),
                len(candidates),
                len(pairs),
                len(frame_truth.objects) - len(matched_truth),
                len(candidates) - len(pairs),
                len(tracked_frame.rejected_proposal_ids),
                tracked_frame.capacity_exceeded,
                mapping,
            )
        )
    switches = 0
    fragments = 0
    for history in histories.values():
        previous_track: str | None = None
        previously_matched = False
        in_gap = False
        for track_id in history:
            if track_id is None:
                if previously_matched:
                    in_gap = True
                continue
            if previous_track is not None and track_id != previous_track:
                switches += 1
            if previously_matched and in_gap:
                fragments += 1
            previous_track = track_id
            previously_matched = True
            in_gap = False
    purity_total = sum(sum(counts.values()) for counts in track_truth_counts.values())
    purity_correct = sum(max(counts.values()) for counts in track_truth_counts.values())
    return CandidateTrackingEvaluation(
        cluster_id,
        record_id,
        split,
        system,
        source,
        _truth_population_hash(truth),
        query_frame_id,
        query_target_entity_id,
        iou_threshold,
        tuple(frame_metrics),
        switches,
        fragments,
        purity_correct,
        purity_total,
        query_trials,
        query_hit,
    )


def aggregate_candidate_tracking(
    evaluations: Sequence[CandidateTrackingEvaluation],
    *,
    split: str,
) -> dict[str, object]:
    selected = tuple(item for item in evaluations if item.split == split)
    if not selected:
        raise ValueError("candidate evaluation split is empty")
    systems = {item.system for item in selected}
    if len(systems) != 1:
        raise ValueError("candidate aggregate requires exactly one system")
    record_ids = [item.record_id for item in selected]
    if len(record_ids) != len(set(record_ids)):
        raise ValueError("candidate aggregate record IDs must be unique per system")
    cluster_splits: dict[str, str] = {}
    for item in evaluations:
        previous = cluster_splits.setdefault(item.cluster_id, item.split)
        if previous != item.split:
            raise ValueError("candidate evaluation cluster leaks across splits")
    truth_objects = sum(item.truth_objects for item in selected)
    candidates = sum(item.candidates for item in selected)
    matched = sum(item.matched for item in selected)
    query_trials = sum(item.query_target_trials for item in selected)
    purity_total = sum(item.purity_total for item in selected)
    frames = tuple(frame for item in selected for frame in item.frames)
    return {
        "schema_version": 1,
        "split": split,
        "system": next(iter(systems)),
        "episodes": len(selected),
        "clusters": len({item.cluster_id for item in selected}),
        "frames": len(frames),
        "truth_objects": truth_objects,
        "candidates": candidates,
        "matched": matched,
        "candidate_recall": _divide(matched, truth_objects),
        "candidate_precision": _divide(matched, candidates),
        "mean_candidates_per_frame": _divide(candidates, len(frames)),
        "misses": truth_objects - matched,
        "false_positives": candidates - matched,
        "rejected_proposals": sum(frame.rejected_proposals for frame in frames),
        "capacity_exceeded_frames": sum(frame.capacity_exceeded for frame in frames),
        "identity_switches": sum(item.identity_switches for item in selected),
        "fragmentations": sum(item.fragmentations for item in selected),
        "track_purity": _divide(
            sum(item.purity_correct for item in selected), purity_total
        ),
        "query_target_trials": query_trials,
        "query_target_recall": _divide(
            sum(item.query_target_hits for item in selected), query_trials
        ),
        "all_frames_retained": True,
        "cluster_disjoint_splits": True,
    }


def aggregate_candidate_tracking_matrix(
    evaluations: Sequence[CandidateTrackingEvaluation],
    *,
    split: str,
) -> dict[str, object]:
    systems = sorted({item.system for item in evaluations if item.split == split})
    if not systems:
        raise ValueError("candidate evaluation matrix split is empty")
    return {
        "schema_version": 1,
        "split": split,
        "systems": {
            system: aggregate_candidate_tracking(
                tuple(item for item in evaluations if item.system == system),
                split=split,
            )
            for system in systems
        },
    }


def candidate_evaluation_from_mapping(
    payload: Mapping[str, object],
) -> CandidateTrackingEvaluation:
    raw_frames = payload.get("frames")
    if not isinstance(raw_frames, list):
        raise ValueError("candidate evaluation frames must be an array")
    frames: list[CandidateFrameMetrics] = []
    for row in raw_frames:
        if not isinstance(row, Mapping):
            raise ValueError("candidate evaluation frame must be an object")
        matches = row.get("matched_truth_tracks")
        if not isinstance(matches, list):
            raise ValueError("candidate matched identities must be an array")
        frames.append(
            CandidateFrameMetrics(
                _text(row.get("frame_id"), "candidate metric frame_id"),
                _integer(row.get("truth_objects"), "truth_objects"),
                _integer(row.get("candidates"), "candidates"),
                _integer(row.get("matched"), "matched"),
                _integer(row.get("misses"), "misses"),
                _integer(row.get("false_positives"), "false_positives"),
                _integer(row.get("rejected_proposals"), "rejected_proposals"),
                _boolean(row.get("capacity_exceeded"), "capacity_exceeded"),
                tuple(
                    (
                        _text(item[0], "matched truth ID"),
                        _text(item[1], "matched track ID"),
                    )
                    for item in matches
                    if isinstance(item, list) and len(item) == 2
                ),
            )
        )
        if len(frames[-1].matched_truth_tracks) != len(matches):
            raise ValueError("candidate matched identity pair is malformed")
    return CandidateTrackingEvaluation(
        _text(payload.get("cluster_id"), "cluster_id"),
        _text(payload.get("record_id"), "record_id"),
        _text(payload.get("split"), "split"),
        _text(payload.get("system"), "system"),
        _text(payload.get("source"), "source"),
        _text(payload.get("truth_population_sha256"), "truth_population_sha256"),
        _text(payload.get("query_frame_id"), "query_frame_id"),
        _optional_text(payload.get("query_target_entity_id"), "query_target_entity_id"),
        _number(payload.get("iou_threshold"), "iou_threshold"),
        tuple(frames),
        _integer(payload.get("identity_switches"), "identity_switches"),
        _integer(payload.get("fragmentations"), "fragmentations"),
        _integer(payload.get("purity_correct"), "purity_correct"),
        _integer(payload.get("purity_total"), "purity_total"),
        _integer(payload.get("query_target_trials"), "query_target_trials"),
        _integer(payload.get("query_target_hits"), "query_target_hits"),
    )


def _maximum_iou_matching(
    truth: Sequence[CandidateTruthObject],
    candidates: Sequence[TrackedCandidate],
    threshold: float,
) -> tuple[tuple[int, int], ...]:
    @lru_cache(maxsize=None)
    def solve(index: int, used: int) -> tuple[int, float, tuple[tuple[int, int], ...]]:
        if index == len(truth):
            return 0, 0.0, ()
        best = solve(index + 1, used)
        for candidate_index, candidate in enumerate(candidates):
            bit = 1 << candidate_index
            if used & bit:
                continue
            overlap = _iou(truth[index].region, candidate.region)
            if overlap < threshold:
                continue
            count, total, pairs = solve(index + 1, used | bit)
            option = (count + 1, total + overlap, ((index, candidate_index),) + pairs)
            if (option[0], option[1], _tie(option[2])) > (
                best[0], best[1], _tie(best[2])
            ):
                best = option
        return best

    return tuple(sorted(solve(0, 0)[2]))


def _tie(pairs: Sequence[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    return tuple((-left, -right) for left, right in pairs)


def _region(region: Sequence[float]) -> None:
    if len(region) != 4 or any(not math.isfinite(value) for value in region):
        raise ValueError("candidate truth region must contain four finite coordinates")
    if not (0.0 <= region[0] < region[2] <= 1.0):
        raise ValueError("candidate truth region x coordinates are invalid")
    if not (0.0 <= region[1] < region[3] <= 1.0):
        raise ValueError("candidate truth region y coordinates are invalid")


def _iou(left: Sequence[float], right: Sequence[float]) -> float:
    width = max(0.0, min(left[2], right[2]) - max(left[0], right[0]))
    height = max(0.0, min(left[3], right[3]) - max(left[1], right[1]))
    intersection = width * height
    left_area = (left[2] - left[0]) * (left[3] - left[1])
    right_area = (right[2] - right[0]) * (right[3] - right[1])
    return intersection / (left_area + right_area - intersection)


def _divide(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _truth_population_hash(truth: Sequence[CandidateFrameTruth]) -> str:
    payload = [
        {
            "frame_id": frame.frame_id,
            "objects": [
                {"entity_id": item.entity_id, "region": list(item.region)}
                for item in sorted(frame.objects, key=lambda row: row.entity_id)
            ],
        }
        for frame in truth
    ]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def _integer(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")
    return value


def _boolean(value: object, field: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field} must be boolean")
    return value


def _optional_text(value: object, field: str) -> str | None:
    if value is None:
        return None
    return _text(value, field)


def _number(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    return float(value)
