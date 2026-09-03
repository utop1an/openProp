from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from .candidate_evaluation import (
    CandidateFrameTruth,
    CandidateTrackingEvaluation,
    aggregate_candidate_tracking,
    evaluate_candidate_tracking,
    CandidateTruthObject,
)
from .candidate_replay import track_candidate_input
from .candidate_tracking import CandidateTrackingPolicy


@dataclass(frozen=True, slots=True)
class CandidateCalibrationCase:
    input_payload: Mapping[str, object]
    truth: tuple[CandidateFrameTruth, ...]
    cluster_id: str
    split: str
    source: str
    query_frame_id: str
    query_target_entity_id: str | None

    def __post_init__(self) -> None:
        if not self.cluster_id.strip() or not self.source.strip():
            raise ValueError("candidate calibration case identity cannot be empty")
        if self.split not in {"development", "calibration", "test"}:
            raise ValueError("candidate calibration case split is invalid")
        if not self.query_frame_id.strip():
            raise ValueError("candidate calibration query frame cannot be empty")


@dataclass(frozen=True, slots=True)
class FrozenCandidateTrackingPolicy:
    policy: CandidateTrackingPolicy
    calibration_cases: int
    calibration_frames: int
    calibration_truth_objects: int
    candidate_recall: float
    candidate_precision: float
    query_target_recall: float | None
    identity_switch_rate: float
    fragmentations: int
    minimum_candidate_recall: float
    maximum_identity_switch_rate: float
    searched_policies: int
    feasible_policies: int

    def __post_init__(self) -> None:
        if self.calibration_cases <= 0 or self.calibration_frames <= 0:
            raise ValueError("candidate calibration populations must be positive")
        for name, value in (
            ("candidate_recall", self.candidate_recall),
            ("candidate_precision", self.candidate_precision),
            ("minimum_candidate_recall", self.minimum_candidate_recall),
            ("maximum_identity_switch_rate", self.maximum_identity_switch_rate),
        ):
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and in [0, 1]")
        if self.query_target_recall is not None and not 0.0 <= self.query_target_recall <= 1.0:
            raise ValueError("query_target_recall must be in [0, 1]")
        if not math.isfinite(self.identity_switch_rate) or self.identity_switch_rate < 0.0:
            raise ValueError("identity_switch_rate must be finite and nonnegative")
        if self.fragmentations < 0 or self.searched_policies <= 0 or self.feasible_policies <= 0:
            raise ValueError("candidate calibration counts are invalid")


def calibrate_candidate_tracking_policy(
    cases: Sequence[CandidateCalibrationCase],
    *,
    minimum_proposal_confidences: Sequence[float],
    minimum_link_scores: Sequence[float],
    max_missed_frames: Sequence[int],
    minimum_candidate_recall: float,
    maximum_identity_switch_rate: float,
    base_policy: CandidateTrackingPolicy | None = None,
) -> FrozenCandidateTrackingPolicy:
    """Select proposal/link gates on calibration episodes only."""

    if not cases:
        raise ValueError("candidate calibration cases cannot be empty")
    if any(case.split != "calibration" for case in cases):
        raise ValueError("candidate tracking policy can only use calibration cases")
    episode_ids = [case.input_payload.get("episode_id") for case in cases]
    if any(not isinstance(value, str) for value in episode_ids) or len(set(episode_ids)) != len(cases):
        raise ValueError("candidate calibration cases require unique episode IDs")
    _probability(minimum_candidate_recall, "minimum_candidate_recall")
    _probability(maximum_identity_switch_rate, "maximum_identity_switch_rate")
    confidences = tuple(sorted({float(value) for value in minimum_proposal_confidences}))
    links = tuple(sorted({float(value) for value in minimum_link_scores}))
    gaps = tuple(sorted({int(value) for value in max_missed_frames}))
    if not confidences or not links or not gaps:
        raise ValueError("candidate calibration grids cannot be empty")
    for value in (*confidences, *links):
        _probability(value, "candidate calibration grid value")
    if any(value < 0 for value in gaps):
        raise ValueError("candidate max_missed_frames must be nonnegative")
    template = base_policy or CandidateTrackingPolicy()
    candidates: list[
        tuple[CandidateTrackingPolicy, dict[str, object], tuple[CandidateTrackingEvaluation, ...]]
    ] = []
    searched = 0
    for confidence in confidences:
        for link in links:
            for gap in gaps:
                searched += 1
                policy = CandidateTrackingPolicy(
                    minimum_proposal_confidence=confidence,
                    minimum_link_score=link,
                    iou_weight=template.iou_weight,
                    external_id_weight=template.external_id_weight,
                    external_id_mismatch_veto=template.external_id_mismatch_veto,
                    max_missed_frames=gap,
                    max_active_tracks=template.max_active_tracks,
                    max_proposals_per_frame=template.max_proposals_per_frame,
                )
                evaluations = tuple(_evaluate_case(case, policy) for case in cases)
                report = aggregate_candidate_tracking(evaluations, split="calibration")
                recall = float(report["candidate_recall"] or 0.0)
                matched = int(report["matched"])
                switch_rate = int(report["identity_switches"]) / max(1, matched)
                if (
                    recall >= minimum_candidate_recall
                    and switch_rate <= maximum_identity_switch_rate
                ):
                    candidates.append((policy, report, evaluations))
    if not candidates:
        raise ValueError("no candidate tracking policy satisfies calibration gates")
    selected_policy, report, evaluations = max(
        candidates,
        key=lambda item: (
            float(item[1]["query_target_recall"] or 0.0),
            float(item[1]["candidate_recall"] or 0.0),
            -int(item[1]["identity_switches"]),
            -int(item[1]["fragmentations"]),
            float(item[1]["track_purity"] or 0.0),
            float(item[1]["candidate_precision"] or 0.0),
            item[0].minimum_proposal_confidence,
            item[0].minimum_link_score,
            -item[0].max_missed_frames,
        ),
    )
    matched = int(report["matched"])
    return FrozenCandidateTrackingPolicy(
        selected_policy,
        len(cases),
        int(report["frames"]),
        int(report["truth_objects"]),
        float(report["candidate_recall"] or 0.0),
        float(report["candidate_precision"] or 0.0),
        (
            None
            if report["query_target_recall"] is None
            else float(report["query_target_recall"])
        ),
        int(report["identity_switches"]) / max(1, matched),
        int(report["fragmentations"]),
        minimum_candidate_recall,
        maximum_identity_switch_rate,
        searched,
        len(candidates),
    )


def candidate_calibration_case_from_payloads(
    input_payload: Mapping[str, object],
    truth_payload: Mapping[str, object],
) -> CandidateCalibrationCase:
    if truth_payload.get("schema_version") != 1 or truth_payload.get("evaluation_only") is not True:
        raise ValueError("candidate calibration truth must be schema-version-1 evaluation-only")
    if truth_payload.get("episode_id") != input_payload.get("episode_id"):
        raise ValueError("candidate calibration truth episode does not match input")
    raw_frames = truth_payload.get("frames")
    if not isinstance(raw_frames, list):
        raise ValueError("candidate calibration truth frames must be an array")
    frames = []
    for row in raw_frames:
        if not isinstance(row, Mapping) or not isinstance(row.get("objects"), list):
            raise ValueError("candidate calibration truth frame is malformed")
        objects = []
        for item in row["objects"]:
            if not isinstance(item, Mapping) or not isinstance(item.get("region"), list):
                raise ValueError("candidate calibration truth object is malformed")
            objects.append(
                CandidateTruthObject(
                    _text(item.get("entity_id"), "truth entity_id"),
                    tuple(item["region"]),
                )
            )
        frames.append(
            CandidateFrameTruth(
                _text(row.get("frame_id"), "truth frame_id"), tuple(objects)
            )
        )
    query = truth_payload.get("query")
    if not isinstance(query, Mapping):
        raise ValueError("candidate calibration truth query must be an object")
    target = query.get("target_entity_id")
    if target is not None:
        target = _text(target, "query target_entity_id")
    return CandidateCalibrationCase(
        input_payload,
        tuple(frames),
        _text(truth_payload.get("cluster_id"), "cluster_id"),
        _text(truth_payload.get("split"), "split"),
        _text(truth_payload.get("source"), "source"),
        _text(query.get("frame_id"), "query frame_id"),
        target,
    )


def candidate_tracking_policy_from_frozen_payload(
    payload: Mapping[str, object],
) -> CandidateTrackingPolicy:
    if payload.get("schema_version") != 1:
        raise ValueError("frozen candidate policy must use schema_version 1")
    if payload.get("calibration_only_selection") is not True:
        raise ValueError("candidate policy must be selected on calibration only")
    if payload.get("test_truth_used_for_selection") is not False:
        raise ValueError("candidate policy must declare no test-truth selection")
    frozen = payload.get("frozen")
    if not isinstance(frozen, Mapping) or not isinstance(frozen.get("policy"), Mapping):
        raise ValueError("frozen candidate policy payload is malformed")
    try:
        return CandidateTrackingPolicy(**dict(frozen["policy"]))
    except (TypeError, ValueError) as error:
        raise ValueError(f"frozen candidate tracking policy is invalid: {error}") from error


def _evaluate_case(
    case: CandidateCalibrationCase,
    policy: CandidateTrackingPolicy,
) -> CandidateTrackingEvaluation:
    run = track_candidate_input(case.input_payload, policy=policy)
    return evaluate_candidate_tracking(
        run,
        case.truth,
        cluster_id=case.cluster_id,
        record_id=str(case.input_payload["episode_id"]),
        split=case.split,
        system="candidate-calibration",
        source=case.source,
        query_frame_id=case.query_frame_id,
        query_target_entity_id=case.query_target_entity_id,
    )


def _probability(value: float, name: str) -> None:
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1]")


def _text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()
