from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Mapping, Sequence

from .association import EntityAssociationHypothesis, MultiEntityAssociator
from .visual_evaluation import (
    NULL_ENTITY,
    VisualAssociationResult,
    VisualPropertyResult,
)


@dataclass(frozen=True, slots=True)
class VisualDetectionTruth:
    event_id: str
    property_name: str
    gold_value: object
    target_entity_id: str
    region: tuple[float, float, float, float]

    def __post_init__(self) -> None:
        for name, value in (
            ("event_id", self.event_id),
            ("property_name", self.property_name),
            ("target_entity_id", self.target_entity_id),
        ):
            if not value.strip():
                raise ValueError(f"{name} cannot be empty")
        _region(self.region)


@dataclass(frozen=True, slots=True)
class VisualFrameEvaluationTruth:
    cluster_id: str
    split: str
    frame_id: str
    source: str
    condition: str
    distractor_count: int
    events: tuple[VisualDetectionTruth, ...]
    malformed_response: bool = False

    def __post_init__(self) -> None:
        for name, value in (
            ("cluster_id", self.cluster_id),
            ("frame_id", self.frame_id),
            ("source", self.source),
            ("condition", self.condition),
        ):
            if not value.strip():
                raise ValueError(f"{name} cannot be empty")
        if self.split not in ("development", "calibration", "test"):
            raise ValueError("unknown frame evaluation split")
        if self.distractor_count < 0:
            raise ValueError("distractor_count must be nonnegative")
        event_ids = [event.event_id for event in self.events]
        if len(event_ids) != len(set(event_ids)):
            raise ValueError("frame truth contains duplicate event IDs")


@dataclass(frozen=True, slots=True)
class VisualDetectionEvaluationBatch:
    properties: tuple[VisualPropertyResult, ...]
    associations: tuple[VisualAssociationResult, ...]


def build_visual_detection_results(
    frame_truths: Sequence[VisualFrameEvaluationTruth],
    hypotheses: Sequence[EntityAssociationHypothesis],
    *,
    associator: MultiEntityAssociator,
    system: str,
    iou_threshold: float = 0.5,
    max_detections_per_group: int = 12,
) -> VisualDetectionEvaluationBatch:
    """Match predicted localization to evaluation-only truth, then emit records."""

    if not system.strip():
        raise ValueError("system cannot be empty")
    if not math.isfinite(iou_threshold) or not 0.0 <= iou_threshold <= 1.0:
        raise ValueError("iou_threshold must be finite and in [0, 1]")
    if max_detections_per_group <= 0:
        raise ValueError("max_detections_per_group must be positive")
    contexts = {item.frame_id: item for item in frame_truths}
    if len(contexts) != len(frame_truths) or not contexts:
        raise ValueError("frame evaluation truth IDs must be nonempty and unique")
    detection_ids = [item.detection.detection_id for item in hypotheses]
    if len(detection_ids) != len(set(detection_ids)):
        raise ValueError("hypotheses contain duplicate detection IDs")
    for hypothesis in hypotheses:
        context = contexts.get(hypothesis.detection.frame.frame_id)
        if context is None:
            raise ValueError("hypothesis frame is absent from evaluation truth")
        if hypothesis.detection.frame.source != context.source:
            raise ValueError("hypothesis source differs from evaluation truth")

    property_rows: list[VisualPropertyResult] = []
    association_rows: list[VisualAssociationResult] = []
    hypotheses_by_frame: dict[str, list[EntityAssociationHypothesis]] = {
        frame_id: [] for frame_id in contexts
    }
    for hypothesis in hypotheses:
        hypotheses_by_frame[hypothesis.detection.frame.frame_id].append(hypothesis)

    for frame_id in sorted(contexts):
        context = contexts[frame_id]
        predictions = hypotheses_by_frame[frame_id]
        property_names = sorted(
            {
                *(event.property_name for event in context.events),
                *(item.detection.property_name for item in predictions),
            }
        )
        for property_name in property_names:
            truths = tuple(
                event for event in context.events if event.property_name == property_name
            )
            predicted = tuple(
                sorted(
                    (
                        item
                        for item in predictions
                        if item.detection.property_name == property_name
                    ),
                    key=lambda item: item.detection.detection_id,
                )
            )
            if len(predicted) > max_detections_per_group:
                raise ValueError("detection group exceeds exact matching limit")
            matches = _maximum_iou_matching(truths, predicted, iou_threshold)
            matched_truth = {left for left, _ in matches}
            matched_prediction = {right for _, right in matches}
            for truth_index, prediction_index in matches:
                truth = truths[truth_index]
                hypothesis = predicted[prediction_index]
                property_rows.append(
                    _property_row(
                        context,
                        system,
                        truth.event_id,
                        property_name,
                        expected=True,
                        detected=True,
                        gold_value=truth.gold_value,
                        predicted_value=hypothesis.detection.value,
                        confidence=hypothesis.detection.value_confidence,
                        malformed=context.malformed_response,
                    )
                )
                association_rows.append(
                    _association_row(
                        context,
                        system,
                        hypothesis,
                        associator,
                        record_id=truth.event_id,
                        target_entity_id=truth.target_entity_id,
                    )
                )
            for truth_index, truth in enumerate(truths):
                if truth_index in matched_truth:
                    continue
                property_rows.append(
                    _property_row(
                        context,
                        system,
                        truth.event_id,
                        property_name,
                        expected=True,
                        detected=False,
                        gold_value=truth.gold_value,
                        predicted_value=None,
                        confidence=0.0,
                        malformed=context.malformed_response,
                    )
                )
            for prediction_index, hypothesis in enumerate(predicted):
                if prediction_index in matched_prediction:
                    continue
                duplicate = int(
                    hypothesis.detection.region is not None
                    and any(
                        _iou(hypothesis.detection.region, truth.region) >= iou_threshold
                        for truth in truths
                    )
                )
                record_id = (
                    f"false-positive:{frame_id}:{property_name}:"
                    f"{hypothesis.detection.detection_id}"
                )
                property_rows.append(
                    _property_row(
                        context,
                        system,
                        record_id,
                        property_name,
                        expected=False,
                        detected=True,
                        gold_value=None,
                        predicted_value=hypothesis.detection.value,
                        confidence=hypothesis.detection.value_confidence,
                        duplicate_count=duplicate,
                        malformed=context.malformed_response,
                    )
                )
                association_rows.append(
                    _association_row(
                        context,
                        system,
                        hypothesis,
                        associator,
                        record_id=record_id,
                        target_entity_id=None,
                    )
                )
    return VisualDetectionEvaluationBatch(
        tuple(property_rows),
        tuple(association_rows),
    )


def _maximum_iou_matching(
    truths: Sequence[VisualDetectionTruth],
    hypotheses: Sequence[EntityAssociationHypothesis],
    threshold: float,
) -> tuple[tuple[int, int], ...]:
    @lru_cache(maxsize=None)
    def solve(truth_index: int, used_mask: int) -> tuple[int, float, tuple[tuple[int, int], ...]]:
        if truth_index == len(truths):
            return 0, 0.0, ()
        best = solve(truth_index + 1, used_mask)
        for prediction_index, hypothesis in enumerate(hypotheses):
            bit = 1 << prediction_index
            region = hypothesis.detection.region
            if used_mask & bit or region is None:
                continue
            overlap = _iou(truths[truth_index].region, region)
            if overlap < threshold:
                continue
            count, total, pairs = solve(truth_index + 1, used_mask | bit)
            candidate = (
                count + 1,
                total + overlap,
                ((truth_index, prediction_index),) + pairs,
            )
            if (candidate[0], candidate[1], _tie_key(candidate[2])) > (
                best[0],
                best[1],
                _tie_key(best[2]),
            ):
                best = candidate
        return best

    return tuple(sorted(solve(0, 0)[2]))


def _tie_key(pairs: Sequence[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    return tuple((-left, -right) for left, right in pairs)


def _property_row(
    context: VisualFrameEvaluationTruth,
    system: str,
    record_id: str,
    property_name: str,
    *,
    expected: bool,
    detected: bool,
    gold_value: object | None,
    predicted_value: object | None,
    confidence: float,
    duplicate_count: int = 0,
    malformed: bool,
) -> VisualPropertyResult:
    return VisualPropertyResult(
        record_id,
        context.cluster_id,
        context.split,
        system,
        context.source,
        property_name,
        expected,
        detected,
        gold_value,
        predicted_value,
        confidence,
        duplicate_count,
        malformed,
    )


def _association_row(
    context: VisualFrameEvaluationTruth,
    system: str,
    hypothesis: EntityAssociationHypothesis,
    associator: MultiEntityAssociator,
    *,
    record_id: str,
    target_entity_id: str | None,
) -> VisualAssociationResult:
    detection = hypothesis.detection
    probabilities: dict[str, float] = {
        item.entity_id: item.posterior for item in hypothesis.candidates
    }
    probabilities[NULL_ENTITY] = hypothesis.null_probability
    definition = associator.registry.resolve(detection.property_name).definition
    assert definition is not None
    upstream_eligible = (
        not context.malformed_response
        and detection.detection_confidence
        >= associator.policy.minimum_detection_confidence
        and detection.value_confidence >= associator.policy.minimum_value_confidence
        and definition.update_policy.allow_visual_updates
        and definition.update_policy.permits_source(detection.frame.source)
    )
    return VisualAssociationResult(
        record_id,
        context.cluster_id,
        context.split,
        system,
        context.source,
        detection.property_name,
        detection.detection_id,
        detection.frame.frame_id,
        detection.frame.candidate_entity_ids,
        target_entity_id,
        hypothesis.decision_entity_id,
        hypothesis.accepted_entity_id,
        probabilities,
        context.condition,
        context.distractor_count,
        context.malformed_response,
        hypothesis.reason,
        upstream_eligible,
        (
            detection.detection_confidence
            * detection.value_confidence
            * associator.policy.reliability_for(detection.frame.source)
        ),
        definition.update_policy.minimum_confidence,
    )


def _iou(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> float:
    _region(left)
    _region(right)
    intersection_width = max(0.0, min(left[2], right[2]) - max(left[0], right[0]))
    intersection_height = max(0.0, min(left[3], right[3]) - max(left[1], right[1]))
    intersection = intersection_width * intersection_height
    left_area = (left[2] - left[0]) * (left[3] - left[1])
    right_area = (right[2] - right[0]) * (right[3] - right[1])
    return intersection / (left_area + right_area - intersection)


def _region(region: Sequence[float]) -> None:
    if len(region) != 4 or any(not math.isfinite(value) for value in region):
        raise ValueError("evaluation region must contain four finite coordinates")
    if not (0.0 <= region[0] < region[2] <= 1.0):
        raise ValueError("evaluation region x coordinates are invalid")
    if not (0.0 <= region[1] < region[3] <= 1.0):
        raise ValueError("evaluation region y coordinates are invalid")
