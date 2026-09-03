from __future__ import annotations

import math
from collections.abc import Mapping

from .query_decision import build_visual_query_result
from .visual_evaluation import VisualEvaluationDataset
from .visual_replay import VisualReplayOutcome
from .visual_result_builder import (
    VisualDetectionTruth,
    VisualFrameEvaluationTruth,
    build_visual_detection_results,
)


def evaluate_visual_replay(
    outcome: VisualReplayOutcome,
    truth_payload: Mapping[str, object],
    *,
    system: str,
    latency_seconds: float = 0.0,
    vlm_calls: int = 0,
) -> VisualEvaluationDataset:
    """Attach evaluation-only truth after replay decisions are immutable."""

    if truth_payload.get("schema_version") != 1:
        raise ValueError("visual replay truth must use schema_version 1")
    if truth_payload.get("evaluation_only") is not True:
        raise ValueError("visual replay truth must be marked evaluation-only")
    if truth_payload.get("case_id") != outcome.case_id:
        raise ValueError("visual replay truth case does not match outcome")
    cluster_id = _text(truth_payload.get("cluster_id"), "cluster_id")
    split = truth_payload.get("split")
    if split not in {"development", "calibration", "test"}:
        raise ValueError("visual replay truth has invalid split")
    source = _text(truth_payload.get("source"), "source")
    condition = _text(truth_payload.get("condition"), "condition")
    distractor_count = _integer(
        truth_payload.get("distractor_count"), "distractor_count"
    )
    if distractor_count < 0:
        raise ValueError("distractor_count must be nonnegative")
    if "malformed_response" in truth_payload:
        raise ValueError("malformed_response is a model outcome, not a truth field")
    malformed = outcome.malformed_response

    frame_rows = truth_payload.get("frames")
    if not isinstance(frame_rows, list) or not frame_rows:
        raise ValueError("visual replay truth must contain frames")
    frame_truths: list[VisualFrameEvaluationTruth] = []
    for row in frame_rows:
        if not isinstance(row, Mapping):
            raise ValueError("visual replay frame truth must be an object")
        events_raw = row.get("events")
        if not isinstance(events_raw, list):
            raise ValueError("visual replay frame events must be an array")
        events = []
        for event in events_raw:
            if not isinstance(event, Mapping):
                raise ValueError("visual replay event truth must be an object")
            region = event.get("region")
            if not isinstance(region, list):
                raise ValueError("visual replay event region must be an array")
            events.append(
                VisualDetectionTruth(
                    _text(event.get("event_id"), "event_id"),
                    _text(event.get("property_name"), "event property"),
                    event.get("gold_value"),
                    _text(event.get("target_entity_id"), "event target"),
                    tuple(region),
                )
            )
        frame_truths.append(
            VisualFrameEvaluationTruth(
                cluster_id,
                split,
                _text(row.get("frame_id"), "truth frame_id"),
                source,
                condition,
                distractor_count,
                tuple(events),
                malformed,
            )
        )
    replay_frames = {item.frame.frame_id for item in outcome.run.frame_updates}
    truth_frames = {item.frame_id for item in frame_truths}
    if replay_frames != truth_frames:
        raise ValueError("visual replay truth must cover every replay frame exactly")
    batch = build_visual_detection_results(
        frame_truths,
        outcome.run.hypotheses,
        associator=outcome.associator,
        system=system,
    )

    query_raw = truth_payload.get("query")
    if not isinstance(query_raw, Mapping):
        raise ValueError("visual replay query truth must be an object")
    target = query_raw.get("target_entity_id")
    if target is not None:
        target = _text(target, "query target_entity_id")
    eligible = query_raw.get("eligible", True)
    if not isinstance(eligible, bool):
        raise ValueError("query eligible must be boolean")
    query_row = build_visual_query_result(
        outcome.query_decision,
        record_id=_text(query_raw.get("record_id"), "query record_id"),
        cluster_id=cluster_id,
        split=split,
        system=system,
        source=source,
        property_name=_text(query_raw.get("property_name"), "query property_name"),
        target_entity_id=target,
        horizon_seconds=_number(
            query_raw.get("horizon_seconds"), "query horizon_seconds"
        ),
        distractor_count=distractor_count,
        condition=condition,
        latency_seconds=latency_seconds,
        vlm_calls=vlm_calls,
        malformed=malformed,
        eligible=eligible,
    )
    return VisualEvaluationDataset(batch.properties, batch.associations, (query_row,))


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
