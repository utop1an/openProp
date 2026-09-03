from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

from .ai2thor_adapter import ai2thor_property_registry
from .association import (
    AssociationPolicy,
    MultiEntityAssociator,
    VLMPropertyDetector,
)
from .comparators import default_comparators
from .global_assignment import GlobalOneToOneAssociator
from .models import (
    Entity,
    Observation,
    ObservationState,
    PropertyConstraint,
    QueryFrame,
)
from .query_decision import QueryDecision, QueryDecisionPolicy, decide_query_match
from .selectors import MentionBasedSelector
from .visual_pipeline import EntityStateStore, VisualUpdateOrchestrator, VisualUpdateRun
from .vlm import VisualFrame
from .vlm import VLMError


@dataclass(frozen=True, slots=True)
class VisualReplayOutcome:
    case_id: str
    assignment: str
    query: QueryFrame
    run: VisualUpdateRun
    query_decision: QueryDecision
    associator: MultiEntityAssociator
    malformed_response: bool = False
    response_error: str | None = None


class _ReplayOnlyClient:
    def generate_json(self, **_: object) -> Mapping[str, object]:
        raise RuntimeError("replay-only detector cannot perform model inference")


def replay_visual_case(
    input_payload: Mapping[str, object],
    case_payload: Mapping[str, object],
    captured_response: Mapping[str, object],
    *,
    assignment: str,
    association_policy: AssociationPolicy | None = None,
    query_policy: QueryDecisionPolicy | None = None,
) -> VisualReplayOutcome:
    """Replay one truth-free case through update, memory, matcher, and query decision."""

    if assignment not in {"independent", "global"}:
        raise ValueError("assignment must be independent or global")
    if case_payload.get("schema_version") != 1:
        raise ValueError("visual replay case must use schema_version 1")
    forbidden = {"target", "target_entity_id", "truth", "current_truth", "objects"}
    leaked = sorted(key for key in forbidden if _contains_key(case_payload, key))
    if leaked:
        raise ValueError(f"visual replay case contains evaluation truth fields: {leaked}")
    case_id = _text(case_payload.get("case_id"), "case_id")
    frames = _frames(input_payload)
    first_time = min(frame.captured_at for frame in frames)
    query_time = _number(case_payload.get("query_time"), "query_time")
    if query_time < max(frame.captured_at for frame in frames):
        raise ValueError("query_time cannot precede the visual history")
    registry = ai2thor_property_registry()
    entities = _entities(case_payload.get("initial_entities"), first_time=first_time)
    candidate_ids = {item for frame in frames for item in frame.candidate_entity_ids}
    if {entity.entity_id for entity in entities} != candidate_ids:
        raise ValueError("initial_entities must exactly cover visual candidates")
    query = _query(case_payload.get("query"))
    state = EntityStateStore(registry, entities)
    associator_class = (
        MultiEntityAssociator if assignment == "independent" else GlobalOneToOneAssociator
    )
    associator = associator_class(
        registry,
        default_comparators(),
        MentionBasedSelector(),
        policy=association_policy,
    )
    detector = VLMPropertyDetector(_ReplayOnlyClient())
    orchestrator = VisualUpdateOrchestrator(detector, associator, state)
    malformed_response = False
    response_error: str | None = None
    try:
        detections = detector.parse_response(frames, registry, captured_response)
    except VLMError as error:
        malformed_response = True
        response_error = str(error)
        detections = ()
    run = orchestrator.apply(query, frames, detections)
    query_candidates_raw = case_payload.get("query_candidate_entity_ids")
    if not isinstance(query_candidates_raw, list) or not query_candidates_raw:
        raise ValueError("query_candidate_entity_ids must be a non-empty array")
    query_candidates = tuple(_text(item, "query candidate") for item in query_candidates_raw)
    if len(query_candidates) != len(set(query_candidates)):
        raise ValueError("query candidates cannot contain duplicates")
    if not set(query_candidates).issubset(candidate_ids):
        raise ValueError("query candidate is absent from the visual case")
    matches = associator.matcher.match(
        query,
        state.snapshots(query_candidates, before=math.nextafter(query_time, math.inf)),
        as_of=query_time,
    )
    decision = decide_query_match(matches, policy=query_policy)
    return VisualReplayOutcome(
        case_id,
        assignment,
        query,
        run,
        decision,
        associator,
        malformed_response,
        response_error,
    )


def _frames(payload: Mapping[str, object]) -> tuple[VisualFrame, ...]:
    if payload.get("schema_version") != 1:
        raise ValueError("prepared VLM input must use schema_version 1")
    rows = payload.get("frames")
    if not isinstance(rows, list) or not rows:
        raise ValueError("prepared VLM input must contain frames")
    frames: list[VisualFrame] = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("prepared frame must be an object")
        candidates = row.get("candidate_entity_ids")
        regions = row.get("candidate_regions")
        if not isinstance(candidates, list) or not isinstance(regions, Mapping):
            raise ValueError("prepared frame candidates and regions are malformed")
        frames.append(
            VisualFrame(
                _text(row.get("frame_id"), "frame_id"),
                _text(row.get("image_url"), "image_url"),
                _number(row.get("captured_at"), "captured_at"),
                _text(row.get("source"), "source"),
                tuple(_text(item, "candidate_entity_id") for item in candidates),
                {
                    _text(key, "candidate region ID"): tuple(value)
                    for key, value in regions.items()
                    if isinstance(value, list)
                },
            )
        )
    return tuple(frames)


def _entities(raw: object, *, first_time: float) -> tuple[Entity, ...]:
    if not isinstance(raw, list) or not raw:
        raise ValueError("initial_entities must be a non-empty array")
    result: list[Entity] = []
    for row in raw:
        if not isinstance(row, Mapping):
            raise ValueError("initial entity must be an object")
        properties_raw = row.get("properties", {})
        if not isinstance(properties_raw, Mapping):
            raise ValueError("initial entity properties must be an object")
        properties: dict[str, Observation] = {}
        for name, value in properties_raw.items():
            if not isinstance(value, Mapping):
                raise ValueError("initial observation must be an object")
            timestamp = _number(value.get("timestamp"), "observation timestamp")
            if timestamp >= first_time:
                raise ValueError("initial observation must strictly precede all frames")
            try:
                state = ObservationState(value.get("state", "observed"))
            except ValueError as error:
                raise ValueError("initial observation has invalid state") from error
            properties[_text(name, "property name")] = Observation(
                value.get("value"),
                state,
                _number(value.get("confidence", 1.0), "observation confidence"),
                value.get("source"),
                timestamp,
            )
        result.append(Entity(_text(row.get("entity_id"), "entity_id"), properties))
    return tuple(result)


def _query(raw: object) -> QueryFrame:
    if not isinstance(raw, Mapping):
        raise ValueError("query must be an object")
    constraints_raw = raw.get("constraints")
    if not isinstance(constraints_raw, list) or not constraints_raw:
        raise ValueError("query constraints must be a non-empty array")
    constraints = []
    for row in constraints_raw:
        if not isinstance(row, Mapping):
            raise ValueError("query constraint must be an object")
        constraints.append(
            PropertyConstraint(
                _text(row.get("property_name"), "constraint property"),
                row.get("desired_value"),
                _number(row.get("relevance", 1.0), "constraint relevance"),
                row.get("tolerance"),
            )
        )
    return QueryFrame(_text(raw.get("text"), "query text"), tuple(constraints))


def _contains_key(value: object, target: str) -> bool:
    if isinstance(value, Mapping):
        return target in value or any(_contains_key(item, target) for item in value.values())
    if isinstance(value, list):
        return any(_contains_key(item, target) for item in value)
    return False


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result
