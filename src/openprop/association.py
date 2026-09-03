from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, replace
from typing import Any, Iterable, Mapping, Sequence

from .comparators import ComparatorRegistry
from .llm import LLMQueryParser, _VALUE_SCHEMA
from .matcher import EntityMatcher
from .models import Entity, Observation, ObservationState, QueryFrame, ValueType
from .persistence import PersistenceModel
from .property_registry import PropertyRegistry
from .selectors import PropertySelector
from .vlm import (
    EntityObservationLedger,
    JSONVLMClient,
    PropertyUpdateProposal,
    VLMError,
    VLMPropertyUpdater,
    VisualFrame,
)


@dataclass(frozen=True, slots=True)
class VisualPropertyDetection:
    """One localized visual fact before it is bound to an entity identity."""

    detection_id: str
    frame: VisualFrame
    property_name: str
    value: object
    detection_confidence: float
    value_confidence: float
    candidate_affinities: Mapping[str, float]
    track_id: str | None = None
    track_affinities: Mapping[str, float] = field(default_factory=dict)
    region: tuple[float, float, float, float] | None = None

    def __post_init__(self) -> None:
        if not self.detection_id.strip() or not self.property_name.strip():
            raise ValueError("detection_id and property_name cannot be empty")
        for name, confidence in (
            ("detection_confidence", self.detection_confidence),
            ("value_confidence", self.value_confidence),
        ):
            if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
                raise ValueError(f"{name} must be finite and in [0, 1]")
        candidates = set(self.frame.candidate_entity_ids)
        if set(self.candidate_affinities) != candidates:
            raise ValueError("candidate_affinities must cover every frame candidate exactly")
        if not set(self.track_affinities).issubset(candidates):
            raise ValueError("track_affinities contains an entity outside the frame")
        for affinity in (*self.candidate_affinities.values(), *self.track_affinities.values()):
            if not math.isfinite(affinity) or not 0.0 <= affinity <= 1.0:
                raise ValueError("candidate affinities must be finite and in [0, 1]")
        if self.track_id is not None and not self.track_id.strip():
            raise ValueError("track_id cannot be empty")
        if self.region is not None:
            if len(self.region) != 4:
                raise ValueError("detection region must contain four coordinates")
            left, top, right, bottom = self.region
            if any(not math.isfinite(value) for value in self.region):
                raise ValueError("detection region coordinates must be finite")
            if not (0.0 <= left < right <= 1.0 and 0.0 <= top < bottom <= 1.0):
                raise ValueError("detection region must be normalized with positive area")


_AFFINITY_SCHEMA: dict[str, Any] = {
    "type": "array",
    "items": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "entity_id": {"type": "string"},
            "affinity": {"type": "number", "minimum": 0, "maximum": 1},
        },
        "required": ["entity_id", "affinity"],
    },
}

VISUAL_DETECTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "detections": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "detection_id": {"type": "string"},
                    "frame_id": {"type": "string"},
                    "track_id": {"type": ["string", "null"]},
                    "property_name": {"type": "string"},
                    "value_type": {
                        "type": "string",
                        "enum": [value_type.value for value_type in ValueType],
                    },
                    "detection_confidence": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                    },
                    "value_confidence": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                    },
                    "candidate_affinities": _AFFINITY_SCHEMA,
                    "track_affinities": _AFFINITY_SCHEMA,
                    "region": {
                        "anyOf": [
                            {
                                "type": "array",
                                "items": {
                                    "type": "number",
                                    "minimum": 0,
                                    "maximum": 1,
                                },
                                "minItems": 4,
                                "maxItems": 4,
                            },
                            {"type": "null"},
                        ]
                    },
                    "value": _VALUE_SCHEMA,
                },
                "required": [
                    "detection_id",
                    "frame_id",
                    "track_id",
                    "property_name",
                    "value_type",
                    "detection_confidence",
                    "value_confidence",
                    "candidate_affinities",
                    "track_affinities",
                    "region",
                    "value",
                ],
            },
        }
    },
    "required": ["detections"],
}


class VLMPropertyDetector:
    """Extract localized, entity-unbound property detections from visual history."""

    def __init__(self, client: JSONVLMClient) -> None:
        self.client = client

    def request(
        self,
        frames: Sequence[VisualFrame],
        registry: PropertyRegistry,
    ) -> Mapping[str, Any]:
        frame_index = VLMPropertyUpdater._frames(frames)
        return self.client.generate_json(
            instructions=self._instructions(),
            input_text=self._input(tuple(frame_index.values()), registry),
            image_urls=tuple(frame.image_url for frame in frame_index.values()),
            schema_name="openprop_visual_detections",
            schema=VISUAL_DETECTION_SCHEMA,
        )

    def detect(
        self,
        frames: Sequence[VisualFrame],
        registry: PropertyRegistry,
    ) -> tuple[VisualPropertyDetection, ...]:
        return self.parse_response(frames, registry, self.request(frames, registry))

    def parse_response(
        self,
        frames: Sequence[VisualFrame],
        registry: PropertyRegistry,
        raw: Mapping[str, Any],
    ) -> tuple[VisualPropertyDetection, ...]:
        frame_index = VLMPropertyUpdater._frames(frames)
        rows = raw.get("detections")
        if not isinstance(rows, list):
            raise VLMError("structured response is missing a detections array")
        detections: list[VisualPropertyDetection] = []
        seen: set[str] = set()
        for row in rows:
            if not isinstance(row, Mapping):
                raise VLMError("each detection must be an object")
            detection_id = self._text(row.get("detection_id"), "detection_id")
            if detection_id in seen:
                raise VLMError(f"duplicate detection_id: {detection_id}")
            seen.add(detection_id)
            frame_id = self._text(row.get("frame_id"), "frame_id")
            if frame_id not in frame_index:
                raise VLMError(f"unknown frame_id: {frame_id}")
            frame = frame_index[frame_id]
            property_name = self._text(row.get("property_name"), "property_name")
            resolution = registry.resolve(property_name)
            definition = resolution.definition
            if definition is None:
                raise VLMError(f"unregistered detection property: {property_name}")
            try:
                claimed_type = ValueType(row.get("value_type"))
            except (TypeError, ValueError) as error:
                raise VLMError(f"invalid value_type: {row.get('value_type')!r}") from error
            if claimed_type is not definition.value_type:
                raise VLMError(
                    f"{definition.name} expects {definition.value_type.value}, "
                    f"not {claimed_type.value}"
                )
            value = LLMQueryParser._value(row.get("value"), definition.value_type)
            VLMPropertyUpdater._validate_value(value, definition)
            track_id = row.get("track_id")
            if track_id is not None:
                track_id = self._text(track_id, "track_id")
            detections.append(
                VisualPropertyDetection(
                    detection_id,
                    frame,
                    definition.name,
                    value,
                    self._number(
                        row.get("detection_confidence"),
                        "detection_confidence",
                    ),
                    self._number(row.get("value_confidence"), "value_confidence"),
                    self._affinities(
                        row.get("candidate_affinities"),
                        "candidate_affinities",
                    ),
                    track_id,
                    self._affinities(
                        row.get("track_affinities"),
                        "track_affinities",
                    ),
                    self._region(row.get("region")),
                )
            )
        return tuple(detections)

    @staticmethod
    def _instructions() -> str:
        return (
            "Extract localized OpenProp property detections from ordered images. "
            "One detection_id represents one physical target; use multiple detections "
            "for multiple objects. Do not choose or emit a final entity identity. Score "
            "every listed frame candidate independently in candidate_affinities. "
            "Emit the detection's normalized [left, top, right, bottom] region when "
            "visually localized; use null only when a temporal change is supported "
            "but its target is not localizable in that frame. "
            "When candidate_regions are supplied, bind each opaque entity ID only to its "
            "normalized [left, top, right, bottom] image box with a top-left origin; "
            "do not infer identity from the ID text. "
            "track_affinities only for visible temporal continuity, otherwise return an "
            "empty array. Keep detection confidence, value confidence, and identity "
            "affinity separate. Use only registered properties and supplied frame IDs. "
            "Do not infer source or capture time."
        )

    @staticmethod
    def _input(
        frames: Sequence[VisualFrame],
        registry: PropertyRegistry,
    ) -> str:
        properties = [
            {
                "name": definition.name,
                "description": definition.description,
                "value_type": definition.value_type.value,
                "aliases": list(definition.aliases),
                "unit": definition.unit,
                "metadata": dict(definition.metadata),
                "visual_updates_allowed": (
                    definition.update_policy.allow_visual_updates
                ),
            }
            for definition in registry.definitions()
            if definition.update_policy.allow_visual_updates
        ]
        history = [
            {
                "image_index": index,
                "frame_id": frame.frame_id,
                "candidate_entity_ids": list(frame.candidate_entity_ids),
                "candidate_regions": [
                    {"entity_id": entity_id, "box": list(frame.candidate_regions[entity_id])}
                    for entity_id in frame.candidate_entity_ids
                    if entity_id in frame.candidate_regions
                ],
            }
            for index, frame in enumerate(frames)
        ]
        return json.dumps(
            {"property_dictionary": properties, "visual_history": history},
            ensure_ascii=False,
            separators=(",", ":"),
        )

    @classmethod
    def _affinities(cls, raw: Any, field_name: str) -> dict[str, float]:
        if not isinstance(raw, list):
            raise VLMError(f"{field_name} must be an array")
        result: dict[str, float] = {}
        for item in raw:
            if not isinstance(item, Mapping):
                raise VLMError(f"{field_name} entries must be objects")
            entity_id = cls._text(item.get("entity_id"), "entity_id")
            if entity_id in result:
                raise VLMError(f"duplicate affinity entity_id: {entity_id}")
            affinity = cls._number(item.get("affinity"), "affinity")
            if not 0.0 <= affinity <= 1.0:
                raise VLMError("affinity must be in [0, 1]")
            result[entity_id] = affinity
        return result


    @classmethod
    def _region(
        cls,
        raw: Any,
    ) -> tuple[float, float, float, float] | None:
        if raw is None:
            return None
        if not isinstance(raw, list) or len(raw) != 4:
            raise VLMError("region must be null or four normalized coordinates")
        region = tuple(
            cls._number(value, f"region[{index}]")
            for index, value in enumerate(raw)
        )
        left, top, right, bottom = region
        if not (0.0 <= left < right <= 1.0 and 0.0 <= top < bottom <= 1.0):
            raise VLMError("region must be normalized with positive area")
        return region

    @staticmethod
    def _text(value: Any, field_name: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise VLMError(f"{field_name} must be a non-empty string")
        return value.strip()

    @staticmethod
    def _number(value: Any, field_name: str) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise VLMError(f"{field_name} must be a number")
        result = float(value)
        if not math.isfinite(result):
            raise VLMError(f"{field_name} must be finite")
        return result

@dataclass(frozen=True, slots=True)
class AssociationPolicy:
    """Predeclared admission and evidence-combination policy."""

    acceptance_threshold: float = 0.80
    margin_threshold: float = 0.15
    minimum_detection_confidence: float = 0.50
    minimum_value_confidence: float = 0.50
    null_weight: float = 0.05
    query_weight: float = 1.0
    visual_weight: float = 1.0
    track_weight: float = 1.0
    source_reliability: Mapping[str, float] = field(default_factory=dict)
    epsilon: float = 1e-9

    def __post_init__(self) -> None:
        probabilities = (
            self.acceptance_threshold,
            self.margin_threshold,
            self.minimum_detection_confidence,
            self.minimum_value_confidence,
        )
        if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in probabilities):
            raise ValueError("association thresholds must be finite and in [0, 1]")
        if not math.isfinite(self.null_weight) or self.null_weight <= 0.0:
            raise ValueError("null_weight must be finite and positive")
        weights = (self.query_weight, self.visual_weight, self.track_weight)
        if any(not math.isfinite(value) or value < 0.0 for value in weights):
            raise ValueError("association weights must be finite and nonnegative")
        if not any(value > 0.0 for value in weights):
            raise ValueError("at least one association weight must be positive")
        if not math.isfinite(self.epsilon) or not 0.0 < self.epsilon < 1.0:
            raise ValueError("epsilon must be finite and in (0, 1)")
        if any(
            not math.isfinite(value) or not 0.0 <= value <= 1.0
            for value in self.source_reliability.values()
        ):
            raise ValueError("source reliability must be finite and in [0, 1]")

    def reliability_for(self, source: str) -> float:
        expected = source.casefold()
        for candidate, reliability in self.source_reliability.items():
            if candidate.casefold() == expected:
                return reliability
        return 1.0


@dataclass(frozen=True, slots=True)
class AssociationCandidate:
    entity_id: str
    query_score: float
    visual_affinity: float
    track_affinity: float | None
    unnormalized_score: float
    posterior: float


@dataclass(frozen=True, slots=True)
class EntityAssociationHypothesis:
    detection: VisualPropertyDetection
    query_text: str
    candidates: tuple[AssociationCandidate, ...]
    null_probability: float
    accepted_entity_id: str | None
    update_confidence: float
    reason: str
    decision_entity_id: str | None = None

    @property
    def accepted(self) -> bool:
        return self.accepted_entity_id is not None


class AssociationAuditLedger:
    """Append-only audit history including abstained association decisions."""

    def __init__(self) -> None:
        self._hypotheses: list[EntityAssociationHypothesis] = []

    def append(self, hypothesis: EntityAssociationHypothesis) -> None:
        self._hypotheses.append(hypothesis)

    def extend(self, hypotheses: Iterable[EntityAssociationHypothesis]) -> None:
        self._hypotheses.extend(tuple(hypotheses))

    def entries(
        self, *, detection_id: str | None = None, accepted: bool | None = None
    ) -> tuple[EntityAssociationHypothesis, ...]:
        return tuple(
            item
            for item in self._hypotheses
            if (detection_id is None or item.detection.detection_id == detection_id)
            and (accepted is None or item.accepted is accepted)
        )


class MultiEntityAssociator:
    """Associate localized detections against a strictly pre-event snapshot."""

    def __init__(
        self,
        registry: PropertyRegistry,
        comparators: ComparatorRegistry,
        selector: PropertySelector,
        *,
        policy: AssociationPolicy | None = None,
        persistence_model: PersistenceModel | None = None,
    ) -> None:
        self.registry = registry
        self.policy = policy or AssociationPolicy()
        self.matcher = EntityMatcher(
            registry,
            comparators,
            selector,
            persistence_model=persistence_model,
        )

    def associate(
        self,
        detection: VisualPropertyDetection,
        query: QueryFrame,
        entities: Sequence[Entity],
    ) -> EntityAssociationHypothesis:
        definition = self.registry.resolve(detection.property_name).definition
        if definition is None:
            raise VLMError(f"unregistered detection property: {detection.property_name}")
        VLMPropertyUpdater._validate_value(detection.value, definition)
        entity_index = self._pre_event_candidates(detection, entities)
        match_results = self.matcher.match(
            query,
            [entity_index[entity_id] for entity_id in detection.frame.candidate_entity_ids],
            as_of=detection.frame.captured_at,
        )
        query_evidence = {
            result.entity_id: (result.score, result.coverage)
            for result in match_results
        }

        rows: list[tuple[str, float, float, float | None, float]] = []
        for entity_id in detection.frame.candidate_entity_ids:
            query_score, query_coverage = query_evidence[entity_id]
            visual = detection.candidate_affinities[entity_id]
            track = detection.track_affinities.get(entity_id)
            raw = (
                1.0
                if query_coverage == 0.0
                else self._factor(query_score, self.policy.query_weight)
            )
            raw *= self._factor(visual, self.policy.visual_weight)
            if track is not None:
                raw *= self._factor(track, self.policy.track_weight)
            rows.append((entity_id, query_score, visual, track, raw))

        denominator = self.policy.null_weight + sum(row[4] for row in rows)
        candidates = tuple(
            sorted(
                (
                    AssociationCandidate(
                        entity_id,
                        query_score,
                        visual,
                        track,
                        raw,
                        raw / denominator,
                    )
                    for entity_id, query_score, visual, track, raw in rows
                ),
                key=lambda item: (-item.posterior, item.entity_id),
            )
        )
        null_probability = self.policy.null_weight / denominator
        top = candidates[0]
        runner_up = max(
            null_probability,
            candidates[1].posterior if len(candidates) > 1 else 0.0,
        )
        update_policy = definition.update_policy
        accepted_entity_id: str | None = None
        reason: str
        if detection.detection_confidence < self.policy.minimum_detection_confidence:
            reason = "detection confidence below policy"
        elif detection.value_confidence < self.policy.minimum_value_confidence:
            reason = "value confidence below policy"
        elif top.posterior < self.policy.acceptance_threshold:
            reason = "top association below acceptance threshold"
        elif top.posterior - runner_up < self.policy.margin_threshold:
            reason = "association margin below policy"
        else:
            accepted_entity_id = top.entity_id
            reason = "accepted by posterior and margin gates"

        if accepted_entity_id is not None and (
            not update_policy.allow_visual_updates
            or not update_policy.permits_source(detection.frame.source)
        ):
            accepted_entity_id = None
            reason = (
                "property update policy rejects the visual source or modality"
            )
        update_confidence = 0.0
        if accepted_entity_id is not None:
            update_confidence = (
                detection.detection_confidence
                * detection.value_confidence
                * top.posterior
                * self.policy.reliability_for(detection.frame.source)
            )
            if update_confidence < definition.update_policy.minimum_confidence:
                accepted_entity_id = None
                update_confidence = 0.0
                reason = "combined confidence below property update policy"

        return EntityAssociationHypothesis(
            detection,
            query.text,
            candidates,
            null_probability,
            accepted_entity_id,
            update_confidence,
            reason,
            top.entity_id,
        )

    def associate_batch(
        self,
        detections: Sequence[VisualPropertyDetection],
        query: QueryFrame,
        entities: Sequence[Entity],
    ) -> tuple[EntityAssociationHypothesis, ...]:
        seen_ids: set[str] = set()
        hypotheses: list[EntityAssociationHypothesis] = []
        for detection in detections:
            if detection.detection_id in seen_ids:
                raise ValueError(f"duplicate detection_id: {detection.detection_id}")
            seen_ids.add(detection.detection_id)
            hypotheses.append(self.associate(detection, query, entities))

        conflicts: dict[tuple[str, str, str], list[int]] = {}
        for index, hypothesis in enumerate(hypotheses):
            if not hypothesis.accepted:
                continue
            definition = self.registry.resolve(
                hypothesis.detection.property_name
            ).definition
            assert definition is not None
            key = (
                hypothesis.detection.frame.frame_id,
                hypothesis.detection.property_name.casefold(),
                hypothesis.accepted_entity_id or "",
            )
            conflicts.setdefault(key, []).append(index)
        for indexes in conflicts.values():
            if len(indexes) < 2:
                continue
            for index in indexes:
                hypotheses[index] = replace(
                    hypotheses[index],
                    accepted_entity_id=None,
                    update_confidence=0.0,
                    reason="conflicting detections target the same entity property",
                )
        return tuple(hypotheses)

    def to_update(
        self, hypothesis: EntityAssociationHypothesis
    ) -> PropertyUpdateProposal | None:
        if not hypothesis.accepted:
            return None
        detection = hypothesis.detection
        observation = Observation(
            detection.value,
            ObservationState.OBSERVED,
            hypothesis.update_confidence,
            detection.frame.source,
            detection.frame.captured_at,
        )
        definition = self.registry.resolve(detection.property_name).definition
        assert definition is not None
        return PropertyUpdateProposal(
            hypothesis.accepted_entity_id or "",
            definition.name,
            observation,
            detection.frame.frame_id,
        )

    def commit(
        self,
        hypotheses: Iterable[EntityAssociationHypothesis],
        ledger: EntityObservationLedger,
    ) -> tuple[PropertyUpdateProposal, ...]:
        proposals = tuple(
            proposal
            for hypothesis in hypotheses
            if (proposal := self.to_update(hypothesis)) is not None
        )
        keys: set[tuple[str, str, str]] = set()
        for proposal in proposals:
            key = (
                proposal.frame_id,
                proposal.entity_id,
                proposal.property_name.casefold(),
            )
            if key in keys:
                raise ValueError("accepted hypotheses contain a duplicate entity update")
            keys.add(key)
        ledger.extend(proposals)
        return proposals

    def _pre_event_candidates(
        self,
        detection: VisualPropertyDetection,
        entities: Sequence[Entity],
    ) -> dict[str, Entity]:
        entity_index: dict[str, Entity] = {}
        for entity in entities:
            if entity.entity_id in entity_index:
                raise ValueError(f"duplicate entity_id: {entity.entity_id}")
            entity_index[entity.entity_id] = entity
        expected = set(detection.frame.candidate_entity_ids)
        missing = expected - set(entity_index)
        if missing:
            raise ValueError(f"candidate snapshot is missing entities: {sorted(missing)}")
        for entity_id in expected:
            for observation in entity_index[entity_id].properties.values():
                timestamp = observation.timestamp
                if timestamp is not None and timestamp >= detection.frame.captured_at:
                    raise ValueError(
                        "association candidates must come from a strictly pre-event snapshot"
                    )
        return entity_index

    def _factor(self, value: float, weight: float) -> float:
        if weight == 0.0:
            return 1.0
        return max(self.policy.epsilon, value) ** weight
