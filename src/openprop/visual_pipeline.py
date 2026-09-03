from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from .association import (
    AssociationAuditLedger,
    EntityAssociationHypothesis,
    MultiEntityAssociator,
    VLMPropertyDetector,
    VisualPropertyDetection,
)
from .models import Entity, EntityEvent, QueryFrame
from .property_registry import PropertyRegistry
from .vlm import EntityObservationLedger, PropertyUpdateProposal, VisualFrame


class EntityStateStore:
    """Materialize pre-event entity state from base facts, observations, and events."""

    def __init__(
        self,
        registry: PropertyRegistry,
        entities: Iterable[Entity],
        *,
        observations: EntityObservationLedger | None = None,
    ) -> None:
        self.registry = registry
        self.observations = observations or EntityObservationLedger(registry)
        self._entities: dict[str, Entity] = {}
        for entity in entities:
            if not entity.entity_id.strip():
                raise ValueError("entity_id cannot be empty")
            if entity.entity_id in self._entities:
                raise ValueError(f"duplicate entity_id: {entity.entity_id}")
            self._entities[entity.entity_id] = Entity(
                entity.entity_id,
                dict(entity.properties),
                list(entity.events),
            )

    def entity_ids(self) -> tuple[str, ...]:
        return tuple(self._entities)

    def ensure_entities(self, entity_ids: Sequence[str]) -> tuple[str, ...]:
        """Create blank open-world entities without inventing property evidence."""

        if len(set(entity_ids)) != len(entity_ids):
            raise ValueError("ensure_entities IDs cannot contain duplicates")
        created: list[str] = []
        for entity_id in entity_ids:
            if not entity_id.strip():
                raise ValueError("entity_id cannot be empty")
            if entity_id not in self._entities:
                self._entities[entity_id] = Entity(entity_id, {})
                created.append(entity_id)
        return tuple(created)

    def record_event(self, entity_id: str, event: EntityEvent) -> None:
        entity = self._entities.get(entity_id)
        if entity is None:
            raise KeyError(f"unknown entity_id: {entity_id}")
        entity.record_event(event)

    def snapshot(
        self,
        entity_id: str,
        *,
        before: float | None = None,
    ) -> Entity:
        """Return state strictly before a frame time, or current state when omitted."""

        base = self._entities.get(entity_id)
        if base is None:
            raise KeyError(f"unknown entity_id: {entity_id}")
        cutoff = None
        if before is not None:
            if not math.isfinite(before):
                raise ValueError("before must be finite")
            cutoff = math.nextafter(before, -math.inf)

        properties = {
            name: observation
            for name, observation in base.properties.items()
            if cutoff is None
            or observation.timestamp is None
            or observation.timestamp <= cutoff
        }
        observed = self.observations.snapshot(entity_id, as_of=cutoff)
        properties.update(observed.properties)
        events = [
            event
            for event in base.events
            if cutoff is None or event.timestamp <= cutoff
        ]
        return Entity(entity_id, properties, events)

    def snapshots(
        self,
        entity_ids: Sequence[str],
        *,
        before: float | None = None,
    ) -> tuple[Entity, ...]:
        if len(set(entity_ids)) != len(entity_ids):
            raise ValueError("snapshot entity_ids cannot contain duplicates")
        return tuple(self.snapshot(entity_id, before=before) for entity_id in entity_ids)


@dataclass(frozen=True, slots=True)
class VisualFrameUpdate:
    frame: VisualFrame
    detections: tuple[VisualPropertyDetection, ...]
    hypotheses: tuple[EntityAssociationHypothesis, ...]
    proposals: tuple[PropertyUpdateProposal, ...]


@dataclass(frozen=True, slots=True)
class VisualUpdateRun:
    frame_updates: tuple[VisualFrameUpdate, ...]

    @property
    def detections(self) -> tuple[VisualPropertyDetection, ...]:
        return tuple(
            detection
            for update in self.frame_updates
            for detection in update.detections
        )

    @property
    def hypotheses(self) -> tuple[EntityAssociationHypothesis, ...]:
        return tuple(
            hypothesis
            for update in self.frame_updates
            for hypothesis in update.hypotheses
        )

    @property
    def proposals(self) -> tuple[PropertyUpdateProposal, ...]:
        return tuple(
            proposal
            for update in self.frame_updates
            for proposal in update.proposals
        )

    @property
    def abstentions(self) -> tuple[EntityAssociationHypothesis, ...]:
        return tuple(item for item in self.hypotheses if not item.accepted)


class VisualUpdateOrchestrator:
    """Run detection, pre-event association, audit, and atomic per-frame commit."""

    def __init__(
        self,
        detector: VLMPropertyDetector,
        associator: MultiEntityAssociator,
        state: EntityStateStore,
        *,
        audit: AssociationAuditLedger | None = None,
    ) -> None:
        if associator.registry is not state.registry:
            raise ValueError("associator and state store must share one registry")
        self.detector = detector
        self.associator = associator
        self.state = state
        self.audit = audit or AssociationAuditLedger()

    def run(
        self,
        query: QueryFrame,
        frames: Sequence[VisualFrame],
    ) -> VisualUpdateRun:
        detections = self.detector.detect(frames, self.state.registry)
        return self.apply(query, frames, detections)

    def replay(
        self,
        query: QueryFrame,
        frames: Sequence[VisualFrame],
        captured_response: Mapping[str, object],
    ) -> VisualUpdateRun:
        """Replay a captured VLM response without performing fresh inference."""

        detections = self.detector.parse_response(
            frames,
            self.state.registry,
            captured_response,
        )
        return self.apply(query, frames, detections)

    def apply(
        self,
        query: QueryFrame,
        frames: Sequence[VisualFrame],
        detections: Sequence[VisualPropertyDetection],
    ) -> VisualUpdateRun:
        frame_index: dict[str, tuple[int, VisualFrame]] = {}
        for index, frame in enumerate(frames):
            if frame.frame_id in frame_index:
                raise ValueError(f"duplicate frame_id: {frame.frame_id}")
            frame_index[frame.frame_id] = (index, frame)
        if not frame_index:
            raise ValueError("visual history cannot be empty")

        grouped: dict[str, list[VisualPropertyDetection]] = {
            frame_id: [] for frame_id in frame_index
        }
        seen_detection_ids: set[str] = set()
        for detection in detections:
            if detection.detection_id in seen_detection_ids:
                raise ValueError(f"duplicate detection_id: {detection.detection_id}")
            seen_detection_ids.add(detection.detection_id)
            registered = frame_index.get(detection.frame.frame_id)
            if registered is None:
                raise ValueError(
                    f"detection references unknown frame: {detection.frame.frame_id}"
                )
            if detection.frame != registered[1]:
                raise ValueError("detection frame metadata differs from trusted input")
            grouped[detection.frame.frame_id].append(detection)

        ordered_frames = sorted(
            frame_index.values(),
            key=lambda item: (item[1].captured_at, item[0]),
        )
        updates: list[VisualFrameUpdate] = []
        for _, frame in ordered_frames:
            frame_detections = tuple(grouped[frame.frame_id])
            if not frame_detections:
                updates.append(VisualFrameUpdate(frame, (), (), ()))
                continue
            snapshots = self.state.snapshots(
                frame.candidate_entity_ids,
                before=frame.captured_at,
            )
            hypotheses = self.associator.associate_batch(
                frame_detections,
                query,
                snapshots,
            )
            self.audit.extend(hypotheses)
            proposals = self.associator.commit(
                hypotheses,
                self.state.observations,
            )
            updates.append(
                VisualFrameUpdate(
                    frame,
                    frame_detections,
                    hypotheses,
                    proposals,
                )
            )
        return VisualUpdateRun(tuple(updates))
