from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Iterable, Mapping

from .benchmark import core_registry
from .comparators import default_comparators
from .matcher import EntityMatcher
from .models import (
    Entity,
    EntityEvent,
    Observation,
    ObservationState,
    PropertyConstraint,
    PropertyDefinition,
    QueryFrame,
    RelationValue,
    TemporalPolicy,
    ValueType,
)
from .persistence import ExponentialPersistenceModel, PersistenceModel
from .property_registry import PropertyRegistry
from .selectors import MentionBasedSelector
from .temporal import FreshnessResult


@dataclass(frozen=True, slots=True)
class TemporalGroundingCase:
    """A query over stale observations with separately held current truth."""

    case_id: str
    query: str
    entities: tuple[Entity, ...]
    target_id: str
    gold_frame: QueryFrame
    as_of: float
    current_truth: Mapping[str, Mapping[str, object]]
    tags: tuple[str, ...] = ()


class TemporalStrategy(str, Enum):
    NO_DECAY = "no-decay"
    FIXED_DECAY = "fixed-decay"
    LEARNED_DECAY = "learned-decay"


@dataclass(frozen=True, slots=True)
class TemporalCaseResult:
    case_id: str
    target_id: str
    predicted_id: str
    rank: int
    target_score: float
    target_coverage: float
    tags: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class TemporalGroundingReport:
    strategy: TemporalStrategy
    cases: int
    top1_accuracy: float
    top3_recall: float
    mean_reciprocal_rank: float
    accuracy_by_tag: Mapping[str, float]
    results: tuple[TemporalCaseResult, ...]


class NoDecayPersistenceModel:
    """Academic ablation: every observed value remains fully current."""

    def predict(
        self,
        definition: PropertyDefinition,
        observation: Observation,
        entity: Entity,
        *,
        as_of: float,
    ) -> FreshnessResult:
        age = None if observation.timestamp is None else max(0.0, as_of - observation.timestamp)
        return FreshnessResult(1.0, age, 1.0, 1.0)


def temporal_grounding_registry() -> PropertyRegistry:
    registry = PropertyRegistry()
    for definition in core_registry().definitions():
        if definition.name != "location":
            registry.register(definition)
    registry.register(PropertyDefinition("scene", "scene containing the entity", ValueType.CATEGORICAL))
    registry.register(
        PropertyDefinition(
            "location",
            "current spatial relation between an entity and another entity",
            ValueType.RELATION,
            aliases=("spatial relation", "position relation"),
            metadata={
                "argument_roles": ["object"],
                "allowed_predicates": ["on", "inside", "under", "near"],
            },
            temporal_policy=TemporalPolicy(
                half_life_seconds=2 * 3600,
                minimum_freshness=0.01,
                event_retention={"moved": 0.02, "picked_up": 0.05},
            ),
        )
    )
    registry.register(
        PropertyDefinition(
            "cleanliness",
            "currently observed cleanliness state",
            ValueType.SEMANTIC,
            temporal_policy=TemporalPolicy(
                half_life_seconds=48 * 3600,
                minimum_freshness=0.02,
                event_retention={"worn": 0.08, "spilled_on": 0.03},
            ),
        )
    )
    return registry


def _relation(predicate: str, object_id: str) -> RelationValue:
    return RelationValue(predicate, {"object": object_id})


def _observed(value: object, timestamp: float | None = None) -> Observation:
    return Observation(value, timestamp=timestamp)


def _entity(
    entity_id: str,
    properties: Mapping[str, Observation],
    events: Iterable[EntityEvent] = (),
) -> Entity:
    return Entity(entity_id, dict(properties), list(events))


def _frame(text: str, constraints: tuple[PropertyConstraint, ...]) -> QueryFrame:
    return QueryFrame(text, constraints)


def temporal_grounding_benchmark(*, repetitions: int = 10) -> tuple[TemporalGroundingCase, ...]:
    """Generate paired temporal challenges and non-temporal controls."""

    if repetitions <= 0:
        raise ValueError("repetitions must be positive")
    cases: list[TemporalGroundingCase] = []
    base_time = 1_800_000_000.0

    for index in range(repetitions):
        as_of = base_time + index * 86_400
        language = "zh" if index % 2 == 0 else "en"
        relation_query = "桌上的红色杯子" if language == "zh" else "the red cup on the table"
        relation_frame = _frame(
            relation_query,
            (
                PropertyConstraint("type", "cup", 0.25),
                PropertyConstraint("color", "red", 0.25),
                PropertyConstraint("location", _relation("on", "table"), 0.50),
            ),
        )

        stale_id = f"stale-cup-{index:02d}"
        target_id = f"current-cup-{index:02d}"
        blue_id = f"blue-cup-{index:02d}"
        stale_timestamp = as_of - (4 + index % 4) * 3600
        recent_timestamp = as_of - (5 + index % 10) * 60
        irrelevant = {
            "material": _observed("ceramic"),
            "temperature": _observed(18 + index % 5),
            "size": _observed(9 + index % 3),
            "scene": _observed("kitchen"),
        }
        stale = _entity(
            stale_id,
            {
                "type": _observed("cup"),
                "color": _observed("red"),
                "location": _observed(_relation("on", "table"), stale_timestamp),
                **irrelevant,
            },
            (EntityEvent("moved", stale_timestamp + 1800, 1.0, "tracker"),),
        )
        current = _entity(
            target_id,
            {
                "type": _observed("cup"),
                "color": Observation("red", confidence=0.95),
                "location": _observed(_relation("on", "table"), recent_timestamp),
                **irrelevant,
            },
        )
        blue = _entity(
            blue_id,
            {
                "type": _observed("cup"),
                "color": _observed("blue"),
                "location": Observation(
                    _relation("on", "table"),
                    confidence=0.9,
                    timestamp=recent_timestamp,
                ),
                **irrelevant,
            },
        )
        cases.append(
            TemporalGroundingCase(
                f"stale-location-{index:02d}",
                relation_query,
                (stale, current, blue),
                target_id,
                relation_frame,
                as_of,
                {
                    stale_id: {"location": _relation("on", "shelf")},
                    target_id: {"location": _relation("on", "table")},
                    blue_id: {"location": _relation("on", "table")},
                },
                (language, "stale-location", "event", "irrelevant-properties"),
            )
        )

        missing_id = f"missing-color-cup-{index:02d}"
        missing_target = _entity(
            missing_id,
            {
                "type": _observed("cup"),
                "color": Observation.unknown(source="occluded-camera"),
                "location": _observed(_relation("on", "table"), recent_timestamp),
                **irrelevant,
            },
        )
        cases.append(
            TemporalGroundingCase(
                f"missing-observation-{index:02d}",
                relation_query,
                (stale, missing_target, blue),
                missing_id,
                relation_frame,
                as_of,
                {
                    stale_id: {"location": _relation("on", "shelf")},
                    missing_id: {"color": "red", "location": _relation("on", "table")},
                    blue_id: {"color": "blue", "location": _relation("on", "table")},
                },
                (language, "missing-observation", "stale-location", "irrelevant-properties"),
            )
        )

        clean_query = "干净的蓝色衬衫" if language == "zh" else "the clean blue shirt"
        clean_frame = _frame(
            clean_query,
            (
                PropertyConstraint("type", "shirt", 0.25),
                PropertyConstraint("color", "blue", 0.20),
                PropertyConstraint("cleanliness", "clean", 0.55),
            ),
        )
        worn_id = f"worn-shirt-{index:02d}"
        clean_id = f"clean-shirt-{index:02d}"
        old_clean = as_of - (48 + index % 24) * 3600
        worn = _entity(
            worn_id,
            {
                "type": _observed("shirt"),
                "color": _observed("blue"),
                "cleanliness": _observed("clean", old_clean),
                "material": _observed("cotton"),
                "owner": _observed("alice"),
                "scene": _observed("wardrobe"),
            },
            (EntityEvent("worn", old_clean + 3600, 1.0, "wardrobe-log"),),
        )
        clean = _entity(
            clean_id,
            {
                "type": _observed("shirt"),
                "color": _observed("blue"),
                "cleanliness": Observation("clean", confidence=0.95, timestamp=as_of - 30 * 60),
                "material": _observed("linen"),
                "owner": _observed("bob"),
                "scene": _observed("wardrobe"),
            },
        )
        cases.append(
            TemporalGroundingCase(
                f"event-invalidated-{index:02d}",
                clean_query,
                (worn, clean),
                clean_id,
                clean_frame,
                as_of,
                {
                    worn_id: {"cleanliness": "dirty"},
                    clean_id: {"cleanliness": "clean"},
                },
                (language, "event-invalidated", "irrelevant-properties"),
            )
        )

        control_query = "红色陶瓷杯" if language == "zh" else "the red ceramic cup"
        control_frame = _frame(
            control_query,
            (
                PropertyConstraint("type", "cup", 0.35),
                PropertyConstraint("color", "red", 0.35),
                PropertyConstraint("material", "ceramic", 0.30),
            ),
        )
        control_target_id = f"control-red-{index:02d}"
        control_other_id = f"control-blue-{index:02d}"
        controls = (
            _entity(control_target_id, {"type": _observed("cup"), "color": _observed("red"), "material": _observed("ceramic"), "temperature": _observed(22), "scene": _observed("office")}),
            _entity(control_other_id, {"type": _observed("cup"), "color": _observed("blue"), "material": _observed("plastic"), "temperature": _observed(22), "scene": _observed("office")}),
        )
        cases.append(
            TemporalGroundingCase(
                f"static-control-{index:02d}",
                control_query,
                controls,
                control_target_id,
                control_frame,
                as_of,
                {
                    control_target_id: {"type": "cup", "color": "red", "material": "ceramic"},
                    control_other_id: {"type": "cup", "color": "blue", "material": "plastic"},
                },
                (language, "static-control", "irrelevant-properties"),
            )
        )

    return tuple(cases)


def evaluate_temporal_grounding(
    cases: Iterable[TemporalGroundingCase],
    registry: PropertyRegistry,
    strategy: TemporalStrategy,
    *,
    learned_model: PersistenceModel | None = None,
) -> TemporalGroundingReport:
    rows = tuple(cases)
    if not rows:
        raise ValueError("at least one temporal grounding case is required")
    if strategy is TemporalStrategy.NO_DECAY:
        persistence: PersistenceModel = NoDecayPersistenceModel()
    elif strategy is TemporalStrategy.FIXED_DECAY:
        persistence = ExponentialPersistenceModel()
    else:
        if learned_model is None:
            raise ValueError("learned_model is required for learned-decay evaluation")
        persistence = learned_model
    matcher = EntityMatcher(
        registry,
        default_comparators(),
        MentionBasedSelector(),
        persistence_model=persistence,
    )
    results: list[TemporalCaseResult] = []
    for case in rows:
        ranking = matcher.match(case.gold_frame, list(case.entities), as_of=case.as_of)
        ids = [result.entity_id for result in ranking]
        rank = ids.index(case.target_id) + 1
        target = ranking[rank - 1]
        results.append(
            TemporalCaseResult(
                case.case_id,
                case.target_id,
                ids[0],
                rank,
                target.score,
                target.coverage,
                case.tags,
            )
        )
    tags = sorted({tag for result in results for tag in result.tags})
    accuracy_by_tag = {
        tag: sum(result.rank == 1 for result in results if tag in result.tags)
        / sum(tag in result.tags for result in results)
        for tag in tags
    }
    count = len(results)
    return TemporalGroundingReport(
        strategy,
        count,
        sum(result.rank == 1 for result in results) / count,
        sum(result.rank <= 3 for result in results) / count,
        sum(1.0 / result.rank for result in results) / count,
        accuracy_by_tag,
        tuple(results),
    )


def _json_value(value: object) -> object:
    if isinstance(value, RelationValue):
        return {"predicate": value.predicate, "arguments": dict(value.arguments)}
    return value


def write_temporal_grounding_jsonl(
    path: str | Path,
    cases: Iterable[TemporalGroundingCase],
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as stream:
        for case in cases:
            payload = {
                "case_id": case.case_id,
                "query": case.query,
                "target_id": case.target_id,
                "as_of": case.as_of,
                "tags": list(case.tags),
                "constraints": [
                    {
                        "property_name": constraint.property_name,
                        "desired_value": _json_value(constraint.desired_value),
                        "relevance": constraint.relevance,
                    }
                    for constraint in case.gold_frame.constraints
                ],
                "entities": [
                    {
                        "entity_id": entity.entity_id,
                        "observations": {
                            name: {
                                "value": _json_value(observation.value),
                                "state": observation.state.value,
                                "confidence": observation.confidence,
                                "timestamp": observation.timestamp,
                                "source": observation.source,
                            }
                            for name, observation in entity.properties.items()
                        },
                        "events": [
                            {
                                "event_type": event.event_type,
                                "timestamp": event.timestamp,
                                "confidence": event.confidence,
                                "source": event.source,
                            }
                            for event in entity.events
                        ],
                    }
                    for entity in case.entities
                ],
                "current_truth": {
                    entity_id: {name: _json_value(value) for name, value in truth.items()}
                    for entity_id, truth in case.current_truth.items()
                },
            }
            stream.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")

