from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping


class ValueType(str, Enum):
    """A value family determines how a property is compared."""

    SEMANTIC = "semantic"
    CATEGORICAL = "categorical"
    NUMERIC = "numeric"
    VECTOR = "vector"
    RELATION = "relation"
    ENTITY_REFERENCE = "entity_reference"
    TEMPORAL = "temporal"


class ObservationState(str, Enum):
    """Missing evidence is distinct from negative evidence."""

    OBSERVED = "observed"
    UNKNOWN = "unknown"
    NOT_APPLICABLE = "not_applicable"


@dataclass(frozen=True, slots=True)
class TemporalPolicy:
    """How long an observation remains useful and which events invalidate it."""

    half_life_seconds: float | None = None
    minimum_freshness: float = 0.0
    event_retention: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.half_life_seconds is not None and self.half_life_seconds <= 0:
            raise ValueError("half_life_seconds must be positive")
        if not 0.0 <= self.minimum_freshness <= 1.0:
            raise ValueError("minimum_freshness must be between 0 and 1")
        if any(not 0.0 <= value <= 1.0 for value in self.event_retention.values()):
            raise ValueError("event retention values must be between 0 and 1")


@dataclass(frozen=True, slots=True)
class PropertyUpdatePolicy:
    """Admission policy for proposals derived from visual observations."""

    allow_visual_updates: bool = True
    minimum_confidence: float = 0.0
    allowed_sources: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not 0.0 <= self.minimum_confidence <= 1.0:
            raise ValueError("minimum_confidence must be between 0 and 1")
        if any(not source.strip() for source in self.allowed_sources):
            raise ValueError("allowed_sources cannot contain empty values")

    def permits_source(self, source: str) -> bool:
        if not self.allowed_sources:
            return True
        expected = source.casefold()
        return any(candidate.casefold() == expected for candidate in self.allowed_sources)


@dataclass(frozen=True, slots=True)
class EntityEvent:
    event_type: str
    timestamp: float
    confidence: float = 1.0
    source: str | None = None

    def __post_init__(self) -> None:
        if not self.event_type.strip():
            raise ValueError("event_type cannot be empty")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("event confidence must be between 0 and 1")

@dataclass(frozen=True, slots=True)
class PropertyDefinition:
    name: str
    description: str
    value_type: ValueType
    aliases: tuple[str, ...] = ()
    comparator: str | None = None
    unit: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    temporal_policy: TemporalPolicy | None = None
    update_policy: PropertyUpdatePolicy = field(default_factory=PropertyUpdatePolicy)


@dataclass(frozen=True, slots=True)
class RelationValue:
    """Keep predicate semantics and argument identity separate."""

    predicate: str
    arguments: Mapping[str, str]


@dataclass(frozen=True, slots=True)
class Observation:
    value: Any = None
    state: ObservationState = ObservationState.OBSERVED
    confidence: float = 1.0
    source: str | None = None
    timestamp: float | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        if self.state is ObservationState.OBSERVED and self.value is None:
            raise ValueError("an observed property must have a value")

    @classmethod
    def unknown(cls, *, source: str | None = None) -> "Observation":
        return cls(state=ObservationState.UNKNOWN, confidence=0.0, source=source)


@dataclass(slots=True)
class Entity:
    entity_id: str
    properties: dict[str, Observation] = field(default_factory=dict)
    events: list[EntityEvent] = field(default_factory=list)

    def observe(self, property_name: str, observation: Observation) -> None:
        self.properties[property_name] = observation

    def record_event(self, event: EntityEvent) -> None:
        self.events.append(event)


@dataclass(frozen=True, slots=True)
class PropertyConstraint:
    property_name: str
    desired_value: Any
    relevance: float = 1.0
    tolerance: float | None = None

    def __post_init__(self) -> None:
        if self.relevance < 0.0:
            raise ValueError("relevance cannot be negative")


@dataclass(frozen=True, slots=True)
class QueryFrame:
    text: str
    constraints: tuple[PropertyConstraint, ...]


@dataclass(frozen=True, slots=True)
class ComparisonResult:
    score: float | None
    reason: str


@dataclass(frozen=True, slots=True)
class PropertyEvidence:
    property_name: str
    weight: float
    score: float | None
    observation_state: ObservationState
    reason: str
    effective_confidence: float = 1.0
    freshness: float = 1.0
    age_seconds: float | None = None


@dataclass(frozen=True, slots=True)
class MatchResult:
    entity_id: str
    score: float
    match_score: float
    coverage: float
    evidence: tuple[PropertyEvidence, ...]

