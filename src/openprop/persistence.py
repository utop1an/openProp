from __future__ import annotations

from typing import Protocol

from .models import Entity, Observation, PropertyDefinition
from .temporal import FreshnessResult, observation_freshness


class PersistenceModel(Protocol):
    """Predict whether an observation's state still holds at a later time."""

    def predict(
        self,
        definition: PropertyDefinition,
        observation: Observation,
        entity: Entity,
        *,
        as_of: float,
    ) -> FreshnessResult: ...


class ExponentialPersistenceModel:
    """Backward-compatible fixed half-life and event-retention baseline."""

    def predict(
        self,
        definition: PropertyDefinition,
        observation: Observation,
        entity: Entity,
        *,
        as_of: float,
    ) -> FreshnessResult:
        return observation_freshness(
            definition,
            observation,
            entity.events,
            as_of=as_of,
        )
