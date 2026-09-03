from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .models import PropertyConstraint, QueryFrame
from .property_registry import PropertyRegistry


@dataclass(frozen=True, slots=True)
class SelectedProperty:
    constraint: PropertyConstraint
    canonical_name: str
    weight: float


class PropertySelector(Protocol):
    """Extension point for an LLM or learned relevance model."""

    def select(self, query: QueryFrame, registry: PropertyRegistry) -> tuple[SelectedProperty, ...]: ...


class MentionBasedSelector:
    """Uses parsed query constraints as relevance weights."""

    def __init__(self, *, minimum_weight: float = 0.0) -> None:
        self.minimum_weight = minimum_weight

    def select(self, query: QueryFrame, registry: PropertyRegistry) -> tuple[SelectedProperty, ...]:
        selected: list[SelectedProperty] = []
        for constraint in query.constraints:
            resolution = registry.resolve(constraint.property_name)
            if resolution.definition is None:
                continue
            weight = constraint.relevance * resolution.score
            if weight > self.minimum_weight:
                selected.append(SelectedProperty(constraint, resolution.definition.name, weight))
        return tuple(selected)

