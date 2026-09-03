from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
from threading import RLock

from .models import PropertyDefinition


def _normalise(text: str) -> str:
    return " ".join(text.casefold().replace("_", " ").replace("-", " ").split())


@dataclass(frozen=True, slots=True)
class Resolution:
    definition: PropertyDefinition | None
    score: float
    created: bool = False


class PropertyRegistry:
    """Extensible schema with conservative alias resolution.

    Semantic/LLM resolution can be added behind ``resolve`` later; the core
    registry deliberately starts deterministic and auditable.
    """

    def __init__(self) -> None:
        self._definitions: dict[str, PropertyDefinition] = {}
        self._lock = RLock()

    def register(self, definition: PropertyDefinition) -> PropertyDefinition:
        key = _normalise(definition.name)
        if not key:
            raise ValueError("property name cannot be empty")
        with self._lock:
            existing = self._definitions.get(key)
            if existing is not None and existing != definition:
                raise ValueError(f"property already registered: {definition.name}")
            self._definitions[key] = definition
        return definition

    def get(self, name: str) -> PropertyDefinition | None:
        return self._definitions.get(_normalise(name))

    def resolve(self, name: str, *, threshold: float = 0.82) -> Resolution:
        query = _normalise(name)
        exact = self._definitions.get(query)
        if exact is not None:
            return Resolution(exact, 1.0)

        best: PropertyDefinition | None = None
        best_score = 0.0
        for definition in self._definitions.values():
            labels = (definition.name, *definition.aliases)
            score = max(SequenceMatcher(None, query, _normalise(label)).ratio() for label in labels)
            if score > best_score:
                best, best_score = definition, score
        if best_score < threshold:
            return Resolution(None, best_score)
        return Resolution(best, best_score)

    def resolve_or_register(
        self, definition: PropertyDefinition, *, threshold: float = 0.82
    ) -> Resolution:
        resolution = self.resolve(definition.name, threshold=threshold)
        if resolution.definition is not None:
            return resolution
        return Resolution(self.register(definition), 1.0, created=True)

    def definitions(self) -> tuple[PropertyDefinition, ...]:
        return tuple(self._definitions.values())

