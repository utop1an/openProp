from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Any

from .models import ComparisonResult, PropertyConstraint, PropertyDefinition, RelationValue, ValueType

Comparator = Callable[[Any, Any, PropertyConstraint, PropertyDefinition], ComparisonResult]


def _tokens(value: Any) -> set[str]:
    return set(str(value).casefold().replace("_", " ").replace("-", " ").split())


def categorical(actual: Any, desired: Any, *_: Any) -> ComparisonResult:
    score = float(str(actual).casefold() == str(desired).casefold())
    return ComparisonResult(score, "exact categorical match" if score else "categorical mismatch")


def semantic(actual: Any, desired: Any, *_: Any) -> ComparisonResult:
    """Dependency-free baseline; replace with an embedding-backed comparator."""
    left, right = _tokens(actual), _tokens(desired)
    if not left or not right:
        return ComparisonResult(0.0, "empty semantic value")
    score = len(left & right) / len(left | right)
    return ComparisonResult(score, "token semantic baseline")


def numeric(
    actual: Any, desired: Any, constraint: PropertyConstraint, definition: PropertyDefinition
) -> ComparisonResult:
    try:
        distance = abs(float(actual) - float(desired))
    except (TypeError, ValueError):
        return ComparisonResult(0.0, "non-numeric value")
    scale = constraint.tolerance or float(definition.metadata.get("scale", 1.0))
    if scale <= 0:
        raise ValueError("numeric comparison scale must be positive")
    return ComparisonResult(math.exp(-distance / scale), f"numeric distance={distance:g}")


def vector(actual: Any, desired: Any, *_: Any) -> ComparisonResult:
    if not isinstance(actual, Sequence) or not isinstance(desired, Sequence):
        return ComparisonResult(0.0, "non-vector value")
    if len(actual) != len(desired) or not actual:
        return ComparisonResult(0.0, "vector dimensions differ")
    try:
        dot = sum(float(a) * float(b) for a, b in zip(actual, desired, strict=True))
        norm_a = math.sqrt(sum(float(a) ** 2 for a in actual))
        norm_b = math.sqrt(sum(float(b) ** 2 for b in desired))
    except (TypeError, ValueError):
        return ComparisonResult(0.0, "invalid vector value")
    if norm_a == 0 or norm_b == 0:
        return ComparisonResult(0.0, "zero vector")
    cosine = max(-1.0, min(1.0, dot / (norm_a * norm_b)))
    return ComparisonResult((cosine + 1.0) / 2.0, "cosine similarity")


def relation(actual: Any, desired: Any, *_: Any) -> ComparisonResult:
    if not isinstance(actual, RelationValue) or not isinstance(desired, RelationValue):
        return ComparisonResult(0.0, "invalid relation structure")
    predicate_score = semantic(actual.predicate, desired.predicate).score or 0.0
    desired_args = desired.arguments
    if not desired_args:
        argument_score = 1.0
    else:
        matches = sum(
            str(actual.arguments.get(role, "")).casefold() == str(value).casefold()
            for role, value in desired_args.items()
        )
        argument_score = matches / len(desired_args)
    return ComparisonResult(
        predicate_score * argument_score,
        f"predicate={predicate_score:.2f}, arguments={argument_score:.2f}",
    )


class ComparatorRegistry:
    def __init__(self) -> None:
        self._comparators: dict[str, Comparator] = {}

    def register(self, name: str, comparator: Comparator) -> None:
        self._comparators[name] = comparator

    def compare(
        self,
        definition: PropertyDefinition,
        actual: Any,
        constraint: PropertyConstraint,
    ) -> ComparisonResult:
        name = definition.comparator or definition.value_type.value
        try:
            comparator = self._comparators[name]
        except KeyError as error:
            raise KeyError(f"no comparator registered for {name!r}") from error
        return comparator(actual, constraint.desired_value, constraint, definition)


def default_comparators() -> ComparatorRegistry:
    registry = ComparatorRegistry()
    registry.register(ValueType.CATEGORICAL.value, categorical)
    registry.register(ValueType.SEMANTIC.value, semantic)
    registry.register(ValueType.NUMERIC.value, numeric)
    registry.register(ValueType.VECTOR.value, vector)
    registry.register(ValueType.RELATION.value, relation)
    registry.register(ValueType.ENTITY_REFERENCE.value, categorical)
    registry.register(ValueType.TEMPORAL.value, numeric)
    return registry

