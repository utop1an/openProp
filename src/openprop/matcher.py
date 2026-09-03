from __future__ import annotations

import math
import time

from .comparators import ComparatorRegistry
from .models import Entity, MatchResult, ObservationState, PropertyEvidence, QueryFrame
from .persistence import ExponentialPersistenceModel, PersistenceModel
from .property_registry import PropertyRegistry
from .selectors import PropertySelector


class EntityMatcher:
    def __init__(
        self,
        properties: PropertyRegistry,
        comparators: ComparatorRegistry,
        selector: PropertySelector,
        *,
        coverage_power: float = 1.0,
        persistence_model: PersistenceModel | None = None,
    ) -> None:
        if not math.isfinite(coverage_power) or coverage_power < 0.0:
            raise ValueError("coverage_power must be finite and nonnegative")
        self.properties = properties
        self.comparators = comparators
        self.selector = selector
        self.coverage_power = float(coverage_power)
        self.persistence_model = persistence_model or ExponentialPersistenceModel()

    def match(
        self,
        query: QueryFrame,
        entities: list[Entity],
        *,
        as_of: float | None = None,
    ) -> list[MatchResult]:
        selected = self.selector.select(query, self.properties)
        if not selected:
            raise ValueError("the query selected no registered properties")
        if any(
            not math.isfinite(item.weight) or item.weight <= 0.0
            for item in selected
        ):
            raise ValueError("each selected property weight must be finite and positive")
        total_weight = sum(item.weight for item in selected)
        if not math.isfinite(total_weight) or total_weight <= 0.0:
            raise ValueError("selected property weight must be positive")

        evaluation_time = time.time() if as_of is None else as_of
        results: list[MatchResult] = []
        for entity in entities:
            evidence: list[PropertyEvidence] = []
            observed_weight = 0.0
            weighted_score = 0.0
            for item in selected:
                observation = entity.properties.get(item.canonical_name)
                if observation is None or observation.state is not ObservationState.OBSERVED:
                    state = observation.state if observation else ObservationState.UNKNOWN
                    evidence.append(
                        PropertyEvidence(item.canonical_name, item.weight, None, state, "no observed evidence")
                    )
                    continue

                definition = self.properties.get(item.canonical_name)
                assert definition is not None
                temporal = self.persistence_model.predict(
                    definition,
                    observation,
                    entity,
                    as_of=evaluation_time,
                )
                if not math.isfinite(temporal.freshness) or not 0.0 <= temporal.freshness <= 1.0:
                    raise ValueError("persistence freshness must be finite and in [0, 1]")
                comparison = self.comparators.compare(
                    definition, observation.value, item.constraint
                )
                if (
                    comparison.score is None
                    or not math.isfinite(comparison.score)
                    or not 0.0 <= comparison.score <= 1.0
                ):
                    raise ValueError("comparison score must be finite and in [0, 1]")
                effective_confidence = observation.confidence * temporal.freshness
                effective_weight = item.weight * effective_confidence
                observed_weight += effective_weight
                weighted_score += effective_weight * comparison.score
                evidence.append(
                    PropertyEvidence(
                        item.canonical_name,
                        item.weight,
                        comparison.score,
                        observation.state,
                        f"{comparison.reason}; {temporal.reason}",
                        effective_confidence,
                        temporal.freshness,
                        temporal.age_seconds,
                    )
                )

            match_score = weighted_score / observed_weight if observed_weight else 0.0
            coverage = observed_weight / total_weight
            score = match_score * (coverage**self.coverage_power)
            results.append(
                MatchResult(entity.entity_id, score, match_score, coverage, tuple(evidence))
            )
        return sorted(
            results,
            key=lambda result: (-result.score, result.entity_id),
        )

