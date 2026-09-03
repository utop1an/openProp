from __future__ import annotations

import math
from dataclasses import replace
from functools import lru_cache
from typing import Sequence

from .association import (
    EntityAssociationHypothesis,
    MultiEntityAssociator,
    VisualPropertyDetection,
)
from .models import Entity, QueryFrame


class GlobalOneToOneAssociator(MultiEntityAssociator):
    """Jointly assign same-frame, same-property detections to entities or null.

    Null is reusable, while each real entity can receive at most one detection
    for a property in a frame. Exact dynamic programming supplies both the MAP
    assignment and assignment marginals used by the ordinary safety gates.
    """

    def __init__(self, *args: object, max_entities: int = 12, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        if max_entities <= 0:
            raise ValueError("max_entities must be positive")
        self.max_entities = max_entities

    def associate_batch(
        self,
        detections: Sequence[VisualPropertyDetection],
        query: QueryFrame,
        entities: Sequence[Entity],
    ) -> tuple[EntityAssociationHypothesis, ...]:
        seen_ids: set[str] = set()
        independent: list[EntityAssociationHypothesis] = []
        for detection in detections:
            if detection.detection_id in seen_ids:
                raise ValueError(f"duplicate detection_id: {detection.detection_id}")
            seen_ids.add(detection.detection_id)
            independent.append(self.associate(detection, query, entities))

        groups: dict[tuple[str, str], list[int]] = {}
        for index, hypothesis in enumerate(independent):
            key = (
                hypothesis.detection.frame.frame_id,
                hypothesis.detection.property_name.casefold(),
            )
            groups.setdefault(key, []).append(index)

        result = list(independent)
        for indexes in groups.values():
            assigned = self._assign_group([independent[index] for index in indexes])
            for index, hypothesis in zip(indexes, assigned):
                result[index] = hypothesis
        return tuple(result)

    def _assign_group(
        self,
        hypotheses: Sequence[EntityAssociationHypothesis],
    ) -> tuple[EntityAssociationHypothesis, ...]:
        if not hypotheses:
            return ()
        ordered = tuple(
            sorted(hypotheses, key=lambda item: item.detection.detection_id)
        )
        entity_ids = tuple(
            sorted(candidate.entity_id for candidate in ordered[0].candidates)
        )
        if len(entity_ids) > self.max_entities:
            return tuple(
                replace(
                    hypothesis,
                    accepted_entity_id=None,
                    update_confidence=0.0,
                    reason="global assignment candidate limit exceeded",
                )
                for hypothesis in hypotheses
            )
        expected = set(entity_ids)
        for hypothesis in ordered:
            if {candidate.entity_id for candidate in hypothesis.candidates} != expected:
                raise ValueError(
                    "global assignment requires identical candidate sets per group"
                )

        probabilities = tuple(
            self._probability_row(hypothesis, entity_ids) for hypothesis in ordered
        )
        assignment, marginals = self._solve(probabilities)
        by_id: dict[str, EntityAssociationHypothesis] = {}
        for hypothesis, assigned_index, row in zip(ordered, assignment, marginals):
            by_id[hypothesis.detection.detection_id] = self._gate_assignment(
                hypothesis,
                entity_ids,
                assigned_index,
                row,
            )
        return tuple(by_id[item.detection.detection_id] for item in hypotheses)

    @staticmethod
    def _probability_row(
        hypothesis: EntityAssociationHypothesis,
        entity_ids: Sequence[str],
    ) -> tuple[float, ...]:
        by_id = {
            candidate.entity_id: candidate.posterior
            for candidate in hypothesis.candidates
        }
        return (hypothesis.null_probability,) + tuple(
            by_id[entity_id] for entity_id in entity_ids
        )

    @staticmethod
    def _solve(
        probabilities: Sequence[Sequence[float]],
    ) -> tuple[tuple[int, ...], tuple[tuple[float, ...], ...]]:
        """Return MAP option indexes and per-detection assignment marginals.

        Option zero is reusable null. Options one onward are one-to-one entity
        assignments. Ties prefer null and then lower stable entity indexes.
        """

        row_count = len(probabilities)
        option_count = len(probabilities[0])
        if option_count < 2:
            raise ValueError("global assignment requires at least one entity")
        if any(len(row) != option_count for row in probabilities):
            raise ValueError("global assignment probability rows must align")
        if any(
            not math.isfinite(value) or value < 0.0
            for row in probabilities
            for value in row
        ):
            raise ValueError("global assignment probabilities must be nonnegative")

        @lru_cache(maxsize=None)
        def suffix(index: int, used_mask: int) -> float:
            if index == row_count:
                return 1.0
            total = probabilities[index][0] * suffix(index + 1, used_mask)
            for option in range(1, option_count):
                bit = 1 << (option - 1)
                if used_mask & bit:
                    continue
                total += probabilities[index][option] * suffix(
                    index + 1, used_mask | bit
                )
            return total

        @lru_cache(maxsize=None)
        def best(index: int, used_mask: int) -> tuple[float, tuple[int, ...]]:
            if index == row_count:
                return 1.0, ()
            choices: list[tuple[float, tuple[int, ...]]] = []
            tail_score, tail = best(index + 1, used_mask)
            choices.append((probabilities[index][0] * tail_score, (0,) + tail))
            for option in range(1, option_count):
                bit = 1 << (option - 1)
                if used_mask & bit:
                    continue
                tail_score, tail = best(index + 1, used_mask | bit)
                choices.append(
                    (probabilities[index][option] * tail_score, (option,) + tail)
                )
            return max(choices, key=lambda item: (item[0], tuple(-x for x in item[1])))

        partition = suffix(0, 0)
        if not math.isfinite(partition) or partition <= 0.0:
            raise ValueError("global assignment has zero probability mass")

        forward: list[dict[int, float]] = [{0: 1.0}]
        for index in range(row_count):
            next_row: dict[int, float] = {}
            for used_mask, prefix_mass in forward[-1].items():
                next_row[used_mask] = (
                    next_row.get(used_mask, 0.0)
                    + prefix_mass * probabilities[index][0]
                )
                for option in range(1, option_count):
                    bit = 1 << (option - 1)
                    if used_mask & bit:
                        continue
                    next_mask = used_mask | bit
                    next_row[next_mask] = (
                        next_row.get(next_mask, 0.0)
                        + prefix_mass * probabilities[index][option]
                    )
            forward.append(next_row)

        marginal_rows: list[tuple[float, ...]] = []
        for index in range(row_count):
            masses = [0.0] * option_count
            for used_mask, prefix_mass in forward[index].items():
                masses[0] += (
                    prefix_mass
                    * probabilities[index][0]
                    * suffix(index + 1, used_mask)
                )
                for option in range(1, option_count):
                    bit = 1 << (option - 1)
                    if used_mask & bit:
                        continue
                    masses[option] += (
                        prefix_mass
                        * probabilities[index][option]
                        * suffix(index + 1, used_mask | bit)
                    )
            marginal_rows.append(
                tuple(min(1.0, max(0.0, value / partition)) for value in masses)
            )
        return best(0, 0)[1], tuple(marginal_rows)

    def _gate_assignment(
        self,
        hypothesis: EntityAssociationHypothesis,
        entity_ids: Sequence[str],
        assigned_index: int,
        marginals: Sequence[float],
    ) -> EntityAssociationHypothesis:
        detection = hypothesis.detection
        candidate_index = {item.entity_id: item for item in hypothesis.candidates}
        candidates = tuple(
            sorted(
                (
                    replace(candidate_index[entity_id], posterior=marginals[index + 1])
                    for index, entity_id in enumerate(entity_ids)
                ),
                key=lambda item: (-item.posterior, item.entity_id),
            )
        )
        null_probability = marginals[0]
        assigned_entity = None if assigned_index == 0 else entity_ids[assigned_index - 1]
        assigned_probability = marginals[assigned_index]
        runner_up = max(
            value for index, value in enumerate(marginals) if index != assigned_index
        )
        definition = self.registry.resolve(detection.property_name).definition
        assert definition is not None
        accepted_entity_id: str | None = None
        reason = "global assignment selected null"
        if assigned_entity is not None:
            if detection.detection_confidence < self.policy.minimum_detection_confidence:
                reason = "detection confidence below policy"
            elif detection.value_confidence < self.policy.minimum_value_confidence:
                reason = "value confidence below policy"
            elif assigned_probability < self.policy.acceptance_threshold:
                reason = "global assignment below acceptance threshold"
            elif assigned_probability - runner_up < self.policy.margin_threshold:
                reason = "global assignment margin below policy"
            elif (
                not definition.update_policy.allow_visual_updates
                or not definition.update_policy.permits_source(detection.frame.source)
            ):
                reason = "property update policy rejects the visual source or modality"
            else:
                accepted_entity_id = assigned_entity
                reason = "accepted by global one-to-one assignment and safety gates"

        update_confidence = 0.0
        if accepted_entity_id is not None:
            update_confidence = (
                detection.detection_confidence
                * detection.value_confidence
                * assigned_probability
                * self.policy.reliability_for(detection.frame.source)
            )
            if update_confidence < definition.update_policy.minimum_confidence:
                accepted_entity_id = None
                update_confidence = 0.0
                reason = "combined confidence below property update policy"
        return replace(
            hypothesis,
            candidates=candidates,
            null_probability=null_probability,
            accepted_entity_id=accepted_entity_id,
            update_confidence=update_confidence,
            reason=reason,
            decision_entity_id=assigned_entity,
        )
