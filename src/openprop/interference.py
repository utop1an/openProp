from __future__ import annotations

from dataclasses import dataclass

from .benchmark import BenchmarkCase, core_benchmark
from .models import Entity, ObservationState, PropertyConstraint, QueryFrame, RelationValue


@dataclass(frozen=True, slots=True)
class InterferenceCase:
    case_id: str
    query: str
    entities: tuple[Entity, ...]
    target_id: str
    gold_frame: QueryFrame
    tags: tuple[str, ...]
    distractor_constraints: tuple[PropertyConstraint, ...]
    distractor_entity_id: str


def _display(value: object) -> str:
    if isinstance(value, RelationValue):
        arguments = ",".join(f"{role}={item}" for role, item in value.arguments.items())
        return f"{value.predicate}({arguments})"
    return str(value)


def _distractor_for(case: BenchmarkCase) -> Entity:
    """Choose a deterministic non-target entity with the richest evidence."""
    candidates = [entity for entity in case.entities if entity.entity_id != case.target_id]
    return max(
        candidates,
        key=lambda entity: (
            sum(
                observation.state is ObservationState.OBSERVED
                for observation in entity.properties.values()
            ),
            entity.entity_id,
        ),
    )


def interference_benchmark(*, distractor_weight: float = 0.03) -> tuple[InterferenceCase, ...]:
    """Add another entity's attributes as explicitly irrelevant context.

    The clean gold frame remains unchanged. Each distractor constraint receives
    a very low relevance for weighted scoring; the equal-weight ablation later
    promotes the same constraints to weight one.
    """
    if not 0.0 <= distractor_weight < 0.1:
        raise ValueError("distractor_weight must be in [0, 0.1)")

    noisy_cases: list[InterferenceCase] = []
    for case in core_benchmark():
        distractor = _distractor_for(case)
        constraints = tuple(
            PropertyConstraint(name, observation.value, distractor_weight)
            for name, observation in distractor.properties.items()
            if observation.state is ObservationState.OBSERVED
        )
        note = ", ".join(
            f"{constraint.property_name}={_display(constraint.desired_value)}"
            for constraint in constraints
        )
        query = (
            f"{case.query}。无关背景记录（不要用于目标指代）: "
            f"entity={distractor.entity_id}, {note}"
        )
        noisy_cases.append(
            InterferenceCase(
                case_id=f"{case.case_id}-noise",
                query=query,
                entities=case.entities,
                target_id=case.target_id,
                gold_frame=QueryFrame(query, case.gold_frame.constraints),
                tags=(*case.tags, "irrelevant-attributes"),
                distractor_constraints=constraints,
                distractor_entity_id=distractor.entity_id,
            )
        )
    return tuple(noisy_cases)
