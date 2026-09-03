from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Iterable, Mapping

from .comparators import default_comparators
from .matcher import EntityMatcher
from .models import (
    Entity,
    Observation,
    PropertyConstraint,
    PropertyDefinition,
    QueryFrame,
    RelationValue,
    TemporalPolicy,
    ValueType,
)
from .persistence import PersistenceModel
from .persistence_data import PersistenceTrainingExample
from .property_registry import PropertyRegistry
from .selectors import MentionBasedSelector
from .temporal_grounding import TemporalGroundingCase


@dataclass(frozen=True, slots=True)
class ContextDynamics:
    subject_type: str
    state_predicate: str
    context_object: str
    scene: str
    hazard_per_hour: float
    split: str

    def features(self) -> tuple[str, ...]:
        return (
            "location",
            self.subject_type,
            self.state_predicate,
            self.context_object,
            self.scene,
        )


@dataclass(frozen=True, slots=True)
class CompositionalPersistenceDataset:
    train: tuple[PersistenceTrainingExample, ...]
    validation: tuple[PersistenceTrainingExample, ...]
    test: tuple[PersistenceTrainingExample, ...]
    contexts: tuple[ContextDynamics, ...]


@dataclass(frozen=True, slots=True)
class GroundingModelReport:
    name: str
    cases: int
    top1_accuracy: float
    mean_reciprocal_rank: float
    accuracy_by_tag: Mapping[str, float]


_SUBJECT_FACTORS = {"cup": 1.2, "book": 0.6, "tool": 1.6}
_RELATION_FACTORS = {
    ("on", "table"): 1.5,
    ("inside", "cabinet"): 0.3,
    ("on", "shelf"): 0.7,
}
_SCENE_FACTORS = {"quiet": 0.45, "busy": 2.2}
_VALIDATION_CONTEXTS = {
    ("cup", "inside", "cabinet", "busy"),
    ("book", "on", "shelf", "quiet"),
    ("tool", "on", "table", "quiet"),
}
_TEST_CONTEXTS = {
    ("cup", "on", "table", "busy"),
    ("book", "inside", "cabinet", "quiet"),
    ("tool", "on", "shelf", "busy"),
}


def _context_dynamics() -> tuple[ContextDynamics, ...]:
    contexts: list[ContextDynamics] = []
    for subject, subject_factor in _SUBJECT_FACTORS.items():
        for (predicate, context_object), relation_factor in _RELATION_FACTORS.items():
            for scene, scene_factor in _SCENE_FACTORS.items():
                key = (subject, predicate, context_object, scene)
                split = (
                    "test"
                    if key in _TEST_CONTEXTS
                    else "validation"
                    if key in _VALIDATION_CONTEXTS
                    else "train"
                )
                contexts.append(
                    ContextDynamics(
                        subject,
                        predicate,
                        context_object,
                        scene,
                        0.12 * subject_factor * relation_factor * scene_factor,
                        split,
                    )
                )
    return tuple(contexts)


def compositional_location_data(
    *,
    samples_per_context: int = 80,
    censor_after_hours: float = 16.0,
    weibull_shape: float = 1.0,
    censor_after_hours_by_split: Mapping[str, float] | None = None,
    seed: int = 41,
) -> CompositionalPersistenceDataset:
    """Generate entity-disjoint histories with held-out feature combinations.

    Every categorical value in validation and test occurs in training, while
    the complete context tuple does not. This distinguishes compositional
    generalisation from lookup-table memorisation.
    weibull_shape=1 recovers exponential dynamics; other positive values expose
    model misspecification under decreasing or increasing hazards.
    Split-specific censor horizons create duration shift without changing the
    latent context dynamics or leaking test outcomes.
    """

    if (
        samples_per_context <= 0
        or censor_after_hours < 0.5
        or not math.isfinite(weibull_shape)
        or weibull_shape <= 0
    ):
        raise ValueError("sample count/shape must be positive and censor horizon >= 0.5")
    rng = random.Random(seed)
    horizons = {
        "train": censor_after_hours,
        "validation": censor_after_hours,
        "test": censor_after_hours,
    }
    if censor_after_hours_by_split is not None:
        if set(censor_after_hours_by_split) != set(horizons) or any(
            not math.isfinite(value) or value < 0.5
            for value in censor_after_hours_by_split.values()
        ):
            raise ValueError("split censor horizons require train/validation/test >= 0.5")
        horizons.update(censor_after_hours_by_split)
    partitions: dict[str, list[PersistenceTrainingExample]] = {
        "train": [],
        "validation": [],
        "test": [],
    }
    contexts = _context_dynamics()
    for context in contexts:
        for index in range(samples_per_context):
            unit_exponential = rng.expovariate(1.0)
            transition_time = unit_exponential ** (1.0 / weibull_shape) / (
                context.hazard_per_hour
            )
            censor_time = rng.uniform(0.5, horizons[context.split])
            event_observed = transition_time <= censor_time
            duration_hours = min(transition_time, censor_time)
            partitions[context.split].append(
                PersistenceTrainingExample(
                    property_name="location",
                    subject_type=context.subject_type,
                    state_predicate=context.state_predicate,
                    context_object=context.context_object,
                    scene=context.scene,
                    duration_seconds=duration_hours * 3600.0,
                    event_observed=event_observed,
                    group_id=(
                        f"{context.split}-{context.subject_type}-"
                        f"{context.state_predicate}-{context.context_object}-"
                        f"{context.scene}-{index:04d}"
                    ),
                )
            )
    for rows in partitions.values():
        rng.shuffle(rows)
    return CompositionalPersistenceDataset(
        tuple(partitions["train"]),
        tuple(partitions["validation"]),
        tuple(partitions["test"]),
        contexts,
    )


def compositional_grounding_registry() -> PropertyRegistry:
    registry = PropertyRegistry()
    registry.register(PropertyDefinition("type", "semantic object category", ValueType.SEMANTIC))
    registry.register(PropertyDefinition("color", "surface color", ValueType.SEMANTIC))
    registry.register(PropertyDefinition("scene", "dynamic scene context", ValueType.CATEGORICAL))
    registry.register(
        PropertyDefinition(
            "location",
            "current spatial relation to a reference entity",
            ValueType.RELATION,
            metadata={"argument_roles": ["object"]},
            temporal_policy=TemporalPolicy(half_life_seconds=4 * 3600),
        )
    )
    return registry


def _relation(predicate: str, context_object: str) -> RelationValue:
    return RelationValue(predicate, {"object": context_object})


def compositional_grounding_benchmark(
    *,
    repetitions: int = 12,
) -> tuple[TemporalGroundingCase, ...]:
    """Create OOD decisions where a global half-life favours a stale entity."""

    if repetitions <= 0:
        raise ValueError("repetitions must be positive")
    pairs = (
        ("cup", "on", "table"),
        ("book", "inside", "cabinet"),
        ("tool", "on", "shelf"),
    )
    cases: list[TemporalGroundingCase] = []
    base_time = 1_900_000_000.0
    for pair_index, (subject, predicate, context_object) in enumerate(pairs):
        desired = _relation(predicate, context_object)
        wrong = _relation("inside" if predicate == "on" else "on", "storage")
        for index in range(repetitions):
            as_of = base_time + (pair_index * repetitions + index) * 3600
            target_age = (4.6 + 0.15 * (index % 5)) * 3600
            distractor_age = (1.8 + 0.1 * (index % 4)) * 3600
            target_id = f"stable-{subject}-{pair_index}-{index:02d}"
            distractor_id = f"volatile-{subject}-{pair_index}-{index:02d}"
            other_id = f"other-{subject}-{pair_index}-{index:02d}"
            target = Entity(
                target_id,
                {
                    "type": Observation(subject),
                    "color": Observation("red"),
                    "scene": Observation("quiet"),
                    "location": Observation(desired, confidence=0.98, timestamp=as_of - target_age),
                },
            )
            distractor = Entity(
                distractor_id,
                {
                    "type": Observation(subject),
                    "color": Observation("red"),
                    "scene": Observation("busy"),
                    "location": Observation(desired, timestamp=as_of - distractor_age),
                },
            )
            other = Entity(
                other_id,
                {
                    "type": Observation(subject),
                    "color": Observation("blue"),
                    "scene": Observation("quiet"),
                    "location": Observation(wrong, timestamp=as_of - 600),
                },
            )
            query = f"the red {subject} {predicate} the {context_object}"
            frame = QueryFrame(
                query,
                (
                    PropertyConstraint("type", subject, 0.20),
                    PropertyConstraint("color", "red", 0.15),
                    PropertyConstraint("location", desired, 0.65),
                ),
            )
            cases.append(
                TemporalGroundingCase(
                    f"compositional-{subject}-{index:02d}",
                    query,
                    (distractor, target, other),
                    target_id,
                    frame,
                    as_of,
                    {
                        distractor_id: {"location": wrong},
                        target_id: {"location": desired},
                        other_id: {"location": wrong},
                    },
                    (
                        "en",
                        "compositional-ood",
                        f"subject-{subject}",
                        "informative-scene-context",
                    ),
                )
            )
    return tuple(cases)


def evaluate_grounding_model(
    name: str,
    model: PersistenceModel,
    cases: Iterable[TemporalGroundingCase],
    registry: PropertyRegistry,
) -> GroundingModelReport:
    rows = tuple(cases)
    if not rows:
        raise ValueError("at least one grounding case is required")
    matcher = EntityMatcher(
        registry,
        default_comparators(),
        MentionBasedSelector(),
        persistence_model=model,
    )
    ranks: list[int] = []
    tagged: dict[str, list[bool]] = {}
    for case in rows:
        ranking = matcher.match(case.gold_frame, list(case.entities), as_of=case.as_of)
        ids = [result.entity_id for result in ranking]
        rank = ids.index(case.target_id) + 1
        ranks.append(rank)
        for tag in case.tags:
            tagged.setdefault(tag, []).append(rank == 1)
    return GroundingModelReport(
        name,
        len(rows),
        sum(rank == 1 for rank in ranks) / len(ranks),
        sum(1.0 / rank for rank in ranks) / len(ranks),
        {tag: sum(values) / len(values) for tag, values in sorted(tagged.items())},
    )
