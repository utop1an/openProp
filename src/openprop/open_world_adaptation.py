from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Mapping

from .compositional_persistence import ContextDynamics
from .persistence_data import PersistenceTrainingExample


OPEN_WORLD_ADAPTATION_CONDITIONS: Mapping[str, str] = {
    "open_world_control": "source mechanism with neutral novel subject effects",
    "calibrated_novel_subject_reversal": (
        "source-unseen subject=bottle is present in target calibration and reverses"
    ),
    "uncalibrated_novel_subject_reversal": (
        "source-unseen subject=plate occurs only in target test and reverses"
    ),
    "pairwise_subject_scene_xor": (
        "known contexts reverse iff subject=cup XOR scene=busy"
    ),
    "three_way_subject_object_scene_latin": (
        "known contexts reverse iff subject, relation-object, and scene indices sum to zero modulo three"
    ),
}

OPEN_WORLD_PAIRWISE_PARTITIONS: tuple[tuple[int, ...], ...] = (
    (),
    (1,),
    (3,),
    (4,),
    (1, 3),
    (1, 4),
    (3, 4),
)
OPEN_WORLD_TRIPLE_PARTITIONS: tuple[tuple[int, ...], ...] = (
    *OPEN_WORLD_PAIRWISE_PARTITIONS,
    (1, 3, 4),
)

SOURCE_SEEN = "source_seen"
TARGET_CALIBRATED_NOVEL = "target_calibrated_novel"
TARGET_UNCALIBRATED_NOVEL = "target_uncalibrated_novel"
CALIBRATED_NOVEL_SUBJECT = "bottle"
UNCALIBRATED_NOVEL_SUBJECT = "plate"

_SUBJECT_FACTORS = {"cup": 1.2, "book": 0.6, "tool": 1.6}
_RELATION_FACTORS = {
    ("on", "table"): 1.5,
    ("inside", "cabinet"): 0.3,
    ("on", "shelf"): 0.7,
}
_SCENE_FACTORS = {"quiet": 0.45, "mixed": 1.0, "busy": 2.2}
_SUBJECT_INDEX = {value: index for index, value in enumerate(_SUBJECT_FACTORS)}
_OBJECT_INDEX = {
    relation[1]: index for index, relation in enumerate(_RELATION_FACTORS)
}
_SCENE_INDEX = {value: index for index, value in enumerate(_SCENE_FACTORS)}


@dataclass(frozen=True, slots=True)
class OpenWorldAdaptationDataset:
    train: tuple[PersistenceTrainingExample, ...]
    validation: tuple[PersistenceTrainingExample, ...]
    tests: Mapping[str, tuple[PersistenceTrainingExample, ...]]
    test_hazards: Mapping[str, Mapping[tuple[str, ...], float]]
    changed_contexts: Mapping[str, frozenset[tuple[str, ...]]]
    support_by_context: Mapping[tuple[str, ...], str]
    calibration_contexts: frozenset[tuple[str, ...]]
    test_only_contexts: frozenset[tuple[str, ...]]
    contexts: tuple[ContextDynamics, ...]

    def __post_init__(self) -> None:
        context_features = {context.features() for context in self.contexts}
        if set(self.support_by_context) != context_features:
            raise ValueError("every open-world context requires one support label")
        if self.calibration_contexts & self.test_only_contexts:
            raise ValueError("calibration and test-only contexts must be disjoint")
        if self.calibration_contexts | self.test_only_contexts != context_features:
            raise ValueError("calibration support must partition all target contexts")
        if set(self.tests) != set(OPEN_WORLD_ADAPTATION_CONDITIONS):
            raise ValueError("open-world condition registry and rows must match")


def _neutral_subject_factor() -> float:
    # Zero effect in a regularized log-linear source model corresponds to the
    # geometric center of the source-seen categorical effects, not an unknown
    # observation or a negative value.
    return math.exp(sum(math.log(value) for value in _SUBJECT_FACTORS.values()) / 3.0)


def _target_contexts() -> tuple[ContextDynamics, ...]:
    contexts: list[ContextDynamics] = []
    subjects = (
        *tuple(_SUBJECT_FACTORS),
        CALIBRATED_NOVEL_SUBJECT,
        UNCALIBRATED_NOVEL_SUBJECT,
    )
    for subject in subjects:
        subject_factor = _SUBJECT_FACTORS.get(subject, _neutral_subject_factor())
        for (predicate, context_object), relation_factor in _RELATION_FACTORS.items():
            for scene, scene_factor in _SCENE_FACTORS.items():
                contexts.append(
                    ContextDynamics(
                        subject,
                        predicate,
                        context_object,
                        scene,
                        0.12 * subject_factor * relation_factor * scene_factor,
                        "target",
                    )
                )
    return tuple(contexts)


def _is_changed(context: ContextDynamics, condition: str) -> bool:
    if condition == "open_world_control":
        return False
    if condition == "calibrated_novel_subject_reversal":
        return context.subject_type == CALIBRATED_NOVEL_SUBJECT
    if condition == "uncalibrated_novel_subject_reversal":
        return context.subject_type == UNCALIBRATED_NOVEL_SUBJECT
    if condition == "pairwise_subject_scene_xor":
        return context.subject_type not in {
            CALIBRATED_NOVEL_SUBJECT,
            UNCALIBRATED_NOVEL_SUBJECT,
        } and ((context.subject_type == "cup") != (context.scene == "busy"))
    if condition == "three_way_subject_object_scene_latin":
        if context.subject_type in {
            CALIBRATED_NOVEL_SUBJECT,
            UNCALIBRATED_NOVEL_SUBJECT,
        }:
            return False
        indices = (
            _SUBJECT_INDEX[context.subject_type],
            _OBJECT_INDEX[context.context_object],
            _SCENE_INDEX[context.scene],
        )
        return sum(indices) % 3 == 0
    raise KeyError(f"unknown open-world condition: {condition}")


def _condition_hazard(context: ContextDynamics, condition: str) -> float:
    if _is_changed(context, condition):
        return 0.12**2 / context.hazard_per_hour
    return context.hazard_per_hour


def _target_row(
    context: ContextDynamics,
    index: int,
    unit_exponential: float,
    censor_hours: float,
    hazard_per_hour: float,
    group_prefix: str = "open-target",
) -> PersistenceTrainingExample:
    transition_hours = unit_exponential / hazard_per_hour
    event_observed = transition_hours <= censor_hours
    return PersistenceTrainingExample(
        property_name="location",
        subject_type=context.subject_type,
        state_predicate=context.state_predicate,
        context_object=context.context_object,
        scene=context.scene,
        duration_seconds=min(transition_hours, censor_hours) * 3600.0,
        event_observed=event_observed,
        group_id=(
            f"{group_prefix}-{context.subject_type}-{context.state_predicate}-"
            f"{context.context_object}-{context.scene}-{index:04d}"
        ),
    )


def open_world_adaptation_data(
    *,
    samples_per_context: int = 48,
    censor_after_hours: float = 16.0,
    seed: int = 41,
) -> OpenWorldAdaptationDataset:
    """Generate paired target rows with calibrated and uncalibrated novel values.

    `bottle` is absent from source training but eligible for target calibration.
    `plate` is absent from both source training and target calibration, providing
    an identifiability control. A novel observed category is never rewritten to
    the missing-observation token `unknown`.
    """

    if samples_per_context <= 1:
        raise ValueError("samples_per_context must exceed one")
    if not math.isfinite(censor_after_hours) or censor_after_hours < 0.5:
        raise ValueError("censor_after_hours must be finite and at least 0.5")

    contexts = _target_contexts()
    known_contexts = tuple(
        context
        for context in contexts
        if context.subject_type in _SUBJECT_FACTORS
    )
    source_train: list[PersistenceTrainingExample] = []
    source_validation: list[PersistenceTrainingExample] = []
    source_rng = random.Random(seed + 10_000_003)
    validation_per_context = max(16, samples_per_context // 4)
    for context in known_contexts:
        for split, count, prefix in (
            (source_train, samples_per_context, "open-source-train"),
            (source_validation, validation_per_context, "open-source-validation"),
        ):
            split.extend(
                _target_row(
                    context,
                    index,
                    source_rng.expovariate(1.0),
                    source_rng.uniform(0.5, censor_after_hours),
                    context.hazard_per_hour,
                    prefix,
                )
                for index in range(count)
            )
    random.Random(seed + 10_100_003).shuffle(source_train)
    random.Random(seed + 10_200_003).shuffle(source_validation)
    support = {
        context.features(): (
            TARGET_CALIBRATED_NOVEL
            if context.subject_type == CALIBRATED_NOVEL_SUBJECT
            else TARGET_UNCALIBRATED_NOVEL
            if context.subject_type == UNCALIBRATED_NOVEL_SUBJECT
            else SOURCE_SEEN
        )
        for context in contexts
    }
    calibration_contexts = frozenset(
        features
        for features, label in support.items()
        if label != TARGET_UNCALIBRATED_NOVEL
    )
    test_only_contexts = frozenset(set(support) - calibration_contexts)

    rows_by_condition: dict[str, list[PersistenceTrainingExample]] = {
        condition: [] for condition in OPEN_WORLD_ADAPTATION_CONDITIONS
    }
    hazards: dict[str, dict[tuple[str, ...], float]] = {
        condition: {} for condition in OPEN_WORLD_ADAPTATION_CONDITIONS
    }
    changed: dict[str, set[tuple[str, ...]]] = {
        condition: set() for condition in OPEN_WORLD_ADAPTATION_CONDITIONS
    }
    rng = random.Random(seed + 11_000_003)
    for context in contexts:
        draws = tuple(
            (rng.expovariate(1.0), rng.uniform(0.5, censor_after_hours))
            for _ in range(samples_per_context)
        )
        for condition in OPEN_WORLD_ADAPTATION_CONDITIONS:
            hazard = _condition_hazard(context, condition)
            hazards[condition][context.features()] = hazard
            if _is_changed(context, condition):
                changed[condition].add(context.features())
            rows_by_condition[condition].extend(
                _target_row(context, index, unit, censor, hazard)
                for index, (unit, censor) in enumerate(draws)
            )

    order = list(range(len(contexts) * samples_per_context))
    random.Random(seed + 12_000_003).shuffle(order)
    tests = {
        condition: tuple(rows[index] for index in order)
        for condition, rows in rows_by_condition.items()
    }
    return OpenWorldAdaptationDataset(
        tuple(source_train),
        tuple(source_validation),
        tests,
        hazards,
        {name: frozenset(values) for name, values in changed.items()},
        support,
        calibration_contexts,
        test_only_contexts,
        contexts,
    )
