from __future__ import annotations

import math
import random
from dataclasses import dataclass, replace
from typing import Iterable, Sequence

from .association import (
    AssociationPolicy,
    EntityAssociationHypothesis,
    MultiEntityAssociator,
    VisualPropertyDetection,
)
from .comparators import default_comparators
from .models import (
    Entity,
    Observation,
    PropertyConstraint,
    PropertyDefinition,
    PropertyUpdatePolicy,
    QueryFrame,
    ValueType,
)
from .property_registry import PropertyRegistry
from .selectors import MentionBasedSelector
from .vlm import VisualFrame


ASSOCIATION_CONDITIONS = ("strong", "ambiguous", "misleading", "null")


@dataclass(frozen=True, slots=True)
class AssociationBenchmarkCase:
    case_id: str
    group_id: str
    condition: str
    query: QueryFrame
    paraphrase_query: QueryFrame
    entities: tuple[Entity, ...]
    detection: VisualPropertyDetection
    target_entity_id: str | None

    def __post_init__(self) -> None:
        if not self.case_id.strip() or not self.group_id.strip():
            raise ValueError("case_id and group_id cannot be empty")
        if self.condition not in ASSOCIATION_CONDITIONS:
            raise ValueError(f"unknown association condition: {self.condition}")
        entity_ids = {entity.entity_id for entity in self.entities}
        if len(entity_ids) != len(self.entities):
            raise ValueError("benchmark entities must have unique IDs")
        if entity_ids != set(self.detection.frame.candidate_entity_ids):
            raise ValueError("benchmark entities must exactly match frame candidates")
        if self.target_entity_id is not None and self.target_entity_id not in entity_ids:
            raise ValueError("target entity must be a frame candidate")
        if self.query.constraints != self.paraphrase_query.constraints:
            raise ValueError("paraphrase queries must preserve constraints")
        for entity in self.entities:
            if "current_truth" in entity.properties or "target" in entity.properties:
                raise ValueError("evaluation truth cannot enter entity properties")


@dataclass(frozen=True, slots=True)
class AssociationBenchmarkSplit:
    calibration: tuple[AssociationBenchmarkCase, ...]
    test: tuple[AssociationBenchmarkCase, ...]

    def __post_init__(self) -> None:
        calibration_groups = {case.group_id for case in self.calibration}
        test_groups = {case.group_id for case in self.test}
        if not self.calibration or not self.test:
            raise ValueError("calibration and test splits cannot be empty")
        if calibration_groups & test_groups:
            raise ValueError("calibration and test groups must be disjoint")


@dataclass(frozen=True, slots=True)
class AssociationDecisionRecord:
    case_id: str
    condition: str
    target_entity_id: str | None
    accepted_entity_id: str | None
    top_posterior: float
    null_probability: float
    reason: str
    correct_update: bool
    false_update: bool
    candidate_order_invariant: bool
    query_paraphrase_invariant: bool


@dataclass(frozen=True, slots=True)
class AssociationEvaluation:
    total: int
    target_present: int
    accepted: int
    correct_updates: int
    false_updates: int
    abstentions: int
    correct_update_rate: float
    target_recall: float
    selective_accuracy: float
    false_update_rate: float
    abstention_rate: float
    null_false_positive_rate: float
    candidate_order_invariance: float
    query_paraphrase_invariance: float
    by_condition: dict[str, dict[str, float | int]]
    records: tuple[AssociationDecisionRecord, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "total": self.total,
            "target_present": self.target_present,
            "accepted": self.accepted,
            "correct_updates": self.correct_updates,
            "false_updates": self.false_updates,
            "abstentions": self.abstentions,
            "correct_update_rate": self.correct_update_rate,
            "target_recall": self.target_recall,
            "selective_accuracy": self.selective_accuracy,
            "false_update_rate": self.false_update_rate,
            "abstention_rate": self.abstention_rate,
            "null_false_positive_rate": self.null_false_positive_rate,
            "candidate_order_invariance": self.candidate_order_invariance,
            "query_paraphrase_invariance": self.query_paraphrase_invariance,
            "by_condition": self.by_condition,
            "records": [
                {
                    "case_id": row.case_id,
                    "condition": row.condition,
                    "target_entity_id": row.target_entity_id,
                    "accepted_entity_id": row.accepted_entity_id,
                    "top_posterior": row.top_posterior,
                    "null_probability": row.null_probability,
                    "reason": row.reason,
                    "correct_update": row.correct_update,
                    "false_update": row.false_update,
                    "candidate_order_invariant": row.candidate_order_invariant,
                    "query_paraphrase_invariant": row.query_paraphrase_invariant,
                }
                for row in self.records
            ],
        }


@dataclass(frozen=True, slots=True)
class AssociationCalibrationResult:
    policy: AssociationPolicy
    validation: AssociationEvaluation
    searched_policies: int
    feasible_policies: int
    max_false_update_rate: float


def association_benchmark_registry() -> PropertyRegistry:
    registry = PropertyRegistry()
    registry.register(PropertyDefinition("type", "object category", ValueType.CATEGORICAL))
    registry.register(PropertyDefinition("color", "surface color", ValueType.CATEGORICAL))
    registry.register(
        PropertyDefinition(
            "motion_state",
            "observed object motion",
            ValueType.CATEGORICAL,
            metadata={"allowed_values": ("moved", "stationary")},
            update_policy=PropertyUpdatePolicy(minimum_confidence=0.25),
        )
    )
    return registry


def default_association_benchmark_associator(
    registry: PropertyRegistry,
    *,
    policy: AssociationPolicy | None = None,
) -> MultiEntityAssociator:
    return MultiEntityAssociator(
        registry,
        default_comparators(),
        MentionBasedSelector(),
        policy=policy,
    )


def association_benchmark_split(
    *,
    seed: int = 20260831,
    calibration_per_condition: int = 20,
    test_per_condition: int = 40,
) -> AssociationBenchmarkSplit:
    if calibration_per_condition <= 0 or test_per_condition <= 0:
        raise ValueError("split case counts must be positive")
    calibration = _association_cases(
        seed=seed,
        split_name="calibration",
        cases_per_condition=calibration_per_condition,
    )
    test = _association_cases(
        seed=seed + 1,
        split_name="test",
        cases_per_condition=test_per_condition,
    )
    return AssociationBenchmarkSplit(calibration, test)


def _association_cases(
    *,
    seed: int,
    split_name: str,
    cases_per_condition: int,
) -> tuple[AssociationBenchmarkCase, ...]:
    rng = random.Random(seed)
    cases: list[AssociationBenchmarkCase] = []
    base_time = 2_200_000_000.0
    for condition_index, condition in enumerate(ASSOCIATION_CONDITIONS):
        for repetition in range(cases_per_condition):
            case_id = f"association-{split_name}-{condition}-{repetition:03d}"
            entity_ids = tuple(f"{case_id}-cup-{index}" for index in range(3))
            target_index = None if condition == "null" else rng.randrange(2)
            target_id = None if target_index is None else entity_ids[target_index]
            red_other = (
                1 - target_index if target_index is not None else rng.randrange(2)
            )
            frame_time = base_time + (
                condition_index * cases_per_condition + repetition
            ) * 10.0
            candidate_order = list(entity_ids)
            rng.shuffle(candidate_order)
            frame = VisualFrame(
                f"{case_id}-frame",
                "data:image/png;base64,AA==",
                frame_time,
                "synthetic-camera",
                tuple(candidate_order),
            )
            visual, track = _condition_affinities(
                rng,
                condition,
                entity_ids,
                target_index,
                red_other,
            )
            detection = VisualPropertyDetection(
                f"{case_id}-detection",
                frame,
                "motion_state",
                "moved",
                rng.uniform(0.88, 0.98),
                rng.uniform(0.86, 0.97),
                visual,
                track_id=None if not track else f"{case_id}-track",
                track_affinities=track,
            )
            entities = tuple(
                Entity(
                    entity_id,
                    {
                        "type": Observation("cup", timestamp=frame_time - 20.0),
                        "color": Observation(
                            "red" if index < 2 else "blue",
                            timestamp=frame_time - 20.0,
                        ),
                    },
                )
                for index, entity_id in enumerate(entity_ids)
            )
            query = QueryFrame(
                "the red cup",
                (
                    PropertyConstraint("type", "cup", 0.45),
                    PropertyConstraint("color", "red", 0.55),
                ),
            )
            paraphrase = QueryFrame("move that crimson cup", query.constraints)
            cases.append(
                AssociationBenchmarkCase(
                    case_id,
                    f"{split_name}-group-{condition}-{repetition:03d}",
                    condition,
                    query,
                    paraphrase,
                    entities,
                    detection,
                    target_id,
                )
            )
    return tuple(cases)


def _condition_affinities(
    rng: random.Random,
    condition: str,
    entity_ids: tuple[str, ...],
    target_index: int | None,
    red_other: int,
) -> tuple[dict[str, float], dict[str, float]]:
    visual = {entity_id: rng.uniform(0.01, 0.08) for entity_id in entity_ids}
    track: dict[str, float] = {}
    if condition == "strong":
        assert target_index is not None
        visual[entity_ids[target_index]] = rng.uniform(0.86, 0.98)
        visual[entity_ids[red_other]] = rng.uniform(0.05, 0.22)
        track = {
            entity_ids[target_index]: rng.uniform(0.88, 0.98),
            entity_ids[red_other]: rng.uniform(0.04, 0.20),
            entity_ids[2]: rng.uniform(0.01, 0.08),
        }
    elif condition == "ambiguous":
        assert target_index is not None
        centre = rng.uniform(0.62, 0.82)
        visual[entity_ids[target_index]] = centre
        visual[entity_ids[red_other]] = max(
            0.0,
            min(1.0, centre + rng.uniform(-0.035, 0.035)),
        )
    elif condition == "misleading":
        assert target_index is not None
        visual[entity_ids[target_index]] = rng.uniform(0.35, 0.55)
        visual[entity_ids[red_other]] = rng.uniform(0.72, 0.92)
        track = {
            entity_ids[target_index]: rng.uniform(0.45, 0.65),
            entity_ids[red_other]: rng.uniform(0.68, 0.90),
            entity_ids[2]: rng.uniform(0.01, 0.08),
        }
    elif condition != "null":
        raise ValueError(f"unknown association condition: {condition}")
    return visual, track


def evaluate_association(
    associator: MultiEntityAssociator,
    cases: Iterable[AssociationBenchmarkCase],
) -> AssociationEvaluation:
    rows = tuple(cases)
    if not rows:
        raise ValueError("at least one association case is required")
    records: list[AssociationDecisionRecord] = []
    for case in rows:
        hypothesis = associator.associate(
            case.detection,
            case.query,
            case.entities,
        )
        order_invariant = _candidate_order_invariant(associator, case, hypothesis)
        paraphrase = associator.associate(
            case.detection,
            case.paraphrase_query,
            case.entities,
        )
        paraphrase_invariant = _same_decision(hypothesis, paraphrase)
        accepted_id = hypothesis.accepted_entity_id
        correct = case.target_entity_id is not None and accepted_id == case.target_entity_id
        false = accepted_id is not None and not correct
        records.append(
            AssociationDecisionRecord(
                case.case_id,
                case.condition,
                case.target_entity_id,
                accepted_id,
                hypothesis.candidates[0].posterior,
                hypothesis.null_probability,
                hypothesis.reason,
                correct,
                false,
                order_invariant,
                paraphrase_invariant,
            )
        )
    return _summarize_records(tuple(records))


def _candidate_order_invariant(
    associator: MultiEntityAssociator,
    case: AssociationBenchmarkCase,
    original: EntityAssociationHypothesis,
) -> bool:
    reversed_frame = replace(
        case.detection.frame,
        candidate_entity_ids=tuple(reversed(case.detection.frame.candidate_entity_ids)),
    )
    reversed_detection = replace(case.detection, frame=reversed_frame)
    reversed_hypothesis = associator.associate(
        reversed_detection,
        case.query,
        tuple(reversed(case.entities)),
    )
    return _same_decision(original, reversed_hypothesis)


def _same_decision(
    left: EntityAssociationHypothesis,
    right: EntityAssociationHypothesis,
    *,
    tolerance: float = 1e-12,
) -> bool:
    if left.accepted_entity_id != right.accepted_entity_id:
        return False
    if not math.isclose(
        left.null_probability,
        right.null_probability,
        rel_tol=0.0,
        abs_tol=tolerance,
    ):
        return False
    left_scores = {row.entity_id: row.posterior for row in left.candidates}
    right_scores = {row.entity_id: row.posterior for row in right.candidates}
    return left_scores.keys() == right_scores.keys() and all(
        math.isclose(
            left_scores[entity_id],
            right_scores[entity_id],
            rel_tol=0.0,
            abs_tol=tolerance,
        )
        for entity_id in left_scores
    )


def _summarize_records(
    records: tuple[AssociationDecisionRecord, ...],
) -> AssociationEvaluation:
    total = len(records)
    target_present = sum(row.target_entity_id is not None for row in records)
    accepted = sum(row.accepted_entity_id is not None for row in records)
    correct = sum(row.correct_update for row in records)
    false = sum(row.false_update for row in records)
    abstentions = total - accepted
    null_rows = tuple(row for row in records if row.target_entity_id is None)
    null_false = sum(row.accepted_entity_id is not None for row in null_rows)

    by_condition: dict[str, dict[str, float | int]] = {}
    for condition in ASSOCIATION_CONDITIONS:
        subset = tuple(row for row in records if row.condition == condition)
        count = len(subset)
        by_condition[condition] = {
            "total": count,
            "accepted": sum(row.accepted_entity_id is not None for row in subset),
            "correct_updates": sum(row.correct_update for row in subset),
            "false_updates": sum(row.false_update for row in subset),
            "abstention_rate": (
                sum(row.accepted_entity_id is None for row in subset) / count
                if count
                else 0.0
            ),
        }

    return AssociationEvaluation(
        total,
        target_present,
        accepted,
        correct,
        false,
        abstentions,
        correct / total,
        correct / target_present if target_present else 0.0,
        correct / accepted if accepted else 0.0,
        false / total,
        abstentions / total,
        null_false / len(null_rows) if null_rows else 0.0,
        sum(row.candidate_order_invariant for row in records) / total,
        sum(row.query_paraphrase_invariant for row in records) / total,
        by_condition,
        records,
    )


def calibrate_association_policy(
    base_associator: MultiEntityAssociator,
    calibration_cases: Sequence[AssociationBenchmarkCase],
    *,
    acceptance_thresholds: Sequence[float] = (0.55, 0.65, 0.75, 0.80, 0.85, 0.90),
    margin_thresholds: Sequence[float] = (0.05, 0.10, 0.15, 0.20, 0.25),
    max_false_update_rate: float = 0.0,
) -> AssociationCalibrationResult:
    if not calibration_cases:
        raise ValueError("calibration cases cannot be empty")
    if not 0.0 <= max_false_update_rate <= 1.0:
        raise ValueError("max_false_update_rate must be in [0, 1]")
    if not acceptance_thresholds or not margin_thresholds:
        raise ValueError("threshold grids cannot be empty")
    candidates: list[tuple[AssociationPolicy, AssociationEvaluation]] = []
    for acceptance in acceptance_thresholds:
        for margin in margin_thresholds:
            policy = replace(
                base_associator.policy,
                acceptance_threshold=float(acceptance),
                margin_threshold=float(margin),
            )
            associator = MultiEntityAssociator(
                base_associator.registry,
                base_associator.matcher.comparators,
                base_associator.matcher.selector,
                policy=policy,
                persistence_model=base_associator.matcher.persistence_model,
            )
            candidates.append(
                (policy, evaluate_association(associator, calibration_cases))
            )
    feasible = [
        item
        for item in candidates
        if item[1].false_update_rate <= max_false_update_rate
    ]
    if not feasible:
        raise ValueError(
            "no association policy satisfies the calibration false-update gate"
        )
    pool = feasible or candidates
    selected_policy, selected_report = min(
        pool,
        key=lambda item: (
            item[1].false_updates,
            -item[1].correct_updates,
            item[1].abstentions,
            -item[0].acceptance_threshold,
            -item[0].margin_threshold,
        ),
    )
    return AssociationCalibrationResult(
        selected_policy,
        selected_report,
        len(candidates),
        len(feasible),
        max_false_update_rate,
    )
