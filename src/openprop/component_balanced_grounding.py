from __future__ import annotations

import math
import random
import statistics
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from .compositional_persistence import (
    CompositionalPersistenceDataset,
    GroundingModelReport,
    compositional_grounding_registry,
    evaluate_grounding_model,
)
from .models import Entity, Observation, PropertyConstraint, QueryFrame, RelationValue
from .property_registry import PropertyRegistry
from .statistical_persistence import FactorizedExponentialPersistenceModel
from .temporal_grounding import TemporalGroundingCase
from .simultaneous_inference import paired_bootstrap_simultaneous_intervals


DEVELOPMENT_SEEDS = (31, 41, 53, 67, 79, 83, 97, 109, 127, 149)
CONFIRMATION_SEEDS = (157, 163, 173, 181, 191, 199, 211, 223, 227, 239)

_SUBJECT_FACTORS = {"cup": 1.2, "book": 0.6, "tool": 1.6}
_RELATION_FACTORS = {
    ("on", "table"): 1.5,
    ("inside", "cabinet"): 0.3,
    ("on", "shelf"): 0.7,
}
_SCENE_FACTORS = {"quiet": 0.45, "busy": 2.2}
_THRESHOLD_MULTIPLIERS = (0.90, 0.95, 1.00, 1.05, 1.10)


@dataclass(frozen=True, slots=True)
class GroundingProbe:
    name: str
    base_threshold_per_hour: float
    subjects: tuple[str, ...]
    relations: tuple[tuple[str, str], ...]
    scenes: tuple[str, ...]


GROUNDING_PROBES = (
    GroundingProbe(
        "subject",
        0.16,
        tuple(_SUBJECT_FACTORS),
        (("on", "shelf"),),
        ("busy",),
    ),
    GroundingProbe(
        "relation",
        0.09,
        ("tool",),
        tuple(_RELATION_FACTORS),
        ("quiet",),
    ),
    GroundingProbe(
        "scene",
        0.12,
        ("cup",),
        (("on", "shelf"),),
        tuple(_SCENE_FACTORS),
    ),
)

GROUNDING_MODEL_CONDITIONS: Mapping[str, tuple[int, ...]] = {
    "intercept_only": (),
    "no_subject": (2, 3, 4),
    "no_relation": (1, 4),
    "no_scene": (1, 2, 3),
    "full_context": (1, 2, 3, 4),
}


def _relation(predicate: str, context_object: str) -> RelationValue:
    return RelationValue(predicate, {"object": context_object})


def _true_hazard(
    subject: str,
    predicate: str,
    context_object: str,
    scene: str,
) -> float:
    return (
        0.12
        * _SUBJECT_FACTORS[subject]
        * _RELATION_FACTORS[(predicate, context_object)]
        * _SCENE_FACTORS[scene]
    )


def component_balanced_grounding_benchmark(
    *,
    old_age_hours: float = 5.0,
    new_age_hours: float = 1.0,
    new_confidence: float = 0.45,
) -> tuple[TemporalGroundingCase, ...]:
    """Create analytically identified decisions for each typed context group.

    Within a case, both plausible candidates have exactly the same subject,
    queried relation, and scene. The probed factor changes only across cases.
    The old/new confidence ratio sets a known hazard crossover, so target labels
    follow the declared generator hazard without exposing truth to the matcher.
    """

    if (
        not math.isfinite(old_age_hours)
        or not math.isfinite(new_age_hours)
        or old_age_hours <= new_age_hours
        or new_age_hours < 0.0
        or not 0.0 < new_confidence < 1.0
    ):
        raise ValueError("grounding ages/confidence require old > new >= 0 and 0 < c < 1")
    age_gap = old_age_hours - new_age_hours
    cases: list[TemporalGroundingCase] = []
    base_time = 2_000_000_000.0
    case_index = 0
    for probe in GROUNDING_PROBES:
        for subject in probe.subjects:
            for predicate, context_object in probe.relations:
                for scene in probe.scenes:
                    hazard = _true_hazard(subject, predicate, context_object, scene)
                    desired = _relation(predicate, context_object)
                    wrong = _relation(
                        "inside" if predicate == "on" else "on", "elsewhere"
                    )
                    for repetition, multiplier in enumerate(_THRESHOLD_MULTIPLIERS):
                        threshold = probe.base_threshold_per_hour * multiplier
                        old_confidence = new_confidence * math.exp(threshold * age_gap)
                        if old_confidence > 1.0:
                            raise ValueError("analytic crossover requires confidence <= 1")
                        as_of = base_time + case_index * 3600.0
                        old_id = f"old-{probe.name}-{case_index:03d}"
                        new_id = f"new-{probe.name}-{case_index:03d}"
                        other_id = f"other-{probe.name}-{case_index:03d}"
                        old = Entity(
                            old_id,
                            {
                                "type": Observation(subject),
                                "color": Observation("red"),
                                "scene": Observation(scene),
                                "location": Observation(
                                    desired,
                                    confidence=old_confidence,
                                    timestamp=as_of - old_age_hours * 3600.0,
                                ),
                            },
                        )
                        new = Entity(
                            new_id,
                            {
                                "type": Observation(subject),
                                "color": Observation("red"),
                                "scene": Observation(scene),
                                "location": Observation(
                                    desired,
                                    confidence=new_confidence,
                                    timestamp=as_of - new_age_hours * 3600.0,
                                ),
                            },
                        )
                        other = Entity(
                            other_id,
                            {
                                "type": Observation(subject),
                                "color": Observation("blue"),
                                "scene": Observation(scene),
                                "location": Observation(
                                    wrong,
                                    confidence=1.0,
                                    timestamp=as_of,
                                ),
                            },
                        )
                        target_id = old_id if hazard < threshold else new_id
                        non_target_id = new_id if target_id == old_id else old_id
                        query = f"the red {subject} {predicate} the {context_object}"
                        frame = QueryFrame(
                            query,
                            (
                                PropertyConstraint("type", subject, 0.20),
                                PropertyConstraint("color", "red", 0.15),
                                PropertyConstraint("location", desired, 0.65),
                            ),
                        )
                        entities = (old, new, other)
                        if case_index % 2:
                            entities = (other, new, old)
                        cases.append(
                            TemporalGroundingCase(
                                f"balanced-{probe.name}-{case_index:03d}",
                                query,
                                entities,
                                target_id,
                                frame,
                                as_of,
                                {
                                    target_id: {"location": desired},
                                    non_target_id: {"location": wrong},
                                    other_id: {"location": wrong},
                                },
                                (
                                    "component-balanced",
                                    f"probe-{probe.name}",
                                    f"target-{'old' if target_id == old_id else 'new'}",
                                    f"threshold-{repetition}",
                                ),
                            )
                        )
                        case_index += 1
    return tuple(cases)


def evaluate_component_balanced_seed(
    *,
    seed: int,
    dataset: CompositionalPersistenceDataset,
    cases: Iterable[TemporalGroundingCase],
    registry: PropertyRegistry | None = None,
    epochs: int = 1200,
) -> dict[str, Any]:
    rows = tuple(cases)
    if not rows:
        raise ValueError("at least one component-balanced grounding case is required")
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    active_registry = registry or compositional_grounding_registry()
    conditions: dict[str, dict[str, Any]] = {}
    for name, indices in GROUNDING_MODEL_CONDITIONS.items():
        model = FactorizedExponentialPersistenceModel.fit(
            dataset.train,
            epochs=epochs,
            active_feature_indices=indices,
        )
        scale = model.calibrate(dataset.validation)
        grounding: GroundingModelReport = evaluate_grounding_model(
            name, model, rows, active_registry
        )
        conditions[name] = {
            "active_feature_indices": list(indices),
            "validation_hazard_scale": scale,
            "top1": grounding.top1_accuracy,
            "mean_reciprocal_rank": grounding.mean_reciprocal_rank,
            "top1_by_probe": {
                probe.name: grounding.accuracy_by_tag[f"probe-{probe.name}"]
                for probe in GROUNDING_PROBES
            },
            "top1_by_target_age": {
                age: grounding.accuracy_by_tag[f"target-{age}"]
                for age in ("old", "new")
            },
        }
    return {"seed": seed, "conditions": conditions}


def _bootstrap_mean_interval(
    values: tuple[float, ...], *, samples: int, seed: int
) -> tuple[float, float]:
    if samples <= 0:
        raise ValueError("bootstrap samples must be positive")
    rng = random.Random(seed)
    estimates = sorted(
        statistics.mean(rng.choice(values) for _ in values) for _ in range(samples)
    )
    return (
        estimates[int(0.025 * (samples - 1))],
        estimates[int(0.975 * (samples - 1))],
    )


def aggregate_component_balanced_runs(
    runs: Iterable[Mapping[str, Any]],
    *,
    bootstrap_samples: int = 20_000,
    bootstrap_seed: int = 20260827,
) -> dict[str, Any]:
    rows = tuple(runs)
    if not rows:
        raise ValueError("at least one component-balanced run is required")
    seeds = tuple(int(row["seed"]) for row in rows)
    if len(set(seeds)) != len(seeds):
        raise ValueError("component-balanced run seeds must be unique")
    expected = set(GROUNDING_MODEL_CONDITIONS)
    for row in rows:
        conditions = row.get("conditions")
        if not isinstance(conditions, Mapping) or set(conditions) != expected:
            raise ValueError("every run must contain the frozen grounding matrix")

    aggregate: dict[str, Any] = {}
    for name in GROUNDING_MODEL_CONDITIONS:
        overall = tuple(float(row["conditions"][name]["top1"]) for row in rows)
        aggregate[name] = {
            "top1": {
                "mean": statistics.mean(overall),
                "standard_deviation": statistics.pstdev(overall),
                "minimum": min(overall),
                "maximum": max(overall),
            },
            "top1_by_probe": {},
        }
        for probe in GROUNDING_PROBES:
            values = tuple(
                float(row["conditions"][name]["top1_by_probe"][probe.name])
                for row in rows
            )
            aggregate[name]["top1_by_probe"][probe.name] = {
                "mean": statistics.mean(values),
                "standard_deviation": statistics.pstdev(values),
                "minimum": min(values),
                "maximum": max(values),
            }

    comparisons: dict[str, Any] = {}
    comparison_vectors: dict[str, tuple[float, ...]] = {}
    for index, (probe, ablation) in enumerate(
        (("subject", "no_subject"), ("relation", "no_relation"), ("scene", "no_scene"))
    ):
        full = tuple(
            float(row["conditions"]["full_context"]["top1_by_probe"][probe])
            for row in rows
        )
        reduced = tuple(
            float(row["conditions"][ablation]["top1_by_probe"][probe])
            for row in rows
        )
        advantages = tuple(
            full_value - reduced_value
            for full_value, reduced_value in zip(full, reduced, strict=True)
        )
        comparison_vectors[probe] = advantages
        lower, upper = _bootstrap_mean_interval(
            advantages,
            samples=bootstrap_samples,
            seed=bootstrap_seed + index,
        )
        comparisons[probe] = {
            "ablation": ablation,
            "mean_full_advantage": statistics.mean(advantages),
            "bootstrap_95_ci": [lower, upper],
            "wins": sum(value > 0.0 for value in advantages),
            "ties": sum(value == 0.0 for value in advantages),
            "losses": sum(value < 0.0 for value in advantages),
        }
    simultaneous = paired_bootstrap_simultaneous_intervals(
        comparison_vectors,
        samples=bootstrap_samples,
        seed=bootstrap_seed + 10_000,
    )
    for probe, interval in simultaneous["intervals"].items():
        comparisons[probe]["simultaneous_bootstrap_95_ci"] = interval
    simultaneous_summary = {
        key: value
        for key, value in simultaneous.items()
        if key != "intervals"
    }
    simultaneous_summary["metric"] = "axis-isolated top1 full-model advantage"
    simultaneous_summary["orientation"] = "positive means full_context is better"

    return {
        "seeds": list(seeds),
        "aggregate": aggregate,
        "paired_probe_advantage": comparisons,
        "simultaneous_probe_inference": simultaneous_summary,
    }
