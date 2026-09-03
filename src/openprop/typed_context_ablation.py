from __future__ import annotations

import math
import random
import statistics
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from .advanced_survival_evaluation import evaluate_survival_advanced
from .compositional_persistence import (
    CompositionalPersistenceDataset,
    GroundingModelReport,
    evaluate_grounding_model,
)
from .models import PropertyDefinition
from .property_registry import PropertyRegistry
from .statistical_persistence import FactorizedExponentialPersistenceModel
from .simultaneous_inference import paired_bootstrap_simultaneous_intervals
from .temporal_grounding import TemporalGroundingCase


@dataclass(frozen=True, slots=True)
class TypedContextCondition:
    name: str
    active_feature_indices: tuple[int, ...]
    active_groups: tuple[str, ...]


TYPED_CONTEXT_CONDITIONS = (
    TypedContextCondition("intercept_only", (), ()),
    TypedContextCondition("subject_only", (1,), ("subject",)),
    TypedContextCondition("relation_only", (2, 3), ("relation",)),
    TypedContextCondition("scene_only", (4,), ("scene",)),
    TypedContextCondition(
        "subject_relation", (1, 2, 3), ("subject", "relation")
    ),
    TypedContextCondition("subject_scene", (1, 4), ("subject", "scene")),
    TypedContextCondition(
        "relation_scene", (2, 3, 4), ("relation", "scene")
    ),
    TypedContextCondition(
        "full_context", (1, 2, 3, 4), ("subject", "relation", "scene")
    ),
)

_METRIC_DIRECTIONS = {
    "negative_log_likelihood": "lower",
    "concordance_index": "higher",
    "integrated_brier_score": "lower",
    "grounding_top1": "higher",
}


def evaluate_typed_context_seed(
    *,
    seed: int,
    dataset: CompositionalPersistenceDataset,
    grounding_cases: Iterable[TemporalGroundingCase],
    registry: PropertyRegistry,
    epochs: int = 1200,
) -> dict[str, Any]:
    """Evaluate the frozen typed-context matrix on one paired dataset draw."""

    if epochs <= 0:
        raise ValueError("epochs must be positive")
    cases = tuple(grounding_cases)
    if not cases:
        raise ValueError("at least one grounding case is required")
    conditions: dict[str, dict[str, Any]] = {}
    for condition in TYPED_CONTEXT_CONDITIONS:
        model = FactorizedExponentialPersistenceModel.fit(
            dataset.train,
            epochs=epochs,
            active_feature_indices=condition.active_feature_indices,
        )
        validation_scale = model.calibrate(dataset.validation)
        survival = evaluate_survival_advanced(model, dataset.test)
        grounding: GroundingModelReport = evaluate_grounding_model(
            condition.name,
            model,
            cases,
            registry,
        )
        conditions[condition.name] = {
            "active_feature_indices": list(condition.active_feature_indices),
            "active_groups": list(condition.active_groups),
            "validation_hazard_scale": validation_scale,
            "negative_log_likelihood": survival.negative_log_likelihood,
            "concordance_index": survival.concordance_index,
            "integrated_brier_score": survival.integrated_brier_score,
            "grounding_top1": grounding.top1_accuracy,
            "grounding_by_tag": dict(grounding.accuracy_by_tag),
        }
    return {"seed": seed, "conditions": conditions}


def _summary(values: tuple[float, ...]) -> dict[str, float]:
    return {
        "mean": statistics.mean(values),
        "standard_deviation": statistics.pstdev(values),
        "minimum": min(values),
        "maximum": max(values),
    }


def _bootstrap_mean_interval(
    values: tuple[float, ...],
    *,
    samples: int,
    seed: int,
) -> tuple[float, float]:
    if samples <= 0:
        raise ValueError("bootstrap samples must be positive")
    rng = random.Random(seed)
    estimates = sorted(
        statistics.mean(rng.choice(values) for _ in values) for _ in range(samples)
    )
    lower = estimates[int(0.025 * (samples - 1))]
    upper = estimates[int(0.975 * (samples - 1))]
    return lower, upper


def aggregate_typed_context_runs(
    runs: Iterable[Mapping[str, Any]],
    *,
    bootstrap_samples: int = 20_000,
    bootstrap_seed: int = 20260826,
) -> dict[str, Any]:
    """Aggregate paired seeds and orient every delta as a full-model advantage."""

    rows = tuple(runs)
    if not rows:
        raise ValueError("at least one typed-context run is required")
    seeds = tuple(int(row["seed"]) for row in rows)
    if len(set(seeds)) != len(seeds):
        raise ValueError("typed-context run seeds must be unique")
    expected = {condition.name for condition in TYPED_CONTEXT_CONDITIONS}
    for row in rows:
        conditions = row.get("conditions")
        if not isinstance(conditions, Mapping) or set(conditions) != expected:
            raise ValueError("every run must contain the frozen condition matrix")

    aggregate: dict[str, dict[str, dict[str, float]]] = {}
    for condition in TYPED_CONTEXT_CONDITIONS:
        aggregate[condition.name] = {}
        for metric in _METRIC_DIRECTIONS:
            values = tuple(
                float(row["conditions"][condition.name][metric]) for row in rows
            )
            if any(not math.isfinite(value) for value in values):
                raise ValueError("typed-context metrics must be finite")
            aggregate[condition.name][metric] = _summary(values)

    paired: dict[str, dict[str, Any]] = {}
    full_name = "full_context"
    alternatives = [
        condition for condition in TYPED_CONTEXT_CONDITIONS if condition.name != full_name
    ]
    for condition_index, condition in enumerate(alternatives):
        paired[condition.name] = {}
        for metric_index, (metric, direction) in enumerate(_METRIC_DIRECTIONS.items()):
            full_values = tuple(
                float(row["conditions"][full_name][metric]) for row in rows
            )
            alternative_values = tuple(
                float(row["conditions"][condition.name][metric]) for row in rows
            )
            advantages = tuple(
                alternative - full if direction == "lower" else full - alternative
                for full, alternative in zip(
                    full_values, alternative_values, strict=True
                )
            )
            lower, upper = _bootstrap_mean_interval(
                advantages,
                samples=bootstrap_samples,
                seed=bootstrap_seed + condition_index * 101 + metric_index,
            )
            paired[condition.name][metric] = {
                "orientation": "positive means full_context is better",
                "mean_full_advantage": statistics.mean(advantages),
                "bootstrap_95_ci": [lower, upper],
                "wins": sum(value > 0.0 for value in advantages),
                "ties": sum(value == 0.0 for value in advantages),
                "losses": sum(value < 0.0 for value in advantages),
            }
    primary_family = {}
    for condition_name in ("relation_scene", "subject_scene", "subject_relation"):
        primary_family[condition_name] = tuple(
            float(row["conditions"][condition_name]["negative_log_likelihood"])
            - float(row["conditions"][full_name]["negative_log_likelihood"])
            for row in rows
        )
    simultaneous = paired_bootstrap_simultaneous_intervals(
        primary_family,
        samples=bootstrap_samples,
        seed=bootstrap_seed + 10_000,
    )
    for condition_name, interval in simultaneous["intervals"].items():
        paired[condition_name]["negative_log_likelihood"][
            "simultaneous_bootstrap_95_ci"
        ] = interval
    simultaneous_summary = {
        key: value
        for key, value in simultaneous.items()
        if key != "intervals"
    }
    simultaneous_summary["metric"] = "negative_log_likelihood"
    simultaneous_summary["orientation"] = "positive means full_context is better"

    return {
        "seeds": list(seeds),
        "aggregate": aggregate,
        "paired_full_advantage": paired,
        "simultaneous_primary_component_inference": simultaneous_summary,
    }
