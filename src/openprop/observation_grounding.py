from __future__ import annotations

import math
import random
import statistics
from dataclasses import replace
from typing import Any, Iterable, Mapping

from .compositional_persistence import (
    compositional_grounding_registry,
    evaluate_grounding_model,
)
from .models import Entity, Observation, PropertyConstraint, QueryFrame, RelationValue
from .observation_process import ObservationProcessDataset, observation_process_data
from .property_registry import PropertyRegistry
from .statistical_persistence import PerContextExponentialPersistenceModel
from .temporal_grounding import TemporalGroundingCase


DEVELOPMENT_SEEDS = (101, 211, 307, 401, 503)
CONFIRMATION_SEEDS = (607, 701, 809, 907, 1009, 1103, 1201, 1301, 1409, 1511)
SCENE_SCHEDULES: Mapping[str, float] = {
    "frequent-scene": 0.5,
    "sparse-scene": 4.0,
}
CONDITIONS = ("naive", "interval_aware", "oracle")


def _relation(predicate: str, context_object: str) -> RelationValue:
    return RelationValue(predicate, {"object": context_object})


def scene_conditioned_observation_data(
    *,
    samples_per_scene: int = 600,
    test_samples_per_scene: int = 20,
    true_hazard_per_hour: float = 0.25,
    seed: int = 101,
) -> ObservationProcessDataset:
    """Pair equal scene dynamics with unequal inspection schedules.

    The base observation-process generator records the inspection schedule in
    its final typed context column. This adapter gives those strata semantic
    scene names, so the matcher receives an ordinary scene context while the
    inspection history remains confined to the survival training records.
    """

    dataset = observation_process_data(
        samples_per_schedule=samples_per_scene,
        test_samples_per_schedule=test_samples_per_scene,
        true_hazard_per_hour=true_hazard_per_hour,
        inspection_intervals_hours=tuple(SCENE_SCHEDULES.values()),
        seed=seed,
    )
    schedule_to_scene = {
        f"inspection-{interval:g}h": scene
        for scene, interval in SCENE_SCHEDULES.items()
    }

    def relabel(rows: Iterable[Any]) -> tuple[Any, ...]:
        return tuple(replace(row, scene=schedule_to_scene[row.scene]) for row in rows)

    return ObservationProcessDataset(
        relabel(dataset.interval_train),
        relabel(dataset.naive_train),
        relabel(dataset.exact_test),
        dataset.true_hazard_per_hour,
        dataset.inspection_intervals_hours,
    )


def observation_grounding_benchmark(
    *,
    repetitions_per_target_scene: int = 20,
    target_age_hours: float = 3.0,
    distractor_age_hours: float = 4.0,
) -> tuple[TemporalGroundingCase, ...]:
    """Create target-scene-balanced decisions under equal latent persistence.

    The target always has the more recent matching observation. Under the true
    equal hazard it must therefore outrank the distractor. A detection-time
    estimator can instead prefer the sparsely inspected distractor because it
    learns inspection delay as persistence. Target scene is balanced so this
    directional error remains visible rather than becoming an aggregate gain.
    """

    if repetitions_per_target_scene <= 0:
        raise ValueError("repetitions per target scene must be positive")
    if (
        not math.isfinite(target_age_hours)
        or not math.isfinite(distractor_age_hours)
        or target_age_hours < 0.0
        or distractor_age_hours <= target_age_hours
    ):
        raise ValueError("grounding ages require distractor > target >= 0")

    desired = _relation("on", "table")
    wrong = _relation("inside", "cabinet")
    cases: list[TemporalGroundingCase] = []
    base_time = 2_100_000_000.0
    case_index = 0
    for target_scene in SCENE_SCHEDULES:
        distractor_scene = next(scene for scene in SCENE_SCHEDULES if scene != target_scene)
        for repetition in range(repetitions_per_target_scene):
            as_of = base_time + case_index * 3600.0
            target_id = f"target-{target_scene}-{repetition:02d}"
            distractor_id = f"distractor-{target_scene}-{repetition:02d}"
            other_id = f"other-{target_scene}-{repetition:02d}"
            target = Entity(
                target_id,
                {
                    "type": Observation("cup"),
                    "color": Observation("red"),
                    "scene": Observation(target_scene),
                    "location": Observation(
                        desired,
                        timestamp=as_of - target_age_hours * 3600.0,
                    ),
                },
            )
            distractor = Entity(
                distractor_id,
                {
                    "type": Observation("cup"),
                    "color": Observation("red"),
                    "scene": Observation(distractor_scene),
                    "location": Observation(
                        desired,
                        timestamp=as_of - distractor_age_hours * 3600.0,
                    ),
                },
            )
            other = Entity(
                other_id,
                {
                    "type": Observation("cup"),
                    "color": Observation("blue"),
                    "scene": Observation(target_scene),
                    "location": Observation(wrong, timestamp=as_of),
                },
            )
            entities = (target, distractor, other)
            if case_index % 2:
                entities = (other, distractor, target)
            query = "the red cup on the table"
            frame = QueryFrame(
                query,
                (
                    PropertyConstraint("type", "cup", 0.20),
                    PropertyConstraint("color", "red", 0.15),
                    PropertyConstraint("location", desired, 0.65),
                ),
            )
            cases.append(
                TemporalGroundingCase(
                    f"observation-grounding-{case_index:03d}",
                    query,
                    entities,
                    target_id,
                    frame,
                    as_of,
                    {
                        target_id: {"location": desired},
                        distractor_id: {"location": wrong},
                        other_id: {"location": wrong},
                    },
                    (
                        "observation-grounding",
                        "analytic-decision",
                        f"target-{target_scene}",
                    ),
                )
            )
            case_index += 1
    return tuple(cases)


def observation_grounding_oracle_model(true_hazard_per_hour: float) -> PerContextExponentialPersistenceModel:
    hazards = {
        ("location", "cup", "on", "table", scene): true_hazard_per_hour
        for scene in SCENE_SCHEDULES
    }
    return PerContextExponentialPersistenceModel(
        hazards,
        global_hazard=true_hazard_per_hour,
        trained_properties=frozenset({"location"}),
    )


def evaluate_observation_grounding_seed(
    *,
    seed: int,
    dataset: ObservationProcessDataset,
    cases: Iterable[TemporalGroundingCase],
    registry: PropertyRegistry | None = None,
) -> dict[str, Any]:
    rows = tuple(cases)
    if not rows:
        raise ValueError("at least one observation-grounding case is required")
    models = {
        "naive": PerContextExponentialPersistenceModel.fit(
            dataset.naive_train, prior_exposure_hours=0.0
        ),
        "interval_aware": PerContextExponentialPersistenceModel.fit(
            dataset.interval_train, prior_exposure_hours=0.0
        ),
        "oracle": observation_grounding_oracle_model(dataset.true_hazard_per_hour),
    }
    active_registry = registry or compositional_grounding_registry()
    conditions: dict[str, Any] = {}
    for name, model in models.items():
        report = evaluate_grounding_model(name, model, rows, active_registry)
        by_scene = {
            scene: report.accuracy_by_tag[f"target-{scene}"] for scene in SCENE_SCHEDULES
        }
        hazards = {
            scene: model.hazard_per_hour(("location", "cup", "on", "table", scene))
            for scene in SCENE_SCHEDULES
        }
        conditions[name] = {
            "hazards_per_hour": hazards,
            "top1": report.top1_accuracy,
            "mean_reciprocal_rank": report.mean_reciprocal_rank,
            "top1_by_target_scene": by_scene,
            "worst_target_scene_top1": min(by_scene.values()),
            "target_scene_gap": abs(by_scene["frequent-scene"] - by_scene["sparse-scene"]),
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


def aggregate_observation_grounding_runs(
    runs: Iterable[Mapping[str, Any]],
    *,
    bootstrap_samples: int = 20_000,
    bootstrap_seed: int = 20261001,
) -> dict[str, Any]:
    rows = tuple(runs)
    if not rows:
        raise ValueError("at least one observation-grounding run is required")
    seeds = tuple(int(row["seed"]) for row in rows)
    if len(set(seeds)) != len(seeds):
        raise ValueError("observation-grounding run seeds must be unique")
    for row in rows:
        conditions = row.get("conditions")
        if not isinstance(conditions, Mapping) or set(conditions) != set(CONDITIONS):
            raise ValueError("every run must contain the frozen observation matrix")

    aggregate: dict[str, Any] = {}
    for condition in CONDITIONS:
        aggregate[condition] = {}
        for metric in (
            "top1",
            "mean_reciprocal_rank",
            "worst_target_scene_top1",
            "target_scene_gap",
        ):
            values = tuple(float(row["conditions"][condition][metric]) for row in rows)
            aggregate[condition][metric] = _summary(values)
        aggregate[condition]["top1_by_target_scene"] = {
            scene: _summary(
                tuple(
                    float(row["conditions"][condition]["top1_by_target_scene"][scene])
                    for row in rows
                )
            )
            for scene in SCENE_SCHEDULES
        }

    comparisons: dict[str, Any] = {}
    for offset, (name, orientation) in enumerate(
        (
            ("top1", "interval_aware minus naive"),
            ("worst_target_scene_top1", "interval_aware minus naive"),
            ("target_scene_gap", "naive minus interval_aware"),
        )
    ):
        naive = tuple(float(row["conditions"]["naive"][name]) for row in rows)
        interval = tuple(
            float(row["conditions"]["interval_aware"][name]) for row in rows
        )
        deltas = (
            tuple(i - n for n, i in zip(naive, interval, strict=True))
            if name != "target_scene_gap"
            else tuple(n - i for n, i in zip(naive, interval, strict=True))
        )
        lower, upper = _bootstrap_mean_interval(
            deltas,
            samples=bootstrap_samples,
            seed=bootstrap_seed + offset,
        )
        comparisons[name] = {
            "orientation": orientation,
            "mean_advantage": statistics.mean(deltas),
            "bootstrap_95_ci": [lower, upper],
            "wins": sum(value > 0.0 for value in deltas),
            "ties": sum(value == 0.0 for value in deltas),
            "losses": sum(value < 0.0 for value in deltas),
        }
    return {
        "seeds": list(seeds),
        "aggregate": aggregate,
        "paired_interval_advantage": comparisons,
    }
