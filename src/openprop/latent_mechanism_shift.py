from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Mapping

from .compositional_persistence import ContextDynamics, _context_dynamics
from .persistence_data import PersistenceTrainingExample


MECHANISM_CONDITIONS: Mapping[str, Mapping[str, float | str]] = {
    "in_distribution": {"rate_multiplier": 1.0, "weibull_shape": 1.0},
    "global_rate_acceleration": {"rate_multiplier": 2.0, "weibull_shape": 1.0},
    "decreasing_hazard_shape": {"rate_multiplier": 1.0, "weibull_shape": 0.6},
    "increasing_hazard_shape": {"rate_multiplier": 1.0, "weibull_shape": 1.6},
    "typed_factor_reversal": {
        "rate_transform": "0.12^2 / source_hazard",
        "weibull_shape": 1.0,
    },
}


@dataclass(frozen=True, slots=True)
class LatentMechanismShiftDataset:
    train: tuple[PersistenceTrainingExample, ...]
    validation: tuple[PersistenceTrainingExample, ...]
    tests: Mapping[str, tuple[PersistenceTrainingExample, ...]]
    contexts: tuple[ContextDynamics, ...]
    test_hazards: Mapping[str, Mapping[tuple[str, ...], float]]


def _shifted_hazard(context: ContextDynamics, condition: str) -> float:
    if condition == "typed_factor_reversal":
        return 0.12**2 / context.hazard_per_hour
    multiplier = float(MECHANISM_CONDITIONS[condition]["rate_multiplier"])
    return context.hazard_per_hour * multiplier


def _row(
    context: ContextDynamics,
    index: int,
    unit_exponential: float,
    censor_hours: float,
    *,
    hazard_per_hour: float,
    weibull_shape: float,
) -> PersistenceTrainingExample:
    transition_hours = unit_exponential ** (1.0 / weibull_shape) / hazard_per_hour
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
            f"{context.split}-{context.subject_type}-{context.state_predicate}-"
            f"{context.context_object}-{context.scene}-{index:04d}"
        ),
    )


def latent_mechanism_shift_data(
    *,
    samples_per_context: int = 80,
    censor_after_hours: float = 16.0,
    seed: int = 41,
) -> LatentMechanismShiftDataset:
    """Create paired test sets that differ only in latent state dynamics."""

    if samples_per_context <= 0:
        raise ValueError("samples_per_context must be positive")
    if not math.isfinite(censor_after_hours) or censor_after_hours < 0.5:
        raise ValueError("censor_after_hours must be finite and at least 0.5")

    contexts = _context_dynamics()
    source_rng = random.Random(seed)
    test_rng = random.Random(seed + 1_000_003)
    source_rows: dict[str, list[PersistenceTrainingExample]] = {
        "train": [],
        "validation": [],
    }
    tests: dict[str, list[PersistenceTrainingExample]] = {
        condition: [] for condition in MECHANISM_CONDITIONS
    }
    test_hazards: dict[str, dict[tuple[str, ...], float]] = {
        condition: {} for condition in MECHANISM_CONDITIONS
    }

    for context in contexts:
        if context.split in source_rows:
            for index in range(samples_per_context):
                source_rows[context.split].append(
                    _row(
                        context,
                        index,
                        source_rng.expovariate(1.0),
                        source_rng.uniform(0.5, censor_after_hours),
                        hazard_per_hour=context.hazard_per_hour,
                        weibull_shape=1.0,
                    )
                )
            continue

        latent_draws = [
            (
                test_rng.expovariate(1.0),
                test_rng.uniform(0.5, censor_after_hours),
            )
            for _ in range(samples_per_context)
        ]
        for condition, metadata in MECHANISM_CONDITIONS.items():
            hazard = _shifted_hazard(context, condition)
            shape = float(metadata["weibull_shape"])
            test_hazards[condition][context.features()] = hazard
            tests[condition].extend(
                _row(
                    context,
                    index,
                    unit_exponential,
                    censor_hours,
                    hazard_per_hour=hazard,
                    weibull_shape=shape,
                )
                for index, (unit_exponential, censor_hours) in enumerate(
                    latent_draws
                )
            )

    train_order = list(range(len(source_rows["train"])))
    validation_order = list(range(len(source_rows["validation"])))
    test_order = list(range(len(next(iter(tests.values())))))
    random.Random(seed + 2_000_003).shuffle(train_order)
    random.Random(seed + 3_000_003).shuffle(validation_order)
    random.Random(seed + 4_000_003).shuffle(test_order)
    train = tuple(source_rows["train"][index] for index in train_order)
    validation = tuple(
        source_rows["validation"][index] for index in validation_order
    )
    paired_tests = {
        condition: tuple(rows[index] for index in test_order)
        for condition, rows in tests.items()
    }
    return LatentMechanismShiftDataset(
        train,
        validation,
        paired_tests,
        contexts,
        test_hazards,
    )
