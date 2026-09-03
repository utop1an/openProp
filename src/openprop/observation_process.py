from __future__ import annotations

import math
import random
from dataclasses import dataclass, replace

from .persistence_data import PersistenceTrainingExample


@dataclass(frozen=True, slots=True)
class ObservationProcessDataset:
    """Paired records under correct and naive observation-time semantics."""

    interval_train: tuple[PersistenceTrainingExample, ...]
    naive_train: tuple[PersistenceTrainingExample, ...]
    exact_test: tuple[PersistenceTrainingExample, ...]
    true_hazard_per_hour: float
    inspection_intervals_hours: tuple[float, ...]


def observation_process_data(
    *,
    samples_per_schedule: int = 600,
    test_samples_per_schedule: int = 400,
    true_hazard_per_hour: float = 0.25,
    followup_hours: float = 12.0,
    inspection_intervals_hours: tuple[float, ...] = (0.5, 4.0),
    seed: int = 101,
) -> ObservationProcessDataset:
    """Generate equal dynamics observed under frequent and sparse inspection.

    A detected transition is interval censored between the last negative check
    and the first positive check. The naive view deliberately replaces the
    unknown transition time with its detection time, exposing observation-rate
    bias without changing latent state dynamics.
    """

    if samples_per_schedule <= 0 or test_samples_per_schedule <= 0:
        raise ValueError("sample counts must be positive")
    if true_hazard_per_hour <= 0 or followup_hours <= 0:
        raise ValueError("hazard and follow-up must be positive")
    if not inspection_intervals_hours or any(
        interval <= 0 or interval > followup_hours
        for interval in inspection_intervals_hours
    ):
        raise ValueError("inspection intervals must be positive and within follow-up")

    rng = random.Random(seed)
    interval_rows: list[PersistenceTrainingExample] = []
    test_rows: list[PersistenceTrainingExample] = []
    for interval in inspection_intervals_hours:
        schedule = f"inspection-{interval:g}h"
        for index in range(samples_per_schedule):
            transition = rng.expovariate(true_hazard_per_hour)
            if transition <= followup_hours:
                detection = min(
                    followup_hours,
                    math.ceil(transition / interval) * interval,
                )
                last_confirmed = max(0.0, detection - interval)
                interval_rows.append(
                    PersistenceTrainingExample(
                        "location",
                        "cup",
                        "on",
                        "table",
                        schedule,
                        detection * 3600.0,
                        True,
                        f"train-{schedule}-{index:04d}",
                        last_confirmed * 3600.0,
                    )
                )
            else:
                interval_rows.append(
                    PersistenceTrainingExample(
                        "location",
                        "cup",
                        "on",
                        "table",
                        schedule,
                        followup_hours * 3600.0,
                        False,
                        f"train-{schedule}-{index:04d}",
                    )
                )
        for index in range(test_samples_per_schedule):
            transition = rng.expovariate(true_hazard_per_hour)
            observed = transition <= followup_hours
            test_rows.append(
                PersistenceTrainingExample(
                    "location",
                    "cup",
                    "on",
                    "table",
                    schedule,
                    min(transition, followup_hours) * 3600.0,
                    observed,
                    f"test-{schedule}-{index:04d}",
                )
            )
    rng.shuffle(interval_rows)
    rng.shuffle(test_rows)
    return ObservationProcessDataset(
        tuple(interval_rows),
        tuple(replace(row, interval_start_seconds=None) for row in interval_rows),
        tuple(test_rows),
        true_hazard_per_hour,
        inspection_intervals_hours,
    )
