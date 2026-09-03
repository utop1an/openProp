from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

from .persistence_data import PersistenceTrainingExample


@dataclass(frozen=True, slots=True)
class SyntheticWeibullOracle:
    """Evaluation-only oracle for a known synthetic Weibull mechanism."""

    hazards_per_hour: Mapping[tuple[str, ...], float]
    shape: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.shape) or self.shape <= 0:
            raise ValueError("oracle shape must be positive and finite")
        normalized = {
            tuple(value.casefold() for value in features): hazard
            for features, hazard in self.hazards_per_hour.items()
        }
        if not normalized or any(
            not math.isfinite(hazard) or hazard <= 0
            for hazard in normalized.values()
        ):
            raise ValueError("oracle hazards must be non-empty, positive, and finite")
        object.__setattr__(self, "hazards_per_hour", normalized)

    def hazard_per_hour(self, features: tuple[str, ...]) -> float:
        key = tuple(value.casefold() for value in features)
        if key not in self.hazards_per_hour:
            raise KeyError(f"oracle has no mechanism for features: {features!r}")
        return self.hazards_per_hour[key]

    def risk_score(self, features: tuple[str, ...]) -> float:
        return math.log(self.hazard_per_hour(features))

    def survival_probability_at_hours(
        self,
        features: tuple[str, ...],
        duration_hours: float,
    ) -> float:
        if duration_hours < 0:
            raise ValueError("survival duration cannot be negative")
        hazard = self.hazard_per_hour(features)
        return math.exp(-((hazard * duration_hours) ** self.shape))

    def example_negative_log_likelihood(
        self,
        example: PersistenceTrainingExample,
    ) -> float:
        hazard = self.hazard_per_hour(example.features())
        upper = example.duration_seconds / 3600.0
        upper_cumulative = (hazard * upper) ** self.shape
        if example.is_interval_censored:
            assert example.interval_start_seconds is not None
            lower = example.interval_start_seconds / 3600.0
            lower_cumulative = (hazard * lower) ** self.shape
            probability = math.exp(-lower_cumulative) - math.exp(-upper_cumulative)
            return -math.log(max(probability, 1e-300))
        if not example.event_observed:
            return upper_cumulative
        if upper <= 0:
            raise ValueError("an exact Weibull event must occur after time zero")
        log_density = (
            math.log(self.shape)
            + self.shape * math.log(hazard)
            + (self.shape - 1.0) * math.log(upper)
            - upper_cumulative
        )
        return -log_density
