from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

from .advanced_survival_evaluation import concordance_index
from .persistence_data import PersistenceTrainingExample
from .survival_evaluation import (
    exponential_example_negative_log_likelihood,
    model_risk_score,
)
from .target_adaptation import RiskModel


@dataclass(frozen=True, slots=True)
class ConcordanceSafeguardedModel:
    """Deploy a calibration-selected repair only if ranking does not regress.

    This is a calibration-time utility constraint, not a test-time selector.
    The candidate may optimize likelihood internally, while this outer boundary
    preserves the source whenever its Harrell C-index is lower on the declared
    calibration evidence.
    """

    source_model: RiskModel
    candidate_model: RiskModel
    source_calibration_concordance: float
    candidate_calibration_concordance: float
    minimum_concordance_delta: float
    candidate_activated: bool
    accepted: bool
    calibration_examples: int

    def __post_init__(self) -> None:
        numeric = (
            self.source_calibration_concordance,
            self.candidate_calibration_concordance,
            self.minimum_concordance_delta,
        )
        if not all(math.isfinite(value) for value in numeric):
            raise ValueError("concordance diagnostics must be finite")
        if not 0.0 <= self.source_calibration_concordance <= 1.0 or not (
            0.0 <= self.candidate_calibration_concordance <= 1.0
        ):
            raise ValueError("concordance values must be in [0, 1]")
        if self.calibration_examples <= 0:
            raise ValueError("calibration_examples must be positive")
        expected = self.candidate_activated and (
            self.candidate_calibration_concordance
            - self.source_calibration_concordance
            >= self.minimum_concordance_delta
        )
        if self.accepted != expected:
            raise ValueError("acceptance does not match calibration diagnostics")

    @property
    def activated(self) -> bool:
        return self.accepted

    @property
    def concordance_delta(self) -> float:
        return (
            self.candidate_calibration_concordance
            - self.source_calibration_concordance
        )

    @property
    def selected_partition(self) -> tuple[int, ...] | None:
        if not self.accepted:
            return None
        value = getattr(self.candidate_model, "selected_partition", None)
        return value if isinstance(value, tuple) else ()

    @property
    def significant_groups(self) -> frozenset[tuple[str, ...]]:
        if not self.accepted:
            return frozenset()
        value = getattr(self.candidate_model, "significant_groups", frozenset())
        return frozenset(value)

    def _model(self) -> RiskModel:
        return self.candidate_model if self.accepted else self.source_model

    def hazard_per_hour(self, features: tuple[str, ...]) -> float:
        return self._model().hazard_per_hour(features)

    def risk_score(self, features: tuple[str, ...]) -> float:
        return model_risk_score(self._model(), features)

    def survival_probability_at_hours(
        self,
        features: tuple[str, ...],
        duration_hours: float,
    ) -> float:
        if duration_hours < 0.0 or not math.isfinite(duration_hours):
            raise ValueError("duration_hours must be finite and nonnegative")
        return math.exp(-self.hazard_per_hour(features) * duration_hours)

    def example_negative_log_likelihood(
        self,
        example: PersistenceTrainingExample,
    ) -> float:
        return exponential_example_negative_log_likelihood(
            self.hazard_per_hour(example.features()), example
        )


def apply_concordance_safeguard(
    source_model: RiskModel,
    candidate_model: RiskModel,
    examples: Iterable[PersistenceTrainingExample],
    *,
    candidate_activated: bool | None = None,
    minimum_concordance_delta: float = 0.0,
) -> ConcordanceSafeguardedModel:
    """Apply a deterministic, calibration-only ranking non-inferiority guard."""

    rows = tuple(examples)
    if not rows:
        raise ValueError("concordance safeguard requires calibration examples")
    if not math.isfinite(minimum_concordance_delta):
        raise ValueError("minimum_concordance_delta must be finite")
    active = (
        bool(getattr(candidate_model, "activated", True))
        if candidate_activated is None
        else candidate_activated
    )
    source_c = concordance_index(source_model, rows)
    candidate_c = concordance_index(candidate_model, rows)
    accepted = active and candidate_c - source_c >= minimum_concordance_delta
    return ConcordanceSafeguardedModel(
        source_model=source_model,
        candidate_model=candidate_model,
        source_calibration_concordance=source_c,
        candidate_calibration_concordance=candidate_c,
        minimum_concordance_delta=minimum_concordance_delta,
        candidate_activated=active,
        accepted=accepted,
        calibration_examples=len(rows),
    )

