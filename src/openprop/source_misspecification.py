from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

from .persistence_data import PersistenceTrainingExample
from .survival_evaluation import exponential_example_negative_log_likelihood
from .target_adaptation import RiskModel


SOURCE_MISSPECIFICATION_CONDITIONS: Mapping[str, str] = {
    "correct_source": "fitted source model without deployment distortion",
    "rate_x2": "all source hazards multiplied by two; ordering preserved",
    "risk_compressed": "log-risk power 0.5 around hazard 0.12; ordering preserved",
    "risk_expanded": "log-risk power 1.75 around hazard 0.12; ordering preserved",
    "subject_cycle": "subject labels cyclically permuted before source prediction",
    "scene_swap": "quiet and busy scene labels swapped before source prediction",
    "subject_scene_permutation": "subject cycle and scene swap applied together",
}


@dataclass(frozen=True, slots=True)
class MonotoneRiskTransform:
    """Apply a positive affine transform in log-risk space."""

    source_model: RiskModel
    power: float = 1.0
    pivot_hazard_per_hour: float = 0.12
    scale: float = 1.0

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.power)
            or self.power <= 0.0
            or not math.isfinite(self.pivot_hazard_per_hour)
            or self.pivot_hazard_per_hour <= 0.0
            or not math.isfinite(self.scale)
            or self.scale <= 0.0
        ):
            raise ValueError("power, pivot hazard, and scale must be finite and positive")

    def hazard_per_hour(self, features: tuple[str, ...]) -> float:
        source = self.source_model.hazard_per_hour(features)
        if not math.isfinite(source) or source <= 0.0:
            raise ValueError("source hazard must be finite and positive")
        log_hazard = (
            math.log(self.scale * self.pivot_hazard_per_hour)
            + self.power * (math.log(source) - math.log(self.pivot_hazard_per_hour))
        )
        return math.exp(max(-20.0, min(20.0, log_hazard)))

    def risk_score(self, features: tuple[str, ...]) -> float:
        return self.hazard_per_hour(features)

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


@dataclass(frozen=True, slots=True)
class TypedFeaturePermutationRiskModel:
    """Evaluate a frozen source model after explicit typed-value permutations."""

    source_model: RiskModel
    permutations: Mapping[int, Mapping[str, str]]

    def __post_init__(self) -> None:
        normalized: dict[int, dict[str, str]] = {}
        for index, mapping in self.permutations.items():
            if index < 0 or index >= 5:
                raise ValueError("permutation index must select one of five typed features")
            if not mapping:
                raise ValueError("typed permutation mappings cannot be empty")
            folded = {str(source).casefold(): str(target).casefold() for source, target in mapping.items()}
            if len(folded) != len(mapping) or len(set(folded.values())) != len(folded):
                raise ValueError("typed permutations must be one-to-one")
            normalized[index] = folded
        if not normalized:
            raise ValueError("at least one typed permutation is required")
        object.__setattr__(self, "permutations", normalized)

    def permuted_features(self, features: tuple[str, ...]) -> tuple[str, ...]:
        if len(features) != 5:
            raise ValueError("typed permutation requires five features")
        values = [value.casefold() for value in features]
        for index, mapping in self.permutations.items():
            if values[index] not in mapping:
                raise ValueError(
                    f"typed permutation at feature {index} has no value {values[index]!r}"
                )
            values[index] = mapping[values[index]]
        return tuple(values)

    def hazard_per_hour(self, features: tuple[str, ...]) -> float:
        return self.source_model.hazard_per_hour(self.permuted_features(features))

    def risk_score(self, features: tuple[str, ...]) -> float:
        return self.hazard_per_hour(features)

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


def source_misspecification_models(source_model: RiskModel) -> Mapping[str, RiskModel]:
    """Return the fixed deployment-source variants used by the paired audit."""

    subject_cycle = {"book": "cup", "cup": "tool", "tool": "book"}
    scene_swap = {"quiet": "busy", "busy": "quiet"}
    return {
        "correct_source": source_model,
        "rate_x2": MonotoneRiskTransform(source_model, scale=2.0),
        "risk_compressed": MonotoneRiskTransform(source_model, power=0.5),
        "risk_expanded": MonotoneRiskTransform(source_model, power=1.75),
        "subject_cycle": TypedFeaturePermutationRiskModel(
            source_model, {1: subject_cycle}
        ),
        "scene_swap": TypedFeaturePermutationRiskModel(source_model, {4: scene_swap}),
        "subject_scene_permutation": TypedFeaturePermutationRiskModel(
            source_model,
            {1: subject_cycle, 4: scene_swap},
        ),
    }
