from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Mapping

from .persistence_data import PersistenceTrainingExample
from .survival_evaluation import exponential_example_negative_log_likelihood
from .target_adaptation import RiskModel


NON_AFFINE_MISSPECIFICATION_CONDITIONS: Mapping[str, str] = {
    "correct_source_control": "fitted source model without deployment distortion",
    "local_subject_saturation": (
        "monotone tanh compression in log risk for subject=cup only"
    ),
    "local_scene_fold": (
        "absolute-log fold around hazard 0.12 for scene=busy only"
    ),
    "local_subject_scene_bump": (
        "smooth Gaussian log-risk bump for subject=cup and scene=busy only"
    ),
}


@dataclass(frozen=True, slots=True)
class LocalNonAffineRiskWarp:
    """Apply a frozen nonlinear risk distortion inside one typed region only."""

    source_model: RiskModel
    required_values: Mapping[int, str]
    transform: Literal["tanh_log", "absolute_log_fold", "gaussian_log_bump"]
    pivot_hazard_per_hour: float = 0.12
    magnitude: float = 1.5
    width: float = 0.65

    def __post_init__(self) -> None:
        normalized: dict[int, str] = {}
        for index, value in self.required_values.items():
            folded = str(value).strip().casefold()
            if index < 0 or index >= 5 or not folded:
                raise ValueError(
                    "required typed values need indices in [0, 4] and nonempty values"
                )
            normalized[index] = folded
        if not normalized:
            raise ValueError("at least one required typed value is needed")
        if self.transform not in {
            "tanh_log",
            "absolute_log_fold",
            "gaussian_log_bump",
        }:
            raise ValueError("unknown non-affine risk transform")
        if (
            not math.isfinite(self.pivot_hazard_per_hour)
            or self.pivot_hazard_per_hour <= 0.0
            or not math.isfinite(self.magnitude)
            or self.magnitude <= 0.0
            or not math.isfinite(self.width)
            or self.width <= 0.0
        ):
            raise ValueError("pivot, magnitude, and width must be finite and positive")
        object.__setattr__(self, "required_values", normalized)

    def applies_to(self, features: tuple[str, ...]) -> bool:
        if len(features) != 5:
            raise ValueError("non-affine typed routing requires five features")
        return all(
            features[index].casefold() == required
            for index, required in self.required_values.items()
        )

    def warped_hazard(self, source_hazard: float) -> float:
        if not math.isfinite(source_hazard) or source_hazard <= 0.0:
            raise ValueError("source hazard must be finite and positive")
        z = math.log(source_hazard / self.pivot_hazard_per_hour)
        if self.transform == "tanh_log":
            transformed_z = math.tanh(z)
        elif self.transform == "absolute_log_fold":
            transformed_z = abs(z)
        else:
            transformed_z = z + self.magnitude * math.exp(
                -0.5 * (z / self.width) ** 2
            )
        log_hazard = math.log(self.pivot_hazard_per_hour) + transformed_z
        return math.exp(max(-20.0, min(20.0, log_hazard)))

    def hazard_per_hour(self, features: tuple[str, ...]) -> float:
        source = self.source_model.hazard_per_hour(features)
        return self.warped_hazard(source) if self.applies_to(features) else source

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


def non_affine_misspecification_models(
    source_model: RiskModel,
) -> Mapping[str, RiskModel]:
    """Return the predeclared local nonlinear deployment distortions."""

    return {
        "correct_source_control": source_model,
        "local_subject_saturation": LocalNonAffineRiskWarp(
            source_model,
            {1: "cup"},
            "tanh_log",
        ),
        "local_scene_fold": LocalNonAffineRiskWarp(
            source_model,
            {4: "busy"},
            "absolute_log_fold",
        ),
        "local_subject_scene_bump": LocalNonAffineRiskWarp(
            source_model,
            {1: "cup", 4: "busy"},
            "gaussian_log_bump",
        ),
    }


def affected_contexts(
    deployed_model: RiskModel,
    source_model: RiskModel,
    contexts: tuple[tuple[str, ...], ...],
) -> frozenset[tuple[str, ...]]:
    """Identify distorted contexts from predictions, never from target outcomes."""

    if len(contexts) != len(set(contexts)):
        raise ValueError("typed contexts must be unique")
    changed = {
        features
        for features in contexts
        if not math.isclose(
            deployed_model.hazard_per_hour(features),
            source_model.hazard_per_hour(features),
            rel_tol=1e-12,
            abs_tol=0.0,
        )
    }
    return frozenset(changed)
