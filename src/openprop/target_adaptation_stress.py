from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass, field, replace
from typing import Iterable, Mapping

from .compositional_persistence import ContextDynamics, _context_dynamics
from .persistence_data import PersistenceTrainingExample
from .survival_evaluation import (
    exponential_example_negative_log_likelihood,
    model_risk_score,
)
from .target_adaptation import (
    LogRiskAffineAdapter,
    RiskModel,
    fit_log_risk_affine_adapter,
    select_sign_gated_model,
)


STRESS_CONDITIONS: Mapping[str, str] = {
    "in_distribution": "source hazards",
    "global_reversal": "all context hazards map to 0.12^2 / source hazard",
    "subject_cup_reversal": "only subject_type=cup reverses",
    "scene_busy_reversal": "only scene=busy reverses",
    "subject_scene_xor_reversal": (
        "contexts reverse iff subject_type=cup XOR scene=busy"
    ),
}


@dataclass(frozen=True, slots=True)
class TargetAdaptationStressDataset:
    train: tuple[PersistenceTrainingExample, ...]
    validation: tuple[PersistenceTrainingExample, ...]
    tests: Mapping[str, tuple[PersistenceTrainingExample, ...]]
    test_hazards: Mapping[str, Mapping[tuple[str, ...], float]]
    changed_contexts: Mapping[str, frozenset[tuple[str, ...]]]
    contexts: tuple[ContextDynamics, ...]


def _is_changed(context: ContextDynamics, condition: str) -> bool:
    if condition == "in_distribution":
        return False
    if condition == "global_reversal":
        return True
    if condition == "subject_cup_reversal":
        return context.subject_type == "cup"
    if condition == "scene_busy_reversal":
        return context.scene == "busy"
    if condition == "subject_scene_xor_reversal":
        return (context.subject_type == "cup") != (context.scene == "busy")
    raise KeyError(f"unknown stress condition: {condition}")


def _target_hazard(context: ContextDynamics, condition: str) -> float:
    if _is_changed(context, condition):
        return 0.12**2 / context.hazard_per_hour
    return context.hazard_per_hour


def _target_row(
    context: ContextDynamics,
    index: int,
    unit_exponential: float,
    censor_hours: float,
    hazard_per_hour: float,
) -> PersistenceTrainingExample:
    transition_hours = unit_exponential / hazard_per_hour
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
            f"target-{context.subject_type}-{context.state_predicate}-"
            f"{context.context_object}-{context.scene}-{index:04d}"
        ),
    )


def target_adaptation_stress_data(
    *,
    samples_per_context: int = 80,
    censor_after_hours: float = 16.0,
    seed: int = 41,
) -> TargetAdaptationStressDataset:
    """Generate paired target shifts across all 18 typed contexts."""

    if samples_per_context <= 1:
        raise ValueError("samples_per_context must exceed one")
    if not math.isfinite(censor_after_hours) or censor_after_hours < 0.5:
        raise ValueError("censor_after_hours must be finite and at least 0.5")
    # Reuse the established source split and source generator, but create a new
    # target population with distinct group IDs across every typed context.
    from .latent_mechanism_shift import latent_mechanism_shift_data

    source = latent_mechanism_shift_data(
        samples_per_context=samples_per_context,
        censor_after_hours=censor_after_hours,
        seed=seed,
    )
    contexts = _context_dynamics()
    rng = random.Random(seed + 7_000_003)
    rows_by_condition: dict[str, list[PersistenceTrainingExample]] = {
        condition: [] for condition in STRESS_CONDITIONS
    }
    hazards: dict[str, dict[tuple[str, ...], float]] = {
        condition: {} for condition in STRESS_CONDITIONS
    }
    changed: dict[str, set[tuple[str, ...]]] = {
        condition: set() for condition in STRESS_CONDITIONS
    }
    for context in contexts:
        draws = tuple(
            (rng.expovariate(1.0), rng.uniform(0.5, censor_after_hours))
            for _ in range(samples_per_context)
        )
        for condition in STRESS_CONDITIONS:
            hazard = _target_hazard(context, condition)
            hazards[condition][context.features()] = hazard
            if _is_changed(context, condition):
                changed[condition].add(context.features())
            rows_by_condition[condition].extend(
                _target_row(context, index, unit, censor, hazard)
                for index, (unit, censor) in enumerate(draws)
            )
    order = list(range(len(contexts) * samples_per_context))
    random.Random(seed + 8_000_003).shuffle(order)
    tests = {
        condition: tuple(rows[index] for index in order)
        for condition, rows in rows_by_condition.items()
    }
    return TargetAdaptationStressDataset(
        source.train,
        source.validation,
        tests,
        hazards,
        {name: frozenset(values) for name, values in changed.items()},
        contexts,
    )


def corrupt_calibration_event_labels(
    examples: Iterable[PersistenceTrainingExample],
    *,
    fraction: float,
    seed: int,
) -> tuple[PersistenceTrainingExample, ...]:
    """Deterministically flip an exact fraction of calibration event labels."""

    rows = tuple(examples)
    if not rows:
        raise ValueError("calibration examples cannot be empty")
    if not math.isfinite(fraction) or not 0.0 <= fraction < 0.5:
        raise ValueError("fraction must be finite in [0, 0.5)")
    count = round(fraction * len(rows))
    ranked = sorted(
        rows,
        key=lambda row: (
            hashlib.sha256(f"{seed}|{row.group_id}".encode("utf-8")).digest(),
            row.group_id,
        ),
    )
    flipped = {row.group_id for row in ranked[:count]}
    return tuple(
        replace(
            row,
            event_observed=not row.event_observed,
            interval_start_seconds=None,
        )
        if row.group_id in flipped
        else row
        for row in rows
    )


@dataclass(frozen=True, slots=True)
class FeatureGroupedSignGatedModel:
    """Route target adaptation by one declared typed feature column."""

    source_model: RiskModel
    feature_index: int
    group_models: Mapping[str, RiskModel]
    group_slopes: Mapping[str, float]
    calibration_examples: int
    group_confirmation_slopes: Mapping[str, tuple[float, float]] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        if not 0 <= self.feature_index < 5:
            raise ValueError("feature_index must select one of five typed features")
        if self.calibration_examples <= 0:
            raise ValueError("calibration_examples must be positive")
        if set(self.group_models) != set(self.group_slopes):
            raise ValueError("group models and slopes must have identical keys")
        if not set(self.group_confirmation_slopes).issubset(self.group_models):
            raise ValueError("confirmation slopes must reference known groups")
        if any(
            len(slopes) != 2 or not all(math.isfinite(value) for value in slopes)
            for slopes in self.group_confirmation_slopes.values()
        ):
            raise ValueError("each confirmation group requires two finite slopes")
        object.__setattr__(self, "group_models", dict(self.group_models))
        object.__setattr__(self, "group_slopes", dict(self.group_slopes))
        object.__setattr__(
            self,
            "group_confirmation_slopes",
            dict(self.group_confirmation_slopes),
        )

    @property
    def activated_groups(self) -> frozenset[str]:
        return frozenset(
            group
            for group, model in self.group_models.items()
            if model is not self.source_model
        )

    def _model(self, features: tuple[str, ...]) -> RiskModel:
        if len(features) != 5:
            raise ValueError("group routing requires five typed features")
        return self.group_models.get(
            features[self.feature_index].casefold(),
            self.source_model,
        )

    def hazard_per_hour(self, features: tuple[str, ...]) -> float:
        return self._model(features).hazard_per_hour(features)

    def risk_score(self, features: tuple[str, ...]) -> float:
        return model_risk_score(self._model(features), features)

    def survival_probability_at_hours(
        self,
        features: tuple[str, ...],
        duration_hours: float,
    ) -> float:
        if duration_hours < 0 or not math.isfinite(duration_hours):
            raise ValueError("duration_hours must be finite and nonnegative")
        return math.exp(-self.hazard_per_hour(features) * duration_hours)

    def example_negative_log_likelihood(
        self,
        example: PersistenceTrainingExample,
    ) -> float:
        return exponential_example_negative_log_likelihood(
            self.hazard_per_hour(example.features()),
            example,
        )


def fit_feature_grouped_sign_gate(
    source_model: RiskModel,
    examples: Iterable[PersistenceTrainingExample],
    *,
    feature_index: int,
    epochs: int = 1000,
    learning_rate: float = 0.03,
    slope_l2_penalty: float = 1e-4,
) -> FeatureGroupedSignGatedModel:
    """Fit independent calibration-only sign gates for typed feature groups."""

    rows = tuple(examples)
    if not rows:
        raise ValueError("at least one calibration example is required")
    if not 0 <= feature_index < 5:
        raise ValueError("feature_index must select one of five typed features")
    grouped: dict[str, list[PersistenceTrainingExample]] = {}
    for row in rows:
        grouped.setdefault(row.features()[feature_index].casefold(), []).append(row)
    group_models: dict[str, RiskModel] = {}
    slopes: dict[str, float] = {}
    for group, group_rows in grouped.items():
        source_levels = {
            round(math.log(model_risk_score(source_model, row.features())), 12)
            for row in group_rows
        }
        if len(source_levels) < 2:
            group_models[group] = source_model
            slopes[group] = 1.0
            continue
        affine: LogRiskAffineAdapter = fit_log_risk_affine_adapter(
            source_model,
            group_rows,
            fit_slope=True,
            epochs=epochs,
            learning_rate=learning_rate,
            slope_l2_penalty=slope_l2_penalty,
        )
        group_models[group] = select_sign_gated_model(source_model, affine)
        slopes[group] = affine.slope
    return FeatureGroupedSignGatedModel(
        source_model,
        feature_index,
        group_models,
        slopes,
        len(rows),
    )


def _identity_confirmation_halves(
    rows: list[PersistenceTrainingExample],
    *,
    seed: int,
) -> tuple[list[PersistenceTrainingExample], list[PersistenceTrainingExample]]:
    ranked = sorted(
        rows,
        key=lambda row: (
            hashlib.sha256(f"{seed}|{row.group_id}".encode("utf-8")).digest(),
            row.group_id,
        ),
    )
    return ranked[::2], ranked[1::2]


def fit_confirmed_feature_grouped_sign_gate(
    source_model: RiskModel,
    examples: Iterable[PersistenceTrainingExample],
    *,
    feature_index: int,
    confirmation_seed: int,
    epochs: int = 1000,
    learning_rate: float = 0.03,
    slope_l2_penalty: float = 1e-4,
) -> FeatureGroupedSignGatedModel:
    """Activate a typed group only when two identity-split halves agree."""

    rows = tuple(examples)
    if not rows:
        raise ValueError("at least one calibration example is required")
    if not 0 <= feature_index < 5:
        raise ValueError("feature_index must select one of five typed features")
    grouped: dict[str, list[PersistenceTrainingExample]] = {}
    for row in rows:
        grouped.setdefault(row.features()[feature_index].casefold(), []).append(row)
    group_models: dict[str, RiskModel] = {}
    full_slopes: dict[str, float] = {}
    confirmation_slopes: dict[str, tuple[float, float]] = {}
    for group, group_rows in grouped.items():
        halves = _identity_confirmation_halves(
            group_rows,
            seed=confirmation_seed,
        )
        fitted_halves: list[LogRiskAffineAdapter] = []
        for half in halves:
            source_levels = {
                round(math.log(model_risk_score(source_model, row.features())), 12)
                for row in half
            }
            if len(source_levels) < 2:
                fitted_halves = []
                break
            fitted_halves.append(
                fit_log_risk_affine_adapter(
                    source_model,
                    half,
                    fit_slope=True,
                    epochs=epochs,
                    learning_rate=learning_rate,
                    slope_l2_penalty=slope_l2_penalty,
                )
            )
        full = fit_log_risk_affine_adapter(
            source_model,
            group_rows,
            fit_slope=True,
            epochs=epochs,
            learning_rate=learning_rate,
            slope_l2_penalty=slope_l2_penalty,
        )
        full_slopes[group] = full.slope
        if len(fitted_halves) == 2:
            pair = (fitted_halves[0].slope, fitted_halves[1].slope)
            confirmation_slopes[group] = pair
            group_models[group] = (
                full if pair[0] < 0.0 and pair[1] < 0.0 else source_model
            )
        else:
            group_models[group] = source_model
    return FeatureGroupedSignGatedModel(
        source_model,
        feature_index,
        group_models,
        full_slopes,
        len(rows),
        confirmation_slopes,
    )
