from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Iterable, Mapping, Protocol

from .latent_mechanism_shift import LatentMechanismShiftDataset
from .persistence_data import PersistenceTrainingExample
from .survival_evaluation import (
    exponential_example_negative_log_likelihood,
    model_risk_score,
)


class RiskModel(Protocol):
    def hazard_per_hour(self, features: tuple[str, ...]) -> float: ...


@dataclass(frozen=True, slots=True)
class TargetCalibrationProtocol:
    """Outcome-independent target calibration pool and fixed held-out tests."""

    calibration_pool: Mapping[str, tuple[PersistenceTrainingExample, ...]]
    tests: Mapping[str, tuple[PersistenceTrainingExample, ...]]
    ranked_group_ids_by_context: Mapping[tuple[str, ...], tuple[str, ...]]
    max_calibration_per_context: int
    split_seed: int
    calibration_contexts: frozenset[tuple[str, ...]]
    test_only_contexts: frozenset[tuple[str, ...]]

    def calibration_subset(
        self,
        condition: str,
        samples_per_context: int,
    ) -> tuple[PersistenceTrainingExample, ...]:
        if condition not in self.calibration_pool:
            raise KeyError(f"unknown target condition: {condition}")
        if not 0 < samples_per_context <= self.max_calibration_per_context:
            raise ValueError(
                "samples_per_context must be within the frozen calibration pool"
            )
        selected = {
            group_id
            for group_ids in self.ranked_group_ids_by_context.values()
            for group_id in group_ids[:samples_per_context]
        }
        rows = tuple(
            row
            for row in self.calibration_pool[condition]
            if row.group_id in selected
        )
        expected = samples_per_context * len(self.ranked_group_ids_by_context)
        if len(rows) != expected:
            raise RuntimeError("calibration subset does not cover every context")
        return rows


def _outcome_independent_rank(group_id: str, split_seed: int) -> tuple[bytes, str]:
    digest = hashlib.sha256(f"{split_seed}|{group_id}".encode("utf-8")).digest()
    return digest, group_id


def build_target_calibration_protocol(
    dataset: LatentMechanismShiftDataset,
    *,
    max_calibration_per_context: int,
    split_seed: int,
    calibration_contexts: Iterable[tuple[str, ...]] | None = None,
) -> TargetCalibrationProtocol:
    """Freeze nested calibration subsets and one common test before adaptation.

    Conditions must be paired row-for-row. Membership is selected from group IDs
    only, so event times and censoring labels cannot influence the split. An
    optional predeclared context subset can receive calibration; all excluded
    contexts remain test-only and are exposed in the protocol audit fields.
    """

    if max_calibration_per_context <= 0:
        raise ValueError("max_calibration_per_context must be positive")
    if not dataset.tests:
        raise ValueError("at least one target condition is required")
    condition_names = tuple(dataset.tests)
    reference = dataset.tests[condition_names[0]]
    if not reference:
        raise ValueError("target conditions cannot be empty")
    reference_signature = tuple(
        (row.group_id, row.features()) for row in reference
    )
    if len({group_id for group_id, _ in reference_signature}) != len(reference):
        raise ValueError("target group IDs must be unique")
    for condition in condition_names[1:]:
        signature = tuple(
            (row.group_id, row.features()) for row in dataset.tests[condition]
        )
        if signature != reference_signature:
            raise ValueError("target conditions must have paired groups and features")

    groups_by_context: dict[tuple[str, ...], list[str]] = {}
    for row in reference:
        groups_by_context.setdefault(row.features(), []).append(row.group_id)
    available_contexts = frozenset(groups_by_context)
    if calibration_contexts is None:
        eligible_contexts = available_contexts
    else:
        eligible_contexts = frozenset(
            tuple(value.casefold() for value in features)
            for features in calibration_contexts
        )
        if not eligible_contexts:
            raise ValueError("calibration context subset cannot be empty")
        unknown_contexts = eligible_contexts - available_contexts
        if unknown_contexts:
            raise ValueError("calibration contexts must occur in every condition")
    test_only_contexts = available_contexts - eligible_contexts
    if any(
        len(groups_by_context[features]) <= max_calibration_per_context
        for features in eligible_contexts
    ):
        raise ValueError("each calibrated context needs pool and held-out test rows")
    ranked = {
        features: tuple(
            sorted(
                group_ids,
                key=lambda group_id: _outcome_independent_rank(
                    group_id, split_seed
                ),
            )
        )
        for features, group_ids in groups_by_context.items()
        if features in eligible_contexts
    }
    pool_ids = {
        group_id
        for group_ids in ranked.values()
        for group_id in group_ids[:max_calibration_per_context]
    }
    calibration_pool = {
        condition: tuple(
            row for row in rows if row.group_id in pool_ids
        )
        for condition, rows in dataset.tests.items()
    }
    tests = {
        condition: tuple(
            row for row in rows if row.group_id not in pool_ids
        )
        for condition, rows in dataset.tests.items()
    }
    return TargetCalibrationProtocol(
        calibration_pool,
        tests,
        ranked,
        max_calibration_per_context,
        split_seed,
        eligible_contexts,
        test_only_contexts,
    )


@dataclass(frozen=True, slots=True)
class LogRiskAffineAdapter:
    """Map a frozen source log-risk to a target exponential log-hazard."""

    source_model: RiskModel
    log_risk_center: float
    log_hazard_at_center: float
    slope: float
    calibration_examples: int
    initial_negative_log_likelihood: float
    final_negative_log_likelihood: float
    slope_was_fitted: bool

    def __post_init__(self) -> None:
        numeric = (
            self.log_risk_center,
            self.log_hazard_at_center,
            self.slope,
            self.initial_negative_log_likelihood,
            self.final_negative_log_likelihood,
        )
        if not all(math.isfinite(value) for value in numeric):
            raise ValueError("adapter parameters and diagnostics must be finite")
        if self.calibration_examples <= 0:
            raise ValueError("calibration_examples must be positive")

    def hazard_per_hour(self, features: tuple[str, ...]) -> float:
        source_risk = model_risk_score(self.source_model, features)
        if source_risk <= 0:
            raise ValueError("source risk must be positive")
        eta = self.log_hazard_at_center + self.slope * (
            math.log(source_risk) - self.log_risk_center
        )
        return math.exp(max(-20.0, min(20.0, eta)))

    def risk_score(self, features: tuple[str, ...]) -> float:
        return self.hazard_per_hour(features)

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
            self.hazard_per_hour(example.features()), example
        )

def select_sign_gated_model(
    source_model: RiskModel,
    affine: LogRiskAffineAdapter,
) -> RiskModel:
    """Use affine repair only when target calibration detects rank reversal."""

    if not affine.slope_was_fitted or affine.source_model is not source_model:
        raise ValueError("sign gating requires an adapter of the frozen source model")
    # A negative slope is direct calibration-only evidence that source ordering
    # is reversed. Otherwise leave the source model entirely unchanged, avoiding
    # small-target-sample probability overfitting.
    return affine if affine.slope < 0.0 else source_model



def _log_risks(
    source_model: RiskModel,
    rows: tuple[PersistenceTrainingExample, ...],
) -> tuple[float, ...]:
    values = tuple(
        math.log(model_risk_score(source_model, row.features())) for row in rows
    )
    if not all(math.isfinite(value) for value in values):
        raise ValueError("source model returned invalid risk")
    return values


def fit_log_risk_affine_adapter(
    source_model: RiskModel,
    examples: Iterable[PersistenceTrainingExample],
    *,
    fit_slope: bool,
    epochs: int = 1000,
    learning_rate: float = 0.03,
    slope_l2_penalty: float = 1e-4,
) -> LogRiskAffineAdapter:
    """Fit target-only scale or affine log-risk using censored likelihood.

    The source model is frozen. With ``fit_slope=False`` the slope stays one,
    so calibration can change probability scale but cannot repair ranking.
    """

    rows = tuple(examples)
    if not rows:
        raise ValueError("at least one target calibration example is required")
    if (
        epochs <= 0
        or not math.isfinite(learning_rate)
        or learning_rate <= 0
        or not math.isfinite(slope_l2_penalty)
        or slope_l2_penalty < 0
    ):
        raise ValueError("invalid target adapter optimizer settings")
    log_risks = _log_risks(source_model, rows)
    center = sum(log_risks) / len(log_risks)
    centered = tuple(value - center for value in log_risks)
    if fit_slope and max(centered) - min(centered) < 1e-9:
        raise ValueError("slope fitting requires at least two source risk levels")

    log_hazard = center
    slope = 1.0
    initial_nll = sum(
        exponential_example_negative_log_likelihood(
            math.exp(max(-20.0, min(20.0, log_hazard + slope * value))),
            row,
        )
        for value, row in zip(centered, rows, strict=True)
    ) / len(rows)
    first = [0.0, 0.0]
    second = [0.0, 0.0]
    beta1, beta2, epsilon = 0.9, 0.999, 1e-8
    for step in range(1, epochs + 1):
        gradients = [0.0, 0.0]
        for value, row in zip(centered, rows, strict=True):
            eta = max(-20.0, min(20.0, log_hazard + slope * value))
            hazard = math.exp(eta)
            upper = row.duration_seconds / 3600.0
            if row.is_interval_censored:
                assert row.interval_start_seconds is not None
                lower = row.interval_start_seconds / 3600.0
                width_mass = hazard * (upper - lower)
                ratio = (
                    width_mass / math.expm1(width_mass)
                    if width_mass < 700
                    else 0.0
                )
                eta_derivative = hazard * lower - ratio
            else:
                eta_derivative = hazard * upper - float(row.event_observed)
            gradients[0] += eta_derivative
            gradients[1] += eta_derivative * value
        gradients[0] /= len(rows)
        gradients[1] /= len(rows)
        if fit_slope:
            gradients[1] += slope_l2_penalty * (slope - 1.0)
        else:
            gradients[1] = 0.0
        parameters = [log_hazard, slope]
        for index in range(2):
            first[index] = beta1 * first[index] + (1.0 - beta1) * gradients[index]
            second[index] = (
                beta2 * second[index]
                + (1.0 - beta2) * gradients[index] ** 2
            )
            first_hat = first[index] / (1.0 - beta1**step)
            second_hat = second[index] / (1.0 - beta2**step)
            parameters[index] -= learning_rate * first_hat / (
                math.sqrt(second_hat) + epsilon
            )
        log_hazard = max(-20.0, min(20.0, parameters[0]))
        slope = max(-4.0, min(4.0, parameters[1])) if fit_slope else 1.0

    adapter = LogRiskAffineAdapter(
        source_model,
        center,
        log_hazard,
        slope,
        len(rows),
        initial_nll,
        0.0,
        fit_slope,
    )
    final_nll = sum(adapter.example_negative_log_likelihood(row) for row in rows) / len(rows)
    return LogRiskAffineAdapter(
        source_model,
        center,
        log_hazard,
        slope,
        len(rows),
        initial_nll,
        final_nll,
        fit_slope,
    )
