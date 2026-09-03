from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Protocol

from .persistence_data import PersistenceTrainingExample


class HazardPredictor(Protocol):
    def hazard_per_hour(self, features: tuple[str, ...]) -> float: ...


@dataclass(frozen=True, slots=True)
class HorizonCalibration:
    hours: float
    evaluable_examples: int
    brier_score: float
    expected_calibration_error: float


@dataclass(frozen=True, slots=True)
class SurvivalEvaluation:
    examples: int
    negative_log_likelihood: float
    horizons: tuple[HorizonCalibration, ...]


def exponential_example_negative_log_likelihood(
    hazard_per_hour: float,
    example: PersistenceTrainingExample,
) -> float:
    """Score exact, right-censored, or interval-censored exponential records."""

    hazard = max(hazard_per_hour, 1e-12)
    upper_hours = example.duration_seconds / 3600.0
    if example.is_interval_censored:
        assert example.interval_start_seconds is not None
        lower_hours = example.interval_start_seconds / 3600.0
        width = upper_hours - lower_hours
        interval_probability_term = -math.expm1(-hazard * width)
        return hazard * lower_hours - math.log(max(interval_probability_term, 1e-300))
    loss = hazard * upper_hours
    if example.event_observed:
        loss -= math.log(hazard)
    return loss

def model_survival_probability(
    model: HazardPredictor,
    features: tuple[str, ...],
    duration_hours: float,
) -> float:
    """Use a model survival curve when available, else exponential fallback."""
    if duration_hours < 0:
        raise ValueError("survival duration cannot be negative")
    method = getattr(model, "survival_probability_at_hours", None)
    if callable(method):
        probability = float(method(features, duration_hours))
    else:
        probability = math.exp(-model.hazard_per_hour(features) * duration_hours)
    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError("model returned an invalid survival probability")
    return probability


def model_example_negative_log_likelihood(
    model: HazardPredictor,
    example: PersistenceTrainingExample,
) -> float:
    """Use a model-specific censoring likelihood when one is implemented."""
    method = getattr(model, "example_negative_log_likelihood", None)
    if callable(method):
        loss = float(method(example))
    else:
        loss = exponential_example_negative_log_likelihood(
            model.hazard_per_hour(example.features()), example
        )
    if not math.isfinite(loss):
        raise ValueError("model returned a non-finite likelihood")
    return loss


def model_risk_score(
    model: HazardPredictor,
    features: tuple[str, ...],
) -> float:
    """Return a time-independent ranking score for concordance evaluation."""
    method = getattr(model, "risk_score", None)
    score = float(method(features)) if callable(method) else model.hazard_per_hour(features)
    if not math.isfinite(score):
        raise ValueError("model returned a non-finite risk score")
    return score




def survival_negative_log_likelihood(
    model: HazardPredictor,
    examples: Iterable[PersistenceTrainingExample],
) -> float:
    rows = tuple(examples)
    if not rows:
        raise ValueError("cannot evaluate an empty dataset")
    total = 0.0
    for example in rows:
        total += model_example_negative_log_likelihood(model, example)
    return total / len(rows)


def _ece(predictions: list[float], outcomes: list[float], bins: int) -> float:
    total = len(predictions)
    error = 0.0
    for bin_index in range(bins):
        lower = bin_index / bins
        upper = (bin_index + 1) / bins
        members = [
            index
            for index, prediction in enumerate(predictions)
            if lower <= prediction < upper or (bin_index == bins - 1 and prediction == 1.0)
        ]
        if not members:
            continue
        mean_prediction = sum(predictions[index] for index in members) / len(members)
        mean_outcome = sum(outcomes[index] for index in members) / len(members)
        error += len(members) / total * abs(mean_prediction - mean_outcome)
    return error


def evaluate_survival(
    model: HazardPredictor,
    examples: Iterable[PersistenceTrainingExample],
    *,
    horizons_hours: tuple[float, ...] = (1.0, 5.0, 12.0),
    calibration_bins: int = 10,
) -> SurvivalEvaluation:
    rows = tuple(examples)
    if not rows:
        raise ValueError("cannot evaluate an empty dataset")
    horizon_results: list[HorizonCalibration] = []
    for horizon in horizons_hours:
        if horizon <= 0:
            raise ValueError("evaluation horizons must be positive")
        predictions: list[float] = []
        outcomes: list[float] = []
        for example in rows:
            duration_hours = example.duration_seconds / 3600.0
            if example.is_interval_censored:
                assert example.interval_start_seconds is not None
                lower_hours = example.interval_start_seconds / 3600.0
                if duration_hours <= horizon:
                    outcome = 0.0
                elif lower_hours >= horizon:
                    outcome = 1.0
                else:
                    # The event interval straddles this horizon.
                    continue
            elif example.event_observed and duration_hours <= horizon:
                outcome = 0.0
            elif duration_hours >= horizon:
                outcome = 1.0
            else:
                # Censored before this horizon: truth at the horizon is unknown.
                continue
            predictions.append(model_survival_probability(model, example.features(), horizon))
            outcomes.append(outcome)
        if not predictions:
            raise ValueError(f"no evaluable examples at {horizon:g} hours")
        brier = sum((prediction - outcome) ** 2 for prediction, outcome in zip(predictions, outcomes, strict=True)) / len(predictions)
        horizon_results.append(
            HorizonCalibration(
                horizon,
                len(predictions),
                brier,
                _ece(predictions, outcomes, calibration_bins),
            )
        )
    return SurvivalEvaluation(
        len(rows),
        survival_negative_log_likelihood(model, rows),
        tuple(horizon_results),
    )
