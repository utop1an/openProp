from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .advanced_survival_evaluation import concordance_index, integrated_brier_score
from .persistence_data import PersistenceTrainingExample
from .survival_evaluation import (
    HazardPredictor,
    HorizonCalibration,
    model_survival_probability,
)


@dataclass(frozen=True, slots=True)
class SemiparametricSurvivalEvaluation:
    """Metrics that do not require a continuous baseline event density."""

    examples: int
    concordance_index: float
    integrated_brier_score: float
    horizons: tuple[HorizonCalibration, ...]


def _expected_calibration_error(
    predictions: list[float],
    outcomes: list[float],
    bins: int,
) -> float:
    error = 0.0
    for bin_index in range(bins):
        lower = bin_index / bins
        upper = (bin_index + 1) / bins
        members = [
            index
            for index, prediction in enumerate(predictions)
            if lower <= prediction < upper
            or (bin_index == bins - 1 and prediction == 1.0)
        ]
        if not members:
            continue
        mean_prediction = sum(predictions[index] for index in members) / len(members)
        mean_outcome = sum(outcomes[index] for index in members) / len(members)
        error += len(members) / len(predictions) * abs(
            mean_prediction - mean_outcome
        )
    return error


def evaluate_semiparametric_survival(
    model: HazardPredictor,
    examples: Iterable[PersistenceTrainingExample],
    *,
    horizons_hours: tuple[float, ...] = (1.0, 4.0, 8.0, 12.0),
    calibration_bins: int = 10,
) -> SemiparametricSurvivalEvaluation:
    """Evaluate ranking and calibration without inventing an event-time NLL."""

    rows = tuple(examples)
    if not rows:
        raise ValueError("cannot evaluate an empty dataset")
    if calibration_bins <= 0:
        raise ValueError("calibration_bins must be positive")
    horizon_results: list[HorizonCalibration] = []
    for horizon in horizons_hours:
        if horizon <= 0:
            raise ValueError("evaluation horizons must be positive")
        predictions: list[float] = []
        outcomes: list[float] = []
        for example in rows:
            upper_hours = example.duration_seconds / 3600.0
            if example.is_interval_censored:
                assert example.interval_start_seconds is not None
                lower_hours = example.interval_start_seconds / 3600.0
                if upper_hours <= horizon:
                    outcome = 0.0
                elif lower_hours >= horizon:
                    outcome = 1.0
                else:
                    continue
            elif example.event_observed and upper_hours <= horizon:
                outcome = 0.0
            elif upper_hours >= horizon:
                outcome = 1.0
            else:
                continue
            predictions.append(
                model_survival_probability(model, example.features(), horizon)
            )
            outcomes.append(outcome)
        if not predictions:
            raise ValueError(f"no evaluable examples at {horizon:g} hours")
        brier = sum(
            (prediction - outcome) ** 2
            for prediction, outcome in zip(predictions, outcomes, strict=True)
        ) / len(predictions)
        horizon_results.append(
            HorizonCalibration(
                horizon,
                len(predictions),
                brier,
                _expected_calibration_error(
                    predictions,
                    outcomes,
                    calibration_bins,
                ),
            )
        )
    horizons = tuple(horizon_results)
    return SemiparametricSurvivalEvaluation(
        len(rows),
        concordance_index(model, rows),
        integrated_brier_score(horizons),
        horizons,
    )
