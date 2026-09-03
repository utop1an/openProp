from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .persistence_data import PersistenceTrainingExample
from .survival_evaluation import (
    HazardPredictor,
    HorizonCalibration,
    evaluate_survival,
    model_risk_score,
)


@dataclass(frozen=True, slots=True)
class AdvancedSurvivalEvaluation:
    examples: int
    negative_log_likelihood: float
    concordance_index: float
    integrated_brier_score: float
    horizons: tuple[HorizonCalibration, ...]


def concordance_index(
    model: HazardPredictor,
    examples: Iterable[PersistenceTrainingExample],
) -> float:
    """Compute Harrell's C-index over comparable right-censored pairs."""

    rows = tuple(examples)
    hazards = [model_risk_score(model, example.features()) for example in rows]
    event_upper = [example.duration_seconds for example in rows]
    known_alive_until = [
        example.interval_start_seconds
        if example.is_interval_censored
        else example.duration_seconds
        for example in rows
    ]
    concordant = 0.0
    comparable = 0
    for index, example in enumerate(rows):
        if not example.event_observed:
            continue
        for other in range(len(rows)):
            # Compare only when this event precedes the other's last known
            # event-free time; overlapping uncertainty intervals are omitted.
            if known_alive_until[other] <= event_upper[index]:
                continue
            comparable += 1
            if hazards[index] > hazards[other]:
                concordant += 1.0
            elif hazards[index] == hazards[other]:
                concordant += 0.5
    return concordant / comparable if comparable else 0.5


def integrated_brier_score(horizons: Iterable[HorizonCalibration]) -> float:
    """Return the trapezoidal mean Brier score over evaluated horizons.

    The underlying horizon scores retain the existing conservative rule that
    excludes episodes censored before a horizon because their truth is unknown.
    """

    points = sorted((item.hours, item.brier_score) for item in horizons)
    if not points:
        raise ValueError("at least one Brier horizon is required")
    if len(points) == 1:
        return points[0][1]
    span = points[-1][0] - points[0][0]
    if span <= 0:
        raise ValueError("Brier horizons must be distinct")
    area = sum(
        (right_time - left_time) * (left_score + right_score) / 2.0
        for (left_time, left_score), (right_time, right_score) in zip(points, points[1:])
    )
    return area / span


def evaluate_survival_advanced(
    model: HazardPredictor,
    examples: Iterable[PersistenceTrainingExample],
    *,
    horizons_hours: tuple[float, ...] = (1.0, 4.0, 8.0, 12.0),
    calibration_bins: int = 10,
) -> AdvancedSurvivalEvaluation:
    rows = tuple(examples)
    base = evaluate_survival(
        model,
        rows,
        horizons_hours=horizons_hours,
        calibration_bins=calibration_bins,
    )
    return AdvancedSurvivalEvaluation(
        base.examples,
        base.negative_log_likelihood,
        concordance_index(model, rows),
        integrated_brier_score(base.horizons),
        base.horizons,
    )
