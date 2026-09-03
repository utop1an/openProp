from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass

from .visual_evaluation import VisualAssociationResult


@dataclass(frozen=True, slots=True)
class FrozenIsotonicConfidence:
    upper_bounds: tuple[float, ...]
    values: tuple[float, ...]
    samples: int
    positives: int
    method: str = "laplace-isotonic-pav"

    def __post_init__(self) -> None:
        if not self.upper_bounds or len(self.upper_bounds) != len(self.values):
            raise ValueError("isotonic confidence bins must be non-empty and aligned")
        if tuple(sorted(self.upper_bounds)) != self.upper_bounds:
            raise ValueError("isotonic upper bounds must be sorted")
        if any(not math.isfinite(value) for value in self.upper_bounds):
            raise ValueError("isotonic upper bounds must be finite")
        if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in self.values):
            raise ValueError("isotonic values must be probabilities")
        if any(left > right for left, right in zip(self.values, self.values[1:])):
            raise ValueError("isotonic confidence values must be nondecreasing")
        if self.samples <= 0 or not 0 <= self.positives <= self.samples:
            raise ValueError("isotonic sample counts are invalid")

    def predict(self, raw_confidence: float) -> float:
        if not math.isfinite(raw_confidence) or not 0.0 <= raw_confidence <= 1.0:
            raise ValueError("raw confidence must be finite and in [0, 1]")
        for upper, value in zip(self.upper_bounds, self.values):
            if raw_confidence <= upper:
                return value
        return self.values[-1]


@dataclass(frozen=True, slots=True)
class FrozenCombinedConfidenceCalibration:
    calibration_system: str
    calibration_rows: int
    minimum_source_rows: int
    global_model: FrozenIsotonicConfidence
    source_models: Mapping[str, FrozenIsotonicConfidence]

    def __post_init__(self) -> None:
        if not self.calibration_system.strip():
            raise ValueError("calibration_system cannot be empty")
        if self.calibration_rows <= 0 or self.minimum_source_rows <= 0:
            raise ValueError("confidence calibration row counts must be positive")
        if any(not source.strip() for source in self.source_models):
            raise ValueError("source confidence model names cannot be empty")

    def predict(self, raw_confidence: float, source: str) -> float:
        return self.source_models.get(source, self.global_model).predict(raw_confidence)


def fit_combined_confidence_calibration(
    rows: Sequence[VisualAssociationResult],
    *,
    minimum_source_rows: int = 30,
) -> FrozenCombinedConfidenceCalibration:
    """Fit calibration-only monotone mappings for correlated confidence factors."""

    if not rows:
        raise ValueError("combined confidence calibration rows cannot be empty")
    if any(row.split != "calibration" for row in rows):
        raise ValueError("combined confidence calibration can only use calibration rows")
    systems = {row.system for row in rows}
    if len(systems) != 1:
        raise ValueError("combined confidence calibration requires one system")
    if minimum_source_rows <= 0:
        raise ValueError("minimum_source_rows must be positive")
    samples = tuple(sample for row in rows if (sample := _sample(row)) is not None)
    if not samples:
        raise ValueError("combined confidence calibration has no eligible decisions")
    by_source: dict[str, list[tuple[float, bool]]] = {}
    for row in rows:
        sample = _sample(row)
        if sample is not None:
            by_source.setdefault(row.source, []).append(sample)
    source_models = {
        source: _fit_isotonic(tuple(source_samples))
        for source, source_samples in sorted(by_source.items())
        if len(source_samples) >= minimum_source_rows
    }
    return FrozenCombinedConfidenceCalibration(
        next(iter(systems)),
        len(samples),
        minimum_source_rows,
        _fit_isotonic(samples),
        source_models,
    )


def apply_combined_confidence_calibration(
    rows: Sequence[VisualAssociationResult],
    calibration: FrozenCombinedConfidenceCalibration,
) -> tuple[VisualAssociationResult, ...]:
    """Attach calibrated confidence target-blind and only revoke unsafe updates."""

    result: list[VisualAssociationResult] = []
    for row in rows:
        payload = asdict(row)
        if row.malformed or not row.eligible or row.decision_entity_id is None:
            payload["calibrated_update_confidence"] = None
        else:
            raw = row.confidence_scale * row.probabilities[row.decision_entity_id]
            calibrated = calibration.predict(raw, row.source)
            payload["calibrated_update_confidence"] = calibrated
            if (
                row.accepted_entity_id is not None
                and calibrated < row.minimum_update_confidence
            ):
                payload["accepted_entity_id"] = None
                payload["reason"] = "calibrated combined confidence below property policy"
        result.append(VisualAssociationResult(**payload))
    return tuple(result)


def _sample(row: VisualAssociationResult) -> tuple[float, bool] | None:
    if row.malformed or not row.eligible or row.decision_entity_id is None:
        return None
    raw = row.confidence_scale * row.probabilities[row.decision_entity_id]
    return raw, row.decision_entity_id == row.target_entity_id


def _fit_isotonic(samples: Sequence[tuple[float, bool]]) -> FrozenIsotonicConfidence:
    grouped: dict[float, list[int]] = {}
    for raw, label in samples:
        if not math.isfinite(raw) or not 0.0 <= raw <= 1.0:
            raise ValueError("combined raw confidence must be in [0, 1]")
        bucket = grouped.setdefault(raw, [0, 0])
        bucket[0] += 1
        bucket[1] += int(label)
    blocks: list[dict[str, float]] = []
    for raw in sorted(grouped):
        count, positives = grouped[raw]
        weight = count + 2
        blocks.append(
            {
                "upper": raw,
                "weight": float(weight),
                "sum": float(positives + 1),
            }
        )
        while len(blocks) >= 2:
            left = blocks[-2]
            right = blocks[-1]
            if left["sum"] / left["weight"] <= right["sum"] / right["weight"]:
                break
            blocks[-2:] = [
                {
                    "upper": right["upper"],
                    "weight": left["weight"] + right["weight"],
                    "sum": left["sum"] + right["sum"],
                }
            ]
    return FrozenIsotonicConfidence(
        tuple(block["upper"] for block in blocks),
        tuple(block["sum"] / block["weight"] for block in blocks),
        len(samples),
        sum(label for _, label in samples),
    )
