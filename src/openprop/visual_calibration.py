from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Sequence

from .null_competition import rescale_null_distribution
from .visual_evaluation import VisualAssociationResult


@dataclass(frozen=True, slots=True)
class FrozenAcceptancePolicy:
    acceptance_threshold: float
    margin_threshold: float
    null_scale: float
    candidate_count_power: float
    calibration_system: str
    calibration_rows: int
    candidate_count_levels: int
    supported_candidate_counts: tuple[int, ...]
    correct_updates: int
    false_updates: int
    accepted: int
    max_false_update_rate: float
    searched_policies: int
    feasible_policies: int

    def __post_init__(self) -> None:
        for name, value in (
            ("acceptance_threshold", self.acceptance_threshold),
            ("margin_threshold", self.margin_threshold),
            ("max_false_update_rate", self.max_false_update_rate),
        ):
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and in [0, 1]")
        if not math.isfinite(self.null_scale) or self.null_scale <= 0.0:
            raise ValueError("null_scale must be finite and positive")
        if not math.isfinite(self.candidate_count_power):
            raise ValueError("candidate_count_power must be finite")
        if not self.calibration_system.strip():
            raise ValueError("calibration_system cannot be empty")
        if self.calibration_rows <= 0:
            raise ValueError("calibration_rows must be positive")
        if self.candidate_count_levels <= 0:
            raise ValueError("candidate_count_levels must be positive")
        if (
            len(self.supported_candidate_counts) != self.candidate_count_levels
            or any(value <= 0 for value in self.supported_candidate_counts)
            or tuple(sorted(set(self.supported_candidate_counts)))
            != self.supported_candidate_counts
        ):
            raise ValueError("supported_candidate_counts are invalid")
        if self.searched_policies <= 0 or self.feasible_policies <= 0:
            raise ValueError("calibration policy counts must be positive")


def calibrate_acceptance_policy(
    rows: Sequence[VisualAssociationResult],
    *,
    acceptance_thresholds: Sequence[float],
    margin_thresholds: Sequence[float],
    null_scales: Sequence[float] = (1.0,),
    candidate_count_powers: Sequence[float] = (0.0,),
    max_false_update_rate: float,
) -> FrozenAcceptancePolicy:
    """Fit admission gates on calibration identities only.

    Utility is lexicographic: maximize correct updates, minimize false updates,
    then maximize admitted coverage. Test rows are rejected before any search.
    """

    if not rows:
        raise ValueError("acceptance calibration rows cannot be empty")
    if any(row.split != "calibration" for row in rows):
        raise ValueError("acceptance policy can only be fit on calibration rows")
    systems = {row.system for row in rows}
    if len(systems) != 1:
        raise ValueError("acceptance calibration requires exactly one system")
    _probability(max_false_update_rate, "max_false_update_rate")
    thresholds = tuple(sorted({float(value) for value in acceptance_thresholds}))
    margins = tuple(sorted({float(value) for value in margin_thresholds}))
    scales = tuple(sorted({float(value) for value in null_scales}))
    powers = tuple(sorted({float(value) for value in candidate_count_powers}))
    if not thresholds or not margins or not scales or not powers:
        raise ValueError("acceptance, margin, and null grids cannot be empty")
    for value in (*thresholds, *margins):
        _probability(value, "calibration grid value")
    if any(not math.isfinite(value) or value <= 0.0 for value in scales):
        raise ValueError("null scale grid values must be finite and positive")
    if any(not math.isfinite(value) for value in powers):
        raise ValueError("candidate count powers must be finite")
    supported_candidate_counts = tuple(
        sorted({len(row.candidate_entity_ids) for row in rows})
    )
    candidate_count_levels = len(supported_candidate_counts)
    if candidate_count_levels < 2:
        powers = tuple(value for value in powers if abs(value) <= 1e-12)
        if not powers:
            raise ValueError(
                "nonzero candidate count power requires multiple candidate counts"
            )

    candidates: list[tuple[float, float, float, float, int, int, int]] = []
    for threshold in thresholds:
        for margin in margins:
            for null_scale in scales:
                for count_power in powers:
                    decisions = tuple(
                        _calibrated_decision(
                            row, threshold, margin, null_scale, count_power
                        )
                        for row in rows
                    )
                    accepted = sum(decision is not None for decision in decisions)
                    correct = sum(
                        decision is not None and decision == row.target_entity_id
                        for row, decision in zip(rows, decisions)
                    )
                    false = accepted - correct
                    if false / len(rows) <= max_false_update_rate:
                        candidates.append(
                            (
                                threshold,
                                margin,
                                null_scale,
                                count_power,
                                correct,
                                false,
                                accepted,
                            )
                        )
    if not candidates:
        raise ValueError("no acceptance policy satisfies the calibration safety gate")
    selected = max(
        candidates,
        key=lambda item: (
            item[4],
            -item[5],
            item[6],
            -abs(math.log(item[2])),
            -abs(item[3]),
            item[0],
            item[1],
        ),
    )
    return FrozenAcceptancePolicy(
        selected[0],
        selected[1],
        selected[2],
        selected[3],
        next(iter(systems)),
        len(rows),
        candidate_count_levels,
        supported_candidate_counts,
        selected[4],
        selected[5],
        selected[6],
        max_false_update_rate,
        len(thresholds) * len(margins) * len(scales) * len(powers),
        len(candidates),
    )


def apply_acceptance_policy(
    rows: Sequence[VisualAssociationResult],
    policy: FrozenAcceptancePolicy,
) -> tuple[VisualAssociationResult, ...]:
    """Apply frozen gates without reading target identities."""

    result: list[VisualAssociationResult] = []
    for row in rows:
        probabilities, decision = rescale_null_distribution(
            row.probabilities,
            candidate_count=len(row.candidate_entity_ids),
            null_scale=policy.null_scale,
            candidate_count_power=policy.candidate_count_power,
        )
        count_supported = len(row.candidate_entity_ids) in policy.supported_candidate_counts
        accepted = count_supported and _would_accept(
            row,
            policy.acceptance_threshold,
            policy.margin_threshold,
            probabilities,
            decision,
        )
        payload = asdict(row)
        payload["probabilities"] = probabilities
        payload["decision_entity_id"] = decision
        payload["accepted_entity_id"] = decision if accepted else None
        payload["reason"] = (
            "accepted by frozen calibration policy"
            if accepted
            else (
                "candidate count absent from calibration support"
                if not count_supported
                else "rejected by frozen calibration policy"
            )
        )
        result.append(VisualAssociationResult(**payload))
    return tuple(result)


def _would_accept(
    row: VisualAssociationResult,
    threshold: float,
    margin: float,
    probabilities: dict[str, float] | None = None,
    decision_entity_id: str | None | object = ...,
) -> bool:
    selected_probabilities = probabilities or dict(row.probabilities)
    selected_decision = (
        row.decision_entity_id if decision_entity_id is ... else decision_entity_id
    )
    if row.malformed or not row.eligible or selected_decision is None:
        return False
    assert isinstance(selected_decision, str)
    decision_probability = selected_probabilities[selected_decision]
    runner_up = max(
        value
        for key, value in selected_probabilities.items()
        if key != selected_decision
    )
    return (
        decision_probability >= threshold
        and decision_probability - runner_up >= margin
        and row.confidence_scale * decision_probability >= row.minimum_update_confidence
    )


def _calibrated_decision(
    row: VisualAssociationResult,
    threshold: float,
    margin: float,
    null_scale: float,
    candidate_count_power: float,
) -> str | None:
    probabilities, decision = rescale_null_distribution(
        row.probabilities,
        candidate_count=len(row.candidate_entity_ids),
        null_scale=null_scale,
        candidate_count_power=candidate_count_power,
    )
    return decision if _would_accept(row, threshold, margin, probabilities, decision) else None


def _probability(value: float, name: str) -> None:
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1]")
