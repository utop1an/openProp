from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Mapping, Sequence

from .models import MatchResult
from .null_competition import rescale_null_distribution
from .visual_evaluation import NULL_ENTITY, VisualQueryResult


@dataclass(frozen=True, slots=True)
class QueryDecisionPolicy:
    """Frozen admission policy for deterministic final entity ranking."""

    acceptance_threshold: float = 0.50
    margin_threshold: float = 0.10
    minimum_coverage: float = 0.0
    null_weight: float = 0.05
    score_power: float = 1.0
    epsilon: float = 1e-9

    def __post_init__(self) -> None:
        for name, value in (
            ("acceptance_threshold", self.acceptance_threshold),
            ("margin_threshold", self.margin_threshold),
            ("minimum_coverage", self.minimum_coverage),
        ):
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and in [0, 1]")
        if not math.isfinite(self.null_weight) or self.null_weight <= 0.0:
            raise ValueError("null_weight must be finite and positive")
        if not math.isfinite(self.score_power) or self.score_power <= 0.0:
            raise ValueError("score_power must be finite and positive")
        if not math.isfinite(self.epsilon) or not 0.0 < self.epsilon < 1.0:
            raise ValueError("epsilon must be finite and in (0, 1)")


@dataclass(frozen=True, slots=True)
class QueryDecision:
    candidate_entity_ids: tuple[str, ...]
    ranked_entity_ids: tuple[str, ...]
    decision_entity_id: str | None
    accepted_entity_id: str | None
    probabilities: Mapping[str, float]
    reason: str


@dataclass(frozen=True, slots=True)
class FrozenQueryAcceptancePolicy:
    acceptance_threshold: float
    margin_threshold: float
    null_scale: float
    candidate_count_power: float
    calibration_system: str
    calibration_rows: int
    candidate_count_levels: int
    supported_candidate_counts: tuple[int, ...]
    correct_answers: int
    false_answers: int
    accepted: int
    max_false_answer_rate: float
    searched_policies: int
    feasible_policies: int

    def __post_init__(self) -> None:
        for name, value in (
            ("acceptance_threshold", self.acceptance_threshold),
            ("margin_threshold", self.margin_threshold),
            ("max_false_answer_rate", self.max_false_answer_rate),
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
            raise ValueError("query calibration policy counts must be positive")


def decide_query_match(
    matches: Sequence[MatchResult],
    *,
    policy: QueryDecisionPolicy | None = None,
) -> QueryDecision:
    """Turn matcher scores into one candidate/null decision without truth access."""

    selected_policy = policy or QueryDecisionPolicy()
    if not matches:
        raise ValueError("query decision requires at least one candidate")
    identifiers = [item.entity_id for item in matches]
    if any(not entity_id.strip() for entity_id in identifiers):
        raise ValueError("query candidate entity IDs cannot be empty")
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("query candidate entity IDs must be unique")
    for item in matches:
        for name, value in (("score", item.score), ("coverage", item.coverage)):
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"query match {name} must be finite and in [0, 1]")

    weights = {
        item.entity_id: max(selected_policy.epsilon, item.score)
        ** selected_policy.score_power
        for item in matches
    }
    denominator = selected_policy.null_weight + sum(weights.values())
    probabilities = {
        entity_id: weight / denominator for entity_id, weight in weights.items()
    }
    probabilities[NULL_ENTITY] = selected_policy.null_weight / denominator
    ranked = tuple(
        item.entity_id
        for item in sorted(matches, key=lambda row: (-row.score, row.entity_id))
    )
    ordered_decisions = sorted(
        probabilities.items(),
        key=lambda item: (-item[1], item[0]),
    )
    decision_key, top_probability = ordered_decisions[0]
    runner_up = ordered_decisions[1][1]
    decision_entity_id = None if decision_key == NULL_ENTITY else decision_key
    accepted_entity_id: str | None = None
    if decision_entity_id is None:
        reason = "null has highest probability"
    elif top_probability < selected_policy.acceptance_threshold:
        reason = "query decision below acceptance threshold"
    elif top_probability - runner_up < selected_policy.margin_threshold:
        reason = "query decision margin below policy"
    else:
        selected = next(item for item in matches if item.entity_id == decision_entity_id)
        if selected.coverage < selected_policy.minimum_coverage:
            reason = "query evidence coverage below policy"
        else:
            accepted_entity_id = decision_entity_id
            reason = "accepted by probability, margin, and coverage gates"
    return QueryDecision(
        tuple(sorted(identifiers)),
        ranked,
        decision_entity_id,
        accepted_entity_id,
        dict(sorted(probabilities.items())),
        reason,
    )


def build_visual_query_result(
    decision: QueryDecision,
    *,
    record_id: str,
    cluster_id: str,
    split: str,
    system: str,
    source: str,
    property_name: str,
    target_entity_id: str | None,
    horizon_seconds: float,
    distractor_count: int,
    condition: str,
    latency_seconds: float = 0.0,
    vlm_calls: int = 0,
    malformed: bool = False,
    eligible: bool = True,
) -> VisualQueryResult:
    """Attach evaluation-only target truth after the decision is frozen."""

    return VisualQueryResult(
        record_id,
        cluster_id,
        split,
        system,
        source,
        property_name,
        decision.candidate_entity_ids,
        target_entity_id,
        decision.ranked_entity_ids,
        decision.decision_entity_id,
        decision.accepted_entity_id,
        decision.probabilities,
        horizon_seconds,
        distractor_count,
        condition,
        latency_seconds,
        vlm_calls,
        malformed,
        eligible,
    )


def calibrate_query_acceptance_policy(
    rows: Sequence[VisualQueryResult],
    *,
    acceptance_thresholds: Sequence[float],
    margin_thresholds: Sequence[float],
    null_scales: Sequence[float] = (1.0,),
    candidate_count_powers: Sequence[float] = (0.0,),
    max_false_answer_rate: float,
) -> FrozenQueryAcceptancePolicy:
    """Select query gates on calibration rows only under a safety constraint."""

    if not rows:
        raise ValueError("query acceptance calibration rows cannot be empty")
    if any(row.split != "calibration" for row in rows):
        raise ValueError("query acceptance policy can only use calibration rows")
    systems = {row.system for row in rows}
    if len(systems) != 1:
        raise ValueError("query acceptance calibration requires exactly one system")
    _probability(max_false_answer_rate, "max_false_answer_rate")
    thresholds = tuple(sorted({float(value) for value in acceptance_thresholds}))
    margins = tuple(sorted({float(value) for value in margin_thresholds}))
    scales = tuple(sorted({float(value) for value in null_scales}))
    powers = tuple(sorted({float(value) for value in candidate_count_powers}))
    if not thresholds or not margins or not scales or not powers:
        raise ValueError("query acceptance, margin, and null grids cannot be empty")
    for value in (*thresholds, *margins):
        _probability(value, "query calibration grid value")
    if any(not math.isfinite(value) or value <= 0.0 for value in scales):
        raise ValueError("query null scales must be finite and positive")
    if any(not math.isfinite(value) for value in powers):
        raise ValueError("query candidate count powers must be finite")
    supported_candidate_counts = tuple(
        sorted({len(row.candidate_entity_ids) for row in rows})
    )
    candidate_count_levels = len(supported_candidate_counts)
    if candidate_count_levels < 2:
        powers = tuple(value for value in powers if abs(value) <= 1e-12)
        if not powers:
            raise ValueError(
                "nonzero query candidate count power requires multiple candidate counts"
            )
    candidates: list[tuple[float, float, float, float, int, int, int]] = []
    for threshold in thresholds:
        for margin in margins:
            for null_scale in scales:
                for count_power in powers:
                    decisions = tuple(
                        _calibrated_query_decision(
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
                    if false / len(rows) <= max_false_answer_rate:
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
        raise ValueError("no query policy satisfies the calibration safety gate")
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
    return FrozenQueryAcceptancePolicy(
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
        max_false_answer_rate,
        len(thresholds) * len(margins) * len(scales) * len(powers),
        len(candidates),
    )


def apply_query_acceptance_policy(
    rows: Sequence[VisualQueryResult],
    policy: FrozenQueryAcceptancePolicy,
) -> tuple[VisualQueryResult, ...]:
    """Apply frozen query gates without reading target identities."""

    result: list[VisualQueryResult] = []
    for row in rows:
        probabilities, decision = rescale_null_distribution(
            row.probabilities,
            candidate_count=len(row.candidate_entity_ids),
            null_scale=policy.null_scale,
            candidate_count_power=policy.candidate_count_power,
        )
        count_supported = len(row.candidate_entity_ids) in policy.supported_candidate_counts
        accepted = count_supported and _would_accept_query(
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
        result.append(VisualQueryResult(**payload))
    return tuple(result)


def _would_accept_query(
    row: VisualQueryResult,
    threshold: float,
    margin: float,
    probabilities: Mapping[str, float] | None = None,
    decision_entity_id: str | None | object = ...,
) -> bool:
    selected_probabilities = probabilities or row.probabilities
    selected_decision = (
        row.decision_entity_id if decision_entity_id is ... else decision_entity_id
    )
    if row.malformed or not row.eligible or selected_decision is None:
        return False
    assert isinstance(selected_decision, str)
    probability = selected_probabilities[selected_decision]
    runner_up = max(
        value for key, value in selected_probabilities.items() if key != selected_decision
    )
    return probability >= threshold and probability - runner_up >= margin


def _calibrated_query_decision(
    row: VisualQueryResult,
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
    return (
        decision
        if _would_accept_query(row, threshold, margin, probabilities, decision)
        else None
    )


def _probability(value: float, name: str) -> None:
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1]")
