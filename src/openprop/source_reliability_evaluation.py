from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Iterable

from .recurrent_observation import ctmc_transition_probabilities
from .source_reliability_observation import (
    EstimatedSourceReliabilityProcess,
    SourceEmissionParameters,
    SourceObservationResult,
    SourcedObservationEpisode,
)


@dataclass(frozen=True, slots=True)
class SourceGroundingTestRow:
    """Evaluation row whose current truth must never be passed to the matcher."""

    row_id: str
    observation_history: SourcedObservationEpisode
    current_truth: bool


def sourced_grounding_test_rows(
    *,
    seed: int,
    row_count: int,
    forward_rate_per_hour: float,
    return_rate_per_hour: float,
    source_parameters: tuple[SourceEmissionParameters, ...],
    opportunity_interval_hours: float = 0.5,
    followup_hours: float = 12.0,
) -> tuple[SourceGroundingTestRow, ...]:
    if row_count <= 0 or not source_parameters:
        raise ValueError("row_count and source_parameters must be nonempty")
    steps_float = followup_hours / opportunity_interval_hours
    steps = int(round(steps_float))
    if steps <= 0 or not math.isclose(steps_float, steps, abs_tol=1e-12):
        raise ValueError("followup_hours must be a positive multiple of the interval")
    matrix = ctmc_transition_probabilities(
        forward_rate_per_hour, return_rate_per_hour, opportunity_interval_hours
    )
    transition_rng = random.Random(seed)
    source_rngs = {
        source.source_id: (
            random.Random(seed + 1_000_003 + 2 * index),
            random.Random(seed + 1_000_004 + 2 * index),
        )
        for index, source in enumerate(source_parameters)
    }
    rows = []
    for row_index in range(row_count):
        state = 0
        history = []
        for _ in range(steps):
            draw = transition_rng.random()
            if state == 0 and draw < matrix[0][1]:
                state = 1
            elif state == 1 and draw < matrix[1][0]:
                state = 0
            outcomes = []
            for source in source_parameters:
                inspection_rng, detection_rng = source_rngs[source.source_id]
                q = (
                    source.inspection_probability_state_1
                    if state else source.inspection_probability_state_0
                )
                if inspection_rng.random() >= q:
                    result = "missing"
                else:
                    positive_probability = (
                        source.detection_sensitivity if state else source.false_positive_rate
                    )
                    result = (
                        "positive"
                        if detection_rng.random() < positive_probability
                        else "negative"
                    )
                outcomes.append(SourceObservationResult(source.source_id, result))
            history.append(tuple(outcomes))
        episode = SourcedObservationEpisode(
            f"source-grounding-{row_index:06d}",
            opportunity_interval_hours,
            tuple(history),
        )
        rows.append(SourceGroundingTestRow(episode.group_id, episode, bool(state)))
    return tuple(rows)


def source_filtered_state_one_probability(
    observation_history: SourcedObservationEpisode,
    process: EstimatedSourceReliabilityProcess,
) -> float:
    parameters = {value.source_id: value for value in process.source_parameters}
    if tuple(parameters) != observation_history.source_ids:
        raise ValueError("process and observation history source IDs must align")
    matrix = ctmc_transition_probabilities(
        process.forward_rate_per_hour,
        process.return_rate_per_hour,
        observation_history.opportunity_interval_hours,
    )
    posterior = (1.0, 0.0)
    for step in observation_history.results_by_step:
        predicted = (
            posterior[0] * matrix[0][0] + posterior[1] * matrix[1][0],
            posterior[0] * matrix[0][1] + posterior[1] * matrix[1][1],
        )
        emissions = [1.0, 1.0]
        for item in step:
            source = parameters[item.source_id]
            if item.result == "missing":
                pair = (
                    1.0 - source.inspection_probability_state_0,
                    1.0 - source.inspection_probability_state_1,
                )
            elif item.result == "negative":
                pair = (
                    source.inspection_probability_state_0
                    * (1.0 - source.false_positive_rate),
                    source.inspection_probability_state_1
                    * (1.0 - source.detection_sensitivity),
                )
            else:
                pair = (
                    source.inspection_probability_state_0
                    * source.false_positive_rate,
                    source.inspection_probability_state_1
                    * source.detection_sensitivity,
                )
            emissions[0] *= pair[0]
            emissions[1] *= pair[1]
        unnormalized = (
            predicted[0] * emissions[0], predicted[1] * emissions[1]
        )
        scale = sum(unnormalized)
        if scale <= 0.0:
            raise ValueError("observation history has zero probability under process")
        posterior = (unnormalized[0] / scale, unnormalized[1] / scale)
    return posterior[1]


def source_grounding_negative_log_likelihood(
    rows: Iterable[SourceGroundingTestRow],
    process: EstimatedSourceReliabilityProcess,
    *,
    epsilon: float = 1e-12,
) -> float:
    values = tuple(rows)
    if not values:
        raise ValueError("at least one source grounding test row is required")
    losses = []
    for row in values:
        probability = source_filtered_state_one_probability(
            row.observation_history, process
        )
        probability = max(epsilon, min(1.0 - epsilon, probability))
        losses.append(-math.log(probability if row.current_truth else 1.0 - probability))
    return sum(losses) / len(losses)


def source_grounding_brier_score(
    rows: Iterable[SourceGroundingTestRow],
    process: EstimatedSourceReliabilityProcess,
) -> float:
    values = tuple(rows)
    if not values:
        raise ValueError("at least one source grounding test row is required")
    return sum(
        (
            source_filtered_state_one_probability(row.observation_history, process)
            - float(row.current_truth)
        )
        ** 2
        for row in values
    ) / len(values)
