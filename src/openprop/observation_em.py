from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

from .informative_observation import (
    ObservationAwareExponentialModel,
    ObservationEpisode,
    ObservationResult,
)


@dataclass(frozen=True, slots=True)
class EstimatedObservationProcess:
    """Training-only maximum-likelihood estimate of state and observation rates."""

    hazard_per_hour: float
    pre_transition_inspection_probability: float
    post_transition_inspection_probability: float
    detection_sensitivity: float
    false_positive_rate: float
    iterations: int
    converged: bool
    average_negative_log_likelihood_history: tuple[float, ...]

    def as_persistence_model(self) -> ObservationAwareExponentialModel:
        return ObservationAwareExponentialModel(
            self.hazard_per_hour,
            self.pre_transition_inspection_probability,
            self.post_transition_inspection_probability,
            self.detection_sensitivity,
            self.false_positive_rate,
        )


@dataclass(slots=True)
class _ExpectedCounts:
    transition_00: float = 0.0
    transition_01: float = 0.0
    state_0: float = 0.0
    state_1: float = 0.0
    inspected_state_0: float = 0.0
    inspected_state_1: float = 0.0
    positive_state_0: float = 0.0
    positive_state_1: float = 0.0
    negative_log_likelihood: float = 0.0

    def add(self, other: "_ExpectedCounts") -> None:
        self.transition_00 += other.transition_00
        self.transition_01 += other.transition_01
        self.state_0 += other.state_0
        self.state_1 += other.state_1
        self.inspected_state_0 += other.inspected_state_0
        self.inspected_state_1 += other.inspected_state_1
        self.positive_state_0 += other.positive_state_0
        self.positive_state_1 += other.positive_state_1
        self.negative_log_likelihood += other.negative_log_likelihood


def _emission_probabilities(
    result: ObservationResult,
    pre_inspection: float,
    post_inspection: float,
    sensitivity: float,
    false_positive_rate: float,
) -> tuple[float, float]:
    if result == "missing":
        return 1.0 - pre_inspection, 1.0 - post_inspection
    if result == "negative":
        return (
            pre_inspection * (1.0 - false_positive_rate),
            post_inspection * (1.0 - sensitivity),
        )
    return pre_inspection * false_positive_rate, post_inspection * sensitivity


def _episode_expectations(
    episode: ObservationEpisode,
    transition_probability: float,
    pre_inspection: float,
    post_inspection: float,
    sensitivity: float,
    false_positive_rate: float,
) -> _ExpectedCounts:
    alphas: list[tuple[float, float]] = []
    scales: list[float] = []
    previous = (1.0, 0.0)
    negative_log_likelihood = 0.0
    for result in episode.results:
        predicted = (
            previous[0] * (1.0 - transition_probability),
            previous[0] * transition_probability + previous[1],
        )
        emission = _emission_probabilities(
            result,
            pre_inspection,
            post_inspection,
            sensitivity,
            false_positive_rate,
        )
        unnormalized = (
            predicted[0] * emission[0],
            predicted[1] * emission[1],
        )
        scale = unnormalized[0] + unnormalized[1]
        if scale <= 0.0 or not math.isfinite(scale):
            raise ValueError("observation sequence has zero probability")
        current = (unnormalized[0] / scale, unnormalized[1] / scale)
        alphas.append(current)
        scales.append(scale)
        negative_log_likelihood -= math.log(scale)
        previous = current

    betas: list[tuple[float, float]] = [(1.0, 1.0)] * len(episode.results)
    for index in range(len(episode.results) - 2, -1, -1):
        next_emission = _emission_probabilities(
            episode.results[index + 1],
            pre_inspection,
            post_inspection,
            sensitivity,
            false_positive_rate,
        )
        next_beta = betas[index + 1]
        scale = scales[index + 1]
        betas[index] = (
            (
                (1.0 - transition_probability)
                * next_emission[0]
                * next_beta[0]
                + transition_probability * next_emission[1] * next_beta[1]
            )
            / scale,
            next_emission[1] * next_beta[1] / scale,
        )

    counts = _ExpectedCounts(negative_log_likelihood=negative_log_likelihood)
    for index, result in enumerate(episode.results):
        gamma_unnormalized = (
            alphas[index][0] * betas[index][0],
            alphas[index][1] * betas[index][1],
        )
        gamma_scale = gamma_unnormalized[0] + gamma_unnormalized[1]
        gamma = (
            gamma_unnormalized[0] / gamma_scale,
            gamma_unnormalized[1] / gamma_scale,
        )
        counts.state_0 += gamma[0]
        counts.state_1 += gamma[1]
        if result != "missing":
            counts.inspected_state_0 += gamma[0]
            counts.inspected_state_1 += gamma[1]
        if result == "positive":
            counts.positive_state_0 += gamma[0]
            counts.positive_state_1 += gamma[1]

        previous_alpha = (1.0, 0.0) if index == 0 else alphas[index - 1]
        emission = _emission_probabilities(
            result,
            pre_inspection,
            post_inspection,
            sensitivity,
            false_positive_rate,
        )
        current_beta = betas[index]
        transition_weights = (
            previous_alpha[0]
            * (1.0 - transition_probability)
            * emission[0]
            * current_beta[0],
            previous_alpha[0]
            * transition_probability
            * emission[1]
            * current_beta[1],
            previous_alpha[1] * emission[1] * current_beta[1],
        )
        transition_scale = sum(transition_weights)
        if transition_scale <= 0.0:
            raise ValueError("transition posterior has zero probability")
        counts.transition_00 += transition_weights[0] / transition_scale
        counts.transition_01 += transition_weights[1] / transition_scale
    return counts


def _clamp_probability(value: float, epsilon: float) -> float:
    return max(epsilon, min(1.0 - epsilon, value))


def _initial_parameters(
    episodes: tuple[ObservationEpisode, ...],
    epsilon: float,
) -> tuple[float, float, float, float]:
    interval = episodes[0].opportunity_interval_hours
    episode_hours = len(episodes[0].results) * interval
    positive_fraction = sum(
        any(result == "positive" for result in episode.results)
        for episode in episodes
    ) / len(episodes)
    hazard = -math.log(max(epsilon, 1.0 - positive_fraction)) / episode_hours
    transition_probability = _clamp_probability(
        1.0 - math.exp(-max(0.05, hazard) * interval),
        epsilon,
    )

    before_results: list[ObservationResult] = []
    after_results: list[ObservationResult] = []
    for episode in episodes:
        first_positive = next(
            (
                index
                for index, result in enumerate(episode.results)
                if result == "positive"
            ),
            None,
        )
        if first_positive is None:
            before_results.extend(episode.results)
        else:
            before_results.extend(episode.results[:first_positive])
            after_results.extend(episode.results[first_positive:])
    all_results = before_results + after_results
    overall_inspection = sum(result != "missing" for result in all_results) / len(
        all_results
    )
    pre_inspection = (
        sum(result != "missing" for result in before_results) / len(before_results)
        if before_results
        else overall_inspection
    )
    post_inspection = (
        sum(result != "missing" for result in after_results) / len(after_results)
        if after_results
        else overall_inspection
    )
    inspected_after = sum(result != "missing" for result in after_results)
    sensitivity = (
        sum(result == "positive" for result in after_results) / inspected_after
        if inspected_after
        else 0.8
    )
    return (
        transition_probability,
        _clamp_probability(pre_inspection, epsilon),
        _clamp_probability(post_inspection, epsilon),
        _clamp_probability(sensitivity, epsilon),
    )


def fit_observation_process_em(
    episodes: Iterable[ObservationEpisode],
    *,
    max_iterations: int = 300,
    tolerance: float = 1e-8,
    probability_epsilon: float = 1e-6,
    estimate_false_positive_rate: bool = False,
) -> EstimatedObservationProcess:
    """Jointly estimate persistence and observation parameters from logs only.

    Specificity remains fixed at one unless ``estimate_false_positive_rate`` is
    enabled. This explicit switch preserves the identifiable positive-anchor
    protocol and prevents silently adding a nuisance parameter.
    """

    rows = tuple(episodes)
    if not rows:
        raise ValueError("at least one training episode is required")
    if max_iterations <= 0 or tolerance <= 0:
        raise ValueError("EM iteration count and tolerance must be positive")
    if not 0.0 < probability_epsilon < 0.01:
        raise ValueError("probability_epsilon must be in (0, 0.01)")
    interval = rows[0].opportunity_interval_hours
    length = len(rows[0].results)
    if any(
        not math.isclose(
            episode.opportunity_interval_hours,
            interval,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or len(episode.results) != length
        for episode in rows
    ):
        raise ValueError("EM currently requires a common opportunity grid")
    if not any("positive" in episode.results for episode in rows):
        raise ValueError("at least one positive observation is required for identification")

    transition_probability, pre_inspection, post_inspection, sensitivity = (
        _initial_parameters(rows, probability_epsilon)
    )
    false_positive_rate = 0.05 if estimate_false_positive_rate else 0.0

    def expectation() -> _ExpectedCounts:
        total = _ExpectedCounts()
        for episode in rows:
            total.add(
                _episode_expectations(
                    episode,
                    transition_probability,
                    pre_inspection,
                    post_inspection,
                    sensitivity,
                    false_positive_rate,
                )
            )
        return total

    initial = expectation()
    history = [initial.negative_log_likelihood / len(rows)]
    converged = False
    iterations = 0
    for iterations in range(1, max_iterations + 1):
        counts = expectation()
        transition_probability = _clamp_probability(
            counts.transition_01
            / (counts.transition_00 + counts.transition_01),
            probability_epsilon,
        )
        pre_inspection = _clamp_probability(
            counts.inspected_state_0 / counts.state_0,
            probability_epsilon,
        )
        post_inspection = _clamp_probability(
            counts.inspected_state_1 / counts.state_1,
            probability_epsilon,
        )
        sensitivity = _clamp_probability(
            counts.positive_state_1 / counts.inspected_state_1,
            probability_epsilon,
        )
        if estimate_false_positive_rate:
            false_positive_rate = _clamp_probability(
                counts.positive_state_0 / counts.inspected_state_0,
                probability_epsilon,
            )
        updated = expectation()
        current_nll = updated.negative_log_likelihood / len(rows)
        if current_nll > history[-1] + 1e-9:
            raise RuntimeError("EM observation likelihood decreased")
        history.append(current_nll)
        if history[-2] - history[-1] <= tolerance:
            converged = True
            break

    hazard = -math.log1p(-transition_probability) / interval
    return EstimatedObservationProcess(
        hazard,
        pre_inspection,
        post_inspection,
        sensitivity,
        false_positive_rate,
        iterations,
        converged,
        tuple(history),
    )
