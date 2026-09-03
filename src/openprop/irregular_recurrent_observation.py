from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Iterable

from .informative_observation import ObservationEpisode, ObservationResult
from .recurrent_observation import (
    EstimatedRecurrentObservationProcess,
    ctmc_transition_probabilities,
)


def _emission_probabilities(
    result: ObservationResult,
    q0: float,
    q1: float,
    sensitivity: float,
    false_positive: float,
) -> tuple[float, float]:
    if result == "missing":
        return 1.0 - q0, 1.0 - q1
    if result == "negative":
        return q0 * (1.0 - false_positive), q1 * (1.0 - sensitivity)
    return q0 * false_positive, q1 * sensitivity


@dataclass(frozen=True, slots=True)
class IrregularObservationEpisode:
    """Logged outcomes with the elapsed time before each opportunity."""

    group_id: str
    intervals_hours: tuple[float, ...]
    results: tuple[ObservationResult, ...]

    def __post_init__(self) -> None:
        if not self.group_id:
            raise ValueError("group_id must be nonempty")
        if not self.results or len(self.intervals_hours) != len(self.results):
            raise ValueError("irregular intervals and results must have equal nonzero length")
        if any(
            not math.isfinite(interval) or interval <= 0.0
            for interval in self.intervals_hours
        ):
            raise ValueError("irregular intervals must be finite and positive")
        if any(result not in {"missing", "negative", "positive"} for result in self.results):
            raise ValueError("invalid observation result")


@dataclass(frozen=True, slots=True)
class IrregularRecurrentObservationDataset:
    episodes: tuple[IrregularObservationEpisode, ...]
    forward_rate_per_hour: float
    return_rate_per_hour: float
    mean_interval_hours: float
    gap_contrast: float
    inspection_probability_state_0: float
    inspection_probability_state_1: float
    detection_sensitivity: float
    false_positive_rate: float


def irregular_recurrent_observation_data(
    *,
    seed: int,
    episode_count: int,
    observation_count: int = 16,
    forward_rate_per_hour: float = 0.30,
    return_rate_per_hour: float = 0.30,
    mean_interval_hours: float = 0.75,
    gap_contrast: float = 0.80,
    inspection_probability_state_0: float = 0.70,
    inspection_probability_state_1: float = 0.75,
    detection_sensitivity: float = 0.90,
    false_positive_rate: float = 0.04,
) -> IrregularRecurrentObservationDataset:
    """Generate paired short/long schedules without retaining latent paths.

    Every episode has the same total follow-up and mean interval. Only the order
    of short and long gaps varies, so support length cannot explain a comparison
    against a mean-grid fit.
    """

    if episode_count <= 0:
        raise ValueError("episode_count must be positive")
    if observation_count <= 0 or observation_count % 2:
        raise ValueError("observation_count must be positive and even")
    if not math.isfinite(mean_interval_hours) or mean_interval_hours <= 0.0:
        raise ValueError("mean_interval_hours must be finite and positive")
    if not math.isfinite(gap_contrast) or not 0.0 <= gap_contrast < 1.0:
        raise ValueError("gap_contrast must lie in [0, 1)")
    parameters = (
        forward_rate_per_hour,
        return_rate_per_hour,
        inspection_probability_state_0,
        inspection_probability_state_1,
        detection_sensitivity,
        false_positive_rate,
    )
    if any(not math.isfinite(value) for value in parameters[:2]) or any(
        value < 0.0 for value in parameters[:2]
    ):
        raise ValueError("transition rates must be finite and nonnegative")
    if forward_rate_per_hour == 0.0:
        raise ValueError("forward_rate_per_hour must be positive for identification")
    if any(not 0.0 < value < 1.0 for value in parameters[2:]):
        raise ValueError("observation probabilities must lie strictly inside (0, 1)")
    if detection_sensitivity <= false_positive_rate:
        raise ValueError("detection_sensitivity must exceed false_positive_rate")

    long_count = max(1, observation_count // 8)
    short_count = observation_count - long_count
    short = mean_interval_hours * (1.0 - gap_contrast)
    long = mean_interval_hours * (
        1.0 + gap_contrast * short_count / long_count
    )
    schedule_rng = random.Random(seed + 3_000_003)
    transition_rng = random.Random(seed)
    inspection_rng = random.Random(seed + 1_000_003)
    detection_rng = random.Random(seed + 2_000_003)
    episodes: list[IrregularObservationEpisode] = []
    for episode_index in range(episode_count):
        intervals = [short] * short_count + [long] * long_count
        schedule_rng.shuffle(intervals)
        state = 0
        results: list[ObservationResult] = []
        for interval in intervals:
            matrix = ctmc_transition_probabilities(
                forward_rate_per_hour, return_rate_per_hour, interval
            )
            transition_draw = transition_rng.random()
            if state == 0 and transition_draw < matrix[0][1]:
                state = 1
            elif state == 1 and transition_draw < matrix[1][0]:
                state = 0
            inspection_probability = (
                inspection_probability_state_1
                if state
                else inspection_probability_state_0
            )
            if inspection_rng.random() >= inspection_probability:
                results.append("missing")
            else:
                positive_probability = (
                    detection_sensitivity if state else false_positive_rate
                )
                results.append(
                    "positive"
                    if detection_rng.random() < positive_probability
                    else "negative"
                )
        episodes.append(
            IrregularObservationEpisode(
                group_id=f"irregular-{episode_index:06d}",
                intervals_hours=tuple(intervals),
                results=tuple(results),
            )
        )
    return IrregularRecurrentObservationDataset(
        tuple(episodes),
        forward_rate_per_hour,
        return_rate_per_hour,
        mean_interval_hours,
        gap_contrast,
        inspection_probability_state_0,
        inspection_probability_state_1,
        detection_sensitivity,
        false_positive_rate,
    )


def collapse_to_mean_grid(
    episodes: Iterable[IrregularObservationEpisode],
) -> tuple[ObservationEpisode, ...]:
    """Return the deliberately misspecified common-mean-grid baseline."""

    rows = tuple(episodes)
    if not rows:
        raise ValueError("at least one irregular episode is required")
    count = sum(len(episode.results) for episode in rows)
    mean_interval = sum(
        sum(episode.intervals_hours) for episode in rows
    ) / count
    return tuple(
        ObservationEpisode(
            group_id=episode.group_id,
            opportunity_interval_hours=mean_interval,
            results=episode.results,
        )
        for episode in rows
    )


@dataclass(slots=True)
class _IrregularExpectedCounts:
    state_0: float = 0.0
    state_1: float = 0.0
    inspected_state_0: float = 0.0
    inspected_state_1: float = 0.0
    positive_state_0: float = 0.0
    positive_state_1: float = 0.0
    negative_log_likelihood: float = 0.0
    transitions_by_interval: dict[float, list[float]] = field(default_factory=dict)

    def add(self, other: "_IrregularExpectedCounts") -> None:
        self.state_0 += other.state_0
        self.state_1 += other.state_1
        self.inspected_state_0 += other.inspected_state_0
        self.inspected_state_1 += other.inspected_state_1
        self.positive_state_0 += other.positive_state_0
        self.positive_state_1 += other.positive_state_1
        self.negative_log_likelihood += other.negative_log_likelihood
        for interval, values in other.transitions_by_interval.items():
            target = self.transitions_by_interval.setdefault(interval, [0.0] * 4)
            for index, value in enumerate(values):
                target[index] += value


def _episode_expectations(
    episode: IrregularObservationEpisode,
    forward_rate: float,
    return_rate: float,
    q0: float,
    q1: float,
    sensitivity: float,
    false_positive: float,
    matrix_by_interval: dict[float, tuple[tuple[float, float], tuple[float, float]]],
) -> _IrregularExpectedCounts:
    matrices = tuple(
        matrix_by_interval[interval]
        for interval in episode.intervals_hours
    )
    alphas: list[tuple[float, float]] = []
    scales: list[float] = []
    previous = (1.0, 0.0)
    nll = 0.0
    for matrix, result in zip(matrices, episode.results, strict=True):
        predicted = (
            previous[0] * matrix[0][0] + previous[1] * matrix[1][0],
            previous[0] * matrix[0][1] + previous[1] * matrix[1][1],
        )
        emission = _emission_probabilities(
            result, q0, q1, sensitivity, false_positive
        )
        unnormalized = (
            predicted[0] * emission[0],
            predicted[1] * emission[1],
        )
        scale = sum(unnormalized)
        if scale <= 0.0 or not math.isfinite(scale):
            raise ValueError("irregular observation sequence has zero probability")
        previous = (unnormalized[0] / scale, unnormalized[1] / scale)
        alphas.append(previous)
        scales.append(scale)
        nll -= math.log(scale)

    betas: list[tuple[float, float]] = [(1.0, 1.0)] * len(episode.results)
    for index in range(len(episode.results) - 2, -1, -1):
        matrix = matrices[index + 1]
        emission = _emission_probabilities(
            episode.results[index + 1], q0, q1, sensitivity, false_positive
        )
        next_beta = betas[index + 1]
        scale = scales[index + 1]
        betas[index] = (
            (
                matrix[0][0] * emission[0] * next_beta[0]
                + matrix[0][1] * emission[1] * next_beta[1]
            )
            / scale,
            (
                matrix[1][0] * emission[0] * next_beta[0]
                + matrix[1][1] * emission[1] * next_beta[1]
            )
            / scale,
        )

    counts = _IrregularExpectedCounts(negative_log_likelihood=nll)
    for index, (interval, result, matrix) in enumerate(
        zip(episode.intervals_hours, episode.results, matrices, strict=True)
    ):
        gamma_raw = (
            alphas[index][0] * betas[index][0],
            alphas[index][1] * betas[index][1],
        )
        gamma_scale = sum(gamma_raw)
        gamma = (gamma_raw[0] / gamma_scale, gamma_raw[1] / gamma_scale)
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
            result, q0, q1, sensitivity, false_positive
        )
        beta = betas[index]
        weights = (
            previous_alpha[0] * matrix[0][0] * emission[0] * beta[0],
            previous_alpha[0] * matrix[0][1] * emission[1] * beta[1],
            previous_alpha[1] * matrix[1][0] * emission[0] * beta[0],
            previous_alpha[1] * matrix[1][1] * emission[1] * beta[1],
        )
        scale = sum(weights)
        if scale <= 0.0:
            raise ValueError("irregular transition posterior has zero probability")
        target = counts.transitions_by_interval.setdefault(interval, [0.0] * 4)
        for transition_index, weight in enumerate(weights):
            target[transition_index] += weight / scale
    return counts


def _clamp(value: float, epsilon: float) -> float:
    return max(epsilon, min(1.0 - epsilon, value))


def _transition_objective(
    transitions: dict[float, list[float]],
    forward_rate: float,
    return_rate: float,
) -> float:
    value = 0.0
    for interval, counts in transitions.items():
        matrix = ctmc_transition_probabilities(
            forward_rate, return_rate, interval
        )
        probabilities = (
            matrix[0][0], matrix[0][1], matrix[1][0], matrix[1][1]
        )
        value += sum(
            count * math.log(max(1e-300, probability))
            for count, probability in zip(counts, probabilities, strict=True)
        )
    return value


def _golden_log_rate(
    objective,
    current_rate: float,
    *,
    minimum_rate: float = 1e-6,
    maximum_rate: float = 5.0,
    iterations: int = 36,
) -> float:
    lower = math.log(minimum_rate)
    upper = math.log(maximum_rate)
    ratio = (math.sqrt(5.0) - 1.0) / 2.0
    left = upper - ratio * (upper - lower)
    right = lower + ratio * (upper - lower)
    left_value = objective(math.exp(left))
    right_value = objective(math.exp(right))
    for _ in range(iterations):
        if left_value < right_value:
            lower = left
            left = right
            left_value = right_value
            right = lower + ratio * (upper - lower)
            right_value = objective(math.exp(right))
        else:
            upper = right
            right = left
            right_value = left_value
            left = upper - ratio * (upper - lower)
            left_value = objective(math.exp(left))
    candidate = math.exp((lower + upper) / 2.0)
    return max(
        (current_rate, candidate),
        key=objective,
    )


def _maximize_transition_rates(
    transitions: dict[float, list[float]],
    forward_rate: float,
    return_rate: float,
) -> tuple[float, float]:
    for _ in range(8):
        previous = (forward_rate, return_rate)
        forward_rate = _golden_log_rate(
            lambda candidate: _transition_objective(
                transitions, candidate, return_rate
            ),
            forward_rate,
        )
        return_rate = _golden_log_rate(
            lambda candidate: _transition_objective(
                transitions, forward_rate, candidate
            ),
            return_rate,
        )
        if max(
            abs(math.log(forward_rate / previous[0])),
            abs(math.log(return_rate / previous[1])),
        ) < 1e-7:
            break
    return forward_rate, return_rate


def _initial_emissions(
    episodes: tuple[IrregularObservationEpisode, ...], epsilon: float
) -> tuple[float, float, float, float]:
    before: list[ObservationResult] = []
    after: list[ObservationResult] = []
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
            before.extend(episode.results)
        else:
            before.extend(episode.results[:first_positive])
            after.extend(episode.results[first_positive:])
    all_results = before + after
    overall_q = sum(result != "missing" for result in all_results) / len(all_results)

    def inspection(values: list[ObservationResult]) -> float:
        if not values:
            return overall_q
        return sum(result != "missing" for result in values) / len(values)

    inspected_after = sum(result != "missing" for result in after)
    sensitivity = (
        sum(result == "positive" for result in after) / inspected_after
        if inspected_after
        else 0.85
    )
    inspected_before = sum(result != "missing" for result in before)
    false_positive = (
        sum(result == "positive" for result in before) / inspected_before
        if inspected_before
        else 0.03
    )
    return (
        _clamp(inspection(before), epsilon),
        _clamp(inspection(after), epsilon),
        _clamp(max(sensitivity, false_positive + 0.05), epsilon),
        _clamp(false_positive, epsilon),
    )


def _fit_from_start(
    episodes: tuple[IrregularObservationEpisode, ...],
    start: tuple[float, float, float, float, float, float],
    *,
    max_iterations: int,
    tolerance: float,
    epsilon: float,
) -> tuple[tuple[float, ...], tuple[float, ...], int, bool]:
    forward, backward, q0, q1, sensitivity, false_positive = start

    def expectation() -> _IrregularExpectedCounts:
        total = _IrregularExpectedCounts()
        matrix_by_interval = {
            interval: ctmc_transition_probabilities(forward, backward, interval)
            for interval in {
                value for episode in episodes for value in episode.intervals_hours
            }
        }
        for episode in episodes:
            total.add(
                _episode_expectations(
                    episode,
                    forward,
                    backward,
                    q0,
                    q1,
                    sensitivity,
                    false_positive,
                    matrix_by_interval,
                )
            )
        return total

    history = [expectation().negative_log_likelihood / len(episodes)]
    converged = False
    iterations = 0
    for iterations in range(1, max_iterations + 1):
        counts = expectation()
        forward, backward = _maximize_transition_rates(
            counts.transitions_by_interval, forward, backward
        )
        q0 = _clamp(counts.inspected_state_0 / counts.state_0, epsilon)
        q1 = _clamp(counts.inspected_state_1 / counts.state_1, epsilon)
        sensitivity = _clamp(
            counts.positive_state_1 / counts.inspected_state_1, epsilon
        )
        false_positive = _clamp(
            counts.positive_state_0 / counts.inspected_state_0, epsilon
        )
        if sensitivity <= false_positive:
            raise RuntimeError("irregular EM lost the positive-state emission ordering")
        current = expectation().negative_log_likelihood / len(episodes)
        if current > history[-1] + 1e-8:
            raise RuntimeError("irregular EM observation likelihood decreased")
        history.append(current)
        if history[-2] - history[-1] <= tolerance:
            converged = True
            break
    return (
        (forward, backward, q0, q1, sensitivity, false_positive),
        tuple(history),
        iterations,
        converged,
    )


def fit_irregular_recurrent_observation_em(
    episodes: Iterable[IrregularObservationEpisode],
    *,
    max_iterations: int = 200,
    tolerance: float = 1e-7,
    probability_epsilon: float = 1e-6,
) -> EstimatedRecurrentObservationProcess:
    """Estimate a binary CTMC using each logged elapsed interval exactly."""

    rows = tuple(episodes)
    if not rows:
        raise ValueError("at least one irregular training episode is required")
    if len({episode.group_id for episode in rows}) != len(rows):
        raise ValueError("irregular episode group IDs must be unique")
    if max_iterations <= 0 or tolerance <= 0.0:
        raise ValueError("EM iteration count and tolerance must be positive")
    if not 0.0 < probability_epsilon < 0.01:
        raise ValueError("probability_epsilon must lie in (0, 0.01)")
    if not any("positive" in episode.results for episode in rows):
        raise ValueError("irregular EM requires at least one positive observation")

    q0, q1, sensitivity, false_positive = _initial_emissions(
        rows, probability_epsilon
    )
    starts = (
        (0.20, 0.10, q0, q1, sensitivity, false_positive),
        (0.30, 0.30, q0, q1, 0.90, 0.03),
        (0.60, 0.15, q0, q1, 0.90, 0.03),
        (0.15, 0.60, q0, q1, 0.90, 0.03),
    )
    candidates = []
    for start in starts:
        try:
            candidates.append(
                _fit_from_start(
                    rows,
                    start,
                    max_iterations=max_iterations,
                    tolerance=tolerance,
                    epsilon=probability_epsilon,
                )
            )
        except RuntimeError:
            continue
    if not candidates:
        raise RuntimeError("all irregular EM initializations failed")
    parameters, history, iterations, converged = min(
        candidates, key=lambda candidate: candidate[1][-1]
    )
    forward, backward, q0, q1, sensitivity, false_positive = parameters
    return EstimatedRecurrentObservationProcess(
        forward,
        backward,
        q0,
        q1,
        sensitivity,
        false_positive,
        iterations,
        converged,
        history,
        len(starts),
    )


def irregular_observation_negative_log_likelihood(
    episodes: Iterable[IrregularObservationEpisode],
    process: EstimatedRecurrentObservationProcess,
) -> float:
    """Evaluate logged outcomes using their actual elapsed intervals."""

    rows = tuple(episodes)
    if not rows:
        raise ValueError("at least one irregular episode is required")
    matrix_by_interval = {
        interval: ctmc_transition_probabilities(
            process.forward_rate_per_hour,
            process.return_rate_per_hour,
            interval,
        )
        for interval in {
            value for episode in rows for value in episode.intervals_hours
        }
    }
    return sum(
        _episode_expectations(
            episode,
            process.forward_rate_per_hour,
            process.return_rate_per_hour,
            process.inspection_probability_state_0,
            process.inspection_probability_state_1,
            process.detection_sensitivity,
            process.false_positive_rate,
            matrix_by_interval,
        ).negative_log_likelihood
        for episode in rows
    ) / len(rows)
