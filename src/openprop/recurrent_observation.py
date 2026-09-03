from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Iterable, Sequence

from .informative_observation import ObservationEpisode, ObservationResult
from .models import Entity, Observation, PropertyDefinition
from .persistence import ExponentialPersistenceModel
from .temporal import FreshnessResult


def _probability(value: float, *, name: str, allow_one: bool = True) -> float:
    upper_ok = value <= 1.0 if allow_one else value < 1.0
    if not math.isfinite(value) or value < 0.0 or not upper_ok:
        upper = "[0,1]" if allow_one else "[0,1)"
        raise ValueError(f"{name} must be finite and in {upper}")
    return float(value)


def _rate(value: float, *, name: str) -> float:
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return float(value)


def ctmc_transition_probabilities(
    forward_rate_per_hour: float,
    return_rate_per_hour: float,
    interval_hours: float,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Return the exact two-state CTMC transition matrix for one interval."""

    forward = _rate(forward_rate_per_hour, name="forward_rate_per_hour")
    backward = _rate(return_rate_per_hour, name="return_rate_per_hour")
    if not math.isfinite(interval_hours) or interval_hours <= 0.0:
        raise ValueError("interval_hours must be finite and positive")
    total = forward + backward
    if total == 0.0:
        return ((1.0, 0.0), (0.0, 1.0))
    moved = -math.expm1(-total * interval_hours)
    p01 = forward / total * moved
    p10 = backward / total * moved
    return ((1.0 - p01, p01), (p10, 1.0 - p10))


def recurrent_state_one_probability(
    forward_rate_per_hour: float,
    return_rate_per_hour: float,
    horizon_hours: float,
    *,
    initial_state: int = 0,
) -> float:
    """Probability that a binary CTMC is in state one at a future horizon."""

    forward = _rate(forward_rate_per_hour, name="forward_rate_per_hour")
    backward = _rate(return_rate_per_hour, name="return_rate_per_hour")
    if not math.isfinite(horizon_hours) or horizon_hours < 0.0:
        raise ValueError("horizon_hours must be finite and nonnegative")
    if initial_state not in {0, 1}:
        raise ValueError("initial_state must be zero or one")
    total = forward + backward
    if total == 0.0:
        return float(initial_state)
    stationary_one = forward / total
    decay = math.exp(-total * horizon_hours)
    if initial_state == 0:
        return stationary_one * (1.0 - decay)
    return stationary_one + (1.0 - stationary_one) * decay


@dataclass(frozen=True, slots=True)
class RecurrentObservationDataset:
    episodes: tuple[ObservationEpisode, ...]
    forward_rate_per_hour: float
    return_rate_per_hour: float
    inspection_probability_state_0: float
    inspection_probability_state_1: float
    detection_sensitivity: float
    false_positive_rate: float


@dataclass(frozen=True, slots=True)
class RecurrentExactTestRow:
    row_id: str
    horizon_hours: float
    state_one: bool


def recurrent_observation_data(
    *,
    seed: int,
    episode_count: int,
    forward_rate_per_hour: float,
    return_rate_per_hour: float,
    opportunity_interval_hours: float = 0.5,
    followup_hours: float = 12.0,
    inspection_probability_state_0: float = 0.7,
    inspection_probability_state_1: float = 0.7,
    detection_sensitivity: float = 0.9,
    false_positive_rate: float = 0.05,
) -> RecurrentObservationDataset:
    """Generate logged opportunities from a recurrent binary hidden state.

    Training episodes contain only missing/negative/positive observation results;
    latent state paths are deliberately discarded.
    """

    if episode_count <= 0:
        raise ValueError("episode_count must be positive")
    forward = _rate(forward_rate_per_hour, name="forward_rate_per_hour")
    backward = _rate(return_rate_per_hour, name="return_rate_per_hour")
    if forward == 0.0:
        raise ValueError("forward_rate_per_hour must be positive for identification")
    if not math.isfinite(opportunity_interval_hours) or opportunity_interval_hours <= 0:
        raise ValueError("opportunity_interval_hours must be finite and positive")
    if not math.isfinite(followup_hours) or followup_hours <= 0:
        raise ValueError("followup_hours must be finite and positive")
    steps_float = followup_hours / opportunity_interval_hours
    steps = int(round(steps_float))
    if steps <= 0 or not math.isclose(steps_float, steps, abs_tol=1e-12):
        raise ValueError("followup_hours must be a positive multiple of the interval")
    q0 = _probability(
        inspection_probability_state_0,
        name="inspection_probability_state_0",
    )
    q1 = _probability(
        inspection_probability_state_1,
        name="inspection_probability_state_1",
    )
    sensitivity = _probability(detection_sensitivity, name="detection_sensitivity")
    false_positive = _probability(
        false_positive_rate, name="false_positive_rate", allow_one=False
    )
    if sensitivity <= false_positive:
        raise ValueError("detection_sensitivity must exceed false_positive_rate")
    matrix = ctmc_transition_probabilities(
        forward, backward, opportunity_interval_hours
    )
    transition_rng = random.Random(seed)
    inspection_rng = random.Random(seed + 1_000_003)
    detection_rng = random.Random(seed + 2_000_003)
    episodes: list[ObservationEpisode] = []
    for index in range(episode_count):
        state = 0
        results: list[ObservationResult] = []
        for _ in range(steps):
            draw = transition_rng.random()
            if state == 0 and draw < matrix[0][1]:
                state = 1
            elif state == 1 and draw < matrix[1][0]:
                state = 0
            inspection_probability = q1 if state else q0
            if inspection_rng.random() >= inspection_probability:
                results.append("missing")
            else:
                positive_probability = sensitivity if state else false_positive
                results.append(
                    "positive"
                    if detection_rng.random() < positive_probability
                    else "negative"
                )
        episodes.append(
            ObservationEpisode(
                group_id=f"recurrent-{index:06d}",
                opportunity_interval_hours=opportunity_interval_hours,
                results=tuple(results),
            )
        )
    return RecurrentObservationDataset(
        tuple(episodes),
        forward,
        backward,
        q0,
        q1,
        sensitivity,
        false_positive,
    )


def recurrent_exact_test_rows(
    *,
    seed: int,
    row_count: int,
    forward_rate_per_hour: float,
    return_rate_per_hour: float,
    horizons_hours: Sequence[float] = (1.0, 2.0, 4.0, 8.0, 12.0),
) -> tuple[RecurrentExactTestRow, ...]:
    if row_count <= 0:
        raise ValueError("row_count must be positive")
    horizons = tuple(float(value) for value in horizons_hours)
    if not horizons or any(not math.isfinite(value) or value <= 0 for value in horizons):
        raise ValueError("horizons_hours must contain positive finite values")
    horizon_rng = random.Random(seed)
    outcome_rng = random.Random(seed + 1_000_003)
    rows: list[RecurrentExactTestRow] = []
    for index in range(row_count):
        horizon = horizons[horizon_rng.randrange(len(horizons))]
        probability = recurrent_state_one_probability(
            forward_rate_per_hour,
            return_rate_per_hour,
            horizon,
        )
        rows.append(
            RecurrentExactTestRow(
                f"recurrent-test-{index:06d}",
                horizon,
                outcome_rng.random() < probability,
            )
        )
    return tuple(rows)


def recurrent_exact_negative_log_likelihood(
    rows: Iterable[RecurrentExactTestRow],
    *,
    forward_rate_per_hour: float,
    return_rate_per_hour: float,
    epsilon: float = 1e-12,
) -> float:
    values = tuple(rows)
    if not values:
        raise ValueError("at least one exact test row is required")
    losses = []
    for row in values:
        probability = recurrent_state_one_probability(
            forward_rate_per_hour,
            return_rate_per_hour,
            row.horizon_hours,
        )
        probability = max(epsilon, min(1.0 - epsilon, probability))
        losses.append(-math.log(probability if row.state_one else 1.0 - probability))
    return sum(losses) / len(losses)


def recurrent_exact_brier_score(
    rows: Iterable[RecurrentExactTestRow],
    *,
    forward_rate_per_hour: float,
    return_rate_per_hour: float,
) -> float:
    values = tuple(rows)
    if not values:
        raise ValueError("at least one exact test row is required")
    return sum(
        (
            recurrent_state_one_probability(
                forward_rate_per_hour,
                return_rate_per_hour,
                row.horizon_hours,
            )
            - float(row.state_one)
        )
        ** 2
        for row in values
    ) / len(values)


@dataclass(frozen=True, slots=True)
class EstimatedRecurrentObservationProcess:
    forward_rate_per_hour: float
    return_rate_per_hour: float
    inspection_probability_state_0: float
    inspection_probability_state_1: float
    detection_sensitivity: float
    false_positive_rate: float
    iterations: int
    converged: bool
    average_negative_log_likelihood_history: tuple[float, ...]
    initializations_tried: int

    def as_persistence_model(self, property_name: str) -> "ReversibleBinaryPersistenceModel":
        return ReversibleBinaryPersistenceModel(
            property_name,
            self.forward_rate_per_hour,
            self.return_rate_per_hour,
        )


@dataclass(slots=True)
class _ExpectedCounts:
    transition_00: float = 0.0
    transition_01: float = 0.0
    transition_10: float = 0.0
    transition_11: float = 0.0
    state_0: float = 0.0
    state_1: float = 0.0
    inspected_state_0: float = 0.0
    inspected_state_1: float = 0.0
    positive_state_0: float = 0.0
    positive_state_1: float = 0.0
    negative_log_likelihood: float = 0.0

    def add(self, other: "_ExpectedCounts") -> None:
        for name in self.__dataclass_fields__:
            setattr(self, name, getattr(self, name) + getattr(other, name))


def _emission_probabilities(
    result: ObservationResult,
    q0: float,
    q1: float,
    sensitivity: float,
    false_positive_rate: float,
) -> tuple[float, float]:
    if result == "missing":
        return 1.0 - q0, 1.0 - q1
    if result == "negative":
        return q0 * (1.0 - false_positive_rate), q1 * (1.0 - sensitivity)
    return q0 * false_positive_rate, q1 * sensitivity


def _episode_expectations(
    episode: ObservationEpisode,
    p01: float,
    p10: float,
    q0: float,
    q1: float,
    sensitivity: float,
    false_positive_rate: float,
) -> _ExpectedCounts:
    alphas: list[tuple[float, float]] = []
    scales: list[float] = []
    previous = (1.0, 0.0)
    nll = 0.0
    for result in episode.results:
        predicted = (
            previous[0] * (1.0 - p01) + previous[1] * p10,
            previous[0] * p01 + previous[1] * (1.0 - p10),
        )
        emission = _emission_probabilities(
            result, q0, q1, sensitivity, false_positive_rate
        )
        unnormalized = (predicted[0] * emission[0], predicted[1] * emission[1])
        scale = sum(unnormalized)
        if scale <= 0.0 or not math.isfinite(scale):
            raise ValueError("recurrent observation sequence has zero probability")
        current = (unnormalized[0] / scale, unnormalized[1] / scale)
        alphas.append(current)
        scales.append(scale)
        nll -= math.log(scale)
        previous = current

    betas: list[tuple[float, float]] = [(1.0, 1.0)] * len(episode.results)
    for index in range(len(episode.results) - 2, -1, -1):
        emission = _emission_probabilities(
            episode.results[index + 1], q0, q1, sensitivity, false_positive_rate
        )
        next_beta = betas[index + 1]
        scale = scales[index + 1]
        betas[index] = (
            (
                (1.0 - p01) * emission[0] * next_beta[0]
                + p01 * emission[1] * next_beta[1]
            )
            / scale,
            (
                p10 * emission[0] * next_beta[0]
                + (1.0 - p10) * emission[1] * next_beta[1]
            )
            / scale,
        )

    counts = _ExpectedCounts(negative_log_likelihood=nll)
    for index, result in enumerate(episode.results):
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
            result, q0, q1, sensitivity, false_positive_rate
        )
        beta = betas[index]
        weights = (
            previous_alpha[0] * (1.0 - p01) * emission[0] * beta[0],
            previous_alpha[0] * p01 * emission[1] * beta[1],
            previous_alpha[1] * p10 * emission[0] * beta[0],
            previous_alpha[1] * (1.0 - p10) * emission[1] * beta[1],
        )
        scale = sum(weights)
        if scale <= 0.0:
            raise ValueError("recurrent transition posterior has zero probability")
        counts.transition_00 += weights[0] / scale
        counts.transition_01 += weights[1] / scale
        counts.transition_10 += weights[2] / scale
        counts.transition_11 += weights[3] / scale
    return counts


def _clamp(value: float, epsilon: float) -> float:
    return max(epsilon, min(1.0 - epsilon, value))


def _initial_parameters(
    episodes: tuple[ObservationEpisode, ...], epsilon: float
) -> tuple[float, float, float, float, float, float]:
    transition_01 = transition_10 = exposure_0 = exposure_1 = 0.0
    inspected = total = positive = 0
    for episode in episodes:
        previous = 0
        for result in episode.results:
            total += 1
            if result == "missing":
                continue
            inspected += 1
            state = 1 if result == "positive" else 0
            if previous == 0:
                exposure_0 += 1.0
                transition_01 += float(state == 1)
            else:
                exposure_1 += 1.0
                transition_10 += float(state == 0)
            previous = state
            positive += int(result == "positive")
    q = _clamp(inspected / total, epsilon)
    raw_p01 = _clamp(transition_01 / max(1.0, exposure_0), epsilon)
    raw_p10 = _clamp(transition_10 / max(1.0, exposure_1), epsilon)
    positive_fraction = positive / max(1, inspected)
    sensitivity = _clamp(max(0.65, min(0.95, 0.75 + positive_fraction / 4.0)), epsilon)
    false_positive = _clamp(min(0.2, max(0.02, positive_fraction / 5.0)), epsilon)
    return raw_p01, raw_p10, q, q, sensitivity, false_positive


def _transition_probabilities_to_rates(
    p01: float, p10: float, interval_hours: float
) -> tuple[float, float]:
    moved = p01 + p10
    if moved <= 0.0:
        return 0.0, 0.0
    if moved >= 1.0:
        raise ValueError("recurrent transition probabilities must sum below one")
    total_rate = -math.log1p(-moved) / interval_hours
    return total_rate * p01 / moved, total_rate * p10 / moved


def _fit_from_start(
    episodes: tuple[ObservationEpisode, ...],
    start: tuple[float, float, float, float, float, float],
    *,
    max_iterations: int,
    tolerance: float,
    epsilon: float,
) -> tuple[tuple[float, float, float, float, float, float], tuple[float, ...], int, bool]:
    p01, p10, q0, q1, sensitivity, false_positive = start

    def expectation() -> _ExpectedCounts:
        total = _ExpectedCounts()
        for episode in episodes:
            total.add(
                _episode_expectations(
                    episode, p01, p10, q0, q1, sensitivity, false_positive
                )
            )
        return total

    history = [expectation().negative_log_likelihood / len(episodes)]
    converged = False
    iterations = 0
    for iterations in range(1, max_iterations + 1):
        counts = expectation()
        p01 = _clamp(
            counts.transition_01 / (counts.transition_00 + counts.transition_01),
            epsilon,
        )
        p10 = _clamp(
            counts.transition_10 / (counts.transition_10 + counts.transition_11),
            epsilon,
        )
        moved = p01 + p10
        if moved >= 1.0 - epsilon:
            scale = (1.0 - 2.0 * epsilon) / moved
            p01 *= scale
            p10 *= scale
        q0 = _clamp(counts.inspected_state_0 / counts.state_0, epsilon)
        q1 = _clamp(counts.inspected_state_1 / counts.state_1, epsilon)
        sensitivity = _clamp(
            counts.positive_state_1 / counts.inspected_state_1, epsilon
        )
        false_positive = _clamp(
            counts.positive_state_0 / counts.inspected_state_0, epsilon
        )
        if sensitivity <= false_positive:
            raise RuntimeError("recurrent EM lost the positive-state emission ordering")
        current = expectation().negative_log_likelihood / len(episodes)
        if current > history[-1] + 1e-9:
            raise RuntimeError("recurrent EM observation likelihood decreased")
        history.append(current)
        if history[-2] - history[-1] <= tolerance:
            converged = True
            break
    return (
        (p01, p10, q0, q1, sensitivity, false_positive),
        tuple(history),
        iterations,
        converged,
    )


def fit_recurrent_observation_em(
    episodes: Iterable[ObservationEpisode],
    *,
    max_iterations: int = 300,
    tolerance: float = 1e-8,
    probability_epsilon: float = 1e-6,
) -> EstimatedRecurrentObservationProcess:
    """Estimate reversible state and observation rates from logged sequences only."""

    rows = tuple(episodes)
    if not rows:
        raise ValueError("at least one recurrent training episode is required")
    if max_iterations <= 0 or tolerance <= 0:
        raise ValueError("EM iteration count and tolerance must be positive")
    if not 0.0 < probability_epsilon < 0.01:
        raise ValueError("probability_epsilon must be in (0, 0.01)")
    interval = rows[0].opportunity_interval_hours
    length = len(rows[0].results)
    if length == 0 or any(
        len(episode.results) != length
        or not math.isclose(
            episode.opportunity_interval_hours, interval, abs_tol=1e-12
        )
        for episode in rows
    ):
        raise ValueError("recurrent EM requires a common nonempty opportunity grid")
    if not any("positive" in episode.results for episode in rows):
        raise ValueError("recurrent EM requires at least one positive observation")
    base = _initial_parameters(rows, probability_epsilon)
    starts = (
        base,
        (0.03, 0.03, base[2], base[3], 0.9, 0.03),
        (0.12, 0.04, base[2], base[3], 0.9, 0.03),
        (0.04, 0.12, base[2], base[3], 0.9, 0.03),
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
        raise RuntimeError("all recurrent EM initializations failed")
    parameters, history, iterations, converged = min(
        candidates, key=lambda candidate: candidate[1][-1]
    )
    p01, p10, q0, q1, sensitivity, false_positive = parameters
    forward, backward = _transition_probabilities_to_rates(p01, p10, interval)
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


@dataclass(slots=True)
class ReversibleBinaryPersistenceModel:
    """Matcher adapter for a property-specific recurrent binary CTMC."""

    property_name: str
    forward_rate_per_hour: float
    return_rate_per_hour: float
    fallback: ExponentialPersistenceModel = field(
        default_factory=ExponentialPersistenceModel
    )

    def __post_init__(self) -> None:
        if not self.property_name.strip():
            raise ValueError("property_name cannot be empty")
        self.forward_rate_per_hour = _rate(
            self.forward_rate_per_hour, name="forward_rate_per_hour"
        )
        self.return_rate_per_hour = _rate(
            self.return_rate_per_hour, name="return_rate_per_hour"
        )

    def predict(
        self,
        definition: PropertyDefinition,
        observation: Observation,
        entity: Entity,
        *,
        as_of: float,
    ) -> FreshnessResult:
        if (
            observation.timestamp is None
            or definition.name.casefold() != self.property_name.casefold()
        ):
            return self.fallback.predict(definition, observation, entity, as_of=as_of)
        if type(observation.value) is not bool:
            raise ValueError("reversible binary persistence requires a boolean observation")
        age_seconds = max(0.0, as_of - observation.timestamp)
        state_one_probability = recurrent_state_one_probability(
            self.forward_rate_per_hour,
            self.return_rate_per_hour,
            age_seconds / 3600.0,
            initial_state=int(observation.value),
        )
        time_retention = (
            state_one_probability if observation.value else 1.0 - state_one_probability
        )
        baseline = self.fallback.predict(
            definition, observation, entity, as_of=as_of
        )
        event_retention = baseline.event_retention
        return FreshnessResult(
            max(0.0, min(1.0, time_retention * event_retention)),
            age_seconds,
            time_retention,
            event_retention,
            baseline.applied_events,
        )
