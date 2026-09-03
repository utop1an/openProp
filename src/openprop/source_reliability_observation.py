from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Iterable, Literal, Mapping

from .informative_observation import ObservationResult
from .recurrent_observation import (
    ReversibleBinaryPersistenceModel,
    ctmc_transition_probabilities,
)


@dataclass(frozen=True, slots=True)
class SourceObservationResult:
    """One source result; source identity is provenance, not a property value."""

    source_id: str
    result: ObservationResult

    def __post_init__(self) -> None:
        if not self.source_id.strip():
            raise ValueError("source_id cannot be empty")
        if self.result not in {"missing", "negative", "positive"}:
            raise ValueError("invalid source observation result")


@dataclass(frozen=True, slots=True)
class SourcedObservationEpisode:
    group_id: str
    opportunity_interval_hours: float
    results_by_step: tuple[tuple[SourceObservationResult, ...], ...]

    def __post_init__(self) -> None:
        if not self.group_id.strip():
            raise ValueError("group_id cannot be empty")
        if not math.isfinite(self.opportunity_interval_hours) or self.opportunity_interval_hours <= 0:
            raise ValueError("opportunity_interval_hours must be finite and positive")
        if not self.results_by_step:
            raise ValueError("an episode must contain at least one opportunity")
        expected = tuple(item.source_id for item in self.results_by_step[0])
        if not expected or len(set(expected)) != len(expected):
            raise ValueError("each step requires unique nonempty sources")
        if any(tuple(item.source_id for item in step) != expected for step in self.results_by_step):
            raise ValueError("all steps must use the same ordered source IDs")

    @property
    def source_ids(self) -> tuple[str, ...]:
        return tuple(item.source_id for item in self.results_by_step[0])


@dataclass(frozen=True, slots=True)
class SourceEmissionParameters:
    source_id: str
    inspection_probability_state_0: float
    inspection_probability_state_1: float
    detection_sensitivity: float
    false_positive_rate: float


@dataclass(frozen=True, slots=True)
class SourcedObservationDataset:
    episodes: tuple[SourcedObservationEpisode, ...]
    forward_rate_per_hour: float
    return_rate_per_hour: float
    source_parameters: tuple[SourceEmissionParameters, ...]


@dataclass(frozen=True, slots=True)
class EstimatedSourceReliabilityProcess:
    forward_rate_per_hour: float
    return_rate_per_hour: float
    source_parameters: tuple[SourceEmissionParameters, ...]
    pooled_sources: bool
    iterations: int
    converged: bool
    average_negative_log_likelihood_history: tuple[float, ...]
    initializations_tried: int

    def as_persistence_model(self, property_name: str) -> ReversibleBinaryPersistenceModel:
        return ReversibleBinaryPersistenceModel(
            property_name, self.forward_rate_per_hour, self.return_rate_per_hour
        )


def _probability(value: float, name: str) -> float:
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be finite and in [0,1]")
    return float(value)


def sourced_recurrent_observation_data(
    *,
    seed: int,
    episode_count: int,
    forward_rate_per_hour: float,
    return_rate_per_hour: float,
    source_parameters: Iterable[SourceEmissionParameters],
    opportunity_interval_hours: float = 0.5,
    followup_hours: float = 12.0,
) -> SourcedObservationDataset:
    """Generate source-labelled logs while discarding every latent state path."""

    sources = tuple(source_parameters)
    if episode_count <= 0 or not sources:
        raise ValueError("episode_count and source_parameters must be nonempty")
    if len({source.source_id for source in sources}) != len(sources):
        raise ValueError("source IDs must be unique")
    for source in sources:
        if not source.source_id.strip():
            raise ValueError("source_id cannot be empty")
        values = (
            _probability(source.inspection_probability_state_0, "inspection_probability_state_0"),
            _probability(source.inspection_probability_state_1, "inspection_probability_state_1"),
            _probability(source.detection_sensitivity, "detection_sensitivity"),
            _probability(source.false_positive_rate, "false_positive_rate"),
        )
        if values[2] <= values[3]:
            raise ValueError("each source sensitivity must exceed its false-positive rate")
    if not math.isfinite(followup_hours) or followup_hours <= 0:
        raise ValueError("followup_hours must be finite and positive")
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
        for index, source in enumerate(sources)
    }
    episodes: list[SourcedObservationEpisode] = []
    for episode_index in range(episode_count):
        state = 0
        steps_out: list[tuple[SourceObservationResult, ...]] = []
        for _ in range(steps):
            draw = transition_rng.random()
            if state == 0 and draw < matrix[0][1]:
                state = 1
            elif state == 1 and draw < matrix[1][0]:
                state = 0
            source_results = []
            for source in sources:
                inspection_rng, detection_rng = source_rngs[source.source_id]
                q = (
                    source.inspection_probability_state_1
                    if state else source.inspection_probability_state_0
                )
                if inspection_rng.random() >= q:
                    result: ObservationResult = "missing"
                else:
                    positive_probability = (
                        source.detection_sensitivity if state else source.false_positive_rate
                    )
                    result = "positive" if detection_rng.random() < positive_probability else "negative"
                source_results.append(SourceObservationResult(source.source_id, result))
            steps_out.append(tuple(source_results))
        episodes.append(
            SourcedObservationEpisode(
                f"sourced-{episode_index:06d}", opportunity_interval_hours, tuple(steps_out)
            )
        )
    return SourcedObservationDataset(
        tuple(episodes), forward_rate_per_hour, return_rate_per_hour, sources
    )


def _emission(result: ObservationResult, values: tuple[float, float, float, float]) -> tuple[float, float]:
    q0, q1, sensitivity, false_positive = values
    if result == "missing":
        return 1.0 - q0, 1.0 - q1
    if result == "negative":
        return q0 * (1.0 - false_positive), q1 * (1.0 - sensitivity)
    return q0 * false_positive, q1 * sensitivity


def _joint_emission(
    step: tuple[SourceObservationResult, ...],
    parameters: Mapping[str, tuple[float, float, float, float]],
) -> tuple[float, float]:
    state_0 = state_1 = 1.0
    for item in step:
        emission_0, emission_1 = _emission(item.result, parameters[item.source_id])
        state_0 *= emission_0
        state_1 *= emission_1
    return state_0, state_1


@dataclass(slots=True)
class _Counts:
    transitions: list[float]
    state: list[float]
    inspected: dict[str, list[float]]
    positive: dict[str, list[float]]
    nll: float = 0.0


def _expectations(
    episode: SourcedObservationEpisode,
    p01: float,
    p10: float,
    parameters: Mapping[str, tuple[float, float, float, float]],
) -> _Counts:
    alphas: list[tuple[float, float]] = []
    scales: list[float] = []
    previous = (1.0, 0.0)
    nll = 0.0
    for step in episode.results_by_step:
        predicted = (
            previous[0] * (1.0 - p01) + previous[1] * p10,
            previous[0] * p01 + previous[1] * (1.0 - p10),
        )
        emission = _joint_emission(step, parameters)
        values = (predicted[0] * emission[0], predicted[1] * emission[1])
        scale = sum(values)
        if scale <= 0.0 or not math.isfinite(scale):
            raise RuntimeError("source observation sequence has zero probability")
        previous = (values[0] / scale, values[1] / scale)
        alphas.append(previous)
        scales.append(scale)
        nll -= math.log(scale)
    betas: list[tuple[float, float]] = [(1.0, 1.0)] * len(alphas)
    for index in range(len(alphas) - 2, -1, -1):
        emission = _joint_emission(episode.results_by_step[index + 1], parameters)
        next_beta = betas[index + 1]
        scale = scales[index + 1]
        betas[index] = (
            ((1.0 - p01) * emission[0] * next_beta[0] + p01 * emission[1] * next_beta[1]) / scale,
            (p10 * emission[0] * next_beta[0] + (1.0 - p10) * emission[1] * next_beta[1]) / scale,
        )
    counts = _Counts(
        [0.0] * 4,
        [0.0, 0.0],
        {source_id: [0.0, 0.0] for source_id in episode.source_ids},
        {source_id: [0.0, 0.0] for source_id in episode.source_ids},
        nll,
    )
    for index, step in enumerate(episode.results_by_step):
        raw = (alphas[index][0] * betas[index][0], alphas[index][1] * betas[index][1])
        normalizer = sum(raw)
        gamma = (raw[0] / normalizer, raw[1] / normalizer)
        counts.state[0] += gamma[0]
        counts.state[1] += gamma[1]
        for item in step:
            if item.result != "missing":
                counts.inspected[item.source_id][0] += gamma[0]
                counts.inspected[item.source_id][1] += gamma[1]
            if item.result == "positive":
                counts.positive[item.source_id][0] += gamma[0]
                counts.positive[item.source_id][1] += gamma[1]
        prior = (1.0, 0.0) if index == 0 else alphas[index - 1]
        emission = _joint_emission(step, parameters)
        beta = betas[index]
        weights = (
            prior[0] * (1.0 - p01) * emission[0] * beta[0],
            prior[0] * p01 * emission[1] * beta[1],
            prior[1] * p10 * emission[0] * beta[0],
            prior[1] * (1.0 - p10) * emission[1] * beta[1],
        )
        normalizer = sum(weights)
        for cell in range(4):
            counts.transitions[cell] += weights[cell] / normalizer
    return counts


def _clamp(value: float, epsilon: float) -> float:
    return max(epsilon, min(1.0 - epsilon, value))


def _to_rates(p01: float, p10: float, interval: float) -> tuple[float, float]:
    moved = p01 + p10
    if moved <= 0.0 or moved >= 1.0:
        raise RuntimeError("invalid fitted transition probabilities")
    total = -math.log1p(-moved) / interval
    return total * p01 / moved, total * p10 / moved


def _fit_start(
    episodes: tuple[SourcedObservationEpisode, ...],
    start: tuple[float, float, float, float, float, float],
    *,
    pooled_sources: bool,
    max_iterations: int,
    tolerance: float,
    epsilon: float,
) -> tuple[float, float, dict[str, tuple[float, float, float, float]], tuple[float, ...], int, bool]:
    source_ids = episodes[0].source_ids
    p01, p10, q0, q1, sensitivity, false_positive = start
    parameters = {source_id: (q0, q1, sensitivity, false_positive) for source_id in source_ids}

    def total_counts() -> _Counts:
        total = _Counts(
            [0.0] * 4, [0.0, 0.0],
            {source_id: [0.0, 0.0] for source_id in source_ids},
            {source_id: [0.0, 0.0] for source_id in source_ids}, 0.0,
        )
        for episode in episodes:
            current = _expectations(episode, p01, p10, parameters)
            total.transitions = [a + b for a, b in zip(total.transitions, current.transitions)]
            total.state = [a + b for a, b in zip(total.state, current.state)]
            total.nll += current.nll
            for source_id in source_ids:
                total.inspected[source_id] = [a + b for a, b in zip(total.inspected[source_id], current.inspected[source_id])]
                total.positive[source_id] = [a + b for a, b in zip(total.positive[source_id], current.positive[source_id])]
        return total

    history = [total_counts().nll / len(episodes)]
    converged = False
    for iteration in range(1, max_iterations + 1):
        counts = total_counts()
        p01 = _clamp(counts.transitions[1] / (counts.transitions[0] + counts.transitions[1]), epsilon)
        p10 = _clamp(counts.transitions[2] / (counts.transitions[2] + counts.transitions[3]), epsilon)
        moved = p01 + p10
        if moved >= 1.0 - epsilon:
            scale = (1.0 - 2.0 * epsilon) / moved
            p01 *= scale
            p10 *= scale
        if pooled_sources:
            inspected = [sum(counts.inspected[s][z] for s in source_ids) for z in (0, 1)]
            positive = [sum(counts.positive[s][z] for s in source_ids) for z in (0, 1)]
            states = [counts.state[z] * len(source_ids) for z in (0, 1)]
            shared = (
                _clamp(inspected[0] / states[0], epsilon),
                _clamp(inspected[1] / states[1], epsilon),
                _clamp(positive[1] / inspected[1], epsilon),
                _clamp(positive[0] / inspected[0], epsilon),
            )
            parameters = {source_id: shared for source_id in source_ids}
        else:
            parameters = {
                source_id: (
                    _clamp(counts.inspected[source_id][0] / counts.state[0], epsilon),
                    _clamp(counts.inspected[source_id][1] / counts.state[1], epsilon),
                    _clamp(counts.positive[source_id][1] / counts.inspected[source_id][1], epsilon),
                    _clamp(counts.positive[source_id][0] / counts.inspected[source_id][0], epsilon),
                )
                for source_id in source_ids
            }
        if any(values[2] <= values[3] for values in parameters.values()):
            raise RuntimeError("source EM lost the positive-state emission ordering")
        current = total_counts().nll / len(episodes)
        if current > history[-1] + 1e-8:
            raise RuntimeError("source EM observation likelihood decreased")
        history.append(current)
        if history[-2] - history[-1] <= tolerance:
            converged = True
            break
    return p01, p10, parameters, tuple(history), iteration, converged


def fit_source_reliability_em(
    episodes: Iterable[SourcedObservationEpisode],
    *,
    pooled_sources: bool = False,
    max_iterations: int = 300,
    tolerance: float = 1e-8,
    probability_epsilon: float = 1e-6,
) -> EstimatedSourceReliabilityProcess:
    """Fit a shared recurrent state process with source-specific or tied emissions."""

    rows = tuple(episodes)
    if not rows:
        raise ValueError("at least one sourced training episode is required")
    if max_iterations <= 0 or tolerance <= 0 or not 0.0 < probability_epsilon < 0.01:
        raise ValueError("invalid EM controls")
    interval = rows[0].opportunity_interval_hours
    length = len(rows[0].results_by_step)
    source_ids = rows[0].source_ids
    if any(
        len(row.results_by_step) != length
        or row.source_ids != source_ids
        or not math.isclose(row.opportunity_interval_hours, interval, abs_tol=1e-12)
        for row in rows
    ):
        raise ValueError("source EM requires a common nonempty source opportunity grid")
    if not any(item.result == "positive" for row in rows for step in row.results_by_step for item in step):
        raise ValueError("source EM requires at least one positive observation")
    inspected = sum(item.result != "missing" for row in rows for step in row.results_by_step for item in step)
    total = len(rows) * length * len(source_ids)
    positive = sum(item.result == "positive" for row in rows for step in row.results_by_step for item in step)
    q = _clamp(inspected / total, probability_epsilon)
    positive_fraction = positive / inspected
    starts = (
        (0.05, 0.05, q, q, 0.85, min(0.15, positive_fraction / 3.0)),
        (0.12, 0.04, q, q, 0.9, 0.03),
        (0.04, 0.12, q, q, 0.9, 0.03),
        (0.18, 0.10, 0.75, 0.55, 0.92, 0.04),
    )
    candidates = []
    for start in starts:
        try:
            candidates.append(
                _fit_start(
                    rows, start, pooled_sources=pooled_sources,
                    max_iterations=max_iterations, tolerance=tolerance,
                    epsilon=probability_epsilon,
                )
            )
        except RuntimeError:
            continue
    if not candidates:
        raise RuntimeError("all source EM initializations failed")
    p01, p10, parameters, history, iterations, converged = min(
        candidates, key=lambda candidate: candidate[3][-1]
    )
    forward, backward = _to_rates(p01, p10, interval)
    emissions = tuple(
        SourceEmissionParameters(source_id, *parameters[source_id]) for source_id in source_ids
    )
    return EstimatedSourceReliabilityProcess(
        forward, backward, emissions, pooled_sources, iterations, converged,
        history, len(starts)
    )
