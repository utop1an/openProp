from __future__ import annotations

import math
import random
from dataclasses import dataclass, replace
from typing import Literal

from .persistence_data import PersistenceTrainingExample


ObservationResult = Literal["negative", "positive", "missing"]


@dataclass(frozen=True, slots=True)
class ObservationEpisode:
    """Logged observation opportunities without latent transition truth."""

    group_id: str
    opportunity_interval_hours: float
    results: tuple[ObservationResult, ...]

    def __post_init__(self) -> None:
        if not self.group_id:
            raise ValueError("group_id cannot be empty")
        if not math.isfinite(self.opportunity_interval_hours) or self.opportunity_interval_hours <= 0:
            raise ValueError("opportunity interval must be positive and finite")
        if not self.results:
            raise ValueError("an episode requires at least one observation opportunity")
        if any(result not in {"negative", "positive", "missing"} for result in self.results):
            raise ValueError("unknown observation result")


@dataclass(frozen=True, slots=True)
class InformativeObservationDataset:
    """Training logs plus independent exact-time evaluation records."""

    episodes: tuple[ObservationEpisode, ...]
    interval_train: tuple[PersistenceTrainingExample, ...]
    naive_train: tuple[PersistenceTrainingExample, ...]
    exact_test: tuple[PersistenceTrainingExample, ...]
    true_hazard_per_hour: float
    pre_transition_inspection_probability: float
    post_transition_inspection_probability: float
    detection_sensitivity: float
    false_positive_rate: float
    opportunity_interval_hours: float
    followup_hours: float


def _validate_probability(name: str, value: float) -> None:
    if not math.isfinite(value) or not 0.0 < value < 1.0:
        raise ValueError(f"{name} must be finite and strictly between zero and one")


def _validate_sensitivity(value: float) -> None:
    if not math.isfinite(value) or not 0.0 < value <= 1.0:
        raise ValueError("detection_sensitivity must be in (0, 1]")


def _validate_false_positive_rate(value: float) -> None:
    if not math.isfinite(value) or not 0.0 <= value < 1.0:
        raise ValueError("false_positive_rate must be in [0, 1)")


def informative_observation_data(
    *,
    train_samples: int = 1200,
    test_samples: int = 1000,
    true_hazard_per_hour: float = 0.25,
    followup_hours: float = 12.0,
    opportunity_interval_hours: float = 0.5,
    pre_transition_inspection_probability: float = 0.15,
    post_transition_inspection_probability: float = 0.75,
    detection_sensitivity: float = 0.65,
    false_positive_rate: float = 0.0,
    seed: int = 101,
) -> InformativeObservationDataset:
    """Generate state-dependent inspections with imperfect event detection.

    Inspection opportunities occur on a fixed grid, but whether an inspection
    is recorded depends on the hidden state. A recorded post-transition
    inspection detects the changed state only with ``detection_sensitivity``;
    a recorded pre-transition inspection can emit a spurious positive with
    ``false_positive_rate``. Latent transition times are discarded from
    training episodes and retained only in an independently sampled exact-time
    test partition.
    """

    if train_samples <= 0 or test_samples <= 0:
        raise ValueError("sample counts must be positive")
    if not math.isfinite(true_hazard_per_hour) or true_hazard_per_hour <= 0:
        raise ValueError("true hazard must be positive and finite")
    if (
        not math.isfinite(followup_hours)
        or not math.isfinite(opportunity_interval_hours)
        or followup_hours <= 0
        or opportunity_interval_hours <= 0
    ):
        raise ValueError("follow-up and opportunity interval must be positive and finite")
    opportunities = round(followup_hours / opportunity_interval_hours)
    if opportunities <= 0 or not math.isclose(
        opportunities * opportunity_interval_hours,
        followup_hours,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise ValueError("follow-up must be an integer multiple of opportunity interval")
    _validate_probability(
        "pre_transition_inspection_probability",
        pre_transition_inspection_probability,
    )
    _validate_probability(
        "post_transition_inspection_probability",
        post_transition_inspection_probability,
    )
    _validate_sensitivity(detection_sensitivity)
    _validate_false_positive_rate(false_positive_rate)

    transition_rng = random.Random(seed)
    observation_rng = random.Random(seed + 500_003)
    test_rng = random.Random(seed + 1_000_003)
    order_rng = random.Random(seed + 1_500_003)
    episodes: list[ObservationEpisode] = []
    interval_rows: list[PersistenceTrainingExample] = []
    for index in range(train_samples):
        transition = transition_rng.expovariate(true_hazard_per_hour)
        results: list[ObservationResult] = []
        for step in range(1, opportunities + 1):
            time_hours = step * opportunity_interval_hours
            transitioned = transition <= time_hours
            inspection_probability = (
                post_transition_inspection_probability
                if transitioned
                else pre_transition_inspection_probability
            )
            inspection_draw = observation_rng.random()
            detection_draw = observation_rng.random()
            if inspection_draw >= inspection_probability:
                results.append("missing")
            elif transitioned and detection_draw < detection_sensitivity:
                results.append("positive")
            elif not transitioned and detection_draw < false_positive_rate:
                results.append("positive")
            else:
                results.append("negative")

        group_id = f"train-informative-{index:05d}"
        episode = ObservationEpisode(
            group_id,
            opportunity_interval_hours,
            tuple(results),
        )
        episodes.append(episode)
        first_positive = next(
            (step for step, result in enumerate(results, start=1) if result == "positive"),
            None,
        )
        if first_positive is None:
            last_negative = max(
                (step for step, result in enumerate(results, start=1) if result == "negative"),
                default=0,
            )
            interval_rows.append(
                PersistenceTrainingExample(
                    "location",
                    "cup",
                    "on",
                    "table",
                    "informative-policy",
                    last_negative * opportunity_interval_hours * 3600.0,
                    False,
                    group_id,
                )
            )
        else:
            last_negative = max(
                (
                    step
                    for step, result in enumerate(results[: first_positive - 1], start=1)
                    if result == "negative"
                ),
                default=0,
            )
            interval_rows.append(
                PersistenceTrainingExample(
                    "location",
                    "cup",
                    "on",
                    "table",
                    "informative-policy",
                    first_positive * opportunity_interval_hours * 3600.0,
                    True,
                    group_id,
                    last_negative * opportunity_interval_hours * 3600.0,
                )
            )

    test_rows: list[PersistenceTrainingExample] = []
    for index in range(test_samples):
        transition = test_rng.expovariate(true_hazard_per_hour)
        observed = transition <= followup_hours
        test_rows.append(
            PersistenceTrainingExample(
                "location",
                "cup",
                "on",
                "table",
                "informative-policy",
                min(transition, followup_hours) * 3600.0,
                observed,
                f"test-exact-{index:05d}",
            )
        )

    order_rng.shuffle(episodes)
    order_rng.shuffle(interval_rows)
    test_rng.shuffle(test_rows)
    return InformativeObservationDataset(
        tuple(episodes),
        tuple(interval_rows),
        tuple(replace(row, interval_start_seconds=None) for row in interval_rows),
        tuple(test_rows),
        true_hazard_per_hour,
        pre_transition_inspection_probability,
        post_transition_inspection_probability,
        detection_sensitivity,
        false_positive_rate,
        opportunity_interval_hours,
        followup_hours,
    )


@dataclass(frozen=True, slots=True)
class ObservationAwareExponentialModel:
    """Exponential state model fit through a logged observation mechanism."""

    hazard: float
    pre_transition_inspection_probability: float
    post_transition_inspection_probability: float
    detection_sensitivity: float
    false_positive_rate: float = 0.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.hazard) or self.hazard <= 0:
            raise ValueError("hazard must be positive and finite")
        _validate_probability(
            "pre_transition_inspection_probability",
            self.pre_transition_inspection_probability,
        )
        _validate_probability(
            "post_transition_inspection_probability",
            self.post_transition_inspection_probability,
        )
        _validate_sensitivity(self.detection_sensitivity)
        _validate_false_positive_rate(self.false_positive_rate)

    @staticmethod
    def _episode_negative_log_likelihood(
        episode: ObservationEpisode,
        hazard: float,
        pre_transition_inspection_probability: float,
        post_transition_inspection_probability: float,
        detection_sensitivity: float,
        false_positive_rate: float = 0.0,
    ) -> float:
        unchanged = 1.0
        transitioned = 0.0
        log_likelihood = 0.0
        survival_step = math.exp(-hazard * episode.opportunity_interval_hours)
        for result in episode.results:
            transitioned += unchanged * (1.0 - survival_step)
            unchanged *= survival_step
            if result == "missing":
                unchanged *= 1.0 - pre_transition_inspection_probability
                transitioned *= 1.0 - post_transition_inspection_probability
            elif result == "negative":
                unchanged *= pre_transition_inspection_probability * (
                    1.0 - false_positive_rate
                )
                transitioned *= post_transition_inspection_probability * (
                    1.0 - detection_sensitivity
                )
            else:
                unchanged *= pre_transition_inspection_probability * false_positive_rate
                transitioned *= (
                    post_transition_inspection_probability * detection_sensitivity
                )
            scale = unchanged + transitioned
            if scale <= 0.0 or not math.isfinite(scale):
                return math.inf
            log_likelihood += math.log(scale)
            unchanged /= scale
            transitioned /= scale
        return -log_likelihood

    @classmethod
    def fit(
        cls,
        episodes: tuple[ObservationEpisode, ...],
        *,
        pre_transition_inspection_probability: float,
        post_transition_inspection_probability: float,
        detection_sensitivity: float,
        false_positive_rate: float = 0.0,
    ) -> "ObservationAwareExponentialModel":
        if not episodes:
            raise ValueError("at least one training episode is required")
        _validate_probability(
            "pre_transition_inspection_probability",
            pre_transition_inspection_probability,
        )
        _validate_probability(
            "post_transition_inspection_probability",
            post_transition_inspection_probability,
        )
        _validate_sensitivity(detection_sensitivity)
        _validate_false_positive_rate(false_positive_rate)

        def objective(log_hazard: float) -> float:
            hazard = math.exp(log_hazard)
            return sum(
                cls._episode_negative_log_likelihood(
                    episode,
                    hazard,
                    pre_transition_inspection_probability,
                    post_transition_inspection_probability,
                    detection_sensitivity,
                    false_positive_rate,
                )
                for episode in episodes
            ) / len(episodes)

        left, right = math.log(1e-4), math.log(10.0)
        ratio = (math.sqrt(5.0) - 1.0) / 2.0
        first = right - ratio * (right - left)
        second = left + ratio * (right - left)
        first_value = objective(first)
        second_value = objective(second)
        for _ in range(80):
            if first_value <= second_value:
                right = second
                second = first
                second_value = first_value
                first = right - ratio * (right - left)
                first_value = objective(first)
            else:
                left = first
                first = second
                first_value = second_value
                second = left + ratio * (right - left)
                second_value = objective(second)
        return cls(
            math.exp((left + right) / 2.0),
            pre_transition_inspection_probability,
            post_transition_inspection_probability,
            detection_sensitivity,
            false_positive_rate,
        )

    def hazard_per_hour(self, features: tuple[str, ...]) -> float:
        return self.hazard

    def survival_probability_at_hours(
        self,
        features: tuple[str, ...],
        duration_hours: float,
    ) -> float:
        if duration_hours < 0:
            raise ValueError("survival duration cannot be negative")
        return math.exp(-self.hazard * duration_hours)
