from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Mapping

from .models import Entity, Observation, PropertyDefinition, RelationValue
from .persistence import ExponentialPersistenceModel
from .persistence_data import PersistenceTrainingExample
from .temporal import FreshnessResult

from .survival_evaluation import exponential_example_negative_log_likelihood

def entity_persistence_features(
    definition: PropertyDefinition,
    observation: Observation,
    entity: Entity,
) -> tuple[str, ...]:
    """Build the same typed context tuple used by survival training records."""

    subject = entity.properties.get("type")
    subject_type = str(subject.value) if subject is not None else "unknown"
    scene_observation = entity.properties.get("scene")
    scene = str(scene_observation.value) if scene_observation is not None else "unknown"
    if isinstance(observation.value, RelationValue):
        predicate = observation.value.predicate
        context_object = str(observation.value.arguments.get("object", "unknown"))
    else:
        predicate = str(observation.value)
        context_object = "none"
    return (definition.name, subject_type, predicate, context_object, scene)


def _maximum_likelihood_hazard(
    examples: Iterable[PersistenceTrainingExample],
    *,
    prior_hazard: float | None = None,
    prior_exposure_hours: float = 0.0,
) -> float:
    rows = tuple(examples)
    if any(example.is_interval_censored for example in rows):
        # Exact events admit events/exposure; interval-censored events require
        # maximising P(lower < T <= upper) instead.
        prior_events = (prior_hazard or 0.0) * prior_exposure_hours

        def derivative(hazard: float) -> float:
            score = prior_exposure_hours - prior_events / hazard
            for example in rows:
                upper = example.duration_seconds / 3600.0
                if example.is_interval_censored:
                    assert example.interval_start_seconds is not None
                    lower = example.interval_start_seconds / 3600.0
                    width = upper - lower
                    x = hazard * width
                    interval_term = width / math.expm1(x) if x < 700 else 0.0
                    score += lower - interval_term
                else:
                    score += upper
                    if example.event_observed:
                        score -= 1.0 / hazard
            return score

        lower, upper = 1e-9, 1.0
        while derivative(upper) < 0 and upper < 1e6:
            upper *= 2.0
        for _ in range(100):
            midpoint = (lower + upper) / 2.0
            if derivative(midpoint) < 0:
                lower = midpoint
            else:
                upper = midpoint
        return max((lower + upper) / 2.0, 1e-9)
    exposure = sum(example.duration_seconds for example in rows) / 3600.0
    events = sum(example.event_observed for example in rows)
    if prior_hazard is not None:
        exposure += prior_exposure_hours
        events += prior_hazard * prior_exposure_hours
    if exposure <= 0:
        raise ValueError("hazard estimation requires positive exposure")
    return max(float(events) / exposure, 1e-9)


@dataclass(slots=True)
class GlobalExponentialPersistenceModel:
    """A validation-friendly global exponential survival baseline."""

    hazard: float
    trained_properties: frozenset[str]
    fallback: ExponentialPersistenceModel = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.hazard <= 0:
            raise ValueError("hazard must be positive")
        self.fallback = ExponentialPersistenceModel()

    @classmethod
    def fit(
        cls,
        examples: Iterable[PersistenceTrainingExample],
    ) -> "GlobalExponentialPersistenceModel":
        rows = tuple(examples)
        if not rows:
            raise ValueError("at least one training example is required")
        return cls(
            _maximum_likelihood_hazard(rows),
            frozenset(example.property_name.casefold() for example in rows),
        )

    def hazard_per_hour(self, features: tuple[str, ...]) -> float:
        return self.hazard

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
            or definition.name.casefold() not in self.trained_properties
        ):
            return self.fallback.predict(definition, observation, entity, as_of=as_of)
        return _predict_from_hazard(
            self.hazard,
            definition,
            observation,
            entity,
            as_of=as_of,
            fallback=self.fallback,
        )


@dataclass(slots=True)
class PerContextExponentialPersistenceModel:
    """Per-context MLE baseline with an explicit global OOD backoff."""

    hazards: Mapping[tuple[str, ...], float]
    global_hazard: float
    trained_properties: frozenset[str]
    fallback: ExponentialPersistenceModel = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.global_hazard <= 0 or any(hazard <= 0 for hazard in self.hazards.values()):
            raise ValueError("all hazards must be positive")
        self.hazards = dict(self.hazards)
        self.fallback = ExponentialPersistenceModel()

    @classmethod
    def fit(
        cls,
        examples: Iterable[PersistenceTrainingExample],
        *,
        prior_exposure_hours: float = 1.0,
    ) -> "PerContextExponentialPersistenceModel":
        rows = tuple(examples)
        if not rows:
            raise ValueError("at least one training example is required")
        if prior_exposure_hours < 0:
            raise ValueError("prior_exposure_hours cannot be negative")
        global_hazard = _maximum_likelihood_hazard(rows)
        grouped: dict[tuple[str, ...], list[PersistenceTrainingExample]] = {}
        for example in rows:
            grouped.setdefault(example.features(), []).append(example)
        hazards = {
            tuple(value.casefold() for value in features): _maximum_likelihood_hazard(
                group,
                prior_hazard=global_hazard,
                prior_exposure_hours=prior_exposure_hours,
            )
            for features, group in grouped.items()
        }
        return cls(
            hazards,
            global_hazard,
            frozenset(example.property_name.casefold() for example in rows),
        )

    def hazard_per_hour(self, features: tuple[str, ...]) -> float:
        key = tuple(value.casefold() for value in features)
        return self.hazards.get(key, self.global_hazard)

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
            or definition.name.casefold() not in self.trained_properties
        ):
            return self.fallback.predict(definition, observation, entity, as_of=as_of)
        features = entity_persistence_features(definition, observation, entity)
        return _predict_from_hazard(
            self.hazard_per_hour(features),
            definition,
            observation,
            entity,
            as_of=as_of,
            fallback=self.fallback,
        )


@dataclass(slots=True)
class FactorizedExponentialPersistenceModel:
    """Log-linear proportional-hazards baseline over typed categorical factors.

    Each feature value contributes additively to log hazard. Unseen complete
    tuples are therefore composed from familiar factors instead of backed off
    to a global rate. L2 regularization resolves redundant categorical effects.
    """

    intercept: float
    effects: tuple[Mapping[str, float], ...]
    trained_properties: frozenset[str]
    hazard_scale: float = 1.0
    active_feature_indices: frozenset[int] = frozenset(range(5))
    fallback: ExponentialPersistenceModel = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not math.isfinite(self.intercept) or self.hazard_scale <= 0:
            raise ValueError("intercept must be finite and hazard_scale positive")
        if len(self.effects) != 5:
            raise ValueError("factorized model requires five typed feature columns")
        self.active_feature_indices = frozenset(self.active_feature_indices)
        if not self.active_feature_indices <= frozenset(range(5)):
            raise ValueError("active feature indices must be drawn from 0 through 4")
        self.effects = tuple(dict(column) for column in self.effects)
        self.fallback = ExponentialPersistenceModel()

    @classmethod
    def fit(
        cls,
        examples: Iterable[PersistenceTrainingExample],
        *,
        epochs: int = 1200,
        learning_rate: float = 0.03,
        l2_penalty: float = 1e-3,
        active_feature_indices: Iterable[int] | None = None,
    ) -> "FactorizedExponentialPersistenceModel":
        rows = tuple(examples)
        if not rows:
            raise ValueError("at least one training example is required")
        if epochs <= 0 or learning_rate <= 0 or l2_penalty < 0:
            raise ValueError("optimizer settings must be positive with nonnegative L2")
        active = (
            frozenset(range(5))
            if active_feature_indices is None
            else frozenset(active_feature_indices)
        )
        if not active <= frozenset(range(5)):
            raise ValueError("active feature indices must be drawn from 0 through 4")
        features = [
            tuple(value.casefold() for value in example.features())
            for example in rows
        ]
        values = tuple(
            tuple(sorted({row[index] for row in features})) if index in active else ()
            for index in range(5)
        )
        effects = [{value: 0.0 for value in column} for column in values]
        first = [{value: 0.0 for value in column} for column in values]
        second = [{value: 0.0 for value in column} for column in values]
        intercept = math.log(_maximum_likelihood_hazard(rows))
        intercept_first = 0.0
        intercept_second = 0.0
        beta1, beta2, epsilon = 0.9, 0.999, 1e-8

        for step in range(1, epochs + 1):
            gradients = [{value: 0.0 for value in column} for column in values]
            intercept_gradient = 0.0
            for example, encoded in zip(rows, features, strict=True):
                eta = intercept + sum(
                    effects[index][value] for index, value in enumerate(encoded)
                    if index in active
                )
                hazard = math.exp(max(-20.0, min(20.0, eta)))
                upper = example.duration_seconds / 3600.0
                if example.is_interval_censored:
                    assert example.interval_start_seconds is not None
                    lower = example.interval_start_seconds / 3600.0
                    x = hazard * (upper - lower)
                    ratio = x / math.expm1(x) if x < 700 else 0.0
                    derivative = hazard * lower - ratio
                else:
                    derivative = hazard * upper - float(example.event_observed)
                intercept_gradient += derivative
                for index, value in enumerate(encoded):
                    if index in active:
                        gradients[index][value] += derivative
            intercept_gradient /= len(rows)
            intercept_first = beta1 * intercept_first + (1 - beta1) * intercept_gradient
            intercept_second = (
                beta2 * intercept_second + (1 - beta2) * intercept_gradient**2
            )
            first_hat = intercept_first / (1 - beta1**step)
            second_hat = intercept_second / (1 - beta2**step)
            intercept -= learning_rate * first_hat / (math.sqrt(second_hat) + epsilon)
            for index, column in enumerate(effects):
                for value, coefficient in column.items():
                    gradient = gradients[index][value] / len(rows) + l2_penalty * coefficient
                    first[index][value] = (
                        beta1 * first[index][value] + (1 - beta1) * gradient
                    )
                    second[index][value] = (
                        beta2 * second[index][value] + (1 - beta2) * gradient**2
                    )
                    first_hat = first[index][value] / (1 - beta1**step)
                    second_hat = second[index][value] / (1 - beta2**step)
                    column[value] -= (
                        learning_rate * first_hat / (math.sqrt(second_hat) + epsilon)
                    )

        return cls(
            intercept,
            tuple(effects),
            frozenset(example.property_name.casefold() for example in rows),
            active_feature_indices=active,
        )

    def hazard_per_hour(self, features: tuple[str, ...]) -> float:
        if len(features) != len(self.effects):
            raise ValueError("factorized model requires five typed features")
        eta = self.intercept
        for index, (column, value) in enumerate(zip(self.effects, features, strict=True)):
            if index in self.active_feature_indices:
                eta += column.get(value.casefold(), 0.0)
        return math.exp(max(-20.0, min(20.0, eta))) * self.hazard_scale

    def calibrate(self, examples: Iterable[PersistenceTrainingExample]) -> float:
        """Fit one validation-only multiplier without changing factor effects."""
        rows = tuple(examples)
        if not rows:
            raise ValueError("at least one calibration example is required")
        current = self.hazard_scale
        base = [self.hazard_per_hour(row.features()) / current for row in rows]

        def objective(log_scale: float) -> float:
            scale = math.exp(log_scale)
            return sum(
                exponential_example_negative_log_likelihood(hazard * scale, row)
                for hazard, row in zip(base, rows, strict=True)
            ) / len(rows)

        left, right = -8.0, 8.0
        ratio = (math.sqrt(5.0) - 1.0) / 2.0
        for _ in range(100):
            first = right - ratio * (right - left)
            second = left + ratio * (right - left)
            if objective(first) <= objective(second):
                right = second
            else:
                left = first
        self.hazard_scale = max(0.01, min(100.0, math.exp((left + right) / 2.0)))
        return self.hazard_scale

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
            or definition.name.casefold() not in self.trained_properties
        ):
            return self.fallback.predict(definition, observation, entity, as_of=as_of)
        features = entity_persistence_features(definition, observation, entity)
        return _predict_from_hazard(
            self.hazard_per_hour(features),
            definition,
            observation,
            entity,
            as_of=as_of,
            fallback=self.fallback,
        )



def _weibull_example_negative_log_likelihood(
    rate_per_hour: float,
    shape: float,
    example: PersistenceTrainingExample,
) -> float:
    rate = max(rate_per_hour, 1e-12)
    upper = example.duration_seconds / 3600.0
    upper_hazard = (rate * upper) ** shape
    if example.is_interval_censored:
        assert example.interval_start_seconds is not None
        lower = example.interval_start_seconds / 3600.0
        lower_hazard = (rate * lower) ** shape
        probability_term = -math.expm1(-(upper_hazard - lower_hazard))
        return lower_hazard - math.log(max(probability_term, 1e-300))
    if not example.event_observed:
        return upper_hazard
    if upper <= 0:
        raise ValueError("exact Weibull events require positive duration")
    return (
        upper_hazard
        - math.log(shape)
        - shape * math.log(rate)
        - (shape - 1.0) * math.log(upper)
    )


@dataclass(slots=True)
class FactorizedWeibullPersistenceModel:
    """Factorized Weibull proportional-hazards baseline with a shared shape."""

    intercept: float
    log_shape: float
    effects: tuple[Mapping[str, float], ...]
    trained_properties: frozenset[str]
    rate_scale: float = 1.0
    fallback: ExponentialPersistenceModel = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.intercept)
            or not math.isfinite(self.log_shape)
            or self.rate_scale <= 0
            or len(self.effects) != 5
        ):
            raise ValueError("invalid factorized Weibull parameters")
        self.effects = tuple(dict(column) for column in self.effects)
        self.fallback = ExponentialPersistenceModel()

    @property
    def shape(self) -> float:
        return math.exp(self.log_shape)

    @classmethod
    def fit(
        cls,
        examples: Iterable[PersistenceTrainingExample],
        *,
        epochs: int = 1600,
        learning_rate: float = 0.02,
        l2_penalty: float = 1e-3,
        shape_penalty: float = 1e-4,
    ) -> "FactorizedWeibullPersistenceModel":
        rows = tuple(examples)
        if not rows:
            raise ValueError("at least one training example is required")
        if any(row.event_observed and row.duration_seconds <= 0 for row in rows):
            raise ValueError("observed Weibull events require positive duration")
        if (
            epochs <= 0
            or learning_rate <= 0
            or l2_penalty < 0
            or shape_penalty < 0
        ):
            raise ValueError("invalid Weibull optimizer settings")
        encoded_rows = [
            tuple(value.casefold() for value in row.features()) for row in rows
        ]
        values = tuple(
            tuple(sorted({features[index] for features in encoded_rows}))
            for index in range(5)
        )
        effects = [{value: 0.0 for value in column} for column in values]
        effect_first = [{value: 0.0 for value in column} for column in values]
        effect_second = [{value: 0.0 for value in column} for column in values]
        intercept = math.log(_maximum_likelihood_hazard(rows))
        log_shape = 0.0
        intercept_first = intercept_second = 0.0
        shape_first = shape_second = 0.0
        beta1, beta2, epsilon = 0.9, 0.999, 1e-8

        for step in range(1, epochs + 1):
            gradients = [{value: 0.0 for value in column} for column in values]
            intercept_gradient = 0.0
            shape_gradient = 0.0
            shape = math.exp(log_shape)
            for example, encoded in zip(rows, encoded_rows, strict=True):
                eta = intercept + sum(
                    effects[index][value] for index, value in enumerate(encoded)
                )
                rate = math.exp(max(-20.0, min(20.0, eta)))
                upper = example.duration_seconds / 3600.0
                upper_log_time = math.log(max(rate * upper, 1e-300))
                upper_hazard = math.exp(
                    max(-700.0, min(700.0, shape * upper_log_time))
                )
                if example.is_interval_censored:
                    assert example.interval_start_seconds is not None
                    lower = example.interval_start_seconds / 3600.0
                    if lower == 0:
                        lower_hazard = 0.0
                        lower_shape_derivative = 0.0
                    else:
                        lower_log_time = math.log(rate * lower)
                        lower_hazard = math.exp(
                            max(-700.0, min(700.0, shape * lower_log_time))
                        )
                        lower_shape_derivative = (
                            shape * lower_hazard * lower_log_time
                        )
                    delta = upper_hazard - lower_hazard
                    ratio = delta / math.expm1(delta) if delta < 700 else 0.0
                    eta_derivative = shape * (lower_hazard - ratio)
                    delta_shape_derivative = (
                        shape * upper_hazard * upper_log_time
                        - lower_shape_derivative
                    )
                    shape_derivative = lower_shape_derivative - (
                        delta_shape_derivative / math.expm1(delta)
                        if delta < 700
                        else 0.0
                    )
                else:
                    event = float(example.event_observed)
                    eta_derivative = shape * (upper_hazard - event)
                    shape_derivative = shape * upper_hazard * upper_log_time
                    if example.event_observed:
                        shape_derivative -= 1.0 + shape * upper_log_time
                intercept_gradient += eta_derivative
                shape_gradient += shape_derivative
                for index, value in enumerate(encoded):
                    gradients[index][value] += eta_derivative

            intercept_gradient /= len(rows)
            shape_gradient = shape_gradient / len(rows) + shape_penalty * log_shape
            intercept_first = beta1 * intercept_first + (1 - beta1) * intercept_gradient
            intercept_second = (
                beta2 * intercept_second + (1 - beta2) * intercept_gradient**2
            )
            shape_first = beta1 * shape_first + (1 - beta1) * shape_gradient
            shape_second = beta2 * shape_second + (1 - beta2) * shape_gradient**2
            first_correction = 1 - beta1**step
            second_correction = 1 - beta2**step
            intercept -= learning_rate * (intercept_first / first_correction) / (
                math.sqrt(intercept_second / second_correction) + epsilon
            )
            log_shape -= learning_rate * (shape_first / first_correction) / (
                math.sqrt(shape_second / second_correction) + epsilon
            )
            log_shape = max(math.log(0.15), min(math.log(6.0), log_shape))
            for index, column in enumerate(effects):
                for value, coefficient in column.items():
                    gradient = gradients[index][value] / len(rows) + l2_penalty * coefficient
                    effect_first[index][value] = (
                        beta1 * effect_first[index][value] + (1 - beta1) * gradient
                    )
                    effect_second[index][value] = (
                        beta2 * effect_second[index][value] + (1 - beta2) * gradient**2
                    )
                    column[value] -= learning_rate * (
                        effect_first[index][value] / first_correction
                    ) / (
                        math.sqrt(effect_second[index][value] / second_correction)
                        + epsilon
                    )

        return cls(
            intercept,
            log_shape,
            tuple(effects),
            frozenset(row.property_name.casefold() for row in rows),
        )

    def rate_per_hour(self, features: tuple[str, ...]) -> float:
        if len(features) != len(self.effects):
            raise ValueError("factorized model requires five typed features")
        eta = self.intercept + sum(
            column.get(value.casefold(), 0.0)
            for column, value in zip(self.effects, features, strict=True)
        )
        return math.exp(max(-20.0, min(20.0, eta))) * self.rate_scale

    def hazard_per_hour(self, features: tuple[str, ...]) -> float:
        """Compatibility risk score; Weibull instantaneous hazard is time-varying."""
        return self.rate_per_hour(features)

    def risk_score(self, features: tuple[str, ...]) -> float:
        return self.rate_per_hour(features)

    def survival_probability_at_hours(
        self,
        features: tuple[str, ...],
        duration_hours: float,
    ) -> float:
        if duration_hours < 0:
            raise ValueError("survival duration cannot be negative")
        return math.exp(
            -((self.rate_per_hour(features) * duration_hours) ** self.shape)
        )

    def example_negative_log_likelihood(
        self,
        example: PersistenceTrainingExample,
    ) -> float:
        return _weibull_example_negative_log_likelihood(
            self.rate_per_hour(example.features()), self.shape, example
        )

    def calibrate(self, examples: Iterable[PersistenceTrainingExample]) -> float:
        rows = tuple(examples)
        if not rows:
            raise ValueError("at least one calibration example is required")
        current = self.rate_scale
        base_rates = [
            self.rate_per_hour(row.features()) / current for row in rows
        ]

        def objective(log_scale: float) -> float:
            scale = math.exp(log_scale)
            return sum(
                _weibull_example_negative_log_likelihood(
                    rate * scale, self.shape, row
                )
                for rate, row in zip(base_rates, rows, strict=True)
            ) / len(rows)

        left, right = -8.0, 8.0
        ratio = (math.sqrt(5.0) - 1.0) / 2.0
        for _ in range(100):
            first = right - ratio * (right - left)
            second = left + ratio * (right - left)
            if objective(first) <= objective(second):
                right = second
            else:
                left = first
        self.rate_scale = max(0.01, min(100.0, math.exp((left + right) / 2.0)))
        return self.rate_scale

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
            or definition.name.casefold() not in self.trained_properties
        ):
            return self.fallback.predict(definition, observation, entity, as_of=as_of)
        features = entity_persistence_features(definition, observation, entity)
        age_seconds = max(0.0, as_of - observation.timestamp)
        time_retention = self.survival_probability_at_hours(
            features, age_seconds / 3600.0
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



def _piecewise_bin_durations(
    duration_hours: float,
    bin_edges_hours: tuple[float, ...],
) -> tuple[float, ...]:
    if duration_hours < 0:
        raise ValueError("piecewise duration cannot be negative")
    durations: list[float] = []
    start = 0.0
    for edge in bin_edges_hours:
        durations.append(max(0.0, min(duration_hours, edge) - start))
        start = edge
    durations.append(max(0.0, duration_hours - start))
    return tuple(durations)


def _piecewise_event_bin(
    duration_hours: float,
    bin_edges_hours: tuple[float, ...],
) -> int:
    return sum(duration_hours >= edge for edge in bin_edges_hours)


def _piecewise_example_negative_log_likelihood(
    base_rate_per_hour: float,
    bin_edges_hours: tuple[float, ...],
    log_multipliers: tuple[float, ...],
    example: PersistenceTrainingExample,
) -> float:
    multipliers = tuple(math.exp(value) for value in log_multipliers)

    def cumulative(duration: float) -> float:
        exposures = _piecewise_bin_durations(duration, bin_edges_hours)
        return base_rate_per_hour * sum(
            multiplier * exposure
            for multiplier, exposure in zip(multipliers, exposures, strict=True)
        )

    upper = example.duration_seconds / 3600.0
    upper_hazard = cumulative(upper)
    if example.is_interval_censored:
        assert example.interval_start_seconds is not None
        lower = example.interval_start_seconds / 3600.0
        lower_hazard = cumulative(lower)
        probability_term = -math.expm1(-(upper_hazard - lower_hazard))
        return lower_hazard - math.log(max(probability_term, 1e-300))
    loss = upper_hazard
    if example.event_observed:
        event_bin = _piecewise_event_bin(upper, bin_edges_hours)
        event_hazard = base_rate_per_hour * multipliers[event_bin]
        loss -= math.log(max(event_hazard, 1e-300))
    return loss


@dataclass(slots=True)
class FactorizedPiecewiseExponentialPersistenceModel:
    """Typed log-linear risk with global piecewise-constant time effects."""

    intercept: float
    effects: tuple[Mapping[str, float], ...]
    bin_edges_hours: tuple[float, ...]
    log_multipliers: tuple[float, ...]
    trained_properties: frozenset[str]
    rate_scale: float = 1.0
    fallback: ExponentialPersistenceModel = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.intercept)
            or self.rate_scale <= 0
            or len(self.effects) != 5
            or len(self.log_multipliers) != len(self.bin_edges_hours) + 1
            or any(
                not math.isfinite(edge) or edge <= 0
                for edge in self.bin_edges_hours
            )
            or tuple(sorted(self.bin_edges_hours)) != self.bin_edges_hours
            or len(set(self.bin_edges_hours)) != len(self.bin_edges_hours)
            or any(not math.isfinite(value) for value in self.log_multipliers)
        ):
            raise ValueError("invalid factorized piecewise parameters")
        if abs(self.log_multipliers[0]) > 1e-12:
            raise ValueError("the first piecewise multiplier is the reference zero")
        self.effects = tuple(dict(column) for column in self.effects)
        self.fallback = ExponentialPersistenceModel()

    @classmethod
    def fit(
        cls,
        examples: Iterable[PersistenceTrainingExample],
        *,
        bin_edges_hours: tuple[float, ...] = (2.0, 6.0),
        epochs: int = 1600,
        learning_rate: float = 0.025,
        l2_penalty: float = 1e-3,
        time_penalty: float = 1e-3,
    ) -> "FactorizedPiecewiseExponentialPersistenceModel":
        rows = tuple(examples)
        if not rows:
            raise ValueError("at least one training example is required")
        if (
            not bin_edges_hours
            or tuple(sorted(bin_edges_hours)) != bin_edges_hours
            or len(set(bin_edges_hours)) != len(bin_edges_hours)
            or any(not math.isfinite(edge) or edge <= 0 for edge in bin_edges_hours)
            or epochs <= 0
            or learning_rate <= 0
            or l2_penalty < 0
            or time_penalty < 0
        ):
            raise ValueError("invalid piecewise optimizer settings")
        encoded_rows = [
            tuple(value.casefold() for value in row.features()) for row in rows
        ]
        values = tuple(
            tuple(sorted({features[index] for features in encoded_rows}))
            for index in range(5)
        )
        effects = [{value: 0.0 for value in column} for column in values]
        effect_first = [{value: 0.0 for value in column} for column in values]
        effect_second = [{value: 0.0 for value in column} for column in values]
        bin_count = len(bin_edges_hours) + 1
        log_multipliers = [0.0] * bin_count
        time_first = [0.0] * bin_count
        time_second = [0.0] * bin_count
        intercept = math.log(_maximum_likelihood_hazard(rows))
        intercept_first = intercept_second = 0.0
        beta1, beta2, epsilon = 0.9, 0.999, 1e-8

        for step in range(1, epochs + 1):
            feature_gradients = [
                {value: 0.0 for value in column} for column in values
            ]
            time_gradients = [0.0] * bin_count
            intercept_gradient = 0.0
            multipliers = [math.exp(value) for value in log_multipliers]
            for example, encoded in zip(rows, encoded_rows, strict=True):
                eta = intercept + sum(
                    effects[index][value] for index, value in enumerate(encoded)
                )
                base_rate = math.exp(max(-20.0, min(20.0, eta)))
                upper = example.duration_seconds / 3600.0
                upper_durations = _piecewise_bin_durations(
                    upper, bin_edges_hours
                )
                upper_parts = [
                    base_rate * multiplier * duration
                    for multiplier, duration in zip(
                        multipliers, upper_durations, strict=True
                    )
                ]
                upper_hazard = sum(upper_parts)
                if example.is_interval_censored:
                    assert example.interval_start_seconds is not None
                    lower = example.interval_start_seconds / 3600.0
                    lower_durations = _piecewise_bin_durations(
                        lower, bin_edges_hours
                    )
                    lower_parts = [
                        base_rate * multiplier * duration
                        for multiplier, duration in zip(
                            multipliers, lower_durations, strict=True
                        )
                    ]
                    lower_hazard = sum(lower_parts)
                    delta = upper_hazard - lower_hazard
                    inverse_expm1 = 1.0 / math.expm1(delta) if delta < 700 else 0.0
                    eta_derivative = lower_hazard - delta * inverse_expm1
                    bin_derivatives = [
                        lower_part
                        - (upper_part - lower_part) * inverse_expm1
                        for lower_part, upper_part in zip(
                            lower_parts, upper_parts, strict=True
                        )
                    ]
                else:
                    eta_derivative = upper_hazard - float(example.event_observed)
                    bin_derivatives = list(upper_parts)
                    if example.event_observed:
                        event_bin = _piecewise_event_bin(upper, bin_edges_hours)
                        bin_derivatives[event_bin] -= 1.0
                intercept_gradient += eta_derivative
                for index, value in enumerate(encoded):
                    feature_gradients[index][value] += eta_derivative
                for index, derivative in enumerate(bin_derivatives):
                    time_gradients[index] += derivative

            first_correction = 1 - beta1**step
            second_correction = 1 - beta2**step
            intercept_gradient /= len(rows)
            intercept_first = beta1 * intercept_first + (1 - beta1) * intercept_gradient
            intercept_second = (
                beta2 * intercept_second + (1 - beta2) * intercept_gradient**2
            )
            intercept -= learning_rate * (intercept_first / first_correction) / (
                math.sqrt(intercept_second / second_correction) + epsilon
            )
            for index, column in enumerate(effects):
                for value, coefficient in column.items():
                    gradient = (
                        feature_gradients[index][value] / len(rows)
                        + l2_penalty * coefficient
                    )
                    effect_first[index][value] = (
                        beta1 * effect_first[index][value] + (1 - beta1) * gradient
                    )
                    effect_second[index][value] = (
                        beta2 * effect_second[index][value] + (1 - beta2) * gradient**2
                    )
                    column[value] -= learning_rate * (
                        effect_first[index][value] / first_correction
                    ) / (
                        math.sqrt(effect_second[index][value] / second_correction)
                        + epsilon
                    )
            for index in range(1, bin_count):
                gradient = (
                    time_gradients[index] / len(rows)
                    + time_penalty * log_multipliers[index]
                )
                time_first[index] = (
                    beta1 * time_first[index] + (1 - beta1) * gradient
                )
                time_second[index] = (
                    beta2 * time_second[index] + (1 - beta2) * gradient**2
                )
                log_multipliers[index] -= learning_rate * (
                    time_first[index] / first_correction
                ) / (
                    math.sqrt(time_second[index] / second_correction) + epsilon
                )
                log_multipliers[index] = max(
                    -8.0, min(8.0, log_multipliers[index])
                )

        return cls(
            intercept,
            tuple(effects),
            bin_edges_hours,
            tuple(log_multipliers),
            frozenset(row.property_name.casefold() for row in rows),
        )

    def base_rate_per_hour(self, features: tuple[str, ...]) -> float:
        if len(features) != len(self.effects):
            raise ValueError("factorized model requires five typed features")
        eta = self.intercept + sum(
            column.get(value.casefold(), 0.0)
            for column, value in zip(self.effects, features, strict=True)
        )
        return math.exp(max(-20.0, min(20.0, eta))) * self.rate_scale

    def hazard_per_hour(self, features: tuple[str, ...]) -> float:
        return self.base_rate_per_hour(features)

    def risk_score(self, features: tuple[str, ...]) -> float:
        return self.base_rate_per_hour(features)

    def survival_probability_at_hours(
        self,
        features: tuple[str, ...],
        duration_hours: float,
    ) -> float:
        base_rate = self.base_rate_per_hour(features)
        durations = _piecewise_bin_durations(
            duration_hours, self.bin_edges_hours
        )
        cumulative = base_rate * sum(
            math.exp(log_multiplier) * duration
            for log_multiplier, duration in zip(
                self.log_multipliers, durations, strict=True
            )
        )
        return math.exp(-cumulative)

    def example_negative_log_likelihood(
        self,
        example: PersistenceTrainingExample,
    ) -> float:
        return _piecewise_example_negative_log_likelihood(
            self.base_rate_per_hour(example.features()),
            self.bin_edges_hours,
            self.log_multipliers,
            example,
        )

    def calibrate(self, examples: Iterable[PersistenceTrainingExample]) -> float:
        rows = tuple(examples)
        if not rows:
            raise ValueError("at least one calibration example is required")
        current = self.rate_scale
        base_rates = [
            self.base_rate_per_hour(row.features()) / current for row in rows
        ]

        def objective(log_scale: float) -> float:
            scale = math.exp(log_scale)
            return sum(
                _piecewise_example_negative_log_likelihood(
                    rate * scale,
                    self.bin_edges_hours,
                    self.log_multipliers,
                    row,
                )
                for rate, row in zip(base_rates, rows, strict=True)
            ) / len(rows)

        left, right = -8.0, 8.0
        ratio = (math.sqrt(5.0) - 1.0) / 2.0
        for _ in range(100):
            first = right - ratio * (right - left)
            second = left + ratio * (right - left)
            if objective(first) <= objective(second):
                right = second
            else:
                left = first
        self.rate_scale = max(0.01, min(100.0, math.exp((left + right) / 2.0)))
        return self.rate_scale

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
            or definition.name.casefold() not in self.trained_properties
        ):
            return self.fallback.predict(definition, observation, entity, as_of=as_of)
        features = entity_persistence_features(definition, observation, entity)
        age_seconds = max(0.0, as_of - observation.timestamp)
        time_retention = self.survival_probability_at_hours(
            features, age_seconds / 3600.0
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



def _predict_from_hazard(
    hazard: float,
    definition: PropertyDefinition,
    observation: Observation,
    entity: Entity,
    *,
    as_of: float,
    fallback: ExponentialPersistenceModel,
) -> FreshnessResult:
    assert observation.timestamp is not None
    age_seconds = max(0.0, as_of - observation.timestamp)
    time_retention = math.exp(-hazard * age_seconds / 3600.0)
    # Statistical baselines share the same auditable event policy so the
    # comparison isolates the learned time-retention term.
    baseline = fallback.predict(definition, observation, entity, as_of=as_of)
    event_retention = baseline.event_retention
    return FreshnessResult(
        max(0.0, min(1.0, time_retention * event_retention)),
        age_seconds,
        time_retention,
        event_retention,
        baseline.applied_events,
    )
