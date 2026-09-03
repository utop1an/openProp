from __future__ import annotations

import bisect
import math
from dataclasses import dataclass
from typing import Iterable, Mapping

from .persistence_data import PersistenceTrainingExample


@dataclass(slots=True)
class FactorizedCoxPersistenceModel:
    """Typed categorical Cox model with a Breslow baseline survival curve."""

    effects: tuple[Mapping[str, float], ...]
    event_times_hours: tuple[float, ...]
    cumulative_baseline_hazards: tuple[float, ...]
    initial_partial_nll: float
    final_partial_nll: float
    epochs: int
    baseline_scale: float = 1.0

    def __post_init__(self) -> None:
        if len(self.effects) != 5:
            raise ValueError("factorized Cox model requires five typed feature columns")
        if len(self.event_times_hours) != len(self.cumulative_baseline_hazards):
            raise ValueError("baseline event times and hazards must align")
        if not self.event_times_hours:
            raise ValueError("Cox baseline requires at least one observed event")
        if tuple(sorted(set(self.event_times_hours))) != self.event_times_hours:
            raise ValueError("baseline event times must be strictly increasing")
        if any(
            not math.isfinite(value) or value <= 0
            for value in self.cumulative_baseline_hazards
        ):
            raise ValueError("baseline cumulative hazards must be positive and finite")
        if any(
            right <= left
            for left, right in zip(
                self.cumulative_baseline_hazards,
                self.cumulative_baseline_hazards[1:],
            )
        ):
            raise ValueError("baseline cumulative hazards must be strictly increasing")
        if not math.isfinite(self.baseline_scale) or self.baseline_scale <= 0:
            raise ValueError("baseline_scale must be positive and finite")
        self.effects = tuple(dict(column) for column in self.effects)

    @staticmethod
    def _encoded_rows(
        rows: tuple[PersistenceTrainingExample, ...],
    ) -> tuple[tuple[str, ...], ...]:
        return tuple(
            tuple(value.casefold() for value in row.features()) for row in rows
        )

    @staticmethod
    def _partial_objective_and_gradient(
        rows: tuple[PersistenceTrainingExample, ...],
        encoded: tuple[tuple[str, ...], ...],
        effects: tuple[dict[str, float], ...],
        l2_penalty: float,
    ) -> tuple[float, tuple[dict[str, float], ...]]:
        gradients = tuple(
            {value: 0.0 for value in column} for column in effects
        )
        eta = [
            sum(effects[index][value] for index, value in enumerate(features))
            for features in encoded
        ]
        relative_risk = [math.exp(max(-20.0, min(20.0, value))) for value in eta]
        by_time: dict[float, list[int]] = {}
        for index, row in enumerate(rows):
            by_time.setdefault(row.duration_seconds / 3600.0, []).append(index)

        risk_sum = 0.0
        risk_by_value = tuple(
            {value: 0.0 for value in column} for column in effects
        )
        loss = 0.0
        observed_events = 0
        for time in sorted(by_time, reverse=True):
            indices = by_time[time]
            for row_index in indices:
                weight = relative_risk[row_index]
                risk_sum += weight
                for column_index, value in enumerate(encoded[row_index]):
                    risk_by_value[column_index][value] += weight
            event_indices = [
                row_index for row_index in indices if rows[row_index].event_observed
            ]
            event_count = len(event_indices)
            if not event_count:
                continue
            observed_events += event_count
            loss += event_count * math.log(risk_sum) - sum(
                eta[row_index] for row_index in event_indices
            )
            for column_index, column in enumerate(gradients):
                for value in column:
                    column[value] += (
                        event_count
                        * risk_by_value[column_index][value]
                        / risk_sum
                    )
            for row_index in event_indices:
                for column_index, value in enumerate(encoded[row_index]):
                    gradients[column_index][value] -= 1.0

        if observed_events == 0:
            raise ValueError("Cox fitting requires at least one observed event")
        loss /= observed_events
        for column_index, column in enumerate(effects):
            for value, coefficient in column.items():
                loss += 0.5 * l2_penalty * coefficient * coefficient
                gradients[column_index][value] = (
                    gradients[column_index][value] / observed_events
                    + l2_penalty * coefficient
                )
        return loss, gradients

    @classmethod
    def fit(
        cls,
        examples: Iterable[PersistenceTrainingExample],
        *,
        epochs: int = 1200,
        learning_rate: float = 0.03,
        l2_penalty: float = 1e-3,
    ) -> "FactorizedCoxPersistenceModel":
        rows = tuple(examples)
        if not rows:
            raise ValueError("at least one training example is required")
        if any(row.is_interval_censored for row in rows):
            raise ValueError("Cox partial likelihood does not support interval events")
        if epochs <= 0 or learning_rate <= 0 or l2_penalty < 0:
            raise ValueError("optimizer settings must be positive with nonnegative L2")
        if not any(row.event_observed for row in rows):
            raise ValueError("Cox fitting requires at least one observed event")

        encoded = cls._encoded_rows(rows)
        values = tuple(
            tuple(sorted({features[index] for features in encoded}))
            for index in range(5)
        )
        effects = tuple({value: 0.0 for value in column} for column in values)
        first = tuple({value: 0.0 for value in column} for column in values)
        second = tuple({value: 0.0 for value in column} for column in values)
        beta1, beta2, epsilon = 0.9, 0.999, 1e-8
        initial_loss, _ = cls._partial_objective_and_gradient(
            rows,
            encoded,
            effects,
            l2_penalty,
        )

        for step in range(1, epochs + 1):
            _, gradients = cls._partial_objective_and_gradient(
                rows,
                encoded,
                effects,
                l2_penalty,
            )
            for column_index, column in enumerate(effects):
                for value in column:
                    gradient = gradients[column_index][value]
                    first[column_index][value] = (
                        beta1 * first[column_index][value] + (1.0 - beta1) * gradient
                    )
                    second[column_index][value] = (
                        beta2 * second[column_index][value]
                        + (1.0 - beta2) * gradient * gradient
                    )
                    first_hat = first[column_index][value] / (1.0 - beta1**step)
                    second_hat = second[column_index][value] / (1.0 - beta2**step)
                    column[value] -= learning_rate * first_hat / (
                        math.sqrt(second_hat) + epsilon
                    )

        final_loss, _ = cls._partial_objective_and_gradient(
            rows,
            encoded,
            effects,
            l2_penalty,
        )
        fitted_eta = [
            sum(effects[index][value] for index, value in enumerate(features))
            for features in encoded
        ]
        fitted_risk = [
            math.exp(max(-20.0, min(20.0, value))) for value in fitted_eta
        ]
        event_times = sorted(
            {
                row.duration_seconds / 3600.0
                for row in rows
                if row.event_observed
            }
        )
        cumulative = 0.0
        cumulative_baseline: list[float] = []
        for event_time in event_times:
            event_count = sum(
                row.event_observed
                and math.isclose(
                    row.duration_seconds / 3600.0,
                    event_time,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
                for row in rows
            )
            risk_sum = sum(
                risk
                for row, risk in zip(rows, fitted_risk, strict=True)
                if row.duration_seconds / 3600.0 >= event_time
            )
            cumulative += event_count / risk_sum
            cumulative_baseline.append(cumulative)
        return cls(
            effects,
            tuple(event_times),
            tuple(cumulative_baseline),
            initial_loss,
            final_loss,
            epochs,
        )

    def _linear_predictor(self, features: tuple[str, ...]) -> float:
        if len(features) != len(self.effects):
            raise ValueError("factorized Cox model requires five typed features")
        return sum(
            column.get(value.casefold(), 0.0)
            for column, value in zip(self.effects, features, strict=True)
        )

    def risk_score(self, features: tuple[str, ...]) -> float:
        return self._linear_predictor(features)

    def relative_risk(self, features: tuple[str, ...]) -> float:
        return math.exp(max(-20.0, min(20.0, self._linear_predictor(features))))

    def baseline_cumulative_hazard_at_hours(self, duration_hours: float) -> float:
        if duration_hours < 0:
            raise ValueError("survival duration cannot be negative")
        index = bisect.bisect_right(self.event_times_hours, duration_hours)
        return 0.0 if index == 0 else self.cumulative_baseline_hazards[index - 1]
    def calibrate_baseline(
        self,
        examples: Iterable[PersistenceTrainingExample],
        *,
        horizons_hours: tuple[float, ...] = (1.0, 4.0, 8.0, 12.0),
    ) -> float:
        """Fit one validation-only multiplier using horizon Brier loss."""

        rows = tuple(examples)
        if not rows:
            raise ValueError("at least one calibration example is required")
        if not horizons_hours or any(horizon <= 0 for horizon in horizons_hours):
            raise ValueError("calibration horizons must be positive")
        calibration: list[tuple[float, float]] = []
        for horizon in horizons_hours:
            baseline = self.baseline_cumulative_hazard_at_hours(horizon)
            for row in rows:
                upper = row.duration_seconds / 3600.0
                if row.is_interval_censored:
                    assert row.interval_start_seconds is not None
                    lower = row.interval_start_seconds / 3600.0
                    if upper <= horizon:
                        outcome = 0.0
                    elif lower >= horizon:
                        outcome = 1.0
                    else:
                        continue
                elif row.event_observed and upper <= horizon:
                    outcome = 0.0
                elif upper >= horizon:
                    outcome = 1.0
                else:
                    continue
                calibration.append(
                    (baseline * self.relative_risk(row.features()), outcome)
                )
        if not calibration:
            raise ValueError("no evaluable validation outcomes at calibration horizons")

        def objective(log_scale: float) -> float:
            scale = math.exp(log_scale)
            return sum(
                (math.exp(-scale * cumulative_risk) - outcome) ** 2
                for cumulative_risk, outcome in calibration
            ) / len(calibration)

        left, right = -8.0, 8.0
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
        self.baseline_scale = math.exp((left + right) / 2.0)
        return self.baseline_scale

    def survival_probability_at_hours(
        self,
        features: tuple[str, ...],
        duration_hours: float,
    ) -> float:
        cumulative = self.baseline_cumulative_hazard_at_hours(duration_hours)
        return math.exp(
            -self.baseline_scale * cumulative * self.relative_risk(features)
        )
