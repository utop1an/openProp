from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Literal, Mapping, Sequence

from .persistence_data import PersistenceTrainingExample
from .survival_evaluation import (
    exponential_example_negative_log_likelihood,
    model_risk_score,
)
from .target_adaptation import RiskModel
from .target_interaction_adaptation import (
    _canonical_partitions,
    _group_key,
    _group_label,
    _identity_discovery_confirmation_split,
    _likelihood_ratio_p_bound,
    _partition_label,
)


NONLINEAR_BASIS_FAMILIES: tuple[str, ...] = ("affine", "quadratic", "hinge")
SPARSE_NONLINEAR_PARTITIONS: tuple[tuple[int, ...], ...] = ((1,), (4,), (1, 4))


def _basis_values(
    log_risk: float,
    *,
    family: str,
    pivot_log_risk: float,
) -> tuple[float, ...]:
    z = log_risk - pivot_log_risk
    if family == "affine":
        return (1.0, z)
    if family == "quadratic":
        return (1.0, z, z * z)
    if family == "hinge":
        return (1.0, z, abs(z))
    raise ValueError("unknown log-risk basis family")


@dataclass(frozen=True, slots=True)
class LogRiskBasisAdapter:
    """Map frozen source risk through a small predeclared nonlinear basis."""

    source_model: RiskModel
    family: Literal["affine", "quadratic", "hinge"]
    pivot_log_risk: float
    coefficients: tuple[float, ...]
    calibration_examples: int
    initial_negative_log_likelihood: float
    final_negative_log_likelihood: float

    def __post_init__(self) -> None:
        expected = 2 if self.family == "affine" else 3
        if self.family not in NONLINEAR_BASIS_FAMILIES:
            raise ValueError("unknown log-risk basis family")
        if len(self.coefficients) != expected:
            raise ValueError("basis coefficient count does not match family")
        if self.calibration_examples <= 0:
            raise ValueError("calibration_examples must be positive")
        numeric = (
            self.pivot_log_risk,
            self.initial_negative_log_likelihood,
            self.final_negative_log_likelihood,
            *self.coefficients,
        )
        if not all(math.isfinite(value) for value in numeric):
            raise ValueError("basis adapter parameters and diagnostics must be finite")

    @property
    def parameter_count(self) -> int:
        return len(self.coefficients)

    def hazard_per_hour(self, features: tuple[str, ...]) -> float:
        source_risk = model_risk_score(self.source_model, features)
        if not math.isfinite(source_risk) or source_risk <= 0.0:
            raise ValueError("source risk must be finite and positive")
        basis = _basis_values(
            math.log(source_risk),
            family=self.family,
            pivot_log_risk=self.pivot_log_risk,
        )
        eta = sum(
            coefficient * value
            for coefficient, value in zip(self.coefficients, basis, strict=True)
        )
        return math.exp(max(-20.0, min(20.0, eta)))

    def risk_score(self, features: tuple[str, ...]) -> float:
        return self.hazard_per_hour(features)

    def survival_probability_at_hours(
        self,
        features: tuple[str, ...],
        duration_hours: float,
    ) -> float:
        if duration_hours < 0.0 or not math.isfinite(duration_hours):
            raise ValueError("duration_hours must be finite and nonnegative")
        return math.exp(-self.hazard_per_hour(features) * duration_hours)

    def example_negative_log_likelihood(
        self,
        example: PersistenceTrainingExample,
    ) -> float:
        return exponential_example_negative_log_likelihood(
            self.hazard_per_hour(example.features()), example
        )


def fit_log_risk_basis_adapter(
    source_model: RiskModel,
    examples: Iterable[PersistenceTrainingExample],
    *,
    family: Literal["affine", "quadratic", "hinge"],
    pivot_hazard_per_hour: float = 0.12,
    epochs: int = 1000,
    learning_rate: float = 0.03,
    coefficient_l2_penalty: float = 1e-4,
) -> LogRiskBasisAdapter:
    """Fit a frozen-source basis adapter with exponential censoring likelihood."""

    rows = tuple(examples)
    if not rows:
        raise ValueError("at least one target calibration example is required")
    if (
        family not in NONLINEAR_BASIS_FAMILIES
        or not math.isfinite(pivot_hazard_per_hour)
        or pivot_hazard_per_hour <= 0.0
        or epochs <= 0
        or not math.isfinite(learning_rate)
        or learning_rate <= 0.0
        or not math.isfinite(coefficient_l2_penalty)
        or coefficient_l2_penalty < 0.0
    ):
        raise ValueError("invalid basis family, pivot, or optimizer settings")
    pivot = math.log(pivot_hazard_per_hour)
    log_risks = tuple(
        math.log(model_risk_score(source_model, row.features())) for row in rows
    )
    if not all(math.isfinite(value) for value in log_risks):
        raise ValueError("source model returned invalid risk")
    unique_levels = {round(value, 12) for value in log_risks}
    required_levels = 2 if family == "affine" else 3
    if len(unique_levels) < required_levels:
        raise ValueError(f"{family} fitting requires {required_levels} source risk levels")
    basis_rows = tuple(
        _basis_values(value, family=family, pivot_log_risk=pivot)
        for value in log_risks
    )
    coefficients = [pivot, 1.0] + ([0.0] if family != "affine" else [])
    identity = tuple(coefficients)

    def mean_nll(values: Sequence[float]) -> float:
        total = 0.0
        for basis, row in zip(basis_rows, rows, strict=True):
            eta = sum(
                coefficient * value
                for coefficient, value in zip(values, basis, strict=True)
            )
            total += exponential_example_negative_log_likelihood(
                math.exp(max(-20.0, min(20.0, eta))), row
            )
        return total / len(rows)

    initial_nll = mean_nll(identity)
    first = [0.0] * len(coefficients)
    second = [0.0] * len(coefficients)
    beta1, beta2, epsilon = 0.9, 0.999, 1e-8
    for step in range(1, epochs + 1):
        gradients = [0.0] * len(coefficients)
        for basis, row in zip(basis_rows, rows, strict=True):
            eta = max(
                -20.0,
                min(
                    20.0,
                    sum(
                        coefficient * value
                        for coefficient, value in zip(
                            coefficients, basis, strict=True
                        )
                    ),
                ),
            )
            hazard = math.exp(eta)
            upper = row.duration_seconds / 3600.0
            if row.is_interval_censored:
                assert row.interval_start_seconds is not None
                lower = row.interval_start_seconds / 3600.0
                width_mass = hazard * (upper - lower)
                ratio = (
                    width_mass / math.expm1(width_mass)
                    if width_mass < 700.0
                    else 0.0
                )
                eta_derivative = hazard * lower - ratio
            else:
                eta_derivative = hazard * upper - float(row.event_observed)
            for index, value in enumerate(basis):
                gradients[index] += eta_derivative * value
        for index in range(len(gradients)):
            gradients[index] /= len(rows)
            if index > 0:
                gradients[index] += coefficient_l2_penalty * (
                    coefficients[index] - identity[index]
                )
            first[index] = beta1 * first[index] + (1.0 - beta1) * gradients[index]
            second[index] = (
                beta2 * second[index] + (1.0 - beta2) * gradients[index] ** 2
            )
            first_hat = first[index] / (1.0 - beta1**step)
            second_hat = second[index] / (1.0 - beta2**step)
            coefficients[index] -= learning_rate * first_hat / (
                math.sqrt(second_hat) + epsilon
            )
            bound = 20.0 if index == 0 else 8.0
            coefficients[index] = max(-bound, min(bound, coefficients[index]))

    fitted = tuple(coefficients)
    return LogRiskBasisAdapter(
        source_model,
        family,
        pivot,
        fitted,
        len(rows),
        initial_nll,
        mean_nll(fitted),
    )


@dataclass(frozen=True, slots=True)
class SparseNonlinearTypedGate:
    """Route only confirmed sparse typed regions through small basis adapters."""

    source_model: RiskModel
    selected_partition: tuple[int, ...] | None
    calibration_value_support: tuple[frozenset[str], ...]
    group_models: Mapping[tuple[str, ...], RiskModel]
    significant_groups: frozenset[tuple[str, ...]]
    selected_families: Mapping[str, str]
    discovery_bic: Mapping[str, float]
    confirmation_p_values: Mapping[str, float]
    confirmation_mean_gains: Mapping[str, float]
    partition_mean_gains: Mapping[str, float]
    partition_context_fractions: Mapping[str, float]
    calibration_examples: int
    discovery_examples: int
    confirmation_examples: int
    familywise_alpha: float
    candidate_group_count: int
    max_adapted_context_fraction: float

    def __post_init__(self) -> None:
        if len(self.calibration_value_support) != 5 or any(
            not values for values in self.calibration_value_support
        ):
            raise ValueError("calibration support requires five nonempty feature sets")
        if self.calibration_examples != self.discovery_examples + self.confirmation_examples:
            raise ValueError("calibration split counts must add up")
        if not 0.0 < self.familywise_alpha < 1.0:
            raise ValueError("familywise_alpha must be between zero and one")
        if not 0.0 < self.max_adapted_context_fraction < 1.0:
            raise ValueError("sparse context fraction must be between zero and one")
        if self.candidate_group_count <= 0:
            raise ValueError("candidate_group_count must be positive")
        if self.selected_partition is None:
            if self.group_models or self.significant_groups:
                raise ValueError("inactive sparse gate cannot contain group models")
        elif not self.significant_groups.issubset(self.group_models):
            raise ValueError("significant groups require fitted group models")
        object.__setattr__(self, "group_models", dict(self.group_models))
        object.__setattr__(self, "selected_families", dict(self.selected_families))
        object.__setattr__(self, "discovery_bic", dict(self.discovery_bic))
        object.__setattr__(self, "confirmation_p_values", dict(self.confirmation_p_values))
        object.__setattr__(self, "confirmation_mean_gains", dict(self.confirmation_mean_gains))
        object.__setattr__(self, "partition_mean_gains", dict(self.partition_mean_gains))
        object.__setattr__(self, "partition_context_fractions", dict(self.partition_context_fractions))

    @property
    def activated(self) -> bool:
        return self.selected_partition is not None

    @property
    def bonferroni_threshold(self) -> float:
        return self.familywise_alpha / self.candidate_group_count

    def _model(self, features: tuple[str, ...]) -> RiskModel:
        if len(features) != 5:
            raise ValueError("typed nonlinear routing requires five features")
        if any(
            value not in supported
            for value, supported in zip(
                features, self.calibration_value_support, strict=True
            )
        ):
            return self.source_model
        if self.selected_partition is None:
            return self.source_model
        return self.group_models.get(
            _group_key(features, self.selected_partition), self.source_model
        )

    def hazard_per_hour(self, features: tuple[str, ...]) -> float:
        return self._model(features).hazard_per_hour(features)

    def risk_score(self, features: tuple[str, ...]) -> float:
        return model_risk_score(self._model(features), features)

    def survival_probability_at_hours(
        self,
        features: tuple[str, ...],
        duration_hours: float,
    ) -> float:
        if duration_hours < 0.0 or not math.isfinite(duration_hours):
            raise ValueError("duration_hours must be finite and nonnegative")
        return math.exp(-self.hazard_per_hour(features) * duration_hours)

    def example_negative_log_likelihood(
        self,
        example: PersistenceTrainingExample,
    ) -> float:
        return exponential_example_negative_log_likelihood(
            self.hazard_per_hour(example.features()), example
        )


def fit_sparse_nonlinear_typed_gate(
    source_model: RiskModel,
    examples: Iterable[PersistenceTrainingExample],
    *,
    split_seed: int,
    candidate_partitions: Sequence[tuple[int, ...]] = SPARSE_NONLINEAR_PARTITIONS,
    familywise_alpha: float = 0.05,
    max_adapted_context_fraction: float = 0.5,
    pivot_hazard_per_hour: float = 0.12,
    epochs: int = 1000,
    learning_rate: float = 0.03,
    coefficient_l2_penalty: float = 1e-4,
) -> SparseNonlinearTypedGate:
    """Select one sparse typed partition and one BIC-chosen basis per group.

    Model family and complexity are selected on an identity-disjoint discovery
    third. The chosen family must beat the frozen source on the other two thirds
    at a Bonferroni threshold over typed groups. Partitions that would adapt more
    than the declared fraction of calibrated contexts are rejected. Unsupported
    typed values always route to the source model.
    """

    rows = tuple(examples)
    if not rows or len({row.group_id for row in rows}) != len(rows):
        raise ValueError("calibration requires nonempty unique group IDs")
    if not 0.0 < familywise_alpha < 1.0:
        raise ValueError("familywise_alpha must be between zero and one")
    if not 0.0 < max_adapted_context_fraction < 1.0:
        raise ValueError("max adapted context fraction must be between zero and one")
    partitions = _canonical_partitions(candidate_partitions)
    if any(not partition for partition in partitions):
        raise ValueError("sparse nonlinear adaptation excludes the global partition")
    discovery, confirmation = _identity_discovery_confirmation_split(
        rows, seed=split_seed
    )
    contexts = frozenset(row.features() for row in rows)
    discovery_groups: dict[tuple[int, ...], dict[tuple[str, ...], list[PersistenceTrainingExample]]] = {}
    confirmation_groups: dict[tuple[int, ...], dict[tuple[str, ...], list[PersistenceTrainingExample]]] = {}
    full_groups: dict[tuple[int, ...], dict[tuple[str, ...], list[PersistenceTrainingExample]]] = {}
    for partition in partitions:
        discovery_groups[partition] = {}
        confirmation_groups[partition] = {}
        full_groups[partition] = {}
        for row in discovery:
            discovery_groups[partition].setdefault(
                _group_key(row.features(), partition), []
            ).append(row)
        for row in confirmation:
            confirmation_groups[partition].setdefault(
                _group_key(row.features(), partition), []
            ).append(row)
        for row in rows:
            full_groups[partition].setdefault(
                _group_key(row.features(), partition), []
            ).append(row)

    candidate_count = sum(len(groups) for groups in discovery_groups.values())
    threshold = familywise_alpha / candidate_count
    discovery_models: dict[
        tuple[tuple[int, ...], tuple[str, ...]], LogRiskBasisAdapter
    ] = {}
    selected_families: dict[str, str] = {}
    bic_values: dict[str, float] = {}
    p_values: dict[str, float] = {}
    mean_gains: dict[str, float] = {}
    significant: dict[tuple[int, ...], set[tuple[str, ...]]] = {
        partition: set() for partition in partitions
    }
    for partition in partitions:
        for group in sorted(discovery_groups[partition]):
            label = _group_label(partition, group)
            group_rows = discovery_groups[partition][group]
            source_levels = {
                round(math.log(model_risk_score(source_model, row.features())), 12)
                for row in group_rows
            }
            candidates: list[LogRiskBasisAdapter] = []
            for family in NONLINEAR_BASIS_FAMILIES:
                needed = 2 if family == "affine" else 3
                if len(source_levels) < needed:
                    continue
                candidates.append(
                    fit_log_risk_basis_adapter(
                        source_model,
                        group_rows,
                        family=family,  # type: ignore[arg-type]
                        pivot_hazard_per_hour=pivot_hazard_per_hour,
                        epochs=epochs,
                        learning_rate=learning_rate,
                        coefficient_l2_penalty=coefficient_l2_penalty,
                    )
                )
            if not candidates:
                continue
            chosen = min(
                candidates,
                key=lambda model: (
                    2.0 * len(group_rows) * model.final_negative_log_likelihood
                    + model.parameter_count * math.log(len(group_rows)),
                    model.parameter_count,
                    model.family,
                ),
            )
            bic = (
                2.0 * len(group_rows) * chosen.final_negative_log_likelihood
                + chosen.parameter_count * math.log(len(group_rows))
            )
            discovery_models[(partition, group)] = chosen
            selected_families[label] = chosen.family
            bic_values[label] = bic
            differences = [
                exponential_example_negative_log_likelihood(
                    source_model.hazard_per_hour(row.features()), row
                )
                - chosen.example_negative_log_likelihood(row)
                for row in confirmation_groups[partition].get(group, [])
            ]
            p_value = _likelihood_ratio_p_bound(differences)
            mean_gain = statistics_fmean(differences) if differences else 0.0
            p_values[label] = p_value
            mean_gains[label] = mean_gain
            if mean_gain > 0.0 and p_value <= threshold:
                significant[partition].add(group)

    partition_gains: dict[str, float] = {}
    partition_fractions: dict[str, float] = {}
    eligible: list[tuple[int, ...]] = []
    for partition in partitions:
        adapted_contexts = {
            features
            for features in contexts
            if _group_key(features, partition) in significant[partition]
        }
        fraction = len(adapted_contexts) / len(contexts)
        label = _partition_label(partition)
        partition_fractions[label] = fraction
        total_gain = 0.0
        for group in significant[partition]:
            model = discovery_models[(partition, group)]
            total_gain += sum(
                exponential_example_negative_log_likelihood(
                    source_model.hazard_per_hour(row.features()), row
                )
                - model.example_negative_log_likelihood(row)
                for row in confirmation_groups[partition][group]
            )
        partition_gains[label] = total_gain / len(confirmation)
        if (
            significant[partition]
            and 0.0 < fraction <= max_adapted_context_fraction
            and partition_gains[label] > 0.0
        ):
            eligible.append(partition)

    selected = (
        max(
            eligible,
            key=lambda partition: (
                partition_gains[_partition_label(partition)],
                -partition_fractions[_partition_label(partition)],
                len(partition),
                tuple(-index for index in partition),
            ),
        )
        if eligible
        else None
    )
    group_models: dict[tuple[str, ...], RiskModel] = {}
    selected_groups: frozenset[tuple[str, ...]] = frozenset()
    if selected is not None:
        selected_groups = frozenset(significant[selected])
        for group in selected_groups:
            family = selected_families[_group_label(selected, group)]
            group_models[group] = fit_log_risk_basis_adapter(
                source_model,
                full_groups[selected][group],
                family=family,  # type: ignore[arg-type]
                pivot_hazard_per_hour=pivot_hazard_per_hour,
                epochs=epochs,
                learning_rate=learning_rate,
                coefficient_l2_penalty=coefficient_l2_penalty,
            )

    return SparseNonlinearTypedGate(
        source_model=source_model,
        selected_partition=selected,
        calibration_value_support=tuple(
            frozenset(row.features()[index] for row in rows) for index in range(5)
        ),
        group_models=group_models,
        significant_groups=selected_groups,
        selected_families=selected_families,
        discovery_bic=bic_values,
        confirmation_p_values=p_values,
        confirmation_mean_gains=mean_gains,
        partition_mean_gains=partition_gains,
        partition_context_fractions=partition_fractions,
        calibration_examples=len(rows),
        discovery_examples=len(discovery),
        confirmation_examples=len(confirmation),
        familywise_alpha=familywise_alpha,
        candidate_group_count=candidate_count,
        max_adapted_context_fraction=max_adapted_context_fraction,
    )


def statistics_fmean(values: Sequence[float]) -> float:
    """Small local mean helper that rejects empty sequences at call sites."""

    return sum(values) / len(values)
