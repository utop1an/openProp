from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Iterable, Literal, Mapping, Sequence

from .persistence_data import PersistenceTrainingExample
from .survival_evaluation import (
    exponential_example_negative_log_likelihood,
    model_risk_score,
)
from .target_adaptation import (
    LogRiskAffineAdapter,
    RiskModel,
    fit_log_risk_affine_adapter,
)


DEFAULT_TYPED_PARTITIONS: tuple[tuple[int, ...], ...] = (
    (),
    (1,),
    (4,),
    (1, 4),
)


def _partition_label(partition: tuple[int, ...]) -> str:
    return "global" if not partition else "x".join(f"f{index}" for index in partition)


def _group_key(
    features: tuple[str, ...],
    partition: tuple[int, ...],
) -> tuple[str, ...]:
    return tuple(features[index].casefold() for index in partition)


def _group_label(partition: tuple[int, ...], group: tuple[str, ...]) -> str:
    if not partition:
        return "global"
    return "|".join(
        f"f{index}={value}" for index, value in zip(partition, group, strict=True)
    )


def _canonical_partitions(
    candidate_partitions: Sequence[tuple[int, ...]],
) -> tuple[tuple[int, ...], ...]:
    partitions = tuple(
        sorted(
            {tuple(partition) for partition in candidate_partitions},
            key=lambda partition: (len(partition), partition),
        )
    )
    if not partitions:
        raise ValueError("at least one typed partition is required")
    for partition in partitions:
        if tuple(sorted(set(partition))) != partition:
            raise ValueError("partition indices must be unique and increasing")
        if any(index < 0 or index >= 5 for index in partition):
            raise ValueError("partition indices must select the five typed features")
        if len(partition) > 3:
            raise ValueError("only global through three-way typed partitions are supported")
    return partitions


def _identity_discovery_confirmation_split(
    rows: tuple[PersistenceTrainingExample, ...],
    *,
    seed: int,
) -> tuple[tuple[PersistenceTrainingExample, ...], tuple[PersistenceTrainingExample, ...]]:
    by_context: dict[tuple[str, ...], list[PersistenceTrainingExample]] = {}
    for row in rows:
        by_context.setdefault(row.features(), []).append(row)
    discovery: list[PersistenceTrainingExample] = []
    confirmation: list[PersistenceTrainingExample] = []
    for context in sorted(by_context):
        ranked = sorted(
            by_context[context],
            key=lambda row: (
                hashlib.sha256(f"{seed}|{row.group_id}".encode("utf-8")).digest(),
                row.group_id,
            ),
        )
        discovery.extend(ranked[::3])
        confirmation.extend(row for offset, row in enumerate(ranked) if offset % 3)
    if not discovery or not confirmation:
        raise ValueError("calibration requires at least two records per typed context")
    return tuple(discovery), tuple(confirmation)


def _likelihood_ratio_p_bound(differences: Sequence[float]) -> float:
    """Return the reciprocal likelihood-ratio e-value, clipped as a p-bound."""
    log_e_value = sum(differences)
    return math.exp(-min(700.0, log_e_value)) if log_e_value > 0.0 else 1.0


@dataclass(frozen=True, slots=True)
class HierarchicalTypedInteractionGate:
    """A calibration-selected global, main-effect, pairwise, or three-way repair."""

    source_model: RiskModel
    activation_scope: Literal["reversal_only", "any_predictive_gain"]
    discovery_complexity: Literal["none", "bic"]
    selected_partition: tuple[int, ...] | None
    calibration_value_support: tuple[frozenset[str], ...]
    group_models: Mapping[tuple[str, ...], RiskModel]
    significant_groups: frozenset[tuple[str, ...]]
    discovery_slopes: Mapping[str, float]
    discovery_slope_fitted: Mapping[str, bool]
    discovery_bic_active: Mapping[str, bool]
    confirmation_p_values: Mapping[str, float]
    confirmation_mean_gains: Mapping[str, float]
    partition_mean_gains: Mapping[str, float]
    partition_source_p_values: Mapping[str, float]
    group_heterogeneity_veto: Mapping[str, bool]
    partition_heterogeneity_veto: Mapping[str, bool]
    partition_predictive_veto_p_values: Mapping[str, float]
    calibration_examples: int
    discovery_examples: int
    confirmation_examples: int
    familywise_alpha: float
    candidate_group_count: int

    def __post_init__(self) -> None:
        if self.activation_scope not in {"reversal_only", "any_predictive_gain"}:
            raise ValueError("unknown activation scope")
        if self.discovery_complexity not in {"none", "bic"}:
            raise ValueError("unknown discovery complexity")
        if len(self.calibration_value_support) != 5 or any(
            not values for values in self.calibration_value_support
        ):
            raise ValueError("calibration value support requires five nonempty feature sets")
        if self.calibration_examples != self.discovery_examples + self.confirmation_examples:
            raise ValueError("calibration split counts must add up")
        if self.discovery_examples <= 0 or self.confirmation_examples <= 0:
            raise ValueError("both calibration halves must be nonempty")
        if not 0.0 < self.familywise_alpha < 1.0:
            raise ValueError("familywise_alpha must be between zero and one")
        if self.candidate_group_count <= 0:
            raise ValueError("candidate_group_count must be positive")
        if self.selected_partition is None:
            if self.group_models or self.significant_groups:
                raise ValueError("inactive gate cannot contain adapted groups")
        elif not self.significant_groups.issubset(self.group_models):
            raise ValueError("significant groups must have fitted models")
        object.__setattr__(self, "group_models", dict(self.group_models))
        object.__setattr__(self, "discovery_slopes", dict(self.discovery_slopes))
        object.__setattr__(self, "discovery_slope_fitted", dict(self.discovery_slope_fitted))
        object.__setattr__(self, "discovery_bic_active", dict(self.discovery_bic_active))
        object.__setattr__(self, "confirmation_p_values", dict(self.confirmation_p_values))
        object.__setattr__(self, "confirmation_mean_gains", dict(self.confirmation_mean_gains))
        object.__setattr__(self, "partition_mean_gains", dict(self.partition_mean_gains))
        object.__setattr__(self, "partition_source_p_values", dict(self.partition_source_p_values))
        object.__setattr__(self, "group_heterogeneity_veto", dict(self.group_heterogeneity_veto))
        object.__setattr__(self, "partition_heterogeneity_veto", dict(self.partition_heterogeneity_veto))
        object.__setattr__(self, "partition_predictive_veto_p_values", dict(self.partition_predictive_veto_p_values))

    @property
    def activated(self) -> bool:
        return self.selected_partition is not None

    @property
    def bonferroni_threshold(self) -> float:
        return self.familywise_alpha / self.candidate_group_count

    def _model(self, features: tuple[str, ...]) -> RiskModel:
        if len(features) != 5:
            raise ValueError("typed interaction routing requires five features")
        if any(
            value not in supported
            for value, supported in zip(features, self.calibration_value_support, strict=True)
        ):
            return self.source_model
        if self.selected_partition is None:
            return self.source_model
        return self.group_models.get(
            _group_key(features, self.selected_partition),
            self.source_model,
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
        if duration_hours < 0 or not math.isfinite(duration_hours):
            raise ValueError("duration_hours must be finite and nonnegative")
        return math.exp(-self.hazard_per_hour(features) * duration_hours)

    def example_negative_log_likelihood(
        self,
        example: PersistenceTrainingExample,
    ) -> float:
        return exponential_example_negative_log_likelihood(
            self.hazard_per_hour(example.features()), example
        )


def fit_hierarchical_typed_interaction_gate(
    source_model: RiskModel,
    examples: Iterable[PersistenceTrainingExample],
    *,
    split_seed: int,
    candidate_partitions: Sequence[tuple[int, ...]] = DEFAULT_TYPED_PARTITIONS,
    familywise_alpha: float = 0.05,
    activation_scope: Literal["reversal_only", "any_predictive_gain"] = "reversal_only",
    discovery_complexity: Literal["none", "bic"] = "none",
    epochs: int = 1000,
    learning_rate: float = 0.03,
    slope_l2_penalty: float = 1e-4,
) -> HierarchicalTypedInteractionGate:
    """Select a sparse typed repair using discovery/confirmation calibration only.

    Adapters are fitted on an identity-disjoint discovery third. A group can
    activate only when the predictive likelihood-ratio e-value on the other two
    thirds passes a Bonferroni threshold across every candidate group.
    ``reversal_only`` additionally requires a negative discovery slope;
    ``any_predictive_gain`` also permits independently confirmed order-preserving
    calibration or intercept-only repairs for cells with one source-risk level.
    ``discovery_complexity='bic'`` requires a discovery-fit likelihood gain large
    enough to pay the candidate's Bayesian information criterion penalty before
    confirmation testing. The default preserves the earlier unscreened behavior.
    Prediction fails closed to the source model whenever any typed feature value
    was absent from target calibration, so a coarser selected partition cannot
    silently extrapolate a repair onto a genuinely unseen value.
    The significant partition with the largest confirmation NLL gain is selected,
    unless finer typed cells show mixed slope signs. This heterogeneity veto
    prevents a pooled parent repair from changing an opposed child cell. The
    generalized scope additionally vetoes a parent when discovery-fitted child
    models have a Bonferroni-significant predictive likelihood advantage on
    confirmation data. Local descent requires either prior global evidence or a
    partition-level predictive e-value against the source, then selects the
    coarsest eligible partition. Accepted groups are refitted on all calibration.
    """

    rows = tuple(examples)
    if not rows:
        raise ValueError("at least one calibration example is required")
    if len({row.group_id for row in rows}) != len(rows):
        raise ValueError("calibration group IDs must be unique")
    if not 0.0 < familywise_alpha < 1.0:
        raise ValueError("familywise_alpha must be between zero and one")
    if activation_scope not in {"reversal_only", "any_predictive_gain"}:
        raise ValueError("activation_scope must be reversal_only or any_predictive_gain")
    if discovery_complexity not in {"none", "bic"}:
        raise ValueError("discovery_complexity must be none or bic")
    partitions = _canonical_partitions(candidate_partitions)
    discovery, confirmation = _identity_discovery_confirmation_split(
        rows,
        seed=split_seed,
    )

    discovery_by_partition: dict[
        tuple[int, ...], dict[tuple[str, ...], list[PersistenceTrainingExample]]
    ] = {}
    confirmation_by_partition: dict[
        tuple[int, ...], dict[tuple[str, ...], list[PersistenceTrainingExample]]
    ] = {}
    full_by_partition: dict[
        tuple[int, ...], dict[tuple[str, ...], list[PersistenceTrainingExample]]
    ] = {}
    for partition in partitions:
        discovery_by_partition[partition] = {}
        confirmation_by_partition[partition] = {}
        full_by_partition[partition] = {}
        for row in discovery:
            discovery_by_partition[partition].setdefault(
                _group_key(row.features(), partition), []
            ).append(row)
        for row in confirmation:
            confirmation_by_partition[partition].setdefault(
                _group_key(row.features(), partition), []
            ).append(row)
        for row in rows:
            full_by_partition[partition].setdefault(
                _group_key(row.features(), partition), []
            ).append(row)

    candidate_count = sum(len(groups) for groups in discovery_by_partition.values())
    threshold = familywise_alpha / candidate_count
    slopes: dict[str, float] = {}
    p_values: dict[str, float] = {}
    mean_gains: dict[str, float] = {}
    significant: dict[tuple[int, ...], set[tuple[str, ...]]] = {
        partition: set() for partition in partitions
    }
    candidate_slopes: dict[tuple[tuple[int, ...], tuple[str, ...]], float] = {}
    discovery_models: dict[
        tuple[tuple[int, ...], tuple[str, ...]], LogRiskAffineAdapter
    ] = {}

    candidate_fit_slope: dict[str, bool] = {}
    candidate_bic_active: dict[str, bool] = {}
    for partition in partitions:
        for group in sorted(discovery_by_partition[partition]):
            label = _group_label(partition, group)
            group_discovery = discovery_by_partition[partition][group]
            source_levels = {
                round(math.log(model_risk_score(source_model, row.features())), 12)
                for row in group_discovery
            }
            fit_slope = len(source_levels) >= 2
            candidate_fit_slope[label] = fit_slope
            if not fit_slope and activation_scope == "reversal_only":
                candidate_bic_active[label] = False
                slopes[label] = 1.0
                p_values[label] = 1.0
                mean_gains[label] = 0.0
                continue
            adapter = fit_log_risk_affine_adapter(
                source_model,
                group_discovery,
                fit_slope=fit_slope,
                epochs=epochs,
                learning_rate=learning_rate,
                slope_l2_penalty=slope_l2_penalty,
            )
            discovery_models[(partition, group)] = adapter
            parameter_count = 2 if fit_slope else 1
            discovery_gain = len(group_discovery) * max(
                0.0,
                adapter.initial_negative_log_likelihood
                - adapter.final_negative_log_likelihood,
            )
            candidate_bic_active[label] = (
                activation_scope == "reversal_only"
                or partition == ()
                or discovery_complexity == "none"
                or 2.0 * discovery_gain > parameter_count * math.log(len(group_discovery))
            )
            candidate_slopes[(partition, group)] = adapter.slope
            slopes[label] = adapter.slope
            differences = [
                exponential_example_negative_log_likelihood(
                    source_model.hazard_per_hour(row.features()), row
                )
                - adapter.example_negative_log_likelihood(row)
                for row in confirmation_by_partition[partition].get(group, [])
            ]
            p_value = _likelihood_ratio_p_bound(differences)
            mean_gain = sum(differences) / len(differences) if differences else 0.0
            p_values[label] = p_value
            mean_gains[label] = mean_gain
            slope_allowed = (
                activation_scope == "any_predictive_gain" or adapter.slope < 0.0
            )
            if (
                candidate_bic_active[label]
                and slope_allowed
                and mean_gain > 0.0
                and p_value <= threshold
            ):
                significant[partition].add(group)

    partition_gains: dict[str, float] = {}
    for partition in partitions:
        total_gain = 0.0
        for group in significant[partition]:
            adapter = discovery_models[(partition, group)]
            total_gain += sum(
                exponential_example_negative_log_likelihood(
                    source_model.hazard_per_hour(row.features()), row
                )
                - adapter.example_negative_log_likelihood(row)
                for row in confirmation_by_partition[partition][group]
            )
        partition_gains[_partition_label(partition)] = total_gain / len(confirmation)

    partition_source_p_values: dict[str, float] = {}
    for partition in partitions:
        differences: list[float] = []
        for row in confirmation:
            group = _group_key(row.features(), partition)
            candidate = discovery_models.get((partition, group))
            label = _group_label(partition, group)
            if candidate is None or not candidate_bic_active.get(label, False):
                candidate = source_model
            differences.append(
                exponential_example_negative_log_likelihood(
                    source_model.hazard_per_hour(row.features()), row
                )
                - exponential_example_negative_log_likelihood(
                    candidate.hazard_per_hour(row.features()), row
                )
            )
        partition_source_p_values[_partition_label(partition)] = _likelihood_ratio_p_bound(differences)

    partition_veto: dict[str, bool] = {}
    group_heterogeneity_veto: dict[str, bool] = {}
    predictive_veto_p_values: dict[str, float] = {}
    for partition in partitions:
        original_groups = set(significant[partition])
        vetoed_groups: set[tuple[str, ...]] = set()
        for parent_group in original_groups:
            vetoed = False
            parent_values = dict(zip(partition, parent_group, strict=True))
            for child_partition in partitions:
                if len(child_partition) <= len(partition) or not set(partition).issubset(child_partition):
                    continue
                child_slopes = [
                    slope
                    for (candidate_partition, child_group), slope in candidate_slopes.items()
                    if candidate_partition == child_partition
                    and candidate_fit_slope[
                        _group_label(candidate_partition, child_group)
                    ]
                    and candidate_bic_active[
                        _group_label(candidate_partition, child_group)
                    ]
                    and all(
                        child_group[child_partition.index(index)] == value
                        for index, value in parent_values.items()
                    )
                ]
                if child_slopes and min(child_slopes) < 0.0 <= max(child_slopes):
                    vetoed = True
                    break
            group_heterogeneity_veto[
                _group_label(partition, parent_group)
            ] = vetoed
            if vetoed:
                vetoed_groups.add(parent_group)
        significant[partition].difference_update(vetoed_groups)
        partition_veto[_partition_label(partition)] = bool(
            original_groups and not significant[partition]
        )

    for partition in partitions:
        total_gain = 0.0
        for group in significant[partition]:
            adapter = discovery_models[(partition, group)]
            total_gain += sum(
                exponential_example_negative_log_likelihood(
                    source_model.hazard_per_hour(row.features()), row
                ) - adapter.example_negative_log_likelihood(row)
                for row in confirmation_by_partition[partition][group]
            )
        partition_gains[_partition_label(partition)] = total_gain / len(confirmation)

    if activation_scope == "any_predictive_gain":
        partition_comparisons = tuple(
            (parent, child)
            for parent in partitions
            for child in partitions
            if len(child) > len(parent) and set(parent).issubset(child)
        )
        comparison_threshold = familywise_alpha / max(1, len(partition_comparisons))
        for parent, child in partition_comparisons:
            differences: list[float] = []
            for row in confirmation:
                parent_group = _group_key(row.features(), parent)
                child_group = _group_key(row.features(), child)
                parent_model = discovery_models.get((parent, parent_group))
                child_model = discovery_models.get((child, child_group))
                if parent_model is None or not candidate_bic_active.get(
                    _group_label(parent, parent_group), False
                ):
                    parent_model = source_model
                if child_model is None or not candidate_bic_active.get(
                    _group_label(child, child_group), False
                ):
                    child_model = source_model
                differences.append(
                    exponential_example_negative_log_likelihood(
                        parent_model.hazard_per_hour(row.features()), row
                    )
                    - exponential_example_negative_log_likelihood(
                        child_model.hazard_per_hour(row.features()), row
                    )
                )
            if not differences:
                continue
            p_bound = _likelihood_ratio_p_bound(differences)
            comparison = (
                f"{_partition_label(parent)}->{_partition_label(child)}"
            )
            predictive_veto_p_values[comparison] = p_bound
            if sum(differences) > 0.0 and p_bound <= comparison_threshold:
                partition_veto[_partition_label(parent)] = True

    partition_threshold = familywise_alpha / len(partitions)
    global_supported = bool(significant.get((), set()))
    eligible = [
        partition
        for partition in partitions
        if significant[partition] and partition_gains[_partition_label(partition)] > 0.0
        and not partition_veto[_partition_label(partition)]
        and (
            activation_scope == "reversal_only"
            or partition == ()
            or global_supported
            or partition_source_p_values[_partition_label(partition)]
            <= partition_threshold
        )
    ]
    if not eligible:
        selected = None
    elif activation_scope == "any_predictive_gain":
        selected = min(
            eligible,
            key=lambda partition: (
                len(partition),
                -partition_gains[_partition_label(partition)],
                partition,
            ),
        )
    else:
        selected = max(
            eligible,
            key=lambda partition: (
                partition_gains[_partition_label(partition)],
                -len(partition),
                tuple(-index for index in partition),
            ),
        )
    group_models: dict[tuple[str, ...], RiskModel] = {}
    selected_groups: frozenset[tuple[str, ...]] = frozenset()
    if selected is not None:
        selected_groups = frozenset(significant[selected])
        for group in selected_groups:
            group_models[group] = fit_log_risk_affine_adapter(
                source_model,
                full_by_partition[selected][group],
                fit_slope=candidate_fit_slope[
                    _group_label(selected, group)
                ],
                epochs=epochs,
                learning_rate=learning_rate,
                slope_l2_penalty=slope_l2_penalty,
            )

    return HierarchicalTypedInteractionGate(
        source_model=source_model,
        activation_scope=activation_scope,
        discovery_complexity=discovery_complexity,
        calibration_value_support=tuple(
            frozenset(row.features()[index] for row in rows)
            for index in range(5)
        ),
        selected_partition=selected,
        discovery_bic_active=candidate_bic_active,
        group_heterogeneity_veto=group_heterogeneity_veto,
        group_models=group_models,
        significant_groups=selected_groups,
        discovery_slopes=slopes,
        confirmation_p_values=p_values,
        confirmation_mean_gains=mean_gains,
        partition_mean_gains=partition_gains,
        partition_heterogeneity_veto=partition_veto,
        partition_predictive_veto_p_values=predictive_veto_p_values,
        calibration_examples=len(rows),
        discovery_examples=len(discovery),
        discovery_slope_fitted=candidate_fit_slope,
        confirmation_examples=len(confirmation),
        familywise_alpha=familywise_alpha,
        partition_source_p_values=partition_source_p_values,
        candidate_group_count=candidate_count,
    )
