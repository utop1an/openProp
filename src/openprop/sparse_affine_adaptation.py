from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

from .persistence_data import PersistenceTrainingExample
from .survival_evaluation import (
    exponential_example_negative_log_likelihood,
    model_risk_score,
)
from .target_adaptation import RiskModel
from .target_interaction_adaptation import (
    HierarchicalTypedInteractionGate,
    fit_hierarchical_typed_interaction_gate,
)


SPARSE_AFFINE_PARTITIONS: tuple[tuple[int, ...], ...] = ((1,), (4,), (1, 4))


@dataclass(frozen=True, slots=True)
class SparseCoverageAffineGate:
    """Accept a confirmed typed affine repair only when its coverage is sparse."""

    source_model: RiskModel
    candidate_gate: HierarchicalTypedInteractionGate
    adapted_context_fraction: float
    max_adapted_context_fraction: float
    coverage_rejected: bool

    def __post_init__(self) -> None:
        if self.candidate_gate.source_model is not self.source_model:
            raise ValueError("candidate gate must wrap the selected source model")
        if not 0.0 <= self.adapted_context_fraction <= 1.0:
            raise ValueError("adapted context fraction must be between zero and one")
        if not 0.0 < self.max_adapted_context_fraction < 1.0:
            raise ValueError("maximum adapted context fraction must be between zero and one")
        expected_rejection = (
            self.candidate_gate.activated
            and self.adapted_context_fraction > self.max_adapted_context_fraction
        )
        if self.coverage_rejected != expected_rejection:
            raise ValueError("coverage rejection does not match candidate diagnostics")

    @property
    def activated(self) -> bool:
        return self.candidate_gate.activated and not self.coverage_rejected

    @property
    def selected_partition(self) -> tuple[int, ...] | None:
        return self.candidate_gate.selected_partition if self.activated else None

    @property
    def significant_groups(self) -> frozenset[tuple[str, ...]]:
        return (
            self.candidate_gate.significant_groups
            if self.activated
            else frozenset()
        )

    def _model(self) -> RiskModel:
        return self.candidate_gate if self.activated else self.source_model

    def hazard_per_hour(self, features: tuple[str, ...]) -> float:
        return self._model().hazard_per_hour(features)

    def risk_score(self, features: tuple[str, ...]) -> float:
        return model_risk_score(self._model(), features)

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


def fit_sparse_coverage_affine_gate(
    source_model: RiskModel,
    examples: Iterable[PersistenceTrainingExample],
    *,
    split_seed: int,
    candidate_partitions: Sequence[tuple[int, ...]] = SPARSE_AFFINE_PARTITIONS,
    familywise_alpha: float = 0.05,
    max_adapted_context_fraction: float = 0.5,
    epochs: int = 1000,
    learning_rate: float = 0.03,
    slope_l2_penalty: float = 1e-4,
) -> SparseCoverageAffineGate:
    """Fit the affine gate with global exclusion and sparse-coverage closure.

    Candidate fitting and identity-disjoint confirmation are delegated to the
    multiplicity-controlled typed gate. Only non-global typed partitions are
    permitted. A confirmed candidate is discarded when its selected groups
    cover more than the declared fraction of calibration contexts. The closure
    is computed from typed context membership, never outcomes or test metrics.
    """

    rows = tuple(examples)
    if not rows:
        raise ValueError("at least one calibration example is required")
    partitions = tuple(tuple(value) for value in candidate_partitions)
    if not partitions or any(not partition for partition in partitions):
        raise ValueError("sparse affine closure excludes the global partition")
    if not 0.0 < max_adapted_context_fraction < 1.0:
        raise ValueError("maximum adapted context fraction must be between zero and one")
    gate = fit_hierarchical_typed_interaction_gate(
        source_model,
        rows,
        split_seed=split_seed,
        candidate_partitions=partitions,
        familywise_alpha=familywise_alpha,
        activation_scope="any_predictive_gain",
        discovery_complexity="bic",
        epochs=epochs,
        learning_rate=learning_rate,
        slope_l2_penalty=slope_l2_penalty,
    )
    contexts = frozenset(row.features() for row in rows)
    adapted = 0
    if gate.selected_partition is not None:
        adapted = sum(
            tuple(features[index] for index in gate.selected_partition)
            in gate.significant_groups
            for features in contexts
        )
    fraction = adapted / len(contexts)
    rejected = gate.activated and fraction > max_adapted_context_fraction
    return SparseCoverageAffineGate(
        source_model,
        gate,
        fraction,
        max_adapted_context_fraction,
        rejected,
    )
