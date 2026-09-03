from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .advanced_survival_evaluation import evaluate_survival_advanced
from .compositional_persistence import evaluate_grounding_model
from .observation_history import ObservationHistoryRecord, history_to_examples
from .persistence_data import PersistenceTrainingExample
from .statistical_persistence import (
    FactorizedExponentialPersistenceModel,
    GlobalExponentialPersistenceModel,
    PerContextExponentialPersistenceModel,
)
from .survival_evaluation import survival_negative_log_likelihood
from .teach_adapter import read_teach_replay, teach_visible_observation_history
from .teach_audit import TeachAuditSession
from .teach_feasibility import assign_teach_floorplan_splits
from .teach_grounding import (
    TEACH_BOOLEAN_STATE_PROPERTIES,
    build_teach_gold_grounding_cases,
    teach_grounding_registry,
)
from .temporal_grounding import NoDecayPersistenceModel, TemporalGroundingCase

TEACH_TYPED_FEATURES = (
    "property",
    "subject_type",
    "observed_state",
    "context_object",
    "scene",
)
TEACH_FACTORIZED_ABLATIONS = (
    ("train_factorized_property_only", (0,)),
    ("train_factorized_property_state", (0, 2)),
    ("train_factorized_property_subject_state", (0, 1, 2)),
    ("train_factorized_property_state_scene", (0, 2, 4)),
    ("train_factorized_exponential", (0, 1, 2, 3, 4)),
)
TEACH_PER_CONTEXT_PRIOR_EXPOSURE_HOURS = 1.0


@dataclass(frozen=True, slots=True)
class TeachExperimentPartition:
    name: str
    floorplans: tuple[str, ...]
    session_ids: tuple[str, ...]
    histories: tuple[ObservationHistoryRecord, ...]
    examples: tuple[PersistenceTrainingExample, ...]
    all_cases: tuple[TemporalGroundingCase, ...]
    primary_cases: tuple[TemporalGroundingCase, ...]


@dataclass(frozen=True, slots=True)
class TeachPreparedExperiment:
    train: TeachExperimentPartition
    validation: TeachExperimentPartition
    test: TeachExperimentPartition
    split_audit: Mapping[str, Any]
    property_names: tuple[str, ...]


def _partition(
    name: str,
    floorplans: Sequence[str],
    session_ids: list[str],
    histories: list[ObservationHistoryRecord],
    cases: list[TemporalGroundingCase],
) -> TeachExperimentPartition:
    return TeachExperimentPartition(
        name,
        tuple(sorted(floorplans)),
        tuple(sorted(session_ids)),
        tuple(histories),
        history_to_examples(histories),
        tuple(cases),
        tuple(case for case in cases if "primary-evaluable" in case.tags),
    )


def prepare_teach_layer_b_experiment(
    sessions: Iterable[TeachAuditSession],
    *,
    property_names: tuple[str, ...] = TEACH_BOOLEAN_STATE_PROPERTIES,
    split_seed: int = 23,
) -> TeachPreparedExperiment:
    """Prepare floorplan-disjoint histories and cases without fitting a model."""

    rows = tuple(sessions)
    if not rows:
        raise ValueError("at least one TEACh session is required")
    if not property_names or len(property_names) != len(set(property_names)):
        raise ValueError("property_names must be non-empty and unique")
    counts: dict[str, int] = {}
    for session in rows:
        counts[session.floorplan] = counts.get(session.floorplan, 0) + 1
    split_audit = assign_teach_floorplan_splits(counts, seed=split_seed)
    if not split_audit["feasible"]:
        raise ValueError(f"floorplan-disjoint split is infeasible: {split_audit['reason']}")
    floorplan_to_split = {
        floorplan: split
        for split, details in split_audit["splits"].items()
        for floorplan in details["floorplans"]
    }
    by_split = {
        name: {"sessions": [], "histories": [], "cases": []}
        for name in ("train", "validation", "test")
    }
    for session in rows:
        split = floorplan_to_split[session.floorplan]
        initial = json.loads(session.initial_state.read_text(encoding="utf-8"))
        replay = read_teach_replay(
            initial,
            session.state_directory,
            final_timestamp=session.final_timestamp,
        )
        histories = teach_visible_observation_history(
            session.episode_id,
            replay.observations,
            scene=session.floorplan,
            property_names=property_names,
        )
        cases = build_teach_gold_grounding_cases(
            session.episode_id,
            session.floorplan,
            replay,
            property_names=property_names,
        )
        by_split[split]["sessions"].append(session.episode_id)
        by_split[split]["histories"].extend(histories)
        by_split[split]["cases"].extend(cases)
    partitions = {
        name: _partition(
            name,
            split_audit["splits"][name]["floorplans"],
            by_split[name]["sessions"],
            by_split[name]["histories"],
            by_split[name]["cases"],
        )
        for name in ("train", "validation", "test")
    }
    floorplan_sets = [set(partitions[name].floorplans) for name in partitions]
    session_sets = [set(partitions[name].session_ids) for name in partitions]
    if any(
        floorplan_sets[left] & floorplan_sets[right]
        for left in range(3)
        for right in range(left + 1, 3)
    ) or any(
        session_sets[left] & session_sets[right]
        for left in range(3)
        for right in range(left + 1, 3)
    ):
        raise AssertionError("TEACh experiment partitions are not group disjoint")
    return TeachPreparedExperiment(
        partitions["train"],
        partitions["validation"],
        partitions["test"],
        split_audit,
        property_names,
    )


def _validation_horizons(
    examples: Sequence[PersistenceTrainingExample],
) -> tuple[float, ...]:
    durations = sorted(
        example.duration_seconds / 3600.0
        for example in examples
        if example.duration_seconds > 0
    )
    if not durations:
        raise ValueError("validation requires positive duration support")
    indices = (max(0, len(durations) // 4 - 1), len(durations) // 2, 3 * len(durations) // 4)
    horizons = tuple(sorted({durations[min(index, len(durations) - 1)] for index in indices}))
    if not horizons or horizons[0] <= 0:
        raise ValueError("validation-derived horizons must be positive")
    return horizons


def _model_metrics(model, partition: TeachExperimentPartition, registry, horizons):
    survival = evaluate_survival_advanced(
        model,
        partition.examples,
        horizons_hours=horizons,
    )
    grounding = evaluate_grounding_model(
        model.__class__.__name__,
        model,
        partition.primary_cases,
        registry,
    )
    return {
        "survival": {
            "examples": survival.examples,
            "negative_log_likelihood": survival.negative_log_likelihood,
            "concordance_index": survival.concordance_index,
            "integrated_brier_score": survival.integrated_brier_score,
            "horizons": [asdict(item) for item in survival.horizons],
        },
        "grounding": {
            "cases": grounding.cases,
            "top1_accuracy": grounding.top1_accuracy,
            "mean_reciprocal_rank": grounding.mean_reciprocal_rank,
            "accuracy_by_tag": dict(grounding.accuracy_by_tag),
        },
    }

def _survival_feature_support(
    train_examples: Sequence[PersistenceTrainingExample],
    partition: TeachExperimentPartition,
) -> dict[str, Any]:
    """Audit OOD support without consulting outcomes or current truth."""

    train_features = [
        tuple(value.casefold() for value in example.features())
        for example in train_examples
    ]
    split_features = [
        tuple(value.casefold() for value in example.features())
        for example in partition.examples
    ]
    train_contexts = set(train_features)
    exact_seen = sum(features in train_contexts for features in split_features)
    total = len(split_features)
    feature_rows = {}
    for index, name in enumerate(TEACH_TYPED_FEATURES):
        train_values = {features[index] for features in train_features}
        split_values = {features[index] for features in split_features}
        rows_seen = sum(features[index] in train_values for features in split_features)
        feature_rows[name] = {
            "train_unique_values": len(train_values),
            "split_unique_values": len(split_values),
            "unseen_values": sorted(split_values - train_values),
            "row_coverage": rows_seen / total if total else None,
        }
    return {
        "split": partition.name,
        "examples": total,
        "exact_context": {
            "train_unique_contexts": len(train_contexts),
            "seen_examples": exact_seen,
            "row_coverage": exact_seen / total if total else None,
        },
        "features": feature_rows,
        "uses_outcomes": False,
        "uses_current_truth": False,
    }


def run_teach_layer_b_experiment(
    prepared: TeachPreparedExperiment,
    feasibility_gate: Mapping[str, Any],
    *,
    half_life_grid_hours: tuple[float, ...] = (0.25, 1.0, 4.0, 16.0, 64.0, 256.0),
    factorized_epochs: int = 1200,
) -> dict[str, Any]:
    """Fit on train, select/calibrate on validation, and evaluate test once."""

    if feasibility_gate.get("layer_b_ready") is not True:
        raise ValueError("TEACh Layer B feasibility gate has not passed")
    if not half_life_grid_hours or any(value <= 0 for value in half_life_grid_hours):
        raise ValueError("half-life grid must contain positive values")
    if len(set(half_life_grid_hours)) != len(half_life_grid_hours):
        raise ValueError("half-life grid values must be unique")
    if factorized_epochs <= 0:
        raise ValueError("factorized_epochs must be positive")
    for partition in (prepared.train, prepared.validation, prepared.test):
        if not partition.examples or not partition.primary_cases:
            raise ValueError(
                f"{partition.name} requires survival examples and primary grounding cases"
            )
    trained_properties = frozenset(
        example.property_name.casefold() for example in prepared.train.examples
    )
    validation_models = {
        half_life: GlobalExponentialPersistenceModel(
            math.log(2.0) / half_life,
            trained_properties,
        )
        for half_life in half_life_grid_hours
    }
    validation_nll = {
        half_life: survival_negative_log_likelihood(
            model, prepared.validation.examples
        )
        for half_life, model in validation_models.items()
    }
    selected_half_life = min(
        half_life_grid_hours,
        key=lambda value: (validation_nll[value], value),
    )
    fixed_model = validation_models[selected_half_life]
    global_model = GlobalExponentialPersistenceModel.fit(prepared.train.examples)
    per_context_model = PerContextExponentialPersistenceModel.fit(
        prepared.train.examples,
        prior_exposure_hours=TEACH_PER_CONTEXT_PRIOR_EXPOSURE_HOURS,
    )
    factorized_models = {}
    factorized_scales = {}
    for name, active_indices in TEACH_FACTORIZED_ABLATIONS:
        model = FactorizedExponentialPersistenceModel.fit(
            prepared.train.examples,
            epochs=factorized_epochs,
            active_feature_indices=active_indices,
        )
        factorized_scales[name] = model.calibrate(prepared.validation.examples)
        factorized_models[name] = model
    horizons = _validation_horizons(prepared.validation.examples)
    registry = teach_grounding_registry(prepared.property_names)
    no_decay = evaluate_grounding_model(
        "no-decay",
        NoDecayPersistenceModel(),
        prepared.test.primary_cases,
        registry,
    )
    return {
        "protocol": {
            "fit": "train floorplans only",
            "selection_and_calibration": "validation floorplans only",
            "test_use": "single final evaluation; no model selection",
            "split_seed": prepared.split_audit["seed"],
            "half_life_grid_hours": list(half_life_grid_hours),
            "fixed_selection_metric": "validation interval-aware NLL",
            "evaluation_horizons_hours": list(horizons),
            "horizon_source": "validation duration quartiles only",
            "primary_case_rule": "input-identifiable static or temporal-discriminative cases",
            "excluded_case_policy": "retained in coverage counts, excluded from primary accuracy",
            "current_truth_use": "query and evaluation labels only",
            "frozen_model_matrix": {
                "no_decay": "no temporal discount",
                "validation_selected_fixed": "fixed half-life selected by validation NLL",
                "train_global_exponential": "one train-only hazard",
                "train_per_context_exponential": {
                    "family": "exact-tuple train MLE with global OOD backoff",
                    "global_prior_exposure_hours": TEACH_PER_CONTEXT_PRIOR_EXPOSURE_HOURS,
                },
                **{
                    name: {
                        "family": "validation-calibrated factorized exponential",
                        "active_features": [
                            TEACH_TYPED_FEATURES[index] for index in active_indices
                        ],
                    }
                    for name, active_indices in TEACH_FACTORIZED_ABLATIONS
                },
            },
            "ablation_interpretation": (
                "property-only controls property-specific decay; nested subject, state, "
                "and scene variants attribute any full-model gain"
            ),
            "support_audit_use": (
                "feature values and exact context membership only; no event outcomes or "
                "current truth"
            ),
            "claim_scope": "semi-real Layer B only after official-data gate passes",
        },
        "split": {
            name: {
                "floorplans": list(partition.floorplans),
                "sessions": len(partition.session_ids),
                "survival_examples": len(partition.examples),
                "all_grounding_cases": len(partition.all_cases),
                "primary_grounding_cases": len(partition.primary_cases),
            }
            for name, partition in (
                ("train", prepared.train),
                ("validation", prepared.validation),
                ("test", prepared.test),
            )
        },
        "survival_feature_support": {
            partition.name: _survival_feature_support(
                prepared.train.examples, partition
            )
            for partition in (prepared.validation, prepared.test)
        },
        "validation": {
            "fixed_half_life_nll": {
                str(value): validation_nll[value] for value in half_life_grid_hours
            },
            "selected_fixed_half_life_hours": selected_half_life,
            "factorized_hazard_scales": factorized_scales,
        },
        "test": {
            "no_decay": {
                "grounding": {
                    "cases": no_decay.cases,
                    "top1_accuracy": no_decay.top1_accuracy,
                    "mean_reciprocal_rank": no_decay.mean_reciprocal_rank,
                    "accuracy_by_tag": dict(no_decay.accuracy_by_tag),
                }
            },
            "validation_selected_fixed": _model_metrics(
                fixed_model, prepared.test, registry, horizons
            ),
            "train_global_exponential": _model_metrics(
                global_model, prepared.test, registry, horizons
            ),
            "train_per_context_exponential": _model_metrics(
                per_context_model, prepared.test, registry, horizons
            ),
            **{
                name: _model_metrics(model, prepared.test, registry, horizons)
                for name, model in factorized_models.items()
            },
        },
    }


def write_teach_layer_b_report(path: str | Path, report: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
