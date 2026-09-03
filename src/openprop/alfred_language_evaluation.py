from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Mapping

from .alfred_adapter import AlfredLanguageCase
from .alfred_ontology import (
    AlfredTrainingOntology,
    normalise_alfred_goal_frame,
)
from .alfred_selection import (
    SelectionFusionPolicy,
    extract_alfred_selection_evidence,
    fuse_alfred_selection,
)
from .language_temporal_grounding import RawLanguageResponse
from .llm import JSONLLMClient, LLMQueryParser, ParsedQuery
from .models import PropertyConstraint, PropertyDefinition, QueryFrame, ValueType
from .property_registry import PropertyRegistry
from .schema_repair import repair_redundant_relation_fields


class AlfredLanguageStrategy(str, Enum):
    GOLD = "gold"
    LLM_STRICT = "llm-strict"
    LLM_TOLERANT = "llm-tolerant"
    LLM_SCHEMA_REPAIRED = "llm-schema-repaired"
    LLM_ONTOLOGY_NORMALIZED = "llm-ontology-normalized"
    LLM_EVIDENCE_FUSED = "llm-evidence-fused"


@dataclass(frozen=True, slots=True)
class AlfredLanguageResult:
    case_id: str
    task_type: str
    property_precision: float
    property_recall: float
    property_f1: float
    value_recall: float
    relevance_mae: float
    validation_errors: tuple[str, ...]
    repair_actions: tuple[str, ...]
    selection_actions: tuple[str, ...]
    normalisation_actions: tuple[str, ...]
    ignored_properties: tuple[str, ...]
    selected_properties: tuple[str, ...]
    error: str | None = None


@dataclass(frozen=True, slots=True)
class AlfredLanguageReport:
    strategy: AlfredLanguageStrategy
    cases: int
    completed: int
    failures: int
    parse_success_rate: float
    property_precision: float
    property_recall: float
    property_f1: float
    value_recall: float
    relevance_mae: float
    exact_frame_accuracy: float
    repair_rate: float
    selection_action_rate: float
    normalisation_rate: float
    validation_error_rate: float
    exact_frame_accuracy_by_task: Mapping[str, float]
    results: tuple[AlfredLanguageResult, ...]


def alfred_language_registry() -> PropertyRegistry:
    registry = PropertyRegistry()
    registry.register(
        PropertyDefinition("type", "semantic object category", ValueType.SEMANTIC)
    )
    registry.register(
        PropertyDefinition(
            "location",
            "goal spatial relation between the target object and a receptacle or surface",
            ValueType.RELATION,
            aliases=("destination", "spatial relation", "position relation"),
            metadata={
                "argument_roles": ["object"],
                "allowed_predicates": ["on", "inside"],
            },
        )
    )
    registry.register(
        PropertyDefinition(
            "cleanliness", "desired cleanliness state of the target object", ValueType.SEMANTIC
        )
    )
    registry.register(
        PropertyDefinition(
            "thermal_state",
            "desired thermal state of the target object, such as hot or cold",
            ValueType.SEMANTIC,
            aliases=("thermal state", "temperature state"),
        )
    )
    return registry


def select_stratified_cases(
    cases: Iterable[AlfredLanguageCase],
    *,
    split: str = "valid_unseen",
    trajectories_per_task: int = 10,
    annotation_index: int = 0,
    trajectory_offset: int = 0,
) -> tuple[AlfredLanguageCase, ...]:
    """Select a deterministic, task-balanced sample before model evaluation."""

    if trajectories_per_task <= 0:
        raise ValueError("trajectories_per_task must be positive")
    if trajectory_offset < 0:
        raise ValueError("trajectory_offset cannot be negative")
    grouped: dict[str, list[AlfredLanguageCase]] = {}
    for case in cases:
        if case.split == split and case.annotation_index == annotation_index:
            grouped.setdefault(case.task_type, []).append(case)
    selected: list[AlfredLanguageCase] = []
    for task_type in sorted(grouped):
        candidates = sorted(grouped[task_type], key=lambda item: item.case_id)
        required = trajectory_offset + trajectories_per_task
        if len(candidates) < required:
            raise ValueError(
                f"task {task_type} has only {len(candidates)} eligible trajectories; {required} required"
            )
        selected.extend(candidates[trajectory_offset:required])
    if not selected:
        raise ValueError("stratified selection produced no cases")
    return tuple(selected)


def collect_alfred_language_responses(
    cases: Iterable[AlfredLanguageCase],
    registry: PropertyRegistry,
    client: JSONLLMClient,
) -> Mapping[str, RawLanguageResponse]:
    parser = LLMQueryParser(client)
    responses: dict[str, RawLanguageResponse] = {}
    for case in cases:
        if case.query in responses:
            continue
        started = time.perf_counter()
        try:
            raw = parser.request(case.query, registry)
            responses[case.query] = RawLanguageResponse(
                case.query, raw, time.perf_counter() - started
            )
        except Exception as error:
            responses[case.query] = RawLanguageResponse(
                case.query,
                None,
                time.perf_counter() - started,
                f"{type(error).__name__}: {error}",
            )
    return responses


def _constraint_metrics(
    predicted: QueryFrame, gold: QueryFrame
) -> tuple[float, float, float, float, float]:
    predicted_by_name = {
        item.property_name.casefold(): item for item in predicted.constraints
    }
    gold_by_name = {item.property_name.casefold(): item for item in gold.constraints}
    overlap = predicted_by_name.keys() & gold_by_name.keys()
    precision = len(overlap) / len(predicted_by_name) if predicted_by_name else 0.0
    recall = len(overlap) / len(gold_by_name) if gold_by_name else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    value_recall = (
        sum(
            predicted_by_name[name].desired_value == gold_by_name[name].desired_value
            for name in overlap
        )
        / len(gold_by_name)
        if gold_by_name
        else 0.0
    )
    names = predicted_by_name.keys() | gold_by_name.keys()
    relevance_mae = (
        sum(
            abs(
                predicted_by_name.get(name, PropertyConstraint(name, "", 0.0)).relevance
                - gold_by_name.get(name, PropertyConstraint(name, "", 0.0)).relevance
            )
            for name in names
        )
        / len(names)
        if names
        else 0.0
    )
    return precision, recall, f1, value_recall, relevance_mae


def evaluate_alfred_language(
    cases: Iterable[AlfredLanguageCase],
    registry: PropertyRegistry,
    strategy: AlfredLanguageStrategy,
    *,
    responses: Mapping[str, RawLanguageResponse] | None = None,
    ontology: AlfredTrainingOntology | None = None,
) -> AlfredLanguageReport:
    rows = tuple(cases)
    if not rows:
        raise ValueError("at least one ALFRED language case is required")
    if strategy is not AlfredLanguageStrategy.GOLD and responses is None:
        raise ValueError("raw language responses are required for an LLM strategy")
    if (
        strategy
        in {
            AlfredLanguageStrategy.LLM_ONTOLOGY_NORMALIZED,
            AlfredLanguageStrategy.LLM_EVIDENCE_FUSED,
        }
        and ontology is None
    ):
        raise ValueError("training ontology is required for ontology normalization")
    parser = None
    if strategy is not AlfredLanguageStrategy.GOLD:
        parser = LLMQueryParser(
            _UnusedClient(),
            skip_invalid_constraints=strategy
            in {
                AlfredLanguageStrategy.LLM_TOLERANT,
                AlfredLanguageStrategy.LLM_SCHEMA_REPAIRED,
                AlfredLanguageStrategy.LLM_ONTOLOGY_NORMALIZED,
                AlfredLanguageStrategy.LLM_EVIDENCE_FUSED,
            },
        )
    results: list[AlfredLanguageResult] = []
    for case in rows:
        parsed = ParsedQuery(case.gold_frame)
        repair_actions: tuple[str, ...] = ()
        selection_actions: tuple[str, ...] = ()
        normalisation_actions: tuple[str, ...] = ()
        try:
            if parser is not None:
                assert responses is not None
                captured = responses.get(case.query)
                if captured is None:
                    raise ValueError("missing raw response for query")
                if captured.error is not None:
                    raise RuntimeError(captured.error)
                if captured.response is None:
                    raise ValueError("captured response is empty")
                raw = captured.response
                if strategy in {
                    AlfredLanguageStrategy.LLM_SCHEMA_REPAIRED,
                    AlfredLanguageStrategy.LLM_ONTOLOGY_NORMALIZED,
                    AlfredLanguageStrategy.LLM_EVIDENCE_FUSED,
                }:
                    repair = repair_redundant_relation_fields(raw, registry)
                    raw = repair.response
                    repair_actions = repair.actions
                parsed = parser.parse_response(case.query, registry, raw)
                if strategy is AlfredLanguageStrategy.LLM_EVIDENCE_FUSED:
                    assert ontology is not None
                    fused = fuse_alfred_selection(
                        parsed.frame,
                        extract_alfred_selection_evidence(case.query, ontology),
                        policy=SelectionFusionPolicy(
                            gate_conflicting_states=True
                        ),
                    )
                    parsed = ParsedQuery(
                        fused.frame,
                        parsed.created_properties,
                        parsed.ignored_properties,
                        parsed.validation_errors,
                    )
                    selection_actions = fused.actions
                if strategy in {
                    AlfredLanguageStrategy.LLM_ONTOLOGY_NORMALIZED,
                    AlfredLanguageStrategy.LLM_EVIDENCE_FUSED,
                }:
                    assert ontology is not None
                    normalised = normalise_alfred_goal_frame(parsed.frame, ontology)
                    parsed = ParsedQuery(
                        normalised.frame,
                        parsed.created_properties,
                        parsed.ignored_properties,
                        parsed.validation_errors,
                    )
                    normalisation_actions = normalised.actions
            precision, recall, f1, value_recall, relevance_mae = _constraint_metrics(
                parsed.frame, case.gold_frame
            )
            results.append(
                AlfredLanguageResult(
                    case.case_id,
                    case.task_type,
                    precision,
                    recall,
                    f1,
                    value_recall,
                    relevance_mae,
                    parsed.validation_errors,
                    repair_actions,
                    selection_actions,
                    normalisation_actions,
                    parsed.ignored_properties,
                    tuple(item.property_name for item in parsed.frame.constraints),
                )
            )
        except Exception as error:
            results.append(
                AlfredLanguageResult(
                    case.case_id,
                    case.task_type,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    (),
                    repair_actions,
                    selection_actions,
                    normalisation_actions,
                    (),
                    (),
                    f"{type(error).__name__}: {error}",
                )
            )
    count = len(results)
    completed = [item for item in results if item.error is None]
    exact = lambda item: item.property_f1 == 1.0 and item.value_recall == 1.0
    task_types = sorted({item.task_type for item in results})
    by_task = {
        task_type: sum(exact(item) for item in results if item.task_type == task_type)
        / sum(item.task_type == task_type for item in results)
        for task_type in task_types
    }
    return AlfredLanguageReport(
        strategy,
        count,
        len(completed),
        count - len(completed),
        len(completed) / count,
        sum(item.property_precision for item in results) / count,
        sum(item.property_recall for item in results) / count,
        sum(item.property_f1 for item in results) / count,
        sum(item.value_recall for item in results) / count,
        sum(item.relevance_mae for item in results) / count,
        sum(exact(item) for item in results) / count,
        sum(bool(item.repair_actions) for item in results) / count,
        sum(bool(item.selection_actions) for item in results) / count,
        sum(bool(item.normalisation_actions) for item in results) / count,
        sum(bool(item.validation_errors) for item in results) / count,
        by_task,
        tuple(results),
    )


class _UnusedClient:
    def generate_json(self, **_: object) -> Mapping[str, object]:
        raise AssertionError("parse_response must not call the client")
