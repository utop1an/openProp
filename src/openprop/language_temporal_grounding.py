from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Mapping

from .comparators import default_comparators
from .llm import JSONLLMClient, LLMQueryParser, ParsedQuery
from .matcher import EntityMatcher
from .models import PropertyConstraint, QueryFrame
from .persistence import ExponentialPersistenceModel, PersistenceModel
from .schema_repair import repair_redundant_relation_fields
from .property_registry import PropertyRegistry
from .selectors import MentionBasedSelector
from .temporal_grounding import TemporalGroundingCase


class LanguageTemporalStrategy(str, Enum):
    GOLD = "gold"
    LLM_SCHEMA_REPAIRED = "llm-schema-repaired"
    LLM_STRICT = "llm-strict"
    LLM_TOLERANT = "llm-tolerant"


@dataclass(frozen=True, slots=True)
class RawLanguageResponse:
    query: str
    response: Mapping[str, object] | None
    latency_seconds: float
    error: str | None = None


@dataclass(frozen=True, slots=True)
class LanguageTemporalCaseResult:
    case_id: str
    target_id: str
    predicted_id: str | None
    rank: int | None
    property_precision: float
    property_recall: float
    property_f1: float
    value_recall: float
    relevance_mae: float
    validation_errors: tuple[str, ...]
    repair_actions: tuple[str, ...]
    ignored_properties: tuple[str, ...]
    selected_properties: tuple[str, ...]
    tags: tuple[str, ...]
    error: str | None = None


@dataclass(frozen=True, slots=True)
class LanguageTemporalReport:
    strategy: LanguageTemporalStrategy
    cases: int
    completed: int
    failures: int
    parse_success_rate: float
    top1_accuracy: float
    conditional_top1_accuracy: float
    top3_recall: float
    mean_reciprocal_rank: float
    property_precision: float
    property_recall: float
    property_f1: float
    value_recall: float
    relevance_mae: float
    repair_rate: float
    validation_error_rate: float
    accuracy_by_tag: Mapping[str, float]
    results: tuple[LanguageTemporalCaseResult, ...]


def collect_language_responses(
    cases: Iterable[TemporalGroundingCase],
    registry: PropertyRegistry,
    client: JSONLLMClient,
) -> Mapping[str, RawLanguageResponse]:
    """Request each distinct query once for fair strict/tolerant replay."""

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
    predicted: QueryFrame,
    gold: QueryFrame,
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


def evaluate_language_temporal_grounding(
    cases: Iterable[TemporalGroundingCase],
    registry: PropertyRegistry,
    strategy: LanguageTemporalStrategy,
    *,
    responses: Mapping[str, RawLanguageResponse] | None = None,
    persistence_model: PersistenceModel | None = None,
) -> LanguageTemporalReport:
    rows = tuple(cases)
    if not rows:
        raise ValueError("at least one temporal grounding case is required")
    if strategy is not LanguageTemporalStrategy.GOLD and responses is None:
        raise ValueError("raw language responses are required for an LLM strategy")
    parser = None
    if strategy is not LanguageTemporalStrategy.GOLD:
        parser = LLMQueryParser(
            _UnusedClient(),
            skip_invalid_constraints=strategy
            in {
                LanguageTemporalStrategy.LLM_TOLERANT,
                LanguageTemporalStrategy.LLM_SCHEMA_REPAIRED,
            },
        )
    matcher = EntityMatcher(
        registry,
        default_comparators(),
        MentionBasedSelector(),
        persistence_model=persistence_model or ExponentialPersistenceModel(),
    )
    results: list[LanguageTemporalCaseResult] = []
    for case in rows:
        parsed = ParsedQuery(case.gold_frame)
        repair_actions: tuple[str, ...] = ()
        try:
            if parser is not None:
                assert responses is not None
                captured = responses.get(case.query)
                if captured is None:
                    raise ValueError("missing raw response for query")
                if captured.error is not None:
                    raise RuntimeError(captured.error)
                assert captured.response is not None
                raw = captured.response
                if strategy is LanguageTemporalStrategy.LLM_SCHEMA_REPAIRED:
                    repair = repair_redundant_relation_fields(raw, registry)
                    raw = repair.response
                    repair_actions = repair.actions
                parsed = parser.parse_response(case.query, registry, raw)
            precision, recall, f1, value_recall, relevance_mae = _constraint_metrics(
                parsed.frame, case.gold_frame
            )
            ranking = matcher.match(parsed.frame, list(case.entities), as_of=case.as_of)
            ids = [item.entity_id for item in ranking]
            rank = ids.index(case.target_id) + 1 if case.target_id in ids else None
            results.append(
                LanguageTemporalCaseResult(
                    case.case_id,
                    case.target_id,
                    ids[0] if ids else None,
                    rank,
                    precision,
                    recall,
                    f1,
                    value_recall,
                    relevance_mae,
                    parsed.validation_errors,
                    repair_actions,
                    parsed.ignored_properties,
                    tuple(item.property_name for item in parsed.frame.constraints),
                    case.tags,
                )
            )
        except Exception as error:
            results.append(
                LanguageTemporalCaseResult(
                    case.case_id,
                    case.target_id,
                    None,
                    None,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    (),
                    (),
                    (),
                    (),
                    case.tags,
                    f"{type(error).__name__}: {error}",
                )
            )
    count = len(results)
    completed = [item for item in results if item.error is None]
    tags = sorted({tag for item in results for tag in item.tags})
    accuracy_by_tag = {
        tag: sum(item.rank == 1 for item in results if tag in item.tags)
        / sum(tag in item.tags for item in results)
        for tag in tags
    }
    return LanguageTemporalReport(
        strategy,
        count,
        len(completed),
        count - len(completed),
        len(completed) / count,
        sum(item.rank == 1 for item in results) / count,
        sum(item.rank == 1 for item in completed) / len(completed) if completed else 0.0,
        sum(item.rank is not None and item.rank <= 3 for item in results) / count,
        sum(1.0 / item.rank for item in results if item.rank) / count,
        sum(item.property_precision for item in results) / count,
        sum(item.property_recall for item in results) / count,
        sum(item.property_f1 for item in results) / count,
        sum(item.value_recall for item in results) / count,
        sum(item.relevance_mae for item in results) / count,
        sum(bool(item.repair_actions) for item in results) / count,
        sum(bool(item.validation_errors) for item in results) / count,
        accuracy_by_tag,
        tuple(results),
    )


class _UnusedClient:
    def generate_json(self, **_: object) -> Mapping[str, object]:
        raise AssertionError("parse_response must not call the client")
