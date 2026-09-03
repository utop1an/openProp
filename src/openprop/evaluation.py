from __future__ import annotations

import time
from dataclasses import dataclass, replace
from enum import Enum

from .benchmark import BenchmarkCase
from .comparators import default_comparators
from .llm import JSONLLMClient, LLMQueryParser
from .matcher import EntityMatcher
from .models import PropertyConstraint, QueryFrame
from .property_registry import PropertyRegistry
from .selectors import MentionBasedSelector


class EvaluationStrategy(str, Enum):
    GOLD_WEIGHTED = "gold-weighted"
    GOLD_EQUAL = "gold-equal"
    LLM_WEIGHTED = "llm-weighted"


@dataclass(frozen=True, slots=True)
class CaseResult:
    case_id: str
    target_id: str
    predicted_id: str | None
    rank: int | None
    property_precision: float
    property_recall: float
    property_f1: float
    target_score: float
    target_coverage: float
    latency_seconds: float
    error: str | None = None


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    strategy: EvaluationStrategy
    cases: int
    completed: int
    failures: int
    top1_accuracy: float
    top3_recall: float
    mean_reciprocal_rank: float
    property_precision: float
    property_recall: float
    property_f1: float
    mean_target_coverage: float
    mean_latency_seconds: float
    results: tuple[CaseResult, ...]


def _equal_weights(frame: QueryFrame) -> QueryFrame:
    constraints = tuple(replace(constraint, relevance=1.0) for constraint in frame.constraints)
    return QueryFrame(frame.text, constraints)


def _with_distractors(frame: QueryFrame, case: BenchmarkCase) -> QueryFrame:
    distractors = getattr(case, "distractor_constraints", ())
    return QueryFrame(frame.text, (*frame.constraints, *distractors))


def _selection_metrics(predicted: QueryFrame, gold: QueryFrame) -> tuple[float, float, float]:
    predicted_names = {item.property_name.casefold() for item in predicted.constraints}
    gold_names = {item.property_name.casefold() for item in gold.constraints}
    overlap = len(predicted_names & gold_names)
    precision = overlap / len(predicted_names) if predicted_names else 0.0
    recall = overlap / len(gold_names) if gold_names else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def evaluate(
    cases: tuple[BenchmarkCase, ...],
    registry: PropertyRegistry,
    strategy: EvaluationStrategy,
    *,
    llm_client: JSONLLMClient | None = None,
) -> EvaluationReport:
    if strategy is EvaluationStrategy.LLM_WEIGHTED and llm_client is None:
        raise ValueError("llm_client is required for llm-weighted evaluation")
    parser = (
        LLMQueryParser(llm_client, skip_invalid_constraints=True)
        if llm_client is not None
        else None
    )
    matcher = EntityMatcher(registry, default_comparators(), MentionBasedSelector())
    results: list[CaseResult] = []

    for case in cases:
        started = time.perf_counter()
        try:
            if strategy is EvaluationStrategy.GOLD_WEIGHTED:
                frame = _with_distractors(case.gold_frame, case)
                selection_frame = case.gold_frame
            elif strategy is EvaluationStrategy.GOLD_EQUAL:
                frame = _equal_weights(_with_distractors(case.gold_frame, case))
                selection_frame = case.gold_frame
            else:
                assert parser is not None
                frame = parser.parse(case.query, registry).frame
                selection_frame = frame

            precision, recall, f1 = _selection_metrics(selection_frame, case.gold_frame)
            ranking = matcher.match(frame, list(case.entities))
            ids = [item.entity_id for item in ranking]
            rank = ids.index(case.target_id) + 1 if case.target_id in ids else None
            target = next(item for item in ranking if item.entity_id == case.target_id)
            results.append(
                CaseResult(
                    case.case_id,
                    case.target_id,
                    ids[0] if ids else None,
                    rank,
                    precision,
                    recall,
                    f1,
                    target.score,
                    target.coverage,
                    time.perf_counter() - started,
                )
            )
        except Exception as error:
            results.append(
                CaseResult(
                    case.case_id,
                    case.target_id,
                    None,
                    None,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    time.perf_counter() - started,
                    f"{type(error).__name__}: {error}",
                )
            )

    count = len(results)
    completed = [result for result in results if result.error is None]
    denominator = count or 1
    return EvaluationReport(
        strategy=strategy,
        cases=count,
        completed=len(completed),
        failures=count - len(completed),
        top1_accuracy=sum(result.rank == 1 for result in results) / denominator,
        top3_recall=sum(result.rank is not None and result.rank <= 3 for result in results) / denominator,
        mean_reciprocal_rank=sum(1 / result.rank for result in results if result.rank) / denominator,
        property_precision=sum(result.property_precision for result in results) / denominator,
        property_recall=sum(result.property_recall for result in results) / denominator,
        property_f1=sum(result.property_f1 for result in results) / denominator,
        mean_target_coverage=sum(result.target_coverage for result in results) / denominator,
        mean_latency_seconds=sum(result.latency_seconds for result in results) / denominator,
        results=tuple(results),
    )
