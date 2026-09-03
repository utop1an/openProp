from __future__ import annotations

import argparse
import json
from pathlib import Path

from openprop.alfred_adapter import AlfredLanguageCase, load_alfred_language_dataset
from openprop.alfred_language_evaluation import select_stratified_cases
from openprop.alfred_ontology import fit_alfred_training_ontology
from openprop.alfred_retrieval import AlfredBM25FrameRetriever
from openprop.alfred_selection import (
    SelectionFusionPolicy,
    extract_alfred_selection_evidence,
    fuse_alfred_selection,
)
from openprop.models import PropertyConstraint, QueryFrame


def _metrics(predicted: QueryFrame | None, gold: QueryFrame) -> dict[str, float | bool]:
    if predicted is None:
        return {
            "property_precision": 0.0,
            "property_recall": 0.0,
            "property_f1": 0.0,
            "value_recall": 0.0,
            "exact_frame": False,
        }
    predicted_by_name = {item.property_name: item for item in predicted.constraints}
    gold_by_name = {item.property_name: item for item in gold.constraints}
    overlap = predicted_by_name.keys() & gold_by_name.keys()
    precision = len(overlap) / len(predicted_by_name) if predicted_by_name else 0.0
    recall = len(overlap) / len(gold_by_name)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    value_recall = sum(
        predicted_by_name[name].desired_value == gold_by_name[name].desired_value
        for name in overlap
    ) / len(gold_by_name)
    return {
        "property_precision": precision,
        "property_recall": recall,
        "property_f1": f1,
        "value_recall": value_recall,
        "exact_frame": set(predicted_by_name) == set(gold_by_name)
        and value_recall == 1.0,
    }


def _evidence_frame(case: AlfredLanguageCase, ontology) -> QueryFrame:
    evidence = extract_alfred_selection_evidence(case.query, ontology)
    return QueryFrame(
        case.query,
        tuple(
            PropertyConstraint(
                item.property_name,
                item.desired_value,
                0.35 if item.property_name in {"cleanliness", "thermal_state"} else 0.5,
            )
            for item in evidence.evidence
        ),
    )


def _aggregate(rows: list[dict[str, object]], prefix: str) -> dict[str, object]:
    count = len(rows)
    metric_names = (
        "property_precision",
        "property_recall",
        "property_f1",
        "value_recall",
        "exact_frame",
    )
    return {
        "cases": count,
        **{
            name: sum(float(row[f"{prefix}_{name}"]) for row in rows) / count
            for name in metric_names
        },
    }


def _overlap_breakdown(rows: list[dict[str, object]]) -> dict[str, object]:
    strategies = ("evidence", "bm25", "bm25_evidence", "bm25_oracle_at_5")
    breakdown = {}
    for label, exact_overlap in (("exact_train_query", True), ("novel_query", False)):
        subset = [
            row for row in rows if row["exact_query_in_training"] is exact_overlap
        ]
        breakdown[label] = {
            "cases": len(subset),
            **{
                strategy: _aggregate(subset, strategy) if subset else None
                for strategy in strategies
            },
        }
    return breakdown


def _evaluate(
    cases: tuple[AlfredLanguageCase, ...],
    retriever: AlfredBM25FrameRetriever,
    ontology,
    training_queries: set[str],
) -> dict[str, object]:
    rows = []
    for case in cases:
        evidence = extract_alfred_selection_evidence(case.query, ontology)
        evidence_metrics = _metrics(_evidence_frame(case, ontology), case.gold_frame)
        retrievals = retriever.retrieve(case.query, limit=5)
        top1_metrics = _metrics(
            retrievals[0].frame if retrievals else None, case.gold_frame
        )
        if retrievals and case.query in training_queries:
            fused_metrics = top1_metrics
        elif retrievals:
            fused = fuse_alfred_selection(
                retrievals[0].frame,
                evidence,
                policy=SelectionFusionPolicy(
                    add_missing=True,
                    gate_unsupported_states=False,
                    override_conflicting_values=True,
                ),
            )
            fused_metrics = _metrics(fused.frame, case.gold_frame)
        else:
            fused_metrics = evidence_metrics
        candidate_metrics = [
            _metrics(item.frame, case.gold_frame) for item in retrievals
        ]
        if candidate_metrics:
            oracle_index = max(
                range(len(candidate_metrics)),
                key=lambda index: (
                    float(candidate_metrics[index]["value_recall"]),
                    float(candidate_metrics[index]["property_f1"]),
                    -index,
                ),
            )
            oracle_metrics = candidate_metrics[oracle_index]
        else:
            oracle_index = None
            oracle_metrics = _metrics(None, case.gold_frame)
        row = {
            "case_id": case.case_id,
            "task_id": case.task_id,
            "task_type": case.task_type,
            "query": case.query,
            "exact_query_in_training": case.query in training_queries,
            "retrieved_training_case_id": retrievals[0].training_case_id
            if retrievals
            else None,
            "retrieved_training_task_id": retrievals[0].training_task_id
            if retrievals
            else None,
            "retrieval_score": retrievals[0].score if retrievals else None,
            "oracle_rank_at_5": oracle_index + 1 if oracle_index is not None else None,
        }
        for prefix, metrics in (
            ("evidence", evidence_metrics),
            ("bm25", top1_metrics),
            ("bm25_evidence", fused_metrics),
            ("bm25_oracle_at_5", oracle_metrics),
        ):
            row.update({f"{prefix}_{name}": value for name, value in metrics.items()})
        rows.append(row)
    return {
        "evidence_only": _aggregate(rows, "evidence"),
        "bm25_top1": _aggregate(rows, "bm25"),
        "bm25_evidence": _aggregate(rows, "bm25_evidence"),
        "bm25_oracle_at_5": _aggregate(rows, "bm25_oracle_at_5"),
        "retrieval_coverage": sum(
            row["retrieved_training_case_id"] is not None for row in rows
        ) / len(rows),
        "by_query_overlap": _overlap_breakdown(rows),
        "results": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a train-only BM25 typed-frame retrieval baseline on ALFRED."
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/alfred_retrieval_baseline.json"),
    )
    args = parser.parse_args()
    training = load_alfred_language_dataset(args.root, splits=("train",))
    validation = load_alfred_language_dataset(
        args.root, splits=("valid_seen", "valid_unseen")
    )
    ontology = fit_alfred_training_ontology(args.root)
    retriever = AlfredBM25FrameRetriever(training.cases)
    development = select_stratified_cases(
        validation.cases,
        split="valid_unseen",
        trajectories_per_task=10,
        trajectory_offset=0,
        annotation_index=0,
    )
    confirmation = select_stratified_cases(
        validation.cases,
        split="valid_unseen",
        trajectories_per_task=10,
        trajectory_offset=10,
        annotation_index=1,
    )
    train_task_ids = {case.task_id for case in training.cases}
    validation_task_ids = {case.task_id for case in validation.cases}
    train_queries = {case.query for case in training.cases}
    validation_queries = {case.query for case in validation.cases}
    payload = {
        "protocol": {
            "baseline": "BM25 top-1 train typed-frame retrieval",
            "parameters": {"k1": 1.2, "b": 0.75},
            "parameter_source": "standard fixed defaults; no validation tuning",
            "training": training.audit,
            "retriever": retriever.audit(),
            "validation_task_id_overlap_with_train": len(
                train_task_ids & validation_task_ids
            ),
            "exact_query_overlap_with_train": len(train_queries & validation_queries),
            "bm25_oracle_at_5": "evaluation-only retrieval coverage upper bound",
            "candidate_or_matcher_access": False,
            "validation_cases_with_exact_query_overlap": sum(
                case.query in train_queries for case in validation.cases
            ),
            "bm25_evidence_policy": {
                "exact_train_query": "use top-1 training frame unchanged",
                "novel_query": "add and override only span-supported evidence",
                "unsupported_state": "preserve; absence of a recognised cue is unknown",
                "validation_labels_used": False,
            },
            "temporal_claim": "none",
        },
        "development": _evaluate(development, retriever, ontology, train_queries),
        "confirmation": _evaluate(confirmation, retriever, ontology, train_queries),
        "valid_seen": _evaluate(
            tuple(case for case in validation.cases if case.split == "valid_seen"),
            retriever,
            ontology,
            train_queries,
        ),
        "valid_unseen": _evaluate(
            tuple(case for case in validation.cases if case.split == "valid_unseen"),
            retriever,
            ontology,
            train_queries,
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload["protocol"], indent=2, sort_keys=True))
    for name in ("development", "confirmation", "valid_seen", "valid_unseen"):
        result = payload[name]
        print(name)
        for strategy in (
            "evidence_only",
            "bm25_top1",
            "bm25_evidence",
            "bm25_oracle_at_5",
        ):
            metrics = result[strategy]
            print(
                f"  {strategy}: F1={metrics['property_f1']:.3f} "
                f"value={metrics['value_recall']:.3f} "
                f"exact={metrics['exact_frame']:.3f}"
            )
    print(f"report: {args.output}")


if __name__ == "__main__":
    main()
