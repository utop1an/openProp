from __future__ import annotations

import argparse
import json
from pathlib import Path

from openprop.temporal_grounding import (
    TemporalGroundingReport,
    TemporalStrategy,
    evaluate_temporal_grounding,
    temporal_grounding_benchmark,
    temporal_grounding_registry,
    write_temporal_grounding_jsonl,
)


def _print_report(report: TemporalGroundingReport) -> None:
    print(f"strategy:             {report.strategy.value}")
    print(f"cases:                {report.cases}")
    print(f"top-1 accuracy:       {report.top1_accuracy:.3f}")
    print(f"top-3 recall:         {report.top3_recall:.3f}")
    print(f"mean reciprocal rank: {report.mean_reciprocal_rank:.3f}")
    print("accuracy by scenario:")
    for tag, accuracy in report.accuracy_by_tag.items():
        if tag not in {"en", "zh", "irrelevant-properties"}:
            print(f"  {tag:22s} {accuracy:.3f}")


def _report_payload(report: TemporalGroundingReport) -> dict[str, object]:
    return {
        "strategy": report.strategy.value,
        "cases": report.cases,
        "top1_accuracy": report.top1_accuracy,
        "top3_recall": report.top3_recall,
        "mean_reciprocal_rank": report.mean_reciprocal_rank,
        "accuracy_by_tag": dict(report.accuracy_by_tag),
        "results": [
            {
                "case_id": result.case_id,
                "target_id": result.target_id,
                "predicted_id": result.predicted_id,
                "rank": result.rank,
                "target_score": result.target_score,
                "target_coverage": result.target_coverage,
                "tags": list(result.tags),
            }
            for result in report.results
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate time-aware entity grounding.")
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument(
        "--learned-model",
        type=Path,
        default=None,
        help="Optional saved NeuralPersistenceModel artifact.",
    )
    parser.add_argument(
        "--dataset-output",
        type=Path,
        default=Path("artifacts/temporal_grounding_benchmark.jsonl"),
    )
    parser.add_argument(
        "--report-output",
        type=Path,
        default=Path("artifacts/temporal_grounding_results.json"),
    )
    args = parser.parse_args()
    if args.repetitions <= 0:
        parser.error("--repetitions must be positive")

    cases = temporal_grounding_benchmark(repetitions=args.repetitions)
    registry = temporal_grounding_registry()
    write_temporal_grounding_jsonl(args.dataset_output, cases)

    reports = [
        evaluate_temporal_grounding(cases, registry, TemporalStrategy.NO_DECAY),
        evaluate_temporal_grounding(cases, registry, TemporalStrategy.FIXED_DECAY),
    ]
    if args.learned_model is not None:
        from openprop.neural_persistence import NeuralPersistenceModel

        learned_model = NeuralPersistenceModel.load(args.learned_model)
        reports.append(
            evaluate_temporal_grounding(
                cases,
                registry,
                TemporalStrategy.LEARNED_DECAY,
                learned_model=learned_model,
            )
        )

    for index, report in enumerate(reports):
        if index:
            print()
        _print_report(report)
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.write_text(
        json.dumps(
            {"dataset": str(args.dataset_output), "reports": [_report_payload(report) for report in reports]},
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"\ndataset: {args.dataset_output}")
    print(f"report:  {args.report_output}")


if __name__ == "__main__":
    main()

