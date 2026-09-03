from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path

from openprop.language_paraphrases import paraphrased_temporal_grounding_benchmark
from openprop.language_temporal_grounding import (
    LanguageTemporalReport,
    RawLanguageResponse,
    LanguageTemporalStrategy,
    collect_language_responses,
    evaluate_language_temporal_grounding,
)
from openprop.ollama import OllamaClient
from openprop.temporal_grounding import (
    temporal_grounding_benchmark,
    temporal_grounding_registry,
)


def _report_payload(report: LanguageTemporalReport) -> dict[str, object]:
    payload = asdict(report)
    payload["strategy"] = report.strategy.value
    return payload


def _print_report(report: LanguageTemporalReport) -> None:
    print(f"strategy:                 {report.strategy.value}")
    print(f"completed / failures:     {report.completed} / {report.failures}")
    print(f"parse success:            {report.parse_success_rate:.3f}")
    print(f"top-1 (all cases):        {report.top1_accuracy:.3f}")
    print(f"top-1 (parsed only):      {report.conditional_top1_accuracy:.3f}")
    print(f"property F1:              {report.property_f1:.3f}")
    print(f"schema repair rate:       {report.repair_rate:.3f}")
    print(f"constraint value recall: {report.value_recall:.3f}")
    print(f"relevance MAE:            {report.relevance_mae:.3f}")
    print(f"validation-error rate:    {report.validation_error_rate:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate language parsing and temporal grounding end to end."
    )
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--model", default=os.environ.get("OLLAMA_MODEL", "gemma3:4b"))
    parser.add_argument(
        "--query-set",
        choices=("development", "paraphrase"),
        default="development",
    )
    parser.add_argument(
        "--ollama-host",
        default=os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/language_temporal_grounding_results.json"),
    )
    parser.add_argument(
        "--responses-input",
        type=Path,
        default=None,
        help="Replay raw_responses from an existing result without new model calls.",
    )
    args = parser.parse_args()
    if args.repetitions <= 0:
        parser.error("--repetitions must be positive")

    if args.query_set == "paraphrase":
        if args.repetitions != 10:
            parser.error("the paraphrase query set requires --repetitions 10")
        cases = paraphrased_temporal_grounding_benchmark()
    else:
        cases = temporal_grounding_benchmark(repetitions=args.repetitions)
    registry = temporal_grounding_registry()
    if args.responses_input is None:
        client = OllamaClient(model=args.model, base_url=args.ollama_host)
        responses = collect_language_responses(cases, registry, client)
        response_source = "live Ollama"
    else:
        stored = json.loads(args.responses_input.read_text(encoding="utf-8"))
        responses = {
            query: RawLanguageResponse(**payload)
            for query, payload in stored["raw_responses"].items()
        }
        missing = {case.query for case in cases} - responses.keys()
        if missing:
            parser.error(f"responses input is missing {len(missing)} benchmark queries")
        response_source = str(args.responses_input)
    reports = [
        evaluate_language_temporal_grounding(
            cases, registry, LanguageTemporalStrategy.GOLD
        ),
        evaluate_language_temporal_grounding(
            cases,
            registry,
            LanguageTemporalStrategy.LLM_STRICT,
            responses=responses,
        ),
        evaluate_language_temporal_grounding(
            cases,
            registry,
            LanguageTemporalStrategy.LLM_TOLERANT,
            responses=responses,
        ),
        evaluate_language_temporal_grounding(
            cases,
            registry,
            LanguageTemporalStrategy.LLM_SCHEMA_REPAIRED,
            responses=responses,
        ),
    ]
    for index, report in enumerate(reports):
        if index:
            print()
        _print_report(report)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "protocol": {
                    "model": args.model,
                    "cases": len(cases),
                    "unique_queries": len(responses),
                    "query_set": args.query_set,
                    "raw_response_replay": True,
                    "primary_metric_denominator": "all cases including request and parse failures",
                    "matcher_truth_access": "none",
                    "response_source": response_source,
                    "repair_input_boundary": "query schema and raw response only",
                },
                "raw_responses": {
                    query: asdict(response) for query, response in responses.items()
                },
                "reports": [_report_payload(report) for report in reports],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"\nreport: {args.output}")


if __name__ == "__main__":
    main()
