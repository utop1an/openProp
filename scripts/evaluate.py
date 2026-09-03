from __future__ import annotations

import argparse
import os

from openprop.benchmark import core_benchmark, core_registry
from openprop.evaluation import EvaluationReport, EvaluationStrategy, evaluate
from openprop.interference import interference_benchmark
from openprop.ollama import OllamaClient


def _print_report(report: EvaluationReport) -> None:
    print(f"strategy:               {report.strategy.value}")
    print(f"cases:                  {report.cases}")
    print(f"completed / failures:   {report.completed} / {report.failures}")
    print(f"top-1 accuracy:         {report.top1_accuracy:.3f}")
    print(f"top-3 recall:           {report.top3_recall:.3f}")
    print(f"mean reciprocal rank:   {report.mean_reciprocal_rank:.3f}")
    print(f"property precision:     {report.property_precision:.3f}")
    print(f"property recall:        {report.property_recall:.3f}")
    print(f"property F1:            {report.property_f1:.3f}")
    print(f"target coverage:        {report.mean_target_coverage:.3f}")
    print(f"mean latency (seconds): {report.mean_latency_seconds:.3f}")
    misses = [
        result
        for result in report.results
        if result.error is None and result.rank != 1
    ]
    if misses:
        print("non-top-1 cases:")
        for result in misses:
            print(
                f"  {result.case_id}: target={result.target_id}, "
                f"predicted={result.predicted_id}, rank={result.rank}"
            )
    failures = [result for result in report.results if result.error]
    if failures:
        print("failures:")
        for result in failures:
            print(f"  {result.case_id}: {result.error}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate OpenProp benchmarks.")
    parser.add_argument(
        "--strategy",
        choices=[strategy.value for strategy in EvaluationStrategy],
        default=EvaluationStrategy.GOLD_WEIGHTED.value,
    )
    parser.add_argument("--dataset", choices=("core", "interference"), default="core")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--model", default=os.environ.get("OLLAMA_MODEL", "gemma3:4b"))
    parser.add_argument("--ollama-host", default=os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434"))
    args = parser.parse_args()

    cases = (
        core_benchmark()
        if args.dataset == "core"
        else interference_benchmark()
    )
    if args.limit is not None:
        if args.limit <= 0:
            parser.error("--limit must be positive")
        cases = cases[: args.limit]
    strategy = EvaluationStrategy(args.strategy)
    client = None
    if strategy is EvaluationStrategy.LLM_WEIGHTED:
        client = OllamaClient(model=args.model, base_url=args.ollama_host)
    _print_report(evaluate(cases, core_registry(), strategy, llm_client=client))


if __name__ == "__main__":
    main()
