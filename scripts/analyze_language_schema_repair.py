from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

from openprop.language_paraphrases import paraphrased_temporal_grounding_benchmark


def _sign_test(wins: int, losses: int) -> float:
    total = wins + losses
    if total == 0:
        return 1.0
    tail = sum(math.comb(total, index) for index in range(min(wins, losses) + 1))
    return min(1.0, 2.0 * tail / (2**total))


def _cluster_bootstrap(
    clusters: list[tuple[int, int]],
    *,
    samples: int,
    seed: int,
) -> tuple[float, float]:
    generator = random.Random(seed)
    estimates: list[float] = []
    for _ in range(samples):
        draws = [generator.choice(clusters) for _ in clusters]
        estimates.append(sum(item[0] for item in draws) / sum(item[1] for item in draws))
    estimates.sort()
    lower = estimates[int(0.025 * (samples - 1))]
    upper = estimates[int(0.975 * (samples - 1))]
    return lower, upper


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Paired, query-clustered analysis of schema repair artifacts."
    )
    parser.add_argument("artifacts", nargs="+", type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/language_schema_repair_analysis.json"),
    )
    args = parser.parse_args()
    if args.bootstrap_samples <= 0:
        parser.error("--bootstrap-samples must be positive")

    query_by_case = {
        case.case_id: case.query for case in paraphrased_temporal_grounding_benchmark()
    }
    summaries: list[dict[str, object]] = []
    aggregate_clusters: list[tuple[int, int]] = []
    aggregate_wins = 0
    aggregate_losses = 0
    for path in args.artifacts:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload["protocol"].get("query_set") != "paraphrase":
            parser.error(f"{path} is not a paraphrase-set artifact")
        reports = {report["strategy"]: report for report in payload["reports"]}
        tolerant = reports["llm-tolerant"]
        repaired = reports["llm-schema-repaired"]
        before = {item["case_id"]: item for item in tolerant["results"]}
        after = {item["case_id"]: item for item in repaired["results"]}
        if before.keys() != after.keys() or before.keys() != query_by_case.keys():
            parser.error(f"{path} case IDs do not match the benchmark contract")
        wins = sum(before[key]["rank"] != 1 and after[key]["rank"] == 1 for key in before)
        losses = sum(before[key]["rank"] == 1 and after[key]["rank"] != 1 for key in before)
        rank_improvements = sum(after[key]["rank"] < before[key]["rank"] for key in before)
        rank_regressions = sum(after[key]["rank"] > before[key]["rank"] for key in before)
        grouped: dict[str, list[int]] = {}
        for case_id in before:
            grouped.setdefault(query_by_case[case_id], []).append(
                int(after[case_id]["rank"] == 1) - int(before[case_id]["rank"] == 1)
            )
        clusters = [(sum(values), len(values)) for values in grouped.values()]
        lower, upper = _cluster_bootstrap(
            clusters, samples=args.bootstrap_samples, seed=args.seed
        )
        aggregate_clusters.extend(clusters)
        aggregate_wins += wins
        aggregate_losses += losses
        summaries.append(
            {
                "artifact": str(path),
                "model": payload["protocol"]["model"],
                "queries": len(grouped),
                "cases": len(before),
                "tolerant_top1": tolerant["top1_accuracy"],
                "repaired_top1": repaired["top1_accuracy"],
                "paired_top1_delta": repaired["top1_accuracy"]
                - tolerant["top1_accuracy"],
                "top1_improvements": wins,
                "top1_regressions": losses,
                "rank_improvements": rank_improvements,
                "rank_regressions": rank_regressions,
                "repair_cases": sum(bool(item["repair_actions"]) for item in after.values()),
                "paired_sign_test_p": _sign_test(wins, losses),
                "query_clustered_bootstrap_95_ci": [lower, upper],
            }
        )

    aggregate_delta = sum(item[0] for item in aggregate_clusters) / sum(
        item[1] for item in aggregate_clusters
    )
    lower, upper = _cluster_bootstrap(
        aggregate_clusters, samples=args.bootstrap_samples, seed=args.seed
    )
    result = {
        "protocol": {
            "paired_strategy": "llm-schema-repaired minus llm-tolerant",
            "bootstrap_unit": "model-query cluster",
            "bootstrap_samples": args.bootstrap_samples,
            "seed": args.seed,
        },
        "models": summaries,
        "aggregate": {
            "models": len(summaries),
            "cases": sum(item[1] for item in aggregate_clusters),
            "query_clusters": len(aggregate_clusters),
            "paired_top1_delta": aggregate_delta,
            "top1_improvements": aggregate_wins,
            "top1_regressions": aggregate_losses,
            "paired_sign_test_p": _sign_test(aggregate_wins, aggregate_losses),
            "model_query_clustered_bootstrap_95_ci": [lower, upper],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    for item in summaries:
        print(
            f"{item['model']}: {item['tolerant_top1']:.3f} -> "
            f"{item['repaired_top1']:.3f}, delta={item['paired_top1_delta']:.3f}, "
            f"wins/losses={item['top1_improvements']}/{item['top1_regressions']}"
        )
    print(
        f"aggregate delta={aggregate_delta:.3f}, 95% cluster bootstrap "
        f"CI=[{lower:.3f}, {upper:.3f}], sign p={result['aggregate']['paired_sign_test_p']:.3f}"
    )
    print(f"report: {args.output}")


if __name__ == "__main__":
    main()
