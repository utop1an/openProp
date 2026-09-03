from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


METRICS = ("property_f1", "value_recall", "exact_frame")


def _delta(row: dict[str, object], left: str, right: str, metric: str) -> float:
    return float(row[f"{left}_{metric}"]) - float(row[f"{right}_{metric}"])


def _paired_cluster_bootstrap(
    rows: list[dict[str, object]],
    *,
    left: str,
    right: str,
    metric: str,
    samples: int,
    seed: int,
) -> dict[str, object]:
    strata: dict[str, dict[str, list[dict[str, object]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        strata[str(row["task_type"])][str(row["task_id"])].append(row)
    observed = sum(_delta(row, left, right, metric) for row in rows) / len(rows)
    generator = random.Random(seed)
    draws = []
    for _ in range(samples):
        sampled_rows = []
        for task_type in sorted(strata):
            clusters = strata[task_type]
            task_ids = sorted(clusters)
            for _ in task_ids:
                sampled_rows.extend(clusters[generator.choice(task_ids)])
        draws.append(
            sum(_delta(row, left, right, metric) for row in sampled_rows)
            / len(sampled_rows)
        )
    draws.sort()
    lower = draws[int(0.025 * samples)]
    upper = draws[min(samples - 1, int(0.975 * samples))]
    case_deltas = [_delta(row, left, right, metric) for row in rows]
    return {
        "delta": observed,
        "ci_95": [lower, upper],
        "wins": sum(value > 0 for value in case_deltas),
        "ties": sum(value == 0 for value in case_deltas),
        "losses": sum(value < 0 for value in case_deltas),
        "bootstrap_samples": samples,
        "cluster": "task_id",
        "stratum": "task_type",
        "seed": seed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Paired clustered bootstrap for ALFRED retrieval comparisons."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("artifacts/alfred_retrieval_baseline.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/alfred_retrieval_comparison.json"),
    )
    parser.add_argument("--samples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260826)
    args = parser.parse_args()
    if args.samples <= 0:
        raise ValueError("samples must be positive")
    source = json.loads(args.input.read_text(encoding="utf-8"))
    report: dict[str, object] = {
        "protocol": {
            "comparison": "paired case metrics with task-id cluster bootstrap",
            "stratification": "ALFRED task type",
            "input": str(args.input),
            "labels_used_for_method_selection": False,
        }
    }
    comparisons = (
        ("bm25_evidence", "bm25"),
        ("bm25_evidence", "evidence"),
    )
    for split_index, split in enumerate(
        ("development", "confirmation", "valid_seen", "valid_unseen")
    ):
        rows = source[split]["results"]
        split_report = {}
        for comparison_index, (left, right) in enumerate(comparisons):
            split_report[f"{left}_minus_{right}"] = {
                metric: _paired_cluster_bootstrap(
                    rows,
                    left=left,
                    right=right,
                    metric=metric,
                    samples=args.samples,
                    seed=args.seed + split_index * 100 + comparison_index * 10 + index,
                )
                for index, metric in enumerate(METRICS)
            }
        report[split] = split_report
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    for split in ("development", "confirmation", "valid_seen", "valid_unseen"):
        comparison = report[split]["bm25_evidence_minus_bm25"]
        print(split)
        for metric in METRICS:
            result = comparison[metric]
            print(
                f"  {metric}: delta={result['delta']:.3f} "
                f"CI=[{result['ci_95'][0]:.3f},{result['ci_95'][1]:.3f}] "
                f"W/T/L={result['wins']}/{result['ties']}/{result['losses']}"
            )
    print(f"report: {args.output}")


if __name__ == "__main__":
    main()
