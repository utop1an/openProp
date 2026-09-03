from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter
from pathlib import Path


def _sign_test(wins: int, losses: int) -> float:
    total = wins + losses
    if total == 0:
        return 1.0
    tail = sum(math.comb(total, index) for index in range(min(wins, losses) + 1))
    return min(1.0, 2.0 * tail / (2**total))


def _stratified_bootstrap(
    deltas: dict[str, list[float]], *, samples: int, seed: int
) -> tuple[float, float]:
    generator = random.Random(seed)
    estimates = []
    for _ in range(samples):
        draws = [
            generator.choice(values)
            for task_type in sorted(deltas)
            for values in (deltas[task_type],)
            for _ in values
        ]
        estimates.append(sum(draws) / len(draws))
    estimates.sort()
    return (
        estimates[int(0.025 * (samples - 1))],
        estimates[int(0.975 * (samples - 1))],
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Paired analysis of train-only ALFRED ontology normalization."
    )
    parser.add_argument("artifacts", nargs="+", type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/alfred_ontology_ablation_analysis.json"),
    )
    args = parser.parse_args()
    if args.bootstrap_samples <= 0:
        parser.error("--bootstrap-samples must be positive")
    summaries = []
    aggregate_by_case: dict[str, list[float]] = {}
    task_by_case: dict[str, str] = {}
    reference_case_ids: tuple[str, ...] | None = None
    for path in args.artifacts:
        payload = json.loads(path.read_text(encoding="utf-8"))
        ontology_audit = payload["protocol"].get("ontology", {})
        if ontology_audit.get("source_split") != "train":
            parser.error(f"{path} does not declare a train-only ontology")
        if ontology_audit.get("annotation_text_used_for_fit") is not False:
            parser.error(f"{path} ontology fit boundary is not auditable")
        reports = {item["strategy"]: item for item in payload["reports"]}
        before = {
            item["case_id"]: item for item in reports["llm-schema-repaired"]["results"]
        }
        after = {
            item["case_id"]: item
            for item in reports["llm-ontology-normalized"]["results"]
        }
        case_ids = tuple(before)
        if before.keys() != after.keys():
            parser.error(f"{path} paired strategies have different cases")
        if reference_case_ids is None:
            reference_case_ids = case_ids
        elif reference_case_ids != case_ids:
            parser.error("artifacts do not share the same ordered frozen sample")
        deltas_by_task: dict[str, list[float]] = {}
        wins = losses = exact_wins = exact_losses = 0
        action_counts: Counter[str] = Counter()
        for case_id in case_ids:
            old = before[case_id]
            new = after[case_id]
            if old["property_f1"] != new["property_f1"]:
                parser.error(f"ontology changed property selection for {case_id}")
            delta = new["value_recall"] - old["value_recall"]
            task_type = new["task_type"]
            deltas_by_task.setdefault(task_type, []).append(delta)
            aggregate_by_case.setdefault(case_id, []).append(delta)
            task_by_case[case_id] = task_type
            wins += delta > 0
            losses += delta < 0
            old_exact = old["property_f1"] == 1.0 and old["value_recall"] == 1.0
            new_exact = new["property_f1"] == 1.0 and new["value_recall"] == 1.0
            exact_wins += new_exact and not old_exact
            exact_losses += old_exact and not new_exact
            for action in new["normalisation_actions"]:
                action_counts[action.split(":", 1)[0]] += 1
        lower, upper = _stratified_bootstrap(
            deltas_by_task, samples=args.bootstrap_samples, seed=args.seed
        )
        before_report = reports["llm-schema-repaired"]
        after_report = reports["llm-ontology-normalized"]
        summaries.append(
            {
                "artifact": path.as_posix(),
                "model": payload["protocol"]["model"],
                "cases": len(case_ids),
                "schema_repaired_value_recall": before_report["value_recall"],
                "ontology_value_recall": after_report["value_recall"],
                "paired_value_recall_delta": after_report["value_recall"]
                - before_report["value_recall"],
                "value_improvements": wins,
                "value_regressions": losses,
                "paired_sign_test_p": _sign_test(wins, losses),
                "task_stratified_bootstrap_95_ci": [lower, upper],
                "schema_repaired_exact_frame": before_report["exact_frame_accuracy"],
                "ontology_exact_frame": after_report["exact_frame_accuracy"],
                "exact_frame_improvements": exact_wins,
                "exact_frame_regressions": exact_losses,
                "normalisation_rate": after_report["normalisation_rate"],
                "normalisation_actions": dict(sorted(action_counts.items())),
            }
        )
    aggregate_deltas = {
        task_type: [
            sum(aggregate_by_case[case_id]) / len(aggregate_by_case[case_id])
            for case_id in sorted(aggregate_by_case)
            if task_by_case[case_id] == task_type
        ]
        for task_type in sorted(set(task_by_case.values()))
    }
    lower, upper = _stratified_bootstrap(
        aggregate_deltas, samples=args.bootstrap_samples, seed=args.seed
    )
    aggregate_delta = sum(
        sum(values) for values in aggregate_by_case.values()
    ) / sum(len(values) for values in aggregate_by_case.values())
    result = {
        "protocol": {
            "paired_strategy": "llm-ontology-normalized minus llm-schema-repaired",
            "sample": "shared frozen 40-case valid-unseen sample",
            "ontology_fit": "train PDDL labels only; no annotation text",
            "bootstrap_unit": "case cluster, stratified by task type",
            "bootstrap_samples": args.bootstrap_samples,
            "seed": args.seed,
            "primary_denominator": "all selected cases including failures",
        },
        "models": summaries,
        "aggregate": {
            "models": len(summaries),
            "case_clusters": len(aggregate_by_case),
            "model_case_pairs": sum(len(item) for item in aggregate_by_case.values()),
            "paired_value_recall_delta": aggregate_delta,
            "case_clustered_task_stratified_bootstrap_95_ci": [lower, upper],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    for item in summaries:
        print(
            f"{item['model']}: value {item['schema_repaired_value_recall']:.3f} -> "
            f"{item['ontology_value_recall']:.3f}, delta={item['paired_value_recall_delta']:.3f}, "
            f"wins/losses={item['value_improvements']}/{item['value_regressions']}, "
            f"CI=[{item['task_stratified_bootstrap_95_ci'][0]:.3f}, "
            f"{item['task_stratified_bootstrap_95_ci'][1]:.3f}]"
        )
    print(
        f"aggregate delta={aggregate_delta:.3f}, case-clustered 95% CI="
        f"[{lower:.3f}, {upper:.3f}]"
    )
    print(f"report: {args.output}")


if __name__ == "__main__":
    main()
