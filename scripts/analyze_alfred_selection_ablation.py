from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter
from pathlib import Path

from openprop.alfred_adapter import load_alfred_language_dataset


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


def _metric_summary(
    before: dict[str, dict[str, object]],
    after: dict[str, dict[str, object]],
    metric: str,
    *,
    samples: int,
    seed: int,
) -> dict[str, object]:
    deltas_by_task: dict[str, list[float]] = {}
    wins = losses = 0
    for case_id, old in before.items():
        new = after[case_id]
        delta = float(new[metric]) - float(old[metric])
        task_type = str(new["task_type"])
        deltas_by_task.setdefault(task_type, []).append(delta)
        wins += delta > 0
        losses += delta < 0
    lower, upper = _stratified_bootstrap(
        deltas_by_task, samples=samples, seed=seed
    )
    old_mean = sum(float(item[metric]) for item in before.values()) / len(before)
    new_mean = sum(float(item[metric]) for item in after.values()) / len(after)
    return {
        "before": old_mean,
        "after": new_mean,
        "paired_delta": new_mean - old_mean,
        "improvements": wins,
        "regressions": losses,
        "paired_sign_test_p": _sign_test(wins, losses),
        "task_stratified_bootstrap_95_ci": [lower, upper],
    }


def _selection_confusion(
    rows: dict[str, dict[str, object]], cases_by_id: dict[str, object]
) -> dict[str, object]:
    names = ("type", "location", "cleanliness", "thermal_state")
    counts = {name: Counter() for name in names}
    for case_id, row in rows.items():
        case = cases_by_id[case_id]
        gold = {item.property_name for item in case.gold_frame.constraints}
        predicted = set(row["selected_properties"])
        for name in names:
            counts[name]["tp"] += name in gold and name in predicted
            counts[name]["fp"] += name not in gold and name in predicted
            counts[name]["fn"] += name in gold and name not in predicted
    result = {}
    for name, count in counts.items():
        precision = count["tp"] / (count["tp"] + count["fp"]) if count["tp"] + count["fp"] else 0.0
        recall = count["tp"] / (count["tp"] + count["fn"]) if count["tp"] + count["fn"] else 0.0
        result[name] = {
            "tp": count["tp"],
            "fp": count["fp"],
            "fn": count["fn"],
            "precision": precision,
            "recall": recall,
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Paired analysis of evidence-fused ALFRED property selection."
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("artifacts", nargs="+", type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/alfred_selection_ablation_analysis.json"),
    )
    args = parser.parse_args()
    dataset = load_alfred_language_dataset(args.root, splits=("valid_unseen",))
    cases_by_id = {case.case_id: case for case in dataset.cases}
    summaries = []
    aggregate_by_metric: dict[str, dict[str, list[float]]] = {
        name: {} for name in ("property_f1", "value_recall", "exact_frame")
    }
    task_by_case: dict[str, str] = {}
    reference_ids: tuple[str, ...] | None = None
    for path in args.artifacts:
        payload = json.loads(path.read_text(encoding="utf-8"))
        protocol = payload["protocol"]
        if protocol.get("selection_evidence_requirement") is None:
            parser.error(f"{path} does not declare the evidence boundary")
        reports = {item["strategy"]: item for item in payload["reports"]}
        before = {
            item["case_id"]: dict(item)
            for item in reports["llm-ontology-normalized"]["results"]
        }
        after = {
            item["case_id"]: dict(item)
            for item in reports["llm-evidence-fused"]["results"]
        }
        if before.keys() != after.keys():
            parser.error(f"{path} paired strategies have different cases")
        case_ids = tuple(before)
        if reference_ids is None:
            reference_ids = case_ids
        elif reference_ids != case_ids:
            parser.error("artifacts do not share the ordered frozen sample")
        for rows in (before, after):
            for row in rows.values():
                row["exact_frame"] = float(
                    row["property_f1"] == 1.0 and row["value_recall"] == 1.0
                )
        metrics = {
            name: _metric_summary(
                before,
                after,
                name,
                samples=args.bootstrap_samples,
                seed=args.seed,
            )
            for name in ("property_f1", "value_recall", "exact_frame")
        }
        for case_id in case_ids:
            task_by_case[case_id] = str(after[case_id]["task_type"])
            for name in aggregate_by_metric:
                aggregate_by_metric[name].setdefault(case_id, []).append(
                    float(after[case_id][name]) - float(before[case_id][name])
                )
        actions: Counter[str] = Counter()
        for row in after.values():
            for action in row["selection_actions"]:
                if action.startswith("added "):
                    key = "added " + action.split("added ", 1)[1].split(" ", 1)[0]
                elif action.startswith("removed unsupported "):
                    key = "removed unsupported " + action.split(
                        "removed unsupported ", 1
                    )[1].split(":", 1)[0]
                elif action.startswith("removed conflicting "):
                    key = "removed conflicting " + action.split(
                        "removed conflicting ", 1
                    )[1].split(":", 1)[0]
                else:
                    key = "other"
                actions[key] += 1
        summaries.append(
            {
                "artifact": path.as_posix(),
                "model": protocol["model"],
                "cases": len(case_ids),
                "metrics": metrics,
                "selection_action_rate": reports["llm-evidence-fused"]["selection_action_rate"],
                "selection_actions": dict(sorted(actions.items())),
                "selection_before": _selection_confusion(before, cases_by_id),
                "selection_after": _selection_confusion(after, cases_by_id),
            }
        )
    aggregate = {}
    for name, by_case in aggregate_by_metric.items():
        stratified: dict[str, list[float]] = {}
        for case_id, values in by_case.items():
            stratified.setdefault(task_by_case[case_id], []).append(
                sum(values) / len(values)
            )
        lower, upper = _stratified_bootstrap(
            stratified, samples=args.bootstrap_samples, seed=args.seed
        )
        delta = sum(sum(values) for values in by_case.values()) / sum(
            len(values) for values in by_case.values()
        )
        aggregate[name] = {
            "paired_delta": delta,
            "case_clustered_task_stratified_bootstrap_95_ci": [lower, upper],
        }
    output = {
        "protocol": {
            "paired_strategy": "llm-evidence-fused minus llm-ontology-normalized",
            "sample": "shared frozen 40-case valid-unseen sample",
            "selection_inputs": "query spans, train PDDL vocabulary, schema, and parsed frame only",
            "bootstrap_unit": "case cluster, stratified by task type",
            "bootstrap_samples": args.bootstrap_samples,
            "seed": args.seed,
            "primary_denominator": "all selected cases including failures",
        },
        "models": summaries,
        "aggregate": aggregate,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    for item in summaries:
        print(item["model"])
        for name, metric in item["metrics"].items():
            print(
                f"  {name}: {metric['before']:.3f} -> {metric['after']:.3f}, "
                f"delta={metric['paired_delta']:.3f}, "
                f"wins/losses={metric['improvements']}/{metric['regressions']}, "
                f"CI=[{metric['task_stratified_bootstrap_95_ci'][0]:.3f}, "
                f"{metric['task_stratified_bootstrap_95_ci'][1]:.3f}]"
            )
    print(f"report: {args.output}")


if __name__ == "__main__":
    main()
