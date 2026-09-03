from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Mapping

from openprop.alfred_adapter import load_alfred_language_dataset
from openprop.alfred_language_evaluation import alfred_language_registry
from openprop.llm import LLMQueryParser
from openprop.models import RelationValue
from openprop.schema_repair import repair_redundant_relation_fields


class _UnusedClient:
    def generate_json(self, **_: object) -> Mapping[str, object]:
        raise AssertionError("parse_response must not call the client")


def _display(value: object) -> object:
    if isinstance(value, RelationValue):
        return {"predicate": value.predicate, "arguments": dict(value.arguments)}
    return value


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze frozen ALFRED language responses by typed value component."
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/alfred_language_analysis.json"),
    )
    args = parser.parse_args()
    dataset = load_alfred_language_dataset(args.root, splits=("valid_unseen",))
    cases_by_id = {case.case_id: case for case in dataset.cases}
    registry = alfred_language_registry()
    tolerant_parser = LLMQueryParser(_UnusedClient(), skip_invalid_constraints=True)
    analyses = []
    shared_case_ids: tuple[str, ...] | None = None
    for input_path in args.inputs:
        artifact = json.loads(input_path.read_text(encoding="utf-8"))
        case_ids = tuple(item["case_id"] for item in artifact["selected_cases"])
        if shared_case_ids is None:
            shared_case_ids = case_ids
        elif shared_case_ids != case_ids:
            parser.error("input artifacts do not use the same ordered case sample")
        totals: Counter[str] = Counter()
        selected: Counter[str] = Counter()
        exact: Counter[str] = Counter()
        relation_components: Counter[str] = Counter()
        parse_failures: list[dict[str, str]] = []
        mismatches: list[dict[str, object]] = []
        repair_cases = 0
        for case_id in case_ids:
            case = cases_by_id[case_id]
            captured = artifact["raw_responses"].get(case.query)
            if captured is None or captured.get("error") or captured.get("response") is None:
                parse_failures.append(
                    {"case_id": case_id, "error": str((captured or {}).get("error", "missing"))}
                )
                for constraint in case.gold_frame.constraints:
                    totals[constraint.property_name] += 1
                continue
            repair = repair_redundant_relation_fields(captured["response"], registry)
            repair_cases += bool(repair.actions)
            try:
                parsed = tolerant_parser.parse_response(case.query, registry, repair.response)
            except Exception as error:
                parse_failures.append(
                    {"case_id": case_id, "error": f"{type(error).__name__}: {error}"}
                )
                for constraint in case.gold_frame.constraints:
                    totals[constraint.property_name] += 1
                continue
            predicted = {
                item.property_name: item.desired_value for item in parsed.frame.constraints
            }
            for constraint in case.gold_frame.constraints:
                name = constraint.property_name
                totals[name] += 1
                if name not in predicted:
                    if len(mismatches) < 24:
                        mismatches.append(
                            {
                                "case_id": case_id,
                                "task_type": case.task_type,
                                "query": case.query,
                                "property": name,
                                "gold": _display(constraint.desired_value),
                                "predicted": None,
                            }
                        )
                    continue
                selected[name] += 1
                value = predicted[name]
                if value == constraint.desired_value:
                    exact[name] += 1
                    continue
                if isinstance(value, RelationValue) and isinstance(
                    constraint.desired_value, RelationValue
                ):
                    relation_components["predicate_total"] += 1
                    relation_components["argument_total"] += 1
                    relation_components["predicate_exact"] += (
                        value.predicate == constraint.desired_value.predicate
                    )
                    relation_components["argument_exact"] += (
                        value.arguments == constraint.desired_value.arguments
                    )
                if len(mismatches) < 24:
                    mismatches.append(
                        {
                            "case_id": case_id,
                            "task_type": case.task_type,
                            "query": case.query,
                            "property": name,
                            "gold": _display(constraint.desired_value),
                            "predicted": _display(value),
                        }
                    )
        by_property = {
            name: {
                "gold_count": totals[name],
                "selection_recall": selected[name] / totals[name],
                "value_recall": exact[name] / totals[name],
                "conditional_value_accuracy": exact[name] / selected[name]
                if selected[name]
                else 0.0,
            }
            for name in sorted(totals)
        }
        analyses.append(
            {
                "artifact": input_path.as_posix(),
                "model": artifact["protocol"]["model"],
                "cases": len(case_ids),
                "repair_cases": repair_cases,
                "parse_failures": parse_failures,
                "by_property": by_property,
                "relation_mismatch_components": dict(relation_components),
                "mismatch_examples": mismatches,
            }
        )
    output = {
        "protocol": {
            "inputs": [path.as_posix() for path in args.inputs],
            "shared_frozen_sample": True,
            "cases_per_model": len(shared_case_ids or ()),
            "failure_denominator": "all selected cases",
            "analysis_strategy": "schema-repaired tolerant replay",
            "claim_scope": "language to typed goal frame only",
        },
        "models": analyses,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    for analysis in analyses:
        print(analysis["model"])
        print(json.dumps(analysis["by_property"], ensure_ascii=False, indent=2))
    print(f"report: {args.output}")


if __name__ == "__main__":
    main()
