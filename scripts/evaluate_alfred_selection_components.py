from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping

from openprop.alfred_adapter import load_alfred_language_dataset
from openprop.alfred_language_evaluation import alfred_language_registry
from openprop.alfred_ontology import (
    fit_alfred_training_ontology,
    normalise_alfred_goal_frame,
)
from openprop.alfred_selection import (
    SelectionFusionPolicy,
    extract_alfred_selection_evidence,
    fuse_alfred_selection,
)
from openprop.llm import LLMQueryParser
from openprop.models import QueryFrame
from openprop.schema_repair import repair_redundant_relation_fields


class _UnusedClient:
    def generate_json(self, **_: object) -> Mapping[str, object]:
        raise AssertionError("parse_response must not call the client")


POLICIES = {
    "none": SelectionFusionPolicy(add_missing=False),
    "add_only": SelectionFusionPolicy(),
    "absence_gate_only": SelectionFusionPolicy(
        add_missing=False, gate_unsupported_states=True
    ),
    "add_and_absence_gate": SelectionFusionPolicy(
        gate_unsupported_states=True
    ),
    "conflict_gate_only": SelectionFusionPolicy(
        add_missing=False, gate_conflicting_states=True
    ),
    "add_and_conflict_gate": SelectionFusionPolicy(
        gate_conflicting_states=True
    ),
}


def _metrics(predicted: QueryFrame, gold: QueryFrame) -> dict[str, float | bool]:
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ablate evidence addition and absence-based state gating on ALFRED."
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/alfred_selection_component_ablation.json"),
    )
    args = parser.parse_args()
    dataset = load_alfred_language_dataset(args.root, splits=("valid_unseen",))
    cases_by_id = {case.case_id: case for case in dataset.cases}
    ontology = fit_alfred_training_ontology(args.root)
    registry = alfred_language_registry()
    response_parser = LLMQueryParser(_UnusedClient(), skip_invalid_constraints=True)
    models = []
    reference_ids: tuple[str, ...] | None = None
    for path in args.inputs:
        artifact = json.loads(path.read_text(encoding="utf-8"))
        case_ids = tuple(item["case_id"] for item in artifact["selected_cases"])
        if reference_ids is None:
            reference_ids = case_ids
        elif reference_ids != case_ids:
            parser.error("inputs do not share the frozen ordered sample")
        policies = []
        for name, policy in POLICIES.items():
            rows = []
            for case_id in case_ids:
                case = cases_by_id[case_id]
                captured = artifact["raw_responses"].get(case.query)
                try:
                    if captured is None:
                        raise ValueError("missing frozen response")
                    if captured.get("error") is not None:
                        raise RuntimeError(str(captured["error"]))
                    raw = captured.get("response")
                    if raw is None:
                        raise ValueError("empty frozen response")
                    repair = repair_redundant_relation_fields(raw, registry)
                    parsed = response_parser.parse_response(case.query, registry, repair.response)
                    evidence = extract_alfred_selection_evidence(case.query, ontology)
                    fused = fuse_alfred_selection(parsed.frame, evidence, policy=policy)
                    normalised = normalise_alfred_goal_frame(fused.frame, ontology)
                    row = {
                        "case_id": case_id,
                        "task_type": case.task_type,
                        **_metrics(normalised.frame, case.gold_frame),
                        "actions": list(fused.actions),
                        "error": None,
                    }
                except Exception as error:
                    row = {
                        "case_id": case_id,
                        "task_type": case.task_type,
                        "property_precision": 0.0,
                        "property_recall": 0.0,
                        "property_f1": 0.0,
                        "value_recall": 0.0,
                        "exact_frame": False,
                        "actions": [],
                        "error": f"{type(error).__name__}: {error}",
                    }
                rows.append(row)
            count = len(rows)
            policies.append(
                {
                    "policy": name,
                    "settings": {
                        "add_missing": policy.add_missing,
                        "gate_unsupported_states": policy.gate_unsupported_states,
                        "gate_conflicting_states": policy.gate_conflicting_states,
                    },
                    "cases": count,
                    "failures": sum(item["error"] is not None for item in rows),
                    "property_precision": sum(item["property_precision"] for item in rows) / count,
                    "property_recall": sum(item["property_recall"] for item in rows) / count,
                    "property_f1": sum(item["property_f1"] for item in rows) / count,
                    "value_recall": sum(item["value_recall"] for item in rows) / count,
                    "exact_frame_accuracy": sum(item["exact_frame"] for item in rows) / count,
                    "action_rate": sum(bool(item["actions"]) for item in rows) / count,
                    "results": rows,
                }
            )
        models.append(
            {
                "artifact": path.as_posix(),
                "model": artifact["protocol"]["model"],
                "policies": policies,
            }
        )
    payload = {
        "protocol": {
            "sample": "shared frozen 40-case valid-unseen sample",
            "base": "schema-repaired tolerant parse plus controlled ontology",
            "evidence_inputs": "query spans and train PDDL vocabulary only",
            "primary_denominator": "all selected cases including failures",
            "design": "addition, absence gate, and positive-evidence conflict gate ablation",
        },
        "models": models,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    for model in models:
        print(model["model"])
        for result in model["policies"]:
            print(
                f"  {result['policy']}: P/R/F1={result['property_precision']:.3f}/"
                f"{result['property_recall']:.3f}/{result['property_f1']:.3f} "
                f"value={result['value_recall']:.3f} exact={result['exact_frame_accuracy']:.3f}"
            )
    print(f"report: {args.output}")


if __name__ == "__main__":
    main()
