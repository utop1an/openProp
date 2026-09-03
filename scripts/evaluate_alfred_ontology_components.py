from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Mapping

from openprop.alfred_adapter import load_alfred_language_dataset
from openprop.alfred_language_evaluation import alfred_language_registry
from openprop.alfred_ontology import (
    OntologyNormalisationPolicy,
    fit_alfred_training_ontology,
    normalise_alfred_goal_frame,
)
from openprop.llm import LLMQueryParser
from openprop.schema_repair import repair_redundant_relation_fields


class _UnusedClient:
    def generate_json(self, **_: object) -> Mapping[str, object]:
        raise AssertionError("parse_response must not call the client")


POLICIES = {
    "none": OntologyNormalisationPolicy(False, False, False, False),
    "type_only": OntologyNormalisationPolicy(True, False, False, False),
    "relation_argument_only": OntologyNormalisationPolicy(False, True, False, False),
    "relation_predicate_only": OntologyNormalisationPolicy(False, False, True, False),
    "state_only": OntologyNormalisationPolicy(False, False, False, True),
    "full_without_type": OntologyNormalisationPolicy(False, True, True, True),
    "full_without_relation_argument": OntologyNormalisationPolicy(True, False, True, True),
    "full_without_relation_predicate": OntologyNormalisationPolicy(True, True, False, True),
    "full_without_state": OntologyNormalisationPolicy(True, True, True, False),
    "full": OntologyNormalisationPolicy(),
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Component ablation for ALFRED controlled ontology normalization."
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/alfred_ontology_component_ablation.json"),
    )
    args = parser.parse_args()
    dataset = load_alfred_language_dataset(args.root, splits=("valid_unseen",))
    cases_by_id = {case.case_id: case for case in dataset.cases}
    registry = alfred_language_registry()
    ontology = fit_alfred_training_ontology(args.root)
    response_parser = LLMQueryParser(_UnusedClient(), skip_invalid_constraints=True)
    model_results = []
    reference_case_ids: tuple[str, ...] | None = None
    for input_path in args.inputs:
        artifact = json.loads(input_path.read_text(encoding="utf-8"))
        case_ids = tuple(item["case_id"] for item in artifact["selected_cases"])
        if reference_case_ids is None:
            reference_case_ids = case_ids
        elif reference_case_ids != case_ids:
            parser.error("input artifacts do not share the frozen ordered sample")
        strategy_results = []
        for policy_name, policy in POLICIES.items():
            exact_values = 0
            gold_values = 0
            exact_frames = 0
            failures = 0
            property_totals: Counter[str] = Counter()
            property_exact: Counter[str] = Counter()
            action_cases = 0
            case_rows = []
            for case_id in case_ids:
                case = cases_by_id[case_id]
                gold = {
                    item.property_name: item.desired_value
                    for item in case.gold_frame.constraints
                }
                for name in gold:
                    property_totals[name] += 1
                    gold_values += 1
                captured = artifact["raw_responses"].get(case.query)
                try:
                    if captured is None:
                        raise ValueError("missing frozen response")
                    if captured.get("error") is not None:
                        raise RuntimeError(str(captured["error"]))
                    raw = captured.get("response")
                    if raw is None:
                        raise ValueError("empty frozen response")
                    repaired = repair_redundant_relation_fields(raw, registry)
                    parsed = response_parser.parse_response(
                        case.query, registry, repaired.response
                    )
                    normalised = normalise_alfred_goal_frame(
                        parsed.frame, ontology, policy=policy
                    )
                    action_cases += bool(normalised.actions)
                    predicted = {
                        item.property_name: item.desired_value
                        for item in normalised.frame.constraints
                    }
                    matches = {
                        name: predicted.get(name) == value for name, value in gold.items()
                    }
                    for name, matched in matches.items():
                        property_exact[name] += matched
                        exact_values += matched
                    frame_exact = set(predicted) == set(gold) and all(matches.values())
                    exact_frames += frame_exact
                    case_rows.append(
                        {
                            "case_id": case_id,
                            "task_type": case.task_type,
                            "value_recall": sum(matches.values()) / len(gold),
                            "exact_frame": frame_exact,
                            "actions": list(normalised.actions),
                            "error": None,
                        }
                    )
                except Exception as error:
                    failures += 1
                    case_rows.append(
                        {
                            "case_id": case_id,
                            "task_type": case.task_type,
                            "value_recall": 0.0,
                            "exact_frame": False,
                            "actions": [],
                            "error": f"{type(error).__name__}: {error}",
                        }
                    )
            strategy_results.append(
                {
                    "policy": policy_name,
                    "settings": {
                        "type_labels": policy.type_labels,
                        "relation_arguments": policy.relation_arguments,
                        "relation_predicates": policy.relation_predicates,
                        "state_aliases": policy.state_aliases,
                    },
                    "cases": len(case_ids),
                    "failures": failures,
                    "strict_canonical_value_recall": exact_values / gold_values,
                    "exact_frame_accuracy": exact_frames / len(case_ids),
                    "action_rate": action_cases / len(case_ids),
                    "value_recall_by_property": {
                        name: property_exact[name] / property_totals[name]
                        for name in sorted(property_totals)
                    },
                    "results": case_rows,
                }
            )
        baseline = next(item for item in strategy_results if item["policy"] == "none")
        for result in strategy_results:
            result["paired_value_recall_delta_vs_none"] = (
                result["strict_canonical_value_recall"]
                - baseline["strict_canonical_value_recall"]
            )
        model_results.append(
            {
                "artifact": input_path.as_posix(),
                "model": artifact["protocol"]["model"],
                "policies": strategy_results,
            }
        )
    payload = {
        "protocol": {
            "sample": "shared frozen 40-case valid-unseen sample",
            "base_parser": "schema-repaired tolerant replay",
            "ontology": ontology.audit(),
            "primary_denominator": "all gold constraints including failed cases",
            "component_design": "atomic and leave-one-component-out policies",
        },
        "models": model_results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    for model in model_results:
        print(model["model"])
        for result in model["policies"]:
            print(
                f"  {result['policy']}: value={result['strict_canonical_value_recall']:.3f} "
                f"delta={result['paired_value_recall_delta_vs_none']:.3f} "
                f"exact={result['exact_frame_accuracy']:.3f}"
            )
    print(f"report: {args.output}")


if __name__ == "__main__":
    main()
