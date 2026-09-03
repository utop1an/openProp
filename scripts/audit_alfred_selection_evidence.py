from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from openprop.alfred_adapter import load_alfred_language_dataset
from openprop.alfred_ontology import fit_alfred_training_ontology
from openprop.alfred_selection import extract_alfred_selection_evidence


def _summarise(cases, ontology) -> dict[str, object]:
    counts: Counter[str] = Counter()
    properties = {
        name: Counter()
        for name in ("type", "location", "cleanliness", "thermal_state")
    }
    exact_selection = 0
    exact_values = 0
    gold_values = 0
    for case in cases:
        gold = {item.property_name: item.desired_value for item in case.gold_frame.constraints}
        evidence = {
            item.property_name: item.desired_value
            for item in extract_alfred_selection_evidence(case.query, ontology).evidence
        }
        exact_selection += set(evidence) == set(gold)
        for name in properties:
            properties[name]["tp"] += name in gold and name in evidence
            properties[name]["fp"] += name not in gold and name in evidence
            properties[name]["fn"] += name in gold and name not in evidence
        for name, value in gold.items():
            gold_values += 1
            exact_values += evidence.get(name) == value
        counts["predicted"] += len(evidence)
        counts["gold"] += len(gold)
        counts["overlap"] += len(set(evidence) & set(gold))
    precision = counts["overlap"] / counts["predicted"] if counts["predicted"] else 0.0
    recall = counts["overlap"] / counts["gold"]
    result = {
        "cases": len(cases),
        "micro_property_precision": precision,
        "micro_property_recall": recall,
        "micro_property_f1": 2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0,
        "exact_property_set_rate": exact_selection / len(cases),
        "strict_canonical_value_recall": exact_values / gold_values,
        "by_property": {},
    }
    for name, count in properties.items():
        result["by_property"][name] = {
            "tp": count["tp"],
            "fp": count["fp"],
            "fn": count["fn"],
            "precision": count["tp"] / (count["tp"] + count["fp"])
            if count["tp"] + count["fp"]
            else 0.0,
            "recall": count["tp"] / (count["tp"] + count["fn"])
            if count["tp"] + count["fn"]
            else 0.0,
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit query-span selection evidence across all ALFRED validation descriptions."
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/alfred_selection_evidence_audit.json"),
    )
    args = parser.parse_args()
    ontology = fit_alfred_training_ontology(args.root)
    dataset = load_alfred_language_dataset(
        args.root, splits=("valid_seen", "valid_unseen")
    )
    splits = {
        split: _summarise(
            [case for case in dataset.cases if case.split == split], ontology
        )
        for split in ("valid_seen", "valid_unseen")
    }
    payload = {
        "protocol": {
            "scope": "model-independent query evidence coverage audit",
            "ontology_fit": "train PDDL labels only",
            "gold_use": "evaluation only",
            "candidate_or_matcher_access": False,
            "missing_evidence_policy": "no property emitted",
            "ambiguous_evidence_policy": "no property emitted",
            "performance_claim": "selection mechanism only",
        },
        "ontology": ontology.audit(),
        "splits": splits,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(splits, indent=2, sort_keys=True))
    print(f"report: {args.output}")


if __name__ == "__main__":
    main()
