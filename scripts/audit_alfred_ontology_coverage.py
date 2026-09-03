from __future__ import annotations

import argparse
import json
from pathlib import Path

from openprop.alfred_adapter import load_alfred_language_dataset
from openprop.alfred_ontology import fit_alfred_training_ontology
from openprop.models import RelationValue


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit evaluation-label coverage of the frozen train-only ALFRED ontology."
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/alfred_ontology_coverage_audit.json"),
    )
    args = parser.parse_args()
    ontology = fit_alfred_training_ontology(args.root)
    dataset = load_alfred_language_dataset(
        args.root, splits=("valid_seen", "valid_unseen")
    )
    split_results = {}
    for split in ("valid_seen", "valid_unseen"):
        cases = [case for case in dataset.cases if case.split == split]
        object_labels = []
        receptacle_labels = []
        for case in cases:
            gold = {
                item.property_name: item.desired_value
                for item in case.gold_frame.constraints
            }
            object_labels.append(str(gold["type"]))
            location = gold["location"]
            assert isinstance(location, RelationValue)
            receptacle_labels.append(location.arguments["object"])
        unseen_objects = sorted(set(object_labels) - ontology.object_labels)
        unseen_receptacles = sorted(
            set(receptacle_labels) - ontology.receptacle_labels
        )
        split_results[split] = {
            "cases": len(cases),
            "object_case_coverage": sum(
                item in ontology.object_labels for item in object_labels
            )
            / len(cases),
            "receptacle_case_coverage": sum(
                item in ontology.receptacle_labels for item in receptacle_labels
            )
            / len(cases),
            "unseen_object_labels": unseen_objects,
            "unseen_receptacle_labels": unseen_receptacles,
        }
    payload = {
        "protocol": {
            "ontology_fit": "train PDDL labels only",
            "evaluation_use": "coverage reporting only; validation labels never update ontology",
            "performance_metric": False,
        },
        "ontology": ontology.audit(),
        "splits": split_results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(split_results, indent=2, sort_keys=True))
    print(f"report: {args.output}")


if __name__ == "__main__":
    main()
