from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from openprop.alfred_adapter import load_alfred_language_dataset
from openprop.models import RelationValue


STATE_CUES = {
    "pick_clean_then_place_in_recep": ("clean", "wash", "rinse", "washed", "rinsed"),
    "pick_cool_then_place_in_recep": ("cool", "cold", "chill", "fridge", "refrigerator"),
    "pick_heat_then_place_in_recep": ("heat", "hot", "warm", "microwave", "heated"),
}


def _normalise(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", text.casefold()).split())


def _contains(text: str, phrase: str) -> bool:
    return f" {_normalise(phrase)} " in f" {_normalise(text)} "


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit lexical agreement between ALFRED human descriptions and PDDL labels."
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/alfred_label_alignment_audit.json"),
    )
    args = parser.parse_args()
    dataset = load_alfred_language_dataset(args.root)
    known_types = sorted(
        {
            str(next(item.desired_value for item in case.gold_frame.constraints if item.property_name == "type"))
            for case in dataset.cases
        },
        key=lambda item: (-len(item), item),
    )
    counts: Counter[str] = Counter()
    by_split: dict[str, Counter[str]] = {}
    conflict_examples: list[dict[str, object]] = []
    unresolved_examples: list[dict[str, object]] = []
    for case in dataset.cases:
        split_counts = by_split.setdefault(case.split, Counter())
        gold = {item.property_name: item.desired_value for item in case.gold_frame.constraints}
        target = str(gold["type"])
        location = gold["location"]
        assert isinstance(location, RelationValue)
        destination = location.arguments["object"]
        object_exact = _contains(case.query, target)
        destination_exact = _contains(case.query, destination)
        other_types = [
            item
            for item in known_types
            if item != target and _contains(case.query, item)
        ]
        state_cues = STATE_CUES.get(case.task_type, ())
        state_supported = not state_cues or any(
            _contains(case.query, cue) for cue in state_cues
        )
        signals = {
            "object_exact": object_exact,
            "destination_exact": destination_exact,
            "state_cue": state_supported,
            "object_conflict": bool(not object_exact and other_types),
        }
        counts["cases"] += 1
        split_counts["cases"] += 1
        for name, value in signals.items():
            counts[name] += value
            split_counts[name] += value
        row = {
            "case_id": case.case_id,
            "split": case.split,
            "task_type": case.task_type,
            "query": case.query,
            "gold_object": target,
            "gold_destination": destination,
            "other_known_types_mentioned": other_types,
        }
        if signals["object_conflict"] and len(conflict_examples) < 30:
            conflict_examples.append(row)
        elif not object_exact and len(unresolved_examples) < 30:
            unresolved_examples.append(row)

    def rates(counter: Counter[str]) -> dict[str, object]:
        total = counter["cases"]
        return {
            "cases": total,
            "exact_object_label_rate": counter["object_exact"] / total,
            "exact_destination_label_rate": counter["destination_exact"] / total,
            "state_cue_rate": counter["state_cue"] / total,
            "explicit_object_conflict_rate": counter["object_conflict"] / total,
            "explicit_object_conflicts": counter["object_conflict"],
        }

    payload = {
        "protocol": {
            "scope": "lexical support audit, not semantic equivalence scoring",
            "object_support": "normalized whole-phrase match to PDDL object_target",
            "destination_support": "normalized whole-phrase match to PDDL parent_target",
            "explicit_conflict": "gold object phrase absent and another known full object label present",
            "limitations": [
                "aliases and hypernyms are counted as unsupported rather than incorrect",
                "cue presence does not prove full goal-frame correctness",
                "no model outputs are used",
            ],
        },
        "overall": rates(counts),
        "by_split": {name: rates(value) for name, value in sorted(by_split.items())},
        "explicit_conflict_examples": conflict_examples,
        "unsupported_or_alias_examples": unresolved_examples,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload["overall"], indent=2, sort_keys=True))
    print(f"report: {args.output}")


if __name__ == "__main__":
    main()
