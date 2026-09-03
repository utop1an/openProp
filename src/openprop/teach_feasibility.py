from __future__ import annotations

import json
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class TeachFeasibilityCriteria:
    """Predeclared pilot sufficiency thresholds, not performance-tuned values."""

    profile: str = "pilot-v1"
    min_sessions: int = 30
    min_floorplans: int = 3
    min_snapshots: int = 300
    min_visible_entities: int = 50
    min_history_records: int = 100
    min_interval_events: int = 10
    min_transition_properties: int = 3
    min_grounding_cases: int = 50
    min_temporal_discriminative_cases: int = 10
    min_candidate_size: int = 2
    min_dialogue_alignments: int = 50
    min_manual_alignment_labels: int = 50
    min_manual_alignment_precision: float = 0.90

    def __post_init__(self) -> None:
        integer_values = (
            self.min_sessions,
            self.min_floorplans,
            self.min_snapshots,
            self.min_visible_entities,
            self.min_history_records,
            self.min_interval_events,
            self.min_transition_properties,
            self.min_grounding_cases,
            self.min_temporal_discriminative_cases,
            self.min_candidate_size,
            self.min_dialogue_alignments,
            self.min_manual_alignment_labels,
        )
        if not self.profile.strip() or any(value < 0 for value in integer_values):
            raise ValueError("TEACh feasibility criteria must be named and nonnegative")
        if not 0.0 <= self.min_manual_alignment_precision <= 1.0:
            raise ValueError("manual alignment precision must be between zero and one")


def read_teach_dialogue_alignment_audit(
    path: str | Path,
    *,
    expected_manifest_sha256: str | None = None,
    expected_policy_id: str | None = None,
    expected_aligned_case_ids: Any = None,
    expected_aligned_cases: int | None = None,
) -> dict[str, Any]:
    """Validate a frozen manual alignment audit and derive precision from labels."""

    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
        policy_id = str(payload["alignment_policy_id"]).strip()
        manifest_hash = str(payload["frozen_manifest_sha256"]).strip().lower()
        aligned_cases = int(payload["aligned_cases"])
        labels = payload["labels"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        raise ValueError(f"invalid TEACh dialogue alignment audit: {error}") from error
    if not policy_id:
        raise ValueError("alignment_policy_id cannot be empty")
    if re.fullmatch(r"[0-9a-f]{64}", manifest_hash) is None:
        raise ValueError("frozen_manifest_sha256 must be a 64-character hex digest")
    if aligned_cases < 0 or not isinstance(labels, list):
        raise ValueError("aligned_cases must be nonnegative and labels must be a list")
    if expected_manifest_sha256 is not None and manifest_hash != expected_manifest_sha256:
        raise ValueError("manual alignment audit does not match the frozen manifest")
    if expected_policy_id is not None and policy_id != expected_policy_id:
        raise ValueError("manual alignment audit does not match the frozen alignment policy")
    expected_ids: set[str] | None = None
    if expected_aligned_case_ids is not None:
        values = tuple(str(value).strip() for value in expected_aligned_case_ids)
        if any(not value for value in values) or len(values) != len(set(values)):
            raise ValueError("expected automatic alignment case IDs must be unique and nonempty")
        expected_ids = set(values)
        if aligned_cases != len(expected_ids):
            raise ValueError("manual aligned_cases does not match automatic alignment output")
    if expected_aligned_cases is not None and aligned_cases != expected_aligned_cases:
        raise ValueError("manual aligned_cases does not match automatic alignment count")
    seen: set[str] = set()
    correct = 0
    for index, label in enumerate(labels):
        if not isinstance(label, Mapping):
            raise ValueError(f"alignment label {index} must be an object")
        case_id = str(label.get("case_id", "")).strip()
        is_correct = label.get("is_correct")
        if not case_id or case_id in seen:
            raise ValueError("alignment labels require unique nonempty case IDs")
        if not isinstance(is_correct, bool):
            raise ValueError("alignment is_correct labels must be boolean")
        if expected_ids is not None and case_id not in expected_ids:
            raise ValueError("manual alignment label is not an automatic alignment case")
        seen.add(case_id)
        correct += is_correct
    if len(labels) > aligned_cases:
        raise ValueError("manual labels cannot exceed automatically aligned cases")
    validated_against_automatic = all(
        value is not None
        for value in (
            expected_manifest_sha256, expected_policy_id,
            expected_aligned_case_ids, expected_aligned_cases,
        )
    )
    return {
        "alignment_policy_id": policy_id,
        "frozen_manifest_sha256": manifest_hash,
        "aligned_cases": aligned_cases,
        "manually_labeled_cases": len(labels),
        "correct_alignments": correct,
        "manual_precision": correct / len(labels) if labels else None,
        "source": str(source),
        "validated_against_automatic": validated_against_automatic,
    }


def assign_teach_floorplan_splits(
    sessions_per_floorplan: Mapping[str, int],
    *,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
    seed: int = 23,
) -> dict[str, Any]:
    """Create a deterministic, floorplan-disjoint three-way allocation audit."""

    if not 0 < train_fraction < 1 or not 0 < validation_fraction < 1:
        raise ValueError("split fractions must be between zero and one")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("split fractions must leave a test partition")
    groups = [(str(name), int(count)) for name, count in sessions_per_floorplan.items()]
    if any(not name or count <= 0 for name, count in groups):
        raise ValueError("floorplan names must be nonempty with positive session counts")
    if len(groups) < 3:
        return {
            "feasible": False,
            "reason": "at least three floorplans are required",
            "seed": seed,
            "splits": {},
        }
    generator = random.Random(seed)
    generator.shuffle(groups)
    groups.sort(key=lambda item: -item[1])
    names = ("train", "validation", "test")
    fractions = {
        "train": train_fraction,
        "validation": validation_fraction,
        "test": 1.0 - train_fraction - validation_fraction,
    }
    assigned: dict[str, list[tuple[str, int]]] = {name: [] for name in names}
    totals = {name: 0 for name in names}
    total_sessions = sum(count for _, count in groups)
    for split, group in zip(names, groups[:3], strict=True):
        assigned[split].append(group)
        totals[split] += group[1]
    for group in groups[3:]:
        split = max(
            names,
            key=lambda name: (
                fractions[name] * total_sessions - totals[name],
                fractions[name],
                -names.index(name),
            ),
        )
        assigned[split].append(group)
        totals[split] += group[1]
    split_rows = {
        name: {
            "floorplans": sorted(item[0] for item in assigned[name]),
            "floorplan_count": len(assigned[name]),
            "sessions": totals[name],
        }
        for name in names
    }
    floorplan_sets = [set(split_rows[name]["floorplans"]) for name in names]
    disjoint = all(
        floorplan_sets[left].isdisjoint(floorplan_sets[right])
        for left in range(len(floorplan_sets))
        for right in range(left + 1, len(floorplan_sets))
    )
    return {
        "feasible": disjoint and all(totals.values()),
        "reason": "ok" if disjoint and all(totals.values()) else "empty or overlapping split",
        "seed": seed,
        "fractions": fractions,
        "splits": split_rows,
    }


def evaluate_teach_feasibility(
    report: Mapping[str, Any],
    *,
    criteria: TeachFeasibilityCriteria | None = None,
    dialogue_alignment: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return explicit layer readiness; missing evidence always fails closed."""

    selected = criteria or TeachFeasibilityCriteria()
    totals = report.get("totals", {})
    censoring = report.get("censoring", {})
    transitions = report.get("property_transitions", {})
    grounding = report.get("gold_grounding", {})
    split = report.get("floorplan_split", {})
    dialogue = dialogue_alignment or {}
    checks: dict[str, dict[str, Any]] = {}

    def minimum(name: str, observed: Any, required: Any, layer: str) -> None:
        passed = observed is not None and observed >= required
        checks[name] = {
            "layer": layer,
            "passed": bool(passed),
            "observed": observed,
            "required_minimum": required,
        }

    minimum("sessions", totals.get("sessions"), selected.min_sessions, "A")
    minimum("floorplans", totals.get("floorplans"), selected.min_floorplans, "A")
    minimum("snapshots", totals.get("snapshots"), selected.min_snapshots, "A")
    minimum(
        "visible_entities",
        totals.get("unique_visible_entities"),
        selected.min_visible_entities,
        "A",
    )
    minimum(
        "history_records",
        totals.get("history_records"),
        selected.min_history_records,
        "A",
    )
    minimum(
        "interval_events",
        censoring.get("interval_censored_event"),
        selected.min_interval_events,
        "A",
    )
    minimum(
        "transition_properties",
        sum(int(value) > 0 for value in transitions.values()),
        selected.min_transition_properties,
        "A",
    )
    checks["floorplan_disjoint_split"] = {
        "layer": "A",
        "passed": split.get("feasible") is True,
        "observed": split.get("feasible"),
        "required": True,
    }
    minimum(
        "grounding_cases",
        grounding.get("cases"),
        selected.min_grounding_cases,
        "B",
    )
    minimum(
        "temporal_discriminative_cases",
        grounding.get("temporal_discriminative_cases"),
        selected.min_temporal_discriminative_cases,
        "B",
    )
    minimum(
        "candidate_size",
        grounding.get("candidate_size_min"),
        selected.min_candidate_size,
        "B",
    )
    checks["dialogue_audit_bound_to_automatic"] = {
        "layer": "C",
        "passed": dialogue.get("validated_against_automatic") is True,
        "observed": dialogue.get("validated_against_automatic"),
        "required": True,
    }
    checks["unique_final_target"] = {
        "layer": "B",
        "passed": grounding.get("target_ties_in_final_truth") == 0
        and grounding.get("cases", 0) > 0,
        "observed_ties": grounding.get("target_ties_in_final_truth"),
        "required_ties": 0,
    }
    minimum(
        "dialogue_alignments",
        dialogue.get("aligned_cases"),
        selected.min_dialogue_alignments,
        "C",
    )
    minimum(
        "manual_alignment_labels",
        dialogue.get("manually_labeled_cases"),
        selected.min_manual_alignment_labels,
        "C",
    )
    minimum(
        "manual_alignment_precision",
        dialogue.get("manual_precision"),
        selected.min_manual_alignment_precision,
        "C",
    )
    readiness = {
        layer: all(
            item["passed"] for item in checks.values() if item["layer"] == layer
        )
        for layer in ("A", "B", "C")
    }
    failed = [name for name, item in checks.items() if not item["passed"]]
    return {
        "criteria": asdict(selected),
        "threshold_source": "predeclared pilot sufficiency; not tuned on outcomes",
        "checks": checks,
        "layer_a_ready": readiness["A"],
        "layer_b_ready": readiness["A"] and readiness["B"],
        "layer_c_ready": readiness["A"] and readiness["B"] and readiness["C"],
        "main_claim_ready": all(readiness.values()),
        "failed_checks": failed,
        "claim_scope": "feasibility only; never model performance",
    }
