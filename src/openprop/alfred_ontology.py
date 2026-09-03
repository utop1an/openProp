from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from .alfred_adapter import INSIDE_RECEPTACLES
from .models import PropertyConstraint, QueryFrame, RelationValue


THERMAL_ALIASES = {
    "chilled": "cold",
    "cooled": "cold",
    "cool": "cold",
    "cold": "cold",
    "heated": "hot",
    "heat": "hot",
    "hot": "hot",
    "warm": "hot",
    "warmed": "hot",
}

CLEANLINESS_ALIASES = {
    "clean": "clean",
    "cleaned": "clean",
    "rinse": "clean",
    "rinsed": "clean",
    "wash": "clean",
    "washed": "clean",
}


def normalise_label(value: str) -> str:
    spaced = re.sub(r"(?<!^)(?=[A-Z])", " ", value)
    return " ".join(re.sub(r"[^a-z0-9]+", " ", spaced.casefold()).split())


@dataclass(frozen=True, slots=True)
class AlfredTrainingOntology:
    object_labels: frozenset[str]
    receptacle_labels: frozenset[str]
    source_split: str = "train"

    def __post_init__(self) -> None:
        if self.source_split != "train":
            raise ValueError("ALFRED ontology must be fitted on the train split only")
        if not self.object_labels or not self.receptacle_labels:
            raise ValueError("ALFRED ontology requires object and receptacle labels")

    def audit(self) -> Mapping[str, object]:
        return {
            "source_split": self.source_split,
            "object_labels": len(self.object_labels),
            "receptacle_labels": len(self.receptacle_labels),
            "alias_policy": "exact or unique token-containment match",
            "ambiguous_policy": "preserve original value",
            "annotation_text_used_for_fit": False,
            "validation_labels_used_for_fit": False,
            "predeclared_schema_semantics": [
                "thermal and cleanliness aliases",
                "receptacle class to on/inside relation",
            ],
        }


@dataclass(frozen=True, slots=True)
class OntologyNormalisationPolicy:
    type_labels: bool = True
    relation_arguments: bool = True
    relation_predicates: bool = True
    state_aliases: bool = True


@dataclass(frozen=True, slots=True)
class OntologyNormalisationResult:
    frame: QueryFrame
    actions: tuple[str, ...]


def fit_alfred_training_ontology(root: str | Path) -> AlfredTrainingOntology:
    """Fit the closed label vocabulary from train PDDL parameters only."""

    source = Path(root)
    train_root = source / "train"
    if not train_root.is_dir():
        raise ValueError(f"missing ALFRED train split directory: {train_root}")
    objects: set[str] = set()
    receptacles: set[str] = set()
    paths = sorted(train_root.rglob("traj_data.json"))
    if not paths:
        raise ValueError("ALFRED train split contains no trajectories")
    for path in paths:
        row = json.loads(path.read_text(encoding="utf-8"))
        try:
            params = row["pddl_params"]
        except (KeyError, TypeError) as error:
            raise ValueError(f"invalid ALFRED training trajectory schema: {path}") from error
        if not isinstance(params, Mapping):
            raise ValueError(f"invalid ALFRED PDDL parameters: {path}")
        object_target = normalise_label(str(params.get("object_target", "")))
        parent_target = normalise_label(str(params.get("parent_target", "")))
        if object_target:
            objects.add(object_target)
        if parent_target:
            receptacles.add(parent_target)
    return AlfredTrainingOntology(frozenset(objects), frozenset(receptacles))


def resolve_canonical_label(value: str, labels: frozenset[str]) -> str | None:
    """Resolve only exact or unique token-containment aliases.

    Returning ``None`` is deliberate: ambiguous evidence stays unresolved and
    cannot be converted into an apparent match.
    """

    key = normalise_label(value)
    if not key:
        return None
    if key in labels:
        return key
    tokens = frozenset(key.split())
    candidates = []
    for label in labels:
        label_tokens = frozenset(label.split())
        if tokens <= label_tokens or label_tokens <= tokens:
            candidates.append(label)
    return candidates[0] if len(candidates) == 1 else None


def alfred_receptacle_predicate(label: str) -> str:
    compact = label.replace(" ", "")
    return "inside" if compact in INSIDE_RECEPTACLES else "on"


def normalise_alfred_goal_frame(
    frame: QueryFrame,
    ontology: AlfredTrainingOntology,
    *,
    policy: OntologyNormalisationPolicy | None = None,
) -> OntologyNormalisationResult:
    selected_policy = policy or OntologyNormalisationPolicy()
    constraints: list[PropertyConstraint] = []
    actions: list[str] = []
    for constraint in frame.constraints:
        value = constraint.desired_value
        if (
            selected_policy.type_labels
            and constraint.property_name == "type"
            and isinstance(value, str)
        ):
            resolved = resolve_canonical_label(value, ontology.object_labels)
            if resolved is not None and resolved != normalise_label(value):
                actions.append(f"type: {value!r} -> {resolved!r}")
                value = resolved
        elif constraint.property_name == "location" and isinstance(value, RelationValue):
            arguments = dict(value.arguments)
            destination = arguments.get("object")
            if destination is not None:
                resolved = resolve_canonical_label(
                    destination, ontology.receptacle_labels
                )
                if resolved is not None:
                    if (
                        selected_policy.relation_arguments
                        and resolved != normalise_label(destination)
                    ):
                        actions.append(
                            f"location.object: {destination!r} -> {resolved!r}"
                        )
                    if selected_policy.relation_arguments:
                        arguments["object"] = resolved
                    predicate = value.predicate
                    canonical_predicate = alfred_receptacle_predicate(resolved)
                    if (
                        selected_policy.relation_predicates
                        and canonical_predicate != value.predicate.casefold()
                    ):
                        actions.append(
                            f"location.predicate: {value.predicate!r} -> {canonical_predicate!r}"
                        )
                        predicate = canonical_predicate
                    value = RelationValue(predicate, arguments)
        elif selected_policy.state_aliases and constraint.property_name == "thermal_state" and isinstance(value, str):
            resolved = THERMAL_ALIASES.get(normalise_label(value))
            if resolved is not None and resolved != normalise_label(value):
                actions.append(f"thermal_state: {value!r} -> {resolved!r}")
                value = resolved
        elif selected_policy.state_aliases and constraint.property_name == "cleanliness" and isinstance(value, str):
            resolved = CLEANLINESS_ALIASES.get(normalise_label(value))
            if resolved is not None and resolved != normalise_label(value):
                actions.append(f"cleanliness: {value!r} -> {resolved!r}")
                value = resolved
        constraints.append(
            PropertyConstraint(
                constraint.property_name,
                value,
                constraint.relevance,
                constraint.tolerance,
            )
        )
    return OntologyNormalisationResult(
        QueryFrame(frame.text, tuple(constraints)), tuple(actions)
    )
