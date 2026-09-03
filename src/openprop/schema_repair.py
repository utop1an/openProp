from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Mapping

from .models import ValueType
from .property_registry import PropertyRegistry


@dataclass(frozen=True, slots=True)
class SchemaRepairResult:
    response: Mapping[str, object]
    actions: tuple[str, ...]


def repair_redundant_relation_fields(
    response: Mapping[str, object],
    registry: PropertyRegistry,
) -> SchemaRepairResult:
    """Repair relation field permutations using schema metadata only.

    Structured-output schemas contain scalar, predicate, and argument fields for
    every value kind. Small models sometimes place relation tokens in the wrong
    one. This routine uses only the registered relation roles and allowed
    predicates; it never inspects entities, scores, labels, or current truth.
    """

    repaired = deepcopy(dict(response))
    constraints = repaired.get("constraints")
    if not isinstance(constraints, list):
        return SchemaRepairResult(repaired, ())
    actions: list[str] = []
    for index, item in enumerate(constraints):
        if not isinstance(item, dict):
            continue
        property_name = item.get("property_name")
        if not isinstance(property_name, str):
            continue
        definition = registry.resolve(property_name).definition
        if definition is None or definition.value_type is not ValueType.RELATION:
            continue
        roles_raw = definition.metadata.get("argument_roles", ())
        predicates_raw = definition.metadata.get("allowed_predicates", ())
        roles = tuple(role for role in roles_raw if isinstance(role, str))
        allowed = {value.casefold() for value in predicates_raw if isinstance(value, str)}
        if len(roles) != 1 or not allowed:
            continue
        value = item.get("value")
        if not isinstance(value, dict) or value.get("kind") != "relation":
            continue
        predicate = value.get("predicate")
        scalar = value.get("scalar")
        if not isinstance(predicate, str) or not isinstance(scalar, str):
            continue
        predicate_key = predicate.strip().casefold()
        scalar_key = scalar.strip().casefold()
        role = roles[0]
        if predicate_key not in allowed and scalar_key in allowed:
            value["predicate"] = scalar.strip()
            value["arguments"] = [{"role": role, "value": predicate.strip()}]
            value["scalar"] = None
            actions.append(
                f"constraint[{index}] {property_name}: moved scalar to predicate and predicate to {role}"
            )
        elif predicate_key in allowed and scalar_key not in allowed:
            value["arguments"] = [{"role": role, "value": scalar.strip()}]
            value["scalar"] = None
            actions.append(
                f"constraint[{index}] {property_name}: moved redundant scalar to {role}"
            )
    return SchemaRepairResult(repaired, tuple(actions))
