from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import asdict

from .visual_evaluation import VisualEvaluationDataset


def combine_visual_evaluation_datasets(
    datasets: Sequence[VisualEvaluationDataset],
    *,
    required_systems: Sequence[str] = (),
) -> tuple[VisualEvaluationDataset, dict[str, object]]:
    """Combine metric records while enforcing paired core populations."""

    if not datasets:
        raise ValueError("visual matrix requires at least one dataset")
    combined = VisualEvaluationDataset(
        tuple(row for dataset in datasets for row in dataset.properties),
        tuple(row for dataset in datasets for row in dataset.associations),
        tuple(row for dataset in datasets for row in dataset.queries),
    )
    systems = sorted(
        {
            row.system
            for row in (*combined.properties, *combined.associations, *combined.queries)
        }
    )
    required = tuple(required_systems)
    if required:
        if len(required) != len(set(required)) or any(not item.strip() for item in required):
            raise ValueError("required systems must be non-empty and unique")
        if set(systems) != set(required):
            raise ValueError("visual matrix systems differ from required systems")

    query_populations = {
        system: {_query_signature(row) for row in combined.queries if row.system == system}
        for system in systems
    }
    property_populations = {
        system: {
            _property_signature(row)
            for row in combined.properties
            if row.system == system and row.expected
        }
        for system in systems
    }
    _require_paired(query_populations, "query")
    _require_paired(property_populations, "expected property")
    report = {
        "schema_version": 1,
        "systems": systems,
        "paired_query_population": True,
        "paired_expected_property_population": True,
        "model_specific_false_positives_retained": True,
        "denominators": {
            system: {
                "property": sum(row.system == system for row in combined.properties),
                "association": sum(row.system == system for row in combined.associations),
                "query": sum(row.system == system for row in combined.queries),
            }
            for system in systems
        },
        "population_hashes": {
            "query": _population_hash(next(iter(query_populations.values()))),
            "expected_property": _population_hash(
                next(iter(property_populations.values()))
            ),
        },
    }
    return combined, report


def _require_paired(populations: dict[str, set[str]], name: str) -> None:
    if not populations or any(not population for population in populations.values()):
        raise ValueError(f"visual matrix {name} population cannot be empty")
    reference = next(iter(populations.values()))
    if any(population != reference for population in populations.values()):
        raise ValueError(f"visual matrix {name} population is not paired")


def _query_signature(row: object) -> str:
    payload = asdict(row)
    for field in (
        "system",
        "ranked_entity_ids",
        "decision_entity_id",
        "accepted_entity_id",
        "probabilities",
        "latency_seconds",
        "vlm_calls",
        "malformed",
    ):
        payload.pop(field, None)
    return _canonical(payload)


def _property_signature(row: object) -> str:
    payload = asdict(row)
    for field in (
        "system",
        "detected",
        "predicted_value",
        "confidence",
        "duplicate_count",
        "malformed",
    ):
        payload.pop(field, None)
    return _canonical(payload)


def _population_hash(population: set[str]) -> str:
    return hashlib.sha256("\n".join(sorted(population)).encode("utf-8")).hexdigest()


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
