from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

from .models import PropertyConstraint, QueryFrame
from .teach_layer_c import TeachLayerCPrepared


TEACH_LAYER_C_ANNOTATION_PROTOCOL_ID = "explicit-referential-frame-v1"
TEACH_LAYER_C_ANNOTATION_STATUSES = frozenset(
    {"type_only", "explicit_attributes", "uncertain"}
)
TEACH_LAYER_C_MIN_PAIRWISE_AGREEMENT = 0.80
_BLIND_FIELDS = (
    "target object ID",
    "candidate entities and properties",
    "observation timestamps",
    "action result and final truth",
    "model outputs",
    "other annotators' labels",
)


@dataclass(frozen=True, slots=True)
class TeachLayerCAnnotationResolution:
    frames: Mapping[str, QueryFrame]
    audit: Mapping[str, Any]


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _nonempty(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Layer C annotation {field} must be a nonempty string")
    return value.strip()


def _case_views(
    prepared: TeachLayerCPrepared,
    property_names: Sequence[str],
) -> list[dict[str, Any]]:
    allowed = tuple(
        _nonempty(property_name, "property_name") for property_name in property_names
    )
    if not allowed or len(allowed) != len(set(allowed)):
        raise ValueError("annotation property_names must be non-empty and unique")
    reserved = {"type", "scene"}.intersection(
        property_name.casefold() for property_name in allowed
    )
    if reserved:
        raise ValueError(
            "annotation property_names cannot include fixed type or non-discriminative scene"
        )
    views: list[dict[str, Any]] = []
    seen: set[str] = set()
    for case in sorted(prepared.cases, key=lambda item: item.case_id):
        if case.case_id in seen:
            raise ValueError(f"duplicate Layer C annotation case_id: {case.case_id}")
        seen.add(case.case_id)
        constraints = case.gold_frame.constraints
        if (
            len(constraints) != 1
            or constraints[0].property_name.casefold() != "type"
        ):
            raise ValueError("annotation templates require the frozen type-only oracle")
        views.append(
            {
                "case_id": case.case_id,
                "query": case.query,
                "fixed_type_constraint": {
                    "property_name": "type",
                    "desired_value": constraints[0].desired_value,
                },
                "allowed_additional_properties": list(allowed),
            }
        )
    if not views:
        raise ValueError("at least one Layer C case is required for annotation")
    return views


def build_teach_layer_c_annotation_template(
    prepared: TeachLayerCPrepared,
    *,
    annotator_id: str,
    property_names: Sequence[str],
) -> dict[str, Any]:
    """Create a target-blind semantic-frame annotation template."""

    annotator = _nonempty(annotator_id, "annotator_id")
    manifest_hash = str(prepared.audit.get("frozen_manifest_sha256", ""))
    if re.fullmatch(r"[0-9a-f]{64}", manifest_hash) is None:
        raise ValueError("prepared Layer C audit lacks a valid manifest hash")
    views = _case_views(prepared, property_names)
    return {
        "annotation_protocol_id": TEACH_LAYER_C_ANNOTATION_PROTOCOL_ID,
        "frozen_manifest_sha256": manifest_hash,
        "case_view_sha256": _sha256(views),
        "annotator_id": annotator,
        "independence_requirement": "complete without target, candidates, models, or other labels",
        "blind_fields_excluded": list(_BLIND_FIELDS),
        "status_choices": sorted(TEACH_LAYER_C_ANNOTATION_STATUSES),
        "instructions": (
            "Label only attributes explicitly used to refer to the object. "
            "Do not convert the commanded action into a state: for example, "
            "'clean the mug' does not assert isDirty=false, while 'the dirty mug' "
            "can assert isDirty=true. Every additional constraint requires an "
            "exact character span from the Commander text."
        ),
        "cases": views,
        "labels": [
            {
                "case_id": view["case_id"],
                "status": None,
                "constraints": [],
                "notes": "",
            }
            for view in views
        ],
    }


def _validate_annotation_file(
    prepared: TeachLayerCPrepared,
    payload: Mapping[str, Any],
    *,
    property_names: Sequence[str],
) -> tuple[str, dict[str, tuple[str, tuple[tuple[str, bool], ...]]]]:
    views = _case_views(prepared, property_names)
    if payload.get("annotation_protocol_id") != TEACH_LAYER_C_ANNOTATION_PROTOCOL_ID:
        raise ValueError("Layer C annotation protocol does not match")
    if payload.get("frozen_manifest_sha256") != prepared.audit.get(
        "frozen_manifest_sha256"
    ):
        raise ValueError("Layer C annotation does not match the frozen manifest")
    if payload.get("case_view_sha256") != _sha256(views) or payload.get("cases") != views:
        raise ValueError("Layer C annotation case views do not match the frozen blind views")
    if payload.get("blind_fields_excluded") != list(_BLIND_FIELDS):
        raise ValueError("Layer C annotation blinding declaration was changed")
    annotator_id = _nonempty(payload.get("annotator_id"), "annotator_id")
    labels = payload.get("labels")
    if not isinstance(labels, list) or len(labels) != len(views):
        raise ValueError("Layer C annotations must label the complete case population")
    expected_ids = [view["case_id"] for view in views]
    actual_ids = [
        str(label.get("case_id", "")) if isinstance(label, Mapping) else ""
        for label in labels
    ]
    if actual_ids != expected_ids:
        raise ValueError("Layer C annotation labels do not match ordered case IDs")
    allowed = set(property_names)
    query_by_id = {view["case_id"]: view["query"] for view in views}
    canonical: dict[str, tuple[str, tuple[tuple[str, bool], ...]]] = {}
    for label in labels:
        assert isinstance(label, Mapping)
        case_id = str(label["case_id"])
        status = label.get("status")
        if status not in TEACH_LAYER_C_ANNOTATION_STATUSES:
            raise ValueError(f"Layer C annotation {case_id} has invalid or missing status")
        constraints = label.get("constraints")
        if not isinstance(constraints, list):
            raise ValueError(f"Layer C annotation {case_id} constraints must be a list")
        if status == "explicit_attributes" and not constraints:
            raise ValueError("explicit_attributes requires at least one constraint")
        if status != "explicit_attributes" and constraints:
            raise ValueError(f"{status} annotations cannot contain constraints")
        seen_properties: set[str] = set()
        semantic: list[tuple[str, bool]] = []
        query = query_by_id[case_id]
        for constraint in constraints:
            if not isinstance(constraint, Mapping):
                raise ValueError("Layer C annotation constraints must be objects")
            property_name = _nonempty(constraint.get("property_name"), "property_name")
            if property_name not in allowed or property_name in seen_properties:
                raise ValueError("annotation properties must be allowed and unique per case")
            desired = constraint.get("desired_value")
            if not isinstance(desired, bool):
                raise ValueError("Layer C additional state values must be boolean")
            start = constraint.get("span_start")
            end = constraint.get("span_end")
            evidence = constraint.get("evidence_span")
            if (
                type(start) is not int
                or type(end) is not int
                or not 0 <= start < end <= len(query)
                or not isinstance(evidence, str)
                or query[start:end] != evidence
            ):
                raise ValueError("annotation evidence span must exactly match query offsets")
            seen_properties.add(property_name)
            semantic.append((property_name, desired))
        canonical[case_id] = (status, tuple(sorted(semantic)))
    return annotator_id, canonical


def resolve_teach_layer_c_annotations(
    prepared: TeachLayerCPrepared,
    annotations: Sequence[Mapping[str, Any]],
    *,
    property_names: Sequence[str],
    min_pairwise_agreement: float = TEACH_LAYER_C_MIN_PAIRWISE_AGREEMENT,
) -> TeachLayerCAnnotationResolution:
    """Resolve three independent semantic annotations by deterministic majority."""

    if len(annotations) != 3:
        raise ValueError("Layer C rich frames require exactly three annotation files")
    if not 0.0 <= min_pairwise_agreement <= 1.0:
        raise ValueError("min_pairwise_agreement must be between zero and one")
    validated = [
        _validate_annotation_file(prepared, payload, property_names=property_names)
        for payload in annotations
    ]
    annotator_ids = [item[0] for item in validated]
    if len(set(annotator_ids)) != 3:
        raise ValueError("Layer C annotations require three distinct annotator IDs")
    by_case = [item[1] for item in validated]
    cases_by_id = {case.case_id: case for case in prepared.cases}
    pairwise_agreements = 0
    pairwise_comparisons = 0
    unanimous = 0
    majority = 0
    unresolved: list[str] = []
    status_counts: Counter[str] = Counter()
    frames: dict[str, QueryFrame] = {}
    for case_id in sorted(cases_by_id):
        labels = [rows[case_id] for rows in by_case]
        for left in range(3):
            for right in range(left + 1, 3):
                pairwise_comparisons += 1
                pairwise_agreements += labels[left] == labels[right]
        counts = Counter(labels)
        winner, votes = counts.most_common(1)[0]
        if votes == 3:
            unanimous += 1
        elif votes == 2:
            majority += 1
        if votes < 2 or winner[0] == "uncertain":
            unresolved.append(case_id)
            continue
        status, semantic = winner
        status_counts[status] += 1
        case = cases_by_id[case_id]
        base = case.gold_frame.constraints[0]
        values = [
            PropertyConstraint(property_name, desired_value)
            for property_name, desired_value in semantic
        ]
        weight = 1.0 / (1 + len(values))
        frames[case_id] = QueryFrame(
            case.query,
            (
                PropertyConstraint("type", base.desired_value, weight),
                *(replace(item, relevance=weight) for item in values),
            ),
        )
    agreement = pairwise_agreements / pairwise_comparisons
    if unresolved:
        raise ValueError(
            "Layer C annotations have unresolved cases: " + ", ".join(unresolved)
        )
    if agreement < min_pairwise_agreement:
        raise ValueError(
            f"Layer C pairwise semantic agreement {agreement:.3f} is below "
            f"the frozen minimum {min_pairwise_agreement:.3f}"
        )
    return TeachLayerCAnnotationResolution(
        frames,
        {
            "annotation_protocol_id": TEACH_LAYER_C_ANNOTATION_PROTOCOL_ID,
            "frozen_manifest_sha256": prepared.audit["frozen_manifest_sha256"],
            "annotator_ids": annotator_ids,
            "annotation_sha256": [_sha256(payload) for payload in annotations],
            "cases": len(cases_by_id),
            "annotations_per_case": 3,
            "pairwise_semantic_agreements": pairwise_agreements,
            "pairwise_semantic_comparisons": pairwise_comparisons,
            "pairwise_semantic_agreement": agreement,
            "minimum_pairwise_semantic_agreement": min_pairwise_agreement,
            "unanimous_cases": unanimous,
            "majority_resolved_cases": majority,
            "unresolved_cases": 0,
            "status_counts": dict(sorted(status_counts.items())),
            "constraint_relevance": "equal and normalized after annotation",
            "annotation_blinding": list(_BLIND_FIELDS),
            "claim_scope": "independent text-frame annotation; not model performance",
        },
    )


def apply_teach_layer_c_annotation_resolution(
    prepared: TeachLayerCPrepared,
    resolution: TeachLayerCAnnotationResolution,
) -> TeachLayerCPrepared:
    expected = {case.case_id for case in prepared.cases}
    if set(resolution.frames) != expected:
        raise ValueError("resolved Layer C frames do not cover the exact case population")
    cases = tuple(
        replace(
            case,
            gold_frame=resolution.frames[case.case_id],
            tags=(*case.tags, "independent-rich-text-oracle"),
        )
        for case in prepared.cases
    )
    audit = dict(prepared.audit)
    audit["oracle_frame"] = "independently annotated explicit text attributes"
    audit["rich_annotation"] = dict(resolution.audit)
    return TeachLayerCPrepared(cases, audit)

