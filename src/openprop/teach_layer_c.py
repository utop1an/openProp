from __future__ import annotations

import copy
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .models import Entity, Observation, PropertyConstraint, QueryFrame
from .teach_adapter import read_teach_replay
from .teach_audit import TeachAuditSession
from .teach_dialogue_alignment import TEACH_DIALOGUE_ALIGNMENT_POLICY_ID
from .teach_grounding import TEACH_BOOLEAN_STATE_PROPERTIES
from .temporal_grounding import TemporalGroundingCase


@dataclass(frozen=True, slots=True)
class TeachLayerCPrepared:
    """Strictly pre-action language-grounding cases plus coverage accounting."""

    cases: tuple[TemporalGroundingCase, ...]
    audit: Mapping[str, Any]

def validate_teach_layer_c_gate(
    alignment_report: Mapping[str, Any],
    feasibility_report: Mapping[str, Any],
    *,
    expected_manifest_sha256: str,
) -> None:
    """Require a main-ready audit bound to the exact automatic case population."""

    gate = feasibility_report.get("feasibility_gate")
    automatic = feasibility_report.get("dialogue_alignment_auto")
    if not isinstance(gate, Mapping) or gate.get("main_claim_ready") is not True:
        raise ValueError("TEACh Layer C main feasibility gate has not passed")
    if not isinstance(automatic, Mapping):
        raise ValueError("TEACh Layer C feasibility audit lacks automatic alignments")
    for source, label in ((alignment_report, "alignment report"), (automatic, "feasibility audit")):
        if source.get("alignment_policy_id") != TEACH_DIALOGUE_ALIGNMENT_POLICY_ID:
            raise ValueError(f"Layer C {label} uses the wrong alignment policy")
        if source.get("frozen_manifest_sha256") != expected_manifest_sha256:
            raise ValueError(f"Layer C {label} does not match the frozen manifest")
    expected_ids = alignment_report.get("case_ids")
    automatic_ids = automatic.get("case_ids")
    if not isinstance(expected_ids, list) or not isinstance(automatic_ids, list):
        raise ValueError("Layer C bound audits require explicit case_ids")
    if expected_ids != automatic_ids or len(expected_ids) != len(set(expected_ids)):
        raise ValueError("Layer C automatic case IDs do not match the frozen population")
    if alignment_report.get("cases") != automatic.get("cases"):
        raise ValueError(
            "Layer C automatic case contents do not match the frozen population"
        )
    try:
        aligned_cases = int(alignment_report["aligned_cases"])
        automatic_cases = int(automatic["aligned_cases"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Layer C bound audits require aligned_cases") from error
    if aligned_cases != automatic_cases or aligned_cases != len(expected_ids):
        raise ValueError("Layer C automatic case counts do not match the frozen population")
    checks = gate.get("checks")
    if not isinstance(checks, Mapping) or not all(
        isinstance(checks.get(name), Mapping) and checks[name].get("passed") is True
        for name in (
            "dialogue_audit_bound_to_automatic",
            "dialogue_alignments",
            "manual_alignment_labels",
            "manual_alignment_precision",
        )
    ):
        raise ValueError("TEACh Layer C manual alignment checks have not all passed")


def _required_text(row: Mapping[str, Any], field: str) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"TEACh Layer C {field} must be a nonempty string")
    return value.strip()


def _last_visible_before(
    observations,
    *,
    action_time: float,
    scene: str,
    property_names: Sequence[str],
    source: str,
) -> dict[str, dict[str, Observation]]:
    """Return only evidence visible strictly before the target interaction."""

    last: dict[str, dict[str, Observation]] = defaultdict(dict)
    for snapshot in sorted(observations, key=lambda item: item.timestamp):
        if snapshot.is_final:
            raise ValueError("TEACh final truth cannot enter Layer C observations")
        if snapshot.timestamp >= action_time:
            continue
        for raw in snapshot.objects:
            if not raw.get("visible", False) or not raw.get("objectId"):
                continue
            object_id = str(raw["objectId"])
            object_type = str(raw.get("objectType", "unknown"))
            last[object_id]["type"] = Observation(
                object_type,
                timestamp=snapshot.timestamp,
                source=source,
            )
            last[object_id]["scene"] = Observation(
                scene,
                timestamp=snapshot.timestamp,
                source=source,
            )
            for property_name in property_names:
                if property_name in raw:
                    last[object_id][property_name] = Observation(
                        copy.deepcopy(raw[property_name]),
                        timestamp=snapshot.timestamp,
                        source=source,
                    )
    return last


def prepare_teach_layer_c_cases(
    sessions: Iterable[TeachAuditSession],
    alignment_report: Mapping[str, Any],
    *,
    expected_manifest_sha256: str,
    property_names: Sequence[str] = TEACH_BOOLEAN_STATE_PROPERTIES,
    source: str = "teach-egocentric-pre-action-replay",
) -> TeachLayerCPrepared:
    """Build type-oracle Layer C cases without action-result or final-state leakage.

    The automatic alignment certifies which recorded interaction follows a
    dialogue segment; it does not annotate all referring attributes. The only
    defensible automatic oracle frame is therefore the official target type.
    Every aligned case remains in the denominator, including targets that were
    never visible before the action.
    """

    if (
        not isinstance(expected_manifest_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", expected_manifest_sha256) is None
    ):
        raise ValueError("expected_manifest_sha256 must be a 64-character digest")
    if alignment_report.get("alignment_policy_id") != TEACH_DIALOGUE_ALIGNMENT_POLICY_ID:
        raise ValueError("Layer C alignment policy does not match the frozen policy")
    if alignment_report.get("frozen_manifest_sha256") != expected_manifest_sha256:
        raise ValueError("Layer C alignment report does not match the frozen manifest")
    if not property_names or len(property_names) != len(set(property_names)):
        raise ValueError("property_names must be non-empty and unique")
    raw_cases = alignment_report.get("cases")
    if not isinstance(raw_cases, list):
        raise ValueError("Layer C alignment report requires a cases list")
    declared_ids = alignment_report.get("case_ids")
    try:
        declared_count = int(alignment_report["aligned_cases"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Layer C alignment report requires aligned_cases") from error
    if not isinstance(declared_ids, list):
        raise ValueError("Layer C alignment report requires case_ids")
    actual_ids = [
        str(row.get("case_id", "")) for row in raw_cases if isinstance(row, Mapping)
    ]
    if declared_count != len(raw_cases) or declared_ids != actual_ids:
        raise ValueError("Layer C alignment population does not match cases, case_ids, and count")

    by_episode: dict[str, TeachAuditSession] = {}
    for session in sessions:
        if session.episode_id in by_episode:
            raise ValueError(f"duplicate TEACh Layer C episode: {session.episode_id}")
        by_episode[session.episode_id] = session
    if not by_episode:
        raise ValueError("at least one TEACh session is required")

    case_ids: set[str] = set()
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in raw_cases:
        if not isinstance(row, Mapping):
            raise ValueError("TEACh Layer C cases must be objects")
        case_id = _required_text(row, "case_id")
        episode_id = _required_text(row, "episode_id")
        if case_id in case_ids:
            raise ValueError(f"duplicate TEACh Layer C case_id: {case_id}")
        if episode_id not in by_episode:
            raise ValueError(f"Layer C case references unknown episode: {episode_id}")
        case_ids.add(case_id)
        grouped[episode_id].append(row)

    cases: list[TemporalGroundingCase] = []
    candidate_histogram: Counter[int] = Counter()
    same_type_histogram: Counter[int] = Counter()
    tag_counts: Counter[str] = Counter()
    for episode_id in sorted(grouped):
        session = by_episode[episode_id]
        initial = json.loads(session.initial_state.read_text(encoding="utf-8"))
        replay = read_teach_replay(
            initial,
            session.state_directory,
            final_timestamp=session.final_timestamp,
        )
        observation_times = {snapshot.timestamp for snapshot in replay.observations}
        for row in sorted(grouped[episode_id], key=lambda item: str(item["case_id"])):
            try:
                action_time = float(row["action_time"])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError("TEACh Layer C action_time must be numeric") from error
            if not math.isfinite(action_time) or action_time < 0:
                raise ValueError("TEACh Layer C action_time must be finite and nonnegative")
            if action_time not in observation_times:
                raise ValueError(
                    f"Layer C action_time {action_time} has no replay snapshot in {episode_id}"
                )
            case_id = _required_text(row, "case_id")
            target_raw_id = _required_text(row, "target_object_id")
            target_type = _required_text(row, "target_object_type")
            query = _required_text(row, "commander_text")
            last = _last_visible_before(
                replay.observations,
                action_time=action_time,
                scene=session.floorplan,
                property_names=property_names,
                source=source,
            )
            entities = tuple(
                Entity(f"{episode_id}:{object_id}", dict(last[object_id]))
                for object_id in sorted(last)
            )
            target_id = f"{episode_id}:{target_raw_id}"
            same_type = sum(
                str(entity.properties["type"].value).casefold() == target_type.casefold()
                for entity in entities
            )
            tags = ["teach", "layer-c", "type-oracle", "strictly-pre-action"]
            if target_raw_id in last:
                tags.append("target-observed-before-action")
            else:
                tags.extend(("target-unobserved-before-action", "input-coverage-failure"))
            if len(entities) == 0:
                tags.append("zero-candidate")
            elif len(entities) == 1:
                tags.append("single-candidate-trivial")
            else:
                tags.append("multiple-candidates")
            if same_type == 1 and target_raw_id in last:
                tags.append("type-unique")
            elif same_type > 1:
                tags.append("same-type-ambiguity")
            else:
                tags.append("target-type-unsupported")
            candidate_histogram[len(entities)] += 1
            same_type_histogram[same_type] += 1
            tag_counts.update(tags)
            cases.append(
                TemporalGroundingCase(
                    case_id=case_id,
                    query=query,
                    entities=entities,
                    target_id=target_id,
                    gold_frame=QueryFrame(
                        query,
                        (PropertyConstraint("type", target_type, 1.0),),
                    ),
                    as_of=action_time,
                    current_truth={target_id: {"selected_by_recorded_interaction": True}},
                    tags=tuple(tags),
                )
            )

    return TeachLayerCPrepared(
        tuple(cases),
        {
            "alignment_policy_id": TEACH_DIALOGUE_ALIGNMENT_POLICY_ID,
            "frozen_manifest_sha256": expected_manifest_sha256,
            "oracle_frame": "official target object type only",
            "richer_attribute_oracle": "requires independent annotation",
            "candidate_source": "all entities visible at least once strictly before action",
            "target_label_source": "recorded successful object interaction only",
            "action_result_used_as_matcher_evidence": False,
            "final_truth_used": False,
            "primary_metric_denominator": "all automatically aligned cases",
            "cases": len(cases),
            "candidate_size_histogram": {
                str(size): count for size, count in sorted(candidate_histogram.items())
            },
            "same_type_candidate_histogram": {
                str(size): count for size, count in sorted(same_type_histogram.items())
            },
            "tag_counts": dict(sorted(tag_counts.items())),
        },
    )

