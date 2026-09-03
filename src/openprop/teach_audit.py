from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from .teach_adapter import (
    DEFAULT_TEACH_STATE_PROPERTIES,
    read_teach_replay,
    teach_visible_observation_history,
)
from .teach_feasibility import (
    TeachFeasibilityCriteria,
    assign_teach_floorplan_splits,
    evaluate_teach_feasibility,
)
from .teach_grounding import (
    TEACH_BOOLEAN_STATE_PROPERTIES,
    audit_teach_grounding_cases,
    build_teach_gold_grounding_cases,
)


@dataclass(frozen=True, slots=True)
class TeachAuditSession:
    episode_id: str
    floorplan: str
    initial_state: Path
    state_directory: Path
    final_timestamp: float
    game_file: Path | None = None


def _resolve(parent: Path, value: Any) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else parent / path


def read_teach_audit_manifest(path: str | Path) -> tuple[TeachAuditSession, ...]:
    """Read a JSONL manifest without loading evaluation truth into entities."""
    source = Path(path)
    sessions: list[TeachAuditSession] = []
    seen: set[str] = set()
    with source.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                episode_id = str(row["episode_id"]).strip()
                floorplan = str(row["floorplan"]).strip()
                final_timestamp = float(row["final_timestamp"])
                initial_state = _resolve(source.parent, row["initial_state"])
                state_directory = _resolve(source.parent, row["state_directory"])
                game_file = (
                    _resolve(source.parent, row["game_file"])
                    if row.get("game_file") is not None else None
                )
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    f"invalid TEACh audit manifest row {line_number}: {error}"
                ) from error
            if not episode_id or not floorplan:
                raise ValueError(
                    f"invalid TEACh audit manifest row {line_number}: "
                    "episode_id and floorplan cannot be empty"
                )
            if episode_id in seen:
                raise ValueError(f"duplicate episode_id at row {line_number}: {episode_id}")
            if not math.isfinite(final_timestamp) or final_timestamp < 0:
                raise ValueError(
                    f"invalid TEACh audit manifest row {line_number}: "
                    "final_timestamp must be finite and nonnegative"
                )
            seen.add(episode_id)
            sessions.append(
                TeachAuditSession(
                    episode_id,
                    floorplan,
                    initial_state,
                    state_directory,
                    final_timestamp,
                    game_file,
                )
            )
    if not sessions:
        raise ValueError("TEACh audit manifest contains no sessions")
    return tuple(sessions)


def read_teach_manifest(path: str | Path) -> tuple[TeachAuditSession, ...]:
    """Read the strict TEACh manifest used by all audit layers."""

    return read_teach_audit_manifest(path)

def audit_teach_sessions(
    sessions: Iterable[TeachAuditSession],
    *,
    property_names: tuple[str, ...] = DEFAULT_TEACH_STATE_PROPERTIES,
    criteria: TeachFeasibilityCriteria | None = None,
    dialogue_alignment: Mapping[str, Any] | None = None,
    dialogue_alignment_auto: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Summarize observation support before any TEACh performance experiment."""
    rows = tuple(sessions)
    if not rows:
        raise ValueError("at least one TEACh session is required")
    if not property_names or len(property_names) != len(set(property_names)):
        raise ValueError("property_names must be non-empty and unique")

    floorplans: Counter[str] = Counter()
    observations: Counter[str] = Counter()
    transitions: Counter[str] = Counter()
    censoring: Counter[str] = Counter()
    entities: set[str] = set()
    snapshot_count = 0
    visible_count = 0
    session_rows: list[dict[str, Any]] = []
    grounding_cases = []
    grounding_property_names = tuple(
        name for name in property_names if name in TEACH_BOOLEAN_STATE_PROPERTIES
    )

    for session in rows:
        initial = json.loads(session.initial_state.read_text(encoding="utf-8"))
        replay = read_teach_replay(
            initial,
            session.state_directory,
            final_timestamp=session.final_timestamp,
        )
        if replay.final_truth is None:
            raise ValueError(f"session {session.episode_id!r} has no final state diff")
        history = teach_visible_observation_history(
            session.episode_id,
            replay.observations,
            scene=session.floorplan,
            property_names=property_names,
        )
        local_grounding = (
            build_teach_gold_grounding_cases(
                session.episode_id,
                session.floorplan,
                replay,
                property_names=grounding_property_names,
            )
            if grounding_property_names
            else ()
        )
        grounding_cases.extend(local_grounding)
        local_entities: set[str] = set()
        local_visible = 0
        for snapshot in replay.observations:
            snapshot_count += 1
            for obj in snapshot.objects:
                if not obj.get("visible", False) or not obj.get("objectId"):
                    continue
                local_visible += 1
                visible_count += 1
                entity_id = f"{session.episode_id}:{obj['objectId']}"
                local_entities.add(entity_id)
                entities.add(entity_id)
                for name in property_names:
                    if name in obj:
                        observations[name] += 1
        for record in history:
            if not record.state_changed:
                censoring["right_censored"] += 1
            elif record.last_confirmed_at is None:
                censoring["exact_event"] += 1
            else:
                censoring["interval_censored_event"] += 1
            if record.state_changed:
                transitions[record.property_name] += 1
        floorplans[session.floorplan] += 1
        session_rows.append(
            {
                "episode_id": session.episode_id,
                "floorplan": session.floorplan,
                "snapshots": len(replay.observations),
                "visible_object_observations": local_visible,
                "unique_visible_entities": len(local_entities),
                "history_records": len(history),
                "gold_grounding_cases": len(local_grounding),
                "temporal_discriminative_cases": sum(
                    "temporal-discriminative" in case.tags
                    for case in local_grounding
                ),
                "primary_evaluable_cases": sum(
                    "primary-evaluable" in case.tags for case in local_grounding
                ),
            }
        )

    for key in ("exact_event", "interval_censored_event", "right_censored"):
        censoring.setdefault(key, 0)
    for name in property_names:
        observations.setdefault(name, 0)
        transitions.setdefault(name, 0)
    warnings: list[str] = []
    if len(floorplans) < 3:
        warnings.append(
            "fewer than three floorplans: a floorplan-disjoint train/validation/test "
            "split is not feasible"
        )
    if censoring["interval_censored_event"] == 0:
        warnings.append("no interval-censored transitions were observed")
    missing = [name for name in property_names if transitions[name] == 0]
    if missing:
        warnings.append("properties without observed transitions: " + ", ".join(missing))

    report = {
        "protocol": {
            "source": "TEACh egocentric replay state diffs",
            "matcher_input": "visible observation snapshots only",
            "final_truth_use": (
                "evaluation-only query, target, and audit labels; never matcher entities"
            ),
            "property_names": list(property_names),
            "claim_scope": "dataset feasibility audit, not model performance",
            "gate_thresholds": "predeclared pilot sufficiency, not outcome tuned",
            "dialogue_alignment": (
                "separate manual audit; missing evidence fails closed"
            ),
        },
        "totals": {
            "sessions": len(rows),
            "floorplans": len(floorplans),
            "snapshots": snapshot_count,
            "visible_object_observations": visible_count,
            "unique_visible_entities": len(entities),
            "history_records": sum(censoring.values()),
        },
        "censoring": dict(sorted(censoring.items())),
        "property_observations": dict(sorted(observations.items())),
        "property_transitions": dict(sorted(transitions.items())),
        "sessions_per_floorplan": dict(sorted(floorplans.items())),
        "floorplan_split": assign_teach_floorplan_splits(dict(floorplans)),
        "gold_grounding": audit_teach_grounding_cases(grounding_cases),
        "dialogue_alignment_auto": dict(dialogue_alignment_auto or {}),
        "warnings": warnings,
        "sessions": session_rows,
    }
    report["feasibility_gate"] = evaluate_teach_feasibility(
        report,
        criteria=criteria,
        dialogue_alignment=dialogue_alignment,
    )
    return report


def write_teach_audit_report(path: str | Path, report: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
