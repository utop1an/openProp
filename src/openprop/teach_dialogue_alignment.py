from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


TEACH_DIALOGUE_ALIGNMENT_POLICY_ID = "next-successful-object-v1"
TEACH_DIALOGUE_ALIGNMENT_SAMPLE_SEED = 29


@dataclass(frozen=True, slots=True)
class TeachDialogueTurn:
    interaction_index: int
    agent_id: int
    agent_role: str
    time_start: float
    utterance: str


@dataclass(frozen=True, slots=True)
class TeachDialogueAlignmentCase:
    case_id: str
    episode_id: str
    interaction_index: int
    action_id: int
    action_name: str
    action_time: float
    target_object_id: str
    target_object_type: str
    compatible_object_type: str
    commander_text: str
    dialogue: tuple[TeachDialogueTurn, ...]

    def to_dict(self) -> dict[str, Any]:
        row = asdict(self)
        row["dialogue"] = [asdict(turn) for turn in self.dialogue]
        return row


def teach_manifest_sha256(path: str | Path) -> str:
    """Hash the exact frozen manifest bytes used to construct Layer C cases."""

    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _nonempty_string(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"TEACh {field} must be a nonempty string")
    return value.strip()


def _definition_maps(payload: Mapping[str, Any]) -> tuple[dict[int, dict[str, str]], dict[int, str], int]:
    definitions = payload.get("definitions")
    if not isinstance(definitions, Mapping):
        raise ValueError("TEACh game is missing definitions")
    actions = definitions.get("actions")
    agents = definitions.get("agents")
    if not isinstance(actions, list) or not isinstance(agents, list):
        raise ValueError("TEACh definitions require action and agent lists")

    action_map: dict[int, dict[str, str]] = {}
    for row in actions:
        if not isinstance(row, Mapping):
            raise ValueError("TEACh action definitions must be objects")
        try:
            action_id = int(row["action_id"])
            action_name = _nonempty_string(row["action_name"], field="action_name")
            action_type = _nonempty_string(row["action_type"], field="action_type")
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"invalid TEACh action definition: {error}") from error
        if action_id in action_map:
            raise ValueError(f"duplicate TEACh action_id: {action_id}")
        action_map[action_id] = {"name": action_name, "type": action_type}

    agent_roles: dict[int, str] = {}
    commander_ids: list[int] = []
    for row in agents:
        if not isinstance(row, Mapping):
            raise ValueError("TEACh agent definitions must be objects")
        try:
            agent_id = int(row["agent_id"])
            name = _nonempty_string(row["agent_name"], field="agent_name")
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"invalid TEACh agent definition: {error}") from error
        if agent_id in agent_roles:
            raise ValueError(f"duplicate TEACh agent_id: {agent_id}")
        agent_roles[agent_id] = name
        if name.casefold() == "commander":
            commander_ids.append(agent_id)
    if len(commander_ids) != 1:
        raise ValueError("TEACh definitions require exactly one Commander agent")
    return action_map, agent_roles, commander_ids[0]


def _episode(payload: Mapping[str, Any], episode_id: str) -> Mapping[str, Any]:
    tasks = payload.get("tasks")
    if not isinstance(tasks, list):
        raise ValueError("TEACh game is missing tasks")
    matches: list[Mapping[str, Any]] = []
    for task in tasks:
        if not isinstance(task, Mapping) or not isinstance(task.get("episodes"), list):
            raise ValueError("TEACh tasks require episode lists")
        for candidate in task["episodes"]:
            if not isinstance(candidate, Mapping):
                raise ValueError("TEACh episodes must be objects")
            if str(candidate.get("episode_id", "")).strip() == episode_id:
                matches.append(candidate)
    if len(matches) != 1:
        raise ValueError(
            f"TEACh game must contain exactly one episode {episode_id!r}; found {len(matches)}"
        )
    return matches[0]


def _object_type_from_id(object_id: str) -> str:
    head = object_id.split("|", 1)[0].strip()
    head = re.sub(r"\(Clone\)$", "", head).strip()
    if not head:
        raise ValueError("TEACh object interaction has an invalid oid")
    return head


def _type_tokens(object_type: str) -> tuple[str, ...]:
    spaced = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", object_type)
    tokens = tuple(re.findall(r"[a-z0-9]+", spaced.casefold()))
    if tokens and tokens[-1] in {"sliced", "cracked"}:
        tokens = tokens[:-1]
    if not tokens:
        raise ValueError(f"invalid TEACh object type: {object_type!r}")
    return tokens


def _compatible_type(object_type: str) -> str:
    return " ".join(_type_tokens(object_type))


def _surface_forms(object_type: str) -> frozenset[str]:
    tokens = _type_tokens(object_type)
    phrase = " ".join(tokens)
    forms = {phrase, "".join(tokens)}
    if phrase.endswith("y") and len(phrase) > 1:
        forms.add(phrase[:-1] + "ies")
    elif phrase.endswith(("s", "x", "z", "ch", "sh")):
        forms.add(phrase + "es")
    else:
        forms.add(phrase + "s")
    return frozenset(forms)


def _normalize_text(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", value.casefold()))


def _mentioned_types(text: str, vocabulary: Iterable[str]) -> frozenset[str]:
    normalized = f" {_normalize_text(text)} "
    mentioned: set[str] = set()
    for object_type in vocabulary:
        compatible = _compatible_type(object_type)
        if any(f" {form} " in normalized for form in _surface_forms(object_type)):
            mentioned.add(compatible)
    return frozenset(mentioned)


def _successful(value: Any) -> bool:
    return value is True or (type(value) is int and value == 1)


def align_teach_game_dialogue(
    payload: Mapping[str, Any],
    *,
    episode_id: str,
    known_object_types: Iterable[str] = (),
) -> dict[str, Any]:
    """Align dialogue segments to the next successful typed object interaction.

    The policy is deliberately high precision. A case is accepted only when the
    Commander segment since the previous successful object interaction names the
    target's compatible object type and names no other known scene object type.
    Failed object interactions do not consume the pending dialogue segment.
    """

    selected_episode = _episode(payload, episode_id)
    action_map, agent_roles, commander_id = _definition_maps(payload)
    interactions = selected_episode.get("interactions")
    if not isinstance(interactions, list):
        raise ValueError("TEACh episode interactions must be a list")

    action_object_types: set[str] = set()
    last_time = -math.inf
    for index, row in enumerate(interactions):
        if not isinstance(row, Mapping):
            raise ValueError(f"TEACh interaction {index} must be an object")
        try:
            action_id = int(row["action_id"])
            time_start = float(row["time_start"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"invalid TEACh interaction {index}: {error}") from error
        if action_id not in action_map:
            raise ValueError(f"TEACh interaction {index} uses unknown action_id {action_id}")
        if not math.isfinite(time_start) or time_start < 0 or time_start < last_time:
            raise ValueError("TEACh interaction times must be finite, nonnegative, and ordered")
        last_time = time_start
        if action_map[action_id]["type"] == "ObjectInteraction" and _successful(row.get("success")):
            oid = row.get("oid")
            if isinstance(oid, str) and oid.strip():
                action_object_types.add(_object_type_from_id(oid.strip()))

    vocabulary = {
        _nonempty_string(value, field="known object type")
        for value in known_object_types
    }
    vocabulary.update(action_object_types)
    pending_dialogue: list[TeachDialogueTurn] = []
    cases: list[TeachDialogueAlignmentCase] = []
    rejections: Counter[str] = Counter()
    successful_object_interactions = 0

    for index, row in enumerate(interactions):
        action_id = int(row["action_id"])
        definition = action_map[action_id]
        time_start = float(row["time_start"])
        if definition["type"] in {"Keyboard", "Audio"}:
            utterance = row.get("utterance")
            if not isinstance(utterance, str) or not utterance.strip():
                rejections["empty_utterance"] += 1
                continue
            try:
                agent_id = int(row["agent_id"])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"dialogue interaction {index} requires agent_id") from error
            if agent_id not in agent_roles:
                raise ValueError(f"dialogue interaction {index} uses unknown agent_id {agent_id}")
            pending_dialogue.append(
                TeachDialogueTurn(
                    interaction_index=index,
                    agent_id=agent_id,
                    agent_role=agent_roles[agent_id],
                    time_start=time_start,
                    utterance=utterance.strip(),
                )
            )
            continue

        if definition["type"] != "ObjectInteraction" or not _successful(row.get("success")):
            continue
        successful_object_interactions += 1
        oid = row.get("oid")
        if not isinstance(oid, str) or not oid.strip():
            rejections["missing_target_object_id"] += 1
            pending_dialogue.clear()
            continue
        commander_turns = tuple(
            turn for turn in pending_dialogue if turn.agent_id == commander_id
        )
        if not pending_dialogue:
            rejections["no_dialogue_segment"] += 1
            continue
        if not commander_turns:
            rejections["no_commander_utterance"] += 1
            pending_dialogue.clear()
            continue
        target_type = _object_type_from_id(oid.strip())
        compatible = _compatible_type(target_type)
        commander_text = " ".join(turn.utterance for turn in commander_turns)
        mentioned = _mentioned_types(commander_text, vocabulary)
        if compatible not in mentioned:
            rejections["target_type_not_mentioned"] += 1
            pending_dialogue.clear()
            continue
        if mentioned != {compatible}:
            rejections["ambiguous_object_type"] += 1
            pending_dialogue.clear()
            continue
        cases.append(
            TeachDialogueAlignmentCase(
                case_id=f"teach-dialogue:{episode_id}:{index:06d}",
                episode_id=episode_id,
                interaction_index=index,
                action_id=action_id,
                action_name=definition["name"],
                action_time=time_start,
                target_object_id=oid.strip(),
                target_object_type=target_type,
                compatible_object_type=compatible,
                commander_text=commander_text,
                dialogue=tuple(pending_dialogue),
            )
        )
        pending_dialogue.clear()

    return {
        "episode_id": episode_id,
        "interactions": len(interactions),
        "successful_object_interactions": successful_object_interactions,
        "aligned_cases": len(cases),
        "rejection_counts": dict(sorted(rejections.items())),
        "cases": [case.to_dict() for case in cases],
    }


def _initial_object_types(path: Path) -> frozenset[str]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid TEACh initial state {path}: {error}") from error
    objects = payload.get("objects")
    if not isinstance(objects, list):
        raise ValueError(f"TEACh initial state {path} requires an objects list")
    result = {
        str(row["objectType"]).strip()
        for row in objects
        if isinstance(row, Mapping) and str(row.get("objectType", "")).strip()
    }
    return frozenset(result)


def audit_teach_dialogue_alignments(
    sessions: Iterable[Any],
    *,
    frozen_manifest_sha256: str,
) -> dict[str, Any]:
    """Construct all high-precision Layer C cases from manifest-linked games."""

    if re.fullmatch(r"[0-9a-f]{64}", frozen_manifest_sha256) is None:
        raise ValueError("frozen_manifest_sha256 must be a 64-character hex digest")
    rows = tuple(sessions)
    if not rows:
        raise ValueError("at least one TEACh session is required")
    cases: list[dict[str, Any]] = []
    rejections: Counter[str] = Counter()
    episode_rows: list[dict[str, Any]] = []
    missing_games: list[str] = []
    total_interactions = 0
    total_successful_object_interactions = 0
    for session in rows:
        episode_id = str(getattr(session, "episode_id", "")).strip()
        game_file = getattr(session, "game_file", None)
        initial_state = Path(getattr(session, "initial_state"))
        if not episode_id:
            raise ValueError("TEACh dialogue sessions require episode_id")
        if game_file is None:
            missing_games.append(episode_id)
            continue
        game_path = Path(game_file)
        try:
            payload = json.loads(game_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid TEACh game {game_path}: {error}") from error
        if not isinstance(payload, Mapping):
            raise ValueError(f"TEACh game {game_path} must contain an object")
        episode = align_teach_game_dialogue(
            payload,
            episode_id=episode_id,
            known_object_types=_initial_object_types(initial_state),
        )
        cases.extend(episode["cases"])
        rejections.update(episode["rejection_counts"])
        total_interactions += episode["interactions"]
        total_successful_object_interactions += episode["successful_object_interactions"]
        episode_rows.append({key: value for key, value in episode.items() if key != "cases"})
    case_ids = [str(case["case_id"]) for case in cases]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("TEACh dialogue alignment produced duplicate case IDs")
    return {
        "alignment_policy_id": TEACH_DIALOGUE_ALIGNMENT_POLICY_ID,
        "frozen_manifest_sha256": frozen_manifest_sha256,
        "claim_scope": "automatic alignment candidates for manual audit; not performance",
        "sessions": len(rows),
        "sessions_with_game_file": len(episode_rows),
        "sessions_missing_game_file": len(missing_games),
        "missing_game_episode_ids": sorted(missing_games),
        "interactions": total_interactions,
        "successful_object_interactions": total_successful_object_interactions,
        "aligned_cases": len(cases),
        "rejection_counts": dict(sorted(rejections.items())),
        "case_ids": case_ids,
        "cases": cases,
        "episodes": episode_rows,
    }


def freeze_teach_dialogue_alignment_sample(
    cases: Sequence[Mapping[str, Any]],
    *,
    sample_size: int,
    seed: int = TEACH_DIALOGUE_ALIGNMENT_SAMPLE_SEED,
) -> tuple[Mapping[str, Any], ...]:
    """Return an order-invariant deterministic manual-audit sample."""

    if sample_size <= 0:
        raise ValueError("sample_size must be positive")
    by_id: dict[str, Mapping[str, Any]] = {}
    for row in cases:
        case_id = _nonempty_string(row.get("case_id"), field="alignment case_id")
        if case_id in by_id:
            raise ValueError(f"duplicate dialogue alignment case_id: {case_id}")
        by_id[case_id] = row
    ranked = sorted(
        by_id.values(),
        key=lambda row: (
            hashlib.sha256(f"{seed}\0{row['case_id']}".encode("utf-8")).hexdigest(),
            str(row["case_id"]),
        ),
    )
    return tuple(ranked[:sample_size])


def build_teach_dialogue_alignment_label_template(
    report: Mapping[str, Any],
    *,
    sample_size: int,
    seed: int = TEACH_DIALOGUE_ALIGNMENT_SAMPLE_SEED,
) -> dict[str, Any]:
    """Build a frozen, intentionally incomplete human-label template."""

    policy_id = _nonempty_string(report.get("alignment_policy_id"), field="alignment policy")
    manifest_hash = _nonempty_string(
        report.get("frozen_manifest_sha256"), field="frozen manifest hash"
    )
    cases = report.get("cases")
    if not isinstance(cases, list):
        raise ValueError("dialogue alignment report requires a cases list")
    sample = freeze_teach_dialogue_alignment_sample(cases, sample_size=sample_size, seed=seed)
    return {
        "alignment_policy_id": policy_id,
        "frozen_manifest_sha256": manifest_hash,
        "aligned_cases": int(report.get("aligned_cases", len(cases))),
        "sample_seed": seed,
        "sample_size": len(sample),
        "label_instructions": (
            "Set is_correct to true only when the dialogue segment unambiguously "
            "instructs the recorded target object interaction; otherwise false."
        ),
        "labels": [
            {"case_id": str(row["case_id"]), "is_correct": None, "notes": ""}
            for row in sample
        ],
        "cases": list(sample),
    }


def write_teach_dialogue_alignment_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
