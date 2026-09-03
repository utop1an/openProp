from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .alfred_ontology import (
    AlfredTrainingOntology,
    alfred_receptacle_predicate,
    normalise_label,
)
from .models import PropertyConstraint, QueryFrame, RelationValue


DESTINATION_MARKERS = frozenset({"in", "inside", "into", "on", "onto", "to"})
IGNORED_PREPOSITION_TOKENS = frozenset({"a", "an", "the", "top", "of"})

STATE_CUES = {
    "cleanliness": {
        "clean": "clean",
        "cleaned": "clean",
        "rinse": "clean",
        "rinsed": "clean",
        "wash": "clean",
        "washed": "clean",
    },
    "thermal_state": {
        "chill": "cold",
        "chilled": "cold",
        "cold": "cold",
        "cool": "cold",
        "cooled": "cold",
        "heat": "hot",
        "heated": "hot",
        "hot": "hot",
        "warm": "hot",
        "warmed": "hot",
    },
}


@dataclass(frozen=True, slots=True)
class LabelMention:
    label: str
    token_start: int
    token_end: int
    text: str


@dataclass(frozen=True, slots=True)
class PropertySelectionEvidence:
    property_name: str
    desired_value: object
    source_text: str
    token_start: int
    token_end: int
    rule: str


@dataclass(frozen=True, slots=True)
class AlfredSelectionEvidence:
    query: str
    evidence: tuple[PropertySelectionEvidence, ...]


@dataclass(frozen=True, slots=True)
class SelectionFusionPolicy:
    override_conflicting_values: bool = False
    add_missing: bool = True
    gate_unsupported_states: bool = False
    gate_conflicting_states: bool = False


@dataclass(frozen=True, slots=True)
class SelectionFusionResult:
    frame: QueryFrame
    actions: tuple[str, ...]


def _resolve_evidence_label(value: str, labels: frozenset[str]) -> str | None:
    """Resolve exact labels or shortened aliases, never arbitrary supersets."""

    key = normalise_label(value)
    if key in labels:
        return key
    tokens = frozenset(key.split())
    if not tokens:
        return None
    candidates = [
        label for label in labels if tokens <= frozenset(label.split())
    ]
    return candidates[0] if len(candidates) == 1 else None


def _label_mentions(tokens: tuple[str, ...], labels: frozenset[str]) -> list[LabelMention]:
    max_length = max(len(label.split()) for label in labels)
    mentions: list[LabelMention] = []
    for start in range(len(tokens)):
        for end in range(start + 1, min(len(tokens), start + max_length) + 1):
            text = " ".join(tokens[start:end])
            resolved = _resolve_evidence_label(text, labels)
            if resolved is not None:
                mentions.append(LabelMention(resolved, start, end, text))
    deduplicated = {
        (item.label, item.token_start, item.token_end): item for item in mentions
    }
    rows = list(deduplicated.values())
    return [
        item
        for item in rows
        if not any(
            other.token_start <= item.token_start
            and other.token_end >= item.token_end
            and (other.token_end - other.token_start) > (item.token_end - item.token_start)
            for other in rows
        )
    ]


def _is_destination(tokens: tuple[str, ...], mention: LabelMention) -> bool:
    preceding = [
        token
        for token in tokens[max(0, mention.token_start - 4) : mention.token_start]
        if token not in IGNORED_PREPOSITION_TOKENS
    ]
    return bool(preceding and preceding[-1] in DESTINATION_MARKERS)


def _unique_label(mentions: Iterable[LabelMention]) -> LabelMention | None:
    rows = tuple(mentions)
    labels = {item.label for item in rows}
    if len(labels) != 1:
        return None
    return max(rows, key=lambda item: item.token_end - item.token_start)


def extract_alfred_selection_evidence(
    query: str,
    ontology: AlfredTrainingOntology,
) -> AlfredSelectionEvidence:
    """Extract auditable selection evidence without task or target labels."""

    tokens = tuple(normalise_label(query).split())
    if not tokens:
        raise ValueError("query text cannot be empty")
    receptacle_mentions = _label_mentions(tokens, ontology.receptacle_labels)
    destinations = [
        item for item in receptacle_mentions if _is_destination(tokens, item)
    ]
    destination = _unique_label(destinations)
    destination_spans = {
        (item.token_start, item.token_end) for item in destinations
    }
    object_mentions = [
        item
        for item in _label_mentions(tokens, ontology.object_labels)
        if (item.token_start, item.token_end) not in destination_spans
    ]
    target = _unique_label(object_mentions)
    evidence: list[PropertySelectionEvidence] = []
    if target is not None:
        evidence.append(
            PropertySelectionEvidence(
                "type",
                target.label,
                target.text,
                target.token_start,
                target.token_end,
                "unique_train_vocabulary_mention",
            )
        )
    if destination is not None:
        evidence.append(
            PropertySelectionEvidence(
                "location",
                RelationValue(
                    alfred_receptacle_predicate(destination.label),
                    {"object": destination.label},
                ),
                destination.text,
                destination.token_start,
                destination.token_end,
                "preposition_bound_receptacle_mention",
            )
        )
    for property_name, cues in STATE_CUES.items():
        matches = [
            (index, cues[token]) for index, token in enumerate(tokens) if token in cues
        ]
        values = {value for _, value in matches}
        if len(values) == 1:
            index, value = matches[0]
            evidence.append(
                PropertySelectionEvidence(
                    property_name,
                    value,
                    tokens[index],
                    index,
                    index + 1,
                    "predeclared_state_cue",
                )
            )
    return AlfredSelectionEvidence(query, tuple(evidence))


def fuse_alfred_selection(
    frame: QueryFrame,
    evidence: AlfredSelectionEvidence,
    *,
    policy: SelectionFusionPolicy | None = None,
) -> SelectionFusionResult:
    """Fuse explicit query evidence with an LLM frame; never infer without a span."""

    if frame.text != evidence.query:
        raise ValueError("selection evidence query does not match the parsed frame")
    selected_policy = policy or SelectionFusionPolicy()
    evidence_by_name = {item.property_name: item for item in evidence.evidence}
    evidenced_states = {
        name for name in evidence_by_name if name in STATE_CUES
    }
    constraints: list[PropertyConstraint] = []
    actions: list[str] = []
    for constraint in frame.constraints:
        if (
            selected_policy.gate_unsupported_states
            and constraint.property_name in STATE_CUES
            and constraint.property_name not in evidence_by_name
        ):
            actions.append(
                f"removed unsupported {constraint.property_name}: no query cue"
            )
            continue
        if (
            selected_policy.gate_conflicting_states
            and evidenced_states
            and constraint.property_name in STATE_CUES
            and constraint.property_name not in evidenced_states
        ):
            actions.append(
                f"removed conflicting {constraint.property_name}: explicit cue supports "
                f"{','.join(sorted(evidenced_states))}"
            )
            continue
        evidence_item = evidence_by_name.get(constraint.property_name)
        if (
            selected_policy.override_conflicting_values
            and evidence_item is not None
            and constraint.desired_value != evidence_item.desired_value
        ):
            constraints.append(
                PropertyConstraint(
                    constraint.property_name,
                    evidence_item.desired_value,
                    constraint.relevance,
                )
            )
            actions.append(
                f"replaced {constraint.property_name} value from "
                f"tokens[{evidence_item.token_start}:{evidence_item.token_end}] "
                f"{evidence_item.source_text!r} via {evidence_item.rule}"
            )
        else:
            constraints.append(constraint)
    selected_names = {item.property_name for item in constraints}
    if selected_policy.add_missing:
        for item in evidence.evidence:
            if item.property_name in selected_names:
                continue
            relevance = 0.35 if item.property_name in STATE_CUES else 0.5
            constraints.append(
                PropertyConstraint(item.property_name, item.desired_value, relevance)
            )
            selected_names.add(item.property_name)
            actions.append(
                f"added {item.property_name} from tokens[{item.token_start}:{item.token_end}] "
                f"{item.source_text!r} via {item.rule}"
            )
    return SelectionFusionResult(QueryFrame(frame.text, tuple(constraints)), tuple(actions))
