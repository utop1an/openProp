from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Iterable

from .persistence_data import PersistenceTrainingExample


@dataclass(frozen=True, slots=True)
class RepeatedEventEvidence:
    """Independent event-status annotations for one calibration identity.

    The latent event status is deliberately absent. Observation provenance and
    repeated labels live outside the typed property feature tuple.
    """

    property_name: str
    subject_type: str
    state_predicate: str
    context_object: str
    scene: str
    duration_seconds: float
    group_id: str
    annotator_ids: tuple[str, ...]
    event_labels: tuple[bool, ...]

    def __post_init__(self) -> None:
        if not math.isfinite(self.duration_seconds) or self.duration_seconds < 0.0:
            raise ValueError("duration_seconds must be finite and nonnegative")
        if not self.group_id:
            raise ValueError("repeated evidence requires a nonempty group_id")
        count = len(self.annotator_ids)
        if count < 3 or count % 2 == 0:
            raise ValueError("repeated evidence requires an odd number of at least three annotators")
        if count != len(self.event_labels):
            raise ValueError("annotator IDs and event labels must align")
        if len(set(self.annotator_ids)) != count or any(
            not value.strip() for value in self.annotator_ids
        ):
            raise ValueError("annotator IDs must be unique and nonempty")

    def features(self) -> tuple[str, ...]:
        return (
            self.property_name,
            self.subject_type,
            self.state_predicate,
            self.context_object,
            self.scene,
        )

    @property
    def event_votes(self) -> int:
        return sum(self.event_labels)


@dataclass(frozen=True, slots=True)
class SymmetricNoiseEstimate:
    """Calibration-only estimate under homogeneous independent label flips."""

    flip_probability: float
    pairwise_disagreement_rate: float
    compared_annotation_pairs: int

    def __post_init__(self) -> None:
        if not 0.0 <= self.flip_probability <= 0.5:
            raise ValueError("flip probability must be between zero and one half")
        if not 0.0 <= self.pairwise_disagreement_rate <= 1.0:
            raise ValueError("disagreement rate must be between zero and one")
        if self.compared_annotation_pairs <= 0:
            raise ValueError("at least one annotation pair is required")

    @property
    def identifiable(self) -> bool:
        return self.flip_probability < 0.5


@dataclass(frozen=True, slots=True)
class ConsensusCalibration:
    """Hard calibration rows decoded from a declared repeated-label model."""

    examples: tuple[PersistenceTrainingExample, ...]
    noise_estimate: SymmetricNoiseEstimate
    evidence_records: int
    annotations_per_record: int
    retained_records: int
    abstained_records: int
    mean_retained_posterior_confidence: float
    minimum_posterior_confidence: float

    def __post_init__(self) -> None:
        if self.evidence_records <= 0 or self.annotations_per_record < 3:
            raise ValueError("invalid repeated-evidence counts")
        if self.retained_records != len(self.examples):
            raise ValueError("retained count must match decoded examples")
        if self.retained_records + self.abstained_records != self.evidence_records:
            raise ValueError("retained and abstained counts must cover all evidence")
        if not 0.5 <= self.minimum_posterior_confidence < 1.0:
            raise ValueError("minimum posterior confidence must be in [0.5, 1)")
        if self.retained_records:
            if not self.minimum_posterior_confidence <= self.mean_retained_posterior_confidence <= 1.0:
                raise ValueError("mean retained confidence is inconsistent")
        elif self.mean_retained_posterior_confidence != 0.0:
            raise ValueError("empty consensus must have zero mean confidence")

    @property
    def annotation_budget(self) -> int:
        return self.evidence_records * self.annotations_per_record


def _validated_records(
    evidence: Iterable[RepeatedEventEvidence],
) -> tuple[RepeatedEventEvidence, ...]:
    rows = tuple(evidence)
    if not rows:
        raise ValueError("repeated evidence cannot be empty")
    if len({row.group_id for row in rows}) != len(rows):
        raise ValueError("repeated-evidence group IDs must be unique")
    annotators = rows[0].annotator_ids
    if any(row.annotator_ids != annotators for row in rows[1:]):
        raise ValueError("all records must use the same ordered annotator IDs")
    return rows


def estimate_independent_symmetric_noise(
    evidence: Iterable[RepeatedEventEvidence],
) -> SymmetricNoiseEstimate:
    """Estimate flip probability from inter-annotator disagreement only.

    Under conditionally independent annotators with common symmetric error
    ``epsilon < 0.5``, pairwise disagreement is ``2 epsilon (1-epsilon)`` and
    is independent of latent event prevalence. The lower quadratic root is the
    identifiable solution. No entity outcome or test label is required.
    """

    rows = _validated_records(evidence)
    disagreements = 0
    comparisons = 0
    for row in rows:
        for left in range(len(row.event_labels)):
            for right in range(left + 1, len(row.event_labels)):
                disagreements += row.event_labels[left] != row.event_labels[right]
                comparisons += 1
    rate = disagreements / comparisons
    discriminant = max(0.0, 1.0 - 2.0 * rate)
    flip = 0.5 * (1.0 - math.sqrt(discriminant))
    return SymmetricNoiseEstimate(flip, rate, comparisons)


def _posterior_event_probability(
    event_votes: int,
    annotation_count: int,
    flip_probability: float,
) -> float:
    if not 0 <= event_votes <= annotation_count:
        raise ValueError("event vote count is invalid")
    if not 0.0 <= flip_probability < 0.5:
        raise ValueError("posterior decoding requires identifiable noise below one half")
    if flip_probability == 0.0:
        if event_votes == annotation_count:
            return 1.0
        if event_votes == 0:
            return 0.0
        raise ValueError("zero estimated noise is incompatible with discordant labels")
    log_odds = (2 * event_votes - annotation_count) * math.log(
        (1.0 - flip_probability) / flip_probability
    )
    if log_odds >= 0.0:
        return 1.0 / (1.0 + math.exp(-min(700.0, log_odds)))
    odds = math.exp(max(-700.0, log_odds))
    return odds / (1.0 + odds)


def decode_repeated_event_consensus(
    evidence: Iterable[RepeatedEventEvidence],
    *,
    minimum_posterior_confidence: float = 0.5,
) -> ConsensusCalibration:
    """Decode latent event status and abstain when confidence is insufficient."""

    if not 0.5 <= minimum_posterior_confidence < 1.0:
        raise ValueError("minimum posterior confidence must be in [0.5, 1)")
    rows = _validated_records(evidence)
    estimate = estimate_independent_symmetric_noise(rows)
    if not estimate.identifiable:
        raise ValueError("repeated labels are unidentifiable at chance-level noise")
    decoded: list[PersistenceTrainingExample] = []
    confidences: list[float] = []
    for row in rows:
        posterior = _posterior_event_probability(
            row.event_votes,
            len(row.event_labels),
            estimate.flip_probability,
        )
        confidence = max(posterior, 1.0 - posterior)
        if confidence + 1e-15 < minimum_posterior_confidence:
            continue
        decoded.append(
            PersistenceTrainingExample(
                property_name=row.property_name,
                subject_type=row.subject_type,
                state_predicate=row.state_predicate,
                context_object=row.context_object,
                scene=row.scene,
                duration_seconds=row.duration_seconds,
                event_observed=posterior > 0.5,
                group_id=row.group_id,
            )
        )
        confidences.append(confidence)
    return ConsensusCalibration(
        examples=tuple(decoded),
        noise_estimate=estimate,
        evidence_records=len(rows),
        annotations_per_record=len(rows[0].event_labels),
        retained_records=len(decoded),
        abstained_records=len(rows) - len(decoded),
        mean_retained_posterior_confidence=(
            sum(confidences) / len(confidences) if confidences else 0.0
        ),
        minimum_posterior_confidence=minimum_posterior_confidence,
    )


def single_annotation_examples(
    evidence: Iterable[RepeatedEventEvidence],
    *,
    annotator_index: int = 0,
) -> tuple[PersistenceTrainingExample, ...]:
    """Expose one declared annotation as a matched noisy-label baseline."""

    rows = _validated_records(evidence)
    if not 0 <= annotator_index < len(rows[0].event_labels):
        raise ValueError("annotator index is out of range")
    return tuple(
        PersistenceTrainingExample(
            property_name=row.property_name,
            subject_type=row.subject_type,
            state_predicate=row.state_predicate,
            context_object=row.context_object,
            scene=row.scene,
            duration_seconds=row.duration_seconds,
            event_observed=row.event_labels[annotator_index],
            group_id=row.group_id,
        )
        for row in rows
    )


def simulate_repeated_event_evidence(
    examples: Iterable[PersistenceTrainingExample],
    *,
    annotator_count: int,
    flip_fraction: float,
    seed: int,
) -> tuple[RepeatedEventEvidence, ...]:
    """Create synthetic repeated labels without retaining latent truth.

    Each annotator flips an exact rounded fraction selected by an independent
    group-ID hash. Selection is outcome-independent. This helper is benchmark
    scaffolding, not an assumption that real annotators have known error.
    """

    rows = tuple(examples)
    if not rows:
        raise ValueError("calibration examples cannot be empty")
    if annotator_count < 3 or annotator_count % 2 == 0:
        raise ValueError("annotator_count must be odd and at least three")
    if not math.isfinite(flip_fraction) or not 0.0 <= flip_fraction < 0.5:
        raise ValueError("flip_fraction must be finite in [0, 0.5)")
    if any(row.is_interval_censored for row in rows):
        raise ValueError("status-only repeated evidence does not support interval-censored rows")
    if any(not row.group_id for row in rows) or len(
        {row.group_id for row in rows}
    ) != len(rows):
        raise ValueError("simulation requires unique nonempty group IDs")
    annotator_ids = tuple(f"replicate-{index + 1}" for index in range(annotator_count))
    flipped_by_annotator: list[frozenset[str]] = []
    count = round(flip_fraction * len(rows))
    for index, annotator_id in enumerate(annotator_ids):
        ranked = sorted(
            rows,
            key=lambda row: (
                hashlib.sha256(
                    f"{seed}|{index}|{annotator_id}|{row.group_id}".encode("utf-8")
                ).digest(),
                row.group_id,
            ),
        )
        flipped_by_annotator.append(
            frozenset(row.group_id for row in ranked[:count])
        )
    return tuple(
        RepeatedEventEvidence(
            property_name=row.property_name,
            subject_type=row.subject_type,
            state_predicate=row.state_predicate,
            context_object=row.context_object,
            scene=row.scene,
            duration_seconds=row.duration_seconds,
            group_id=row.group_id,
            annotator_ids=annotator_ids,
            event_labels=tuple(
                (not row.event_observed)
                if row.group_id in flipped
                else row.event_observed
                for flipped in flipped_by_annotator
            ),
        )
        for row in rows
    )

