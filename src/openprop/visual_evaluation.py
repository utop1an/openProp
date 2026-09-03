from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence


NULL_ENTITY = "__null__"
VALID_SPLITS = ("development", "calibration", "test")


@dataclass(frozen=True, slots=True)
class VisualPropertyResult:
    record_id: str
    cluster_id: str
    split: str
    system: str
    source: str
    property_name: str
    expected: bool
    detected: bool
    gold_value: object | None
    predicted_value: object | None
    confidence: float
    duplicate_count: int = 0
    malformed: bool = False

    def __post_init__(self) -> None:
        _identity_fields(self)
        _probability(self.confidence, "confidence")
        if self.duplicate_count < 0:
            raise ValueError("duplicate_count must be nonnegative")
        if self.expected and self.gold_value is None:
            raise ValueError("expected property rows require a gold value")
        if not self.expected and self.gold_value is not None:
            raise ValueError("unexpected property rows cannot contain gold value")
        if self.detected and self.predicted_value is None:
            raise ValueError("detected property rows require a predicted value")
        if not self.detected and self.predicted_value is not None:
            raise ValueError("missed property rows cannot contain a predicted value")
        if not self.detected and self.confidence != 0.0:
            raise ValueError("missed property rows must have zero confidence")
        _json_value(self.gold_value)
        _json_value(self.predicted_value)

    @property
    def exact_value_match(self) -> bool:
        return (
            self.expected
            and self.detected
            and _canonical(self.gold_value) == _canonical(self.predicted_value)
        )


@dataclass(frozen=True, slots=True)
class VisualAssociationResult:
    record_id: str
    cluster_id: str
    split: str
    system: str
    source: str
    property_name: str
    detection_id: str
    frame_id: str
    candidate_entity_ids: tuple[str, ...]
    target_entity_id: str | None
    decision_entity_id: str | None
    accepted_entity_id: str | None
    probabilities: Mapping[str, float]
    condition: str
    distractor_count: int
    malformed: bool = False
    reason: str = ""
    eligible: bool = True
    confidence_scale: float = 1.0
    minimum_update_confidence: float = 0.0
    calibrated_update_confidence: float | None = None

    def __post_init__(self) -> None:
        _identity_fields(self)
        _text(self.detection_id, "detection_id")
        _text(self.frame_id, "frame_id")
        _text(self.condition, "condition")
        candidates = _candidate_ids(self.candidate_entity_ids)
        if self.distractor_count < 0:
            raise ValueError("distractor_count must be nonnegative")
        if self.decision_entity_id is not None and self.decision_entity_id not in candidates:
            raise ValueError("association decision must be a candidate or null")
        if self.accepted_entity_id is not None:
            if self.accepted_entity_id not in candidates:
                raise ValueError("accepted association must be a candidate")
            if self.accepted_entity_id != self.decision_entity_id:
                raise ValueError("accepted association must equal the decision identity")
        _distribution(self.probabilities, candidates)
        _probability(self.confidence_scale, "confidence_scale")
        _probability(self.minimum_update_confidence, "minimum_update_confidence")
        if self.calibrated_update_confidence is not None:
            _probability(
                self.calibrated_update_confidence,
                "calibrated_update_confidence",
            )

    @property
    def decision_key(self) -> str:
        return self.decision_entity_id or NULL_ENTITY

    @property
    def truth_key(self) -> str:
        return self.target_entity_id or NULL_ENTITY

    @property
    def accepted(self) -> bool:
        return self.accepted_entity_id is not None

    @property
    def correct_update(self) -> bool:
        return self.accepted and self.accepted_entity_id == self.target_entity_id

    @property
    def false_update(self) -> bool:
        return self.accepted and self.accepted_entity_id != self.target_entity_id


@dataclass(frozen=True, slots=True)
class VisualQueryResult:
    record_id: str
    cluster_id: str
    split: str
    system: str
    source: str
    property_name: str
    candidate_entity_ids: tuple[str, ...]
    target_entity_id: str | None
    ranked_entity_ids: tuple[str, ...]
    decision_entity_id: str | None
    accepted_entity_id: str | None
    probabilities: Mapping[str, float]
    horizon_seconds: float
    distractor_count: int
    condition: str
    latency_seconds: float = 0.0
    vlm_calls: int = 0
    malformed: bool = False
    eligible: bool = True

    def __post_init__(self) -> None:
        _identity_fields(self)
        _text(self.condition, "condition")
        candidates = _candidate_ids(self.candidate_entity_ids)
        if set(self.ranked_entity_ids) != set(candidates):
            raise ValueError("query ranking must contain every candidate exactly once")
        if len(self.ranked_entity_ids) != len(candidates):
            raise ValueError("query ranking contains duplicate candidates")
        if self.decision_entity_id is not None and self.decision_entity_id not in candidates:
            raise ValueError("query decision must be a candidate or null")
        if self.accepted_entity_id is not None:
            if self.accepted_entity_id not in candidates:
                raise ValueError("accepted query identity must be a candidate")
            if self.accepted_entity_id != self.decision_entity_id:
                raise ValueError("accepted query identity must equal the decision")
        _distribution(self.probabilities, candidates)
        if not math.isfinite(self.horizon_seconds) or self.horizon_seconds < 0.0:
            raise ValueError("horizon_seconds must be finite and nonnegative")
        if self.distractor_count < 0:
            raise ValueError("distractor_count must be nonnegative")
        if not math.isfinite(self.latency_seconds) or self.latency_seconds < 0.0:
            raise ValueError("latency_seconds must be finite and nonnegative")
        if self.vlm_calls < 0:
            raise ValueError("vlm_calls must be nonnegative")

    @property
    def target_candidate_present(self) -> bool:
        return (
            self.target_entity_id is None
            or self.target_entity_id in self.candidate_entity_ids
        )

    @property
    def decision_key(self) -> str:
        return self.decision_entity_id or NULL_ENTITY

    @property
    def truth_key(self) -> str:
        if self.target_entity_id is None:
            return NULL_ENTITY
        return self.target_entity_id

    @property
    def accepted(self) -> bool:
        return self.accepted_entity_id is not None

    @property
    def top1_correct(self) -> bool:
        if self.target_entity_id is None:
            return self.accepted_entity_id is None
        return self.accepted_entity_id == self.target_entity_id

    @property
    def reciprocal_rank(self) -> float:
        if self.target_entity_id is None:
            return 1.0 if self.accepted_entity_id is None else 0.0
        if self.target_entity_id not in self.ranked_entity_ids:
            return 0.0
        return 1.0 / (self.ranked_entity_ids.index(self.target_entity_id) + 1)


@dataclass(frozen=True, slots=True)
class VisualEvaluationDataset:
    properties: tuple[VisualPropertyResult, ...] = ()
    associations: tuple[VisualAssociationResult, ...] = ()
    queries: tuple[VisualQueryResult, ...] = ()

    def __post_init__(self) -> None:
        records = (*self.properties, *self.associations, *self.queries)
        if not records:
            raise ValueError("visual evaluation dataset cannot be empty")
        keys = [(type(row).__name__, row.record_id, row.system) for row in records]
        if len(keys) != len(set(keys)):
            raise ValueError("visual evaluation record IDs must be unique per type/system")
        cluster_splits: dict[str, str] = {}
        for row in records:
            previous = cluster_splits.setdefault(row.cluster_id, row.split)
            if previous != row.split:
                raise ValueError("one cluster cannot appear in multiple splits")

    def subset(self, split: str) -> "VisualEvaluationDataset":
        _split(split)
        rows = (
            tuple(row for row in self.properties if row.split == split),
            tuple(row for row in self.associations if row.split == split),
            tuple(row for row in self.queries if row.split == split),
        )
        if not any(rows):
            raise ValueError(f"visual evaluation split is empty: {split}")
        return VisualEvaluationDataset(*rows)


def aggregate_visual_evaluation(
    dataset: VisualEvaluationDataset,
    *,
    split: str,
    ece_bins: int = 10,
) -> dict[str, object]:
    selected = dataset.subset(split)
    systems = sorted(
        {
            row.system
            for row in (*selected.properties, *selected.associations, *selected.queries)
        }
    )
    report: dict[str, object] = {
        "schema_version": 1,
        "split": split,
        "selection_policy": "thresholds and calibration must be frozen before test",
        "all_failures_retained": True,
        "cluster_disjoint_splits": True,
        "systems": {},
    }
    output = report["systems"]
    assert isinstance(output, dict)
    for system in systems:
        properties = tuple(row for row in selected.properties if row.system == system)
        associations = tuple(
            row for row in selected.associations if row.system == system
        )
        queries = tuple(row for row in selected.queries if row.system == system)
        output[system] = {
            "property": _property_metrics(properties),
            "association": _association_metrics(associations, ece_bins),
            "query": _query_metrics(queries, ece_bins),
            "association_risk_coverage": risk_coverage_curve(associations),
            "query_risk_coverage": risk_coverage_curve(queries),
            "slices": {
                "property_by_source": _grouped_metrics(
                    properties,
                    lambda row: row.source,
                    _property_metrics,
                ),
                "association_by_source": _grouped_metrics(
                    associations,
                    lambda row: row.source,
                    lambda rows: _association_metrics(rows, ece_bins),
                ),
                "query_by_source": _grouped_metrics(
                    queries,
                    lambda row: row.source,
                    lambda rows: _query_metrics(rows, ece_bins),
                ),
                "association_by_distractors": _grouped_metrics(
                    associations, lambda row: str(row.distractor_count),
                    lambda rows: _association_metrics(rows, ece_bins),
                ),
                "association_by_candidate_count": _grouped_metrics(
                    associations,
                    lambda row: str(len(row.candidate_entity_ids)),
                    lambda rows: _association_metrics(rows, ece_bins),
                ),
                "query_by_horizon_seconds": _grouped_metrics(
                    queries, lambda row: f"{row.horizon_seconds:g}",
                    lambda rows: _query_metrics(rows, ece_bins),
                ),
                "query_by_distractors": _grouped_metrics(
                    queries, lambda row: str(row.distractor_count),
                    lambda rows: _query_metrics(rows, ece_bins),
                ),
                "query_by_candidate_count": _grouped_metrics(
                    queries,
                    lambda row: str(len(row.candidate_entity_ids)),
                    lambda rows: _query_metrics(rows, ece_bins),
                ),
            },
            "population_hashes": {
                "property": _population_hash(properties),
                "association": _population_hash(associations),
                "query": _population_hash(queries),
            },
        }
    return report


def _grouped_metrics(
    rows: Sequence[object],
    key,
    metric,
) -> dict[str, object]:
    groups: dict[str, list[object]] = {}
    for row in rows:
        groups.setdefault(key(row), []).append(row)
    return {
        name: metric(tuple(groups[name]))
        for name in sorted(
            groups,
            key=lambda item: (
                0,
                float(item),
            )
            if _is_number(item)
            else (1, item),
        )
    }


def _property_metrics(rows: Sequence[VisualPropertyResult]) -> dict[str, object]:
    if not rows:
        return {"total": 0}
    expected = sum(row.expected for row in rows)
    detected = sum(row.detected for row in rows)
    true_detection = sum(row.expected and row.detected for row in rows)
    exact = sum(row.exact_value_match for row in rows)
    by_property: dict[str, dict[str, int]] = {}
    for row in rows:
        bucket = by_property.setdefault(
            row.property_name,
            {"total": 0, "expected": 0, "detected": 0, "exact": 0},
        )
        bucket["total"] += 1
        bucket["expected"] += int(row.expected)
        bucket["detected"] += int(row.detected)
        bucket["exact"] += int(row.exact_value_match)
    return {
        "total": len(rows),
        "expected": expected,
        "detected": detected,
        "true_detections": true_detection,
        "detection_precision": _divide(true_detection, detected),
        "detection_recall": _divide(true_detection, expected),
        "detection_f1": _f1(true_detection, detected, expected),
        "typed_exact_match": _divide(exact, expected),
        "duplicates": sum(row.duplicate_count for row in rows),
        "malformed": sum(row.malformed for row in rows),
        "by_property": by_property,
    }


def _association_metrics(
    rows: Sequence[VisualAssociationResult],
    bins: int,
) -> dict[str, object]:
    if not rows:
        return {"total": 0}
    accepted = sum(row.accepted for row in rows)
    correct = sum(row.correct_update for row in rows)
    false = sum(row.false_update for row in rows)
    target_present = sum(row.target_entity_id is not None for row in rows)
    target_absent = len(rows) - target_present
    false_null = sum(
        row.target_entity_id is not None and row.accepted_entity_id is None
        for row in rows
    )
    no_target_fp = sum(
        row.target_entity_id is None and row.accepted_entity_id is not None
        for row in rows
    )
    calibration = _calibration(rows, bins)
    combined_confidence = _combined_confidence_metrics(rows, bins)
    return {
        "total": len(rows),
        "target_present": target_present,
        "target_absent": target_absent,
        "accepted": accepted,
        "correct_updates": correct,
        "false_updates": false,
        "coverage": _divide(accepted, len(rows)),
        "correct_update_rate": _divide(correct, len(rows)),
        "false_update_rate": _divide(false, len(rows)),
        "selective_accuracy": _divide(correct, accepted),
        "target_recall": _divide(correct, target_present),
        "false_null_rate": _divide(false_null, target_present),
        "no_target_false_positive_rate": _divide(no_target_fp, target_absent),
        "malformed": sum(row.malformed for row in rows),
        "combined_update_confidence": combined_confidence,
        **calibration,
    }


def _query_metrics(
    rows: Sequence[VisualQueryResult],
    bins: int,
) -> dict[str, object]:
    if not rows:
        return {"total": 0}
    target_present = sum(row.target_entity_id is not None for row in rows)
    target_absent = len(rows) - target_present
    accepted = sum(row.accepted for row in rows)
    correct = sum(row.top1_correct for row in rows)
    no_target_fp = sum(
        row.target_entity_id is None and row.accepted_entity_id is not None
        for row in rows
    )
    calibration = _calibration(rows, bins)
    return {
        "total": len(rows),
        "target_present": target_present,
        "target_absent": target_absent,
        "candidate_recall": _divide(
            sum(
                row.target_entity_id is not None and row.target_candidate_present
                for row in rows
            ),
            target_present,
        ),
        "top1": _divide(correct, len(rows)),
        "mrr": sum(row.reciprocal_rank for row in rows) / len(rows),
        "coverage": _divide(accepted, len(rows)),
        "selective_top1": _divide(
            sum(row.top1_correct and row.accepted for row in rows),
            accepted,
        ),
        "no_target_false_positive_rate": _divide(no_target_fp, target_absent),
        "mean_latency_seconds": sum(row.latency_seconds for row in rows) / len(rows),
        "mean_vlm_calls": sum(row.vlm_calls for row in rows) / len(rows),
        "malformed": sum(row.malformed for row in rows),
        **calibration,
    }


def _calibration(
    rows: Sequence[VisualAssociationResult | VisualQueryResult],
    bins: int,
) -> dict[str, object]:
    if bins <= 0:
        raise ValueError("ECE bin count must be positive")
    confidences = [row.probabilities[row.decision_key] for row in rows]
    labels = [float(row.decision_key == row.truth_key) for row in rows]
    reliability: list[dict[str, float | int]] = []
    ece = 0.0
    for index in range(bins):
        lower = index / bins
        upper = (index + 1) / bins
        members = [
            position
            for position, confidence in enumerate(confidences)
            if lower <= confidence <= upper
            and (index == bins - 1 or confidence < upper)
        ]
        if not members:
            reliability.append(
                {
                    "lower": lower,
                    "upper": upper,
                    "count": 0,
                    "mean_confidence": 0.0,
                    "accuracy": 0.0,
                }
            )
            continue
        mean_confidence = sum(confidences[position] for position in members) / len(
            members
        )
        accuracy = sum(labels[position] for position in members) / len(members)
        ece += len(members) / len(rows) * abs(mean_confidence - accuracy)
        reliability.append(
            {
                "lower": lower,
                "upper": upper,
                "count": len(members),
                "mean_confidence": mean_confidence,
                "accuracy": accuracy,
            }
        )
    brier = 0.0
    nll = 0.0
    for row in rows:
        keys = set(row.probabilities)
        truth = row.truth_key
        brier += sum(
            (probability - float(key == truth)) ** 2
            for key, probability in row.probabilities.items()
        )
        if truth not in keys:
            brier += 1.0
        nll -= math.log(max(1e-12, row.probabilities.get(truth, 0.0)))
    return {
        "ece": ece,
        "brier": brier / len(rows),
        "nll": nll / len(rows),
        "reliability": reliability,
    }


def _combined_confidence_metrics(
    rows: Sequence[VisualAssociationResult],
    bins: int,
) -> dict[str, object]:
    selected = tuple(
        row
        for row in rows
        if row.decision_entity_id is not None
        and row.calibrated_update_confidence is not None
    )
    if not selected:
        return {"total": 0}
    labels = tuple(
        float(row.decision_entity_id == row.target_entity_id) for row in selected
    )
    raw = tuple(
        row.confidence_scale * row.probabilities[row.decision_entity_id]
        for row in selected
        if row.decision_entity_id is not None
    )
    calibrated = tuple(
        float(row.calibrated_update_confidence) for row in selected
    )
    return {
        "total": len(selected),
        "raw": _binary_calibration(raw, labels, bins),
        "calibrated": _binary_calibration(calibrated, labels, bins),
    }


def _binary_calibration(
    confidences: Sequence[float],
    labels: Sequence[float],
    bins: int,
) -> dict[str, object]:
    reliability: list[dict[str, float | int]] = []
    ece = 0.0
    for index in range(bins):
        lower = index / bins
        upper = (index + 1) / bins
        members = tuple(
            position
            for position, confidence in enumerate(confidences)
            if lower <= confidence <= upper
            and (index == bins - 1 or confidence < upper)
        )
        if not members:
            continue
        mean_confidence = sum(confidences[position] for position in members) / len(members)
        accuracy = sum(labels[position] for position in members) / len(members)
        ece += len(members) / len(confidences) * abs(mean_confidence - accuracy)
        reliability.append(
            {
                "lower": lower,
                "upper": upper,
                "count": len(members),
                "mean_confidence": mean_confidence,
                "accuracy": accuracy,
            }
        )
    return {
        "ece": ece,
        "brier": sum(
            (confidence - label) ** 2
            for confidence, label in zip(confidences, labels)
        )
        / len(confidences),
        "nll": -sum(
            label * math.log(max(1e-12, confidence))
            + (1.0 - label) * math.log(max(1e-12, 1.0 - confidence))
            for confidence, label in zip(confidences, labels)
        )
        / len(confidences),
        "reliability": reliability,
    }


def _combined_confidence_metrics(
    rows: Sequence[VisualAssociationResult],
    bins: int,
) -> dict[str, object]:
    selected = tuple(
        row
        for row in rows
        if row.decision_entity_id is not None
        and row.calibrated_update_confidence is not None
    )
    if not selected:
        return {"total": 0}
    labels = tuple(
        float(row.decision_entity_id == row.target_entity_id) for row in selected
    )
    raw = tuple(
        row.confidence_scale * row.probabilities[row.decision_entity_id]
        for row in selected
        if row.decision_entity_id is not None
    )
    calibrated = tuple(float(row.calibrated_update_confidence) for row in selected)
    return {
        "total": len(selected),
        "raw": _binary_calibration(raw, labels, bins),
        "calibrated": _binary_calibration(calibrated, labels, bins),
    }


def _binary_calibration(
    confidences: Sequence[float],
    labels: Sequence[float],
    bins: int,
) -> dict[str, object]:
    reliability: list[dict[str, float | int]] = []
    ece = 0.0
    for index in range(bins):
        lower = index / bins
        upper = (index + 1) / bins
        members = tuple(
            position
            for position, confidence in enumerate(confidences)
            if lower <= confidence <= upper
            and (index == bins - 1 or confidence < upper)
        )
        if not members:
            continue
        mean_confidence = sum(confidences[position] for position in members) / len(members)
        accuracy = sum(labels[position] for position in members) / len(members)
        ece += len(members) / len(confidences) * abs(mean_confidence - accuracy)
        reliability.append(
            {
                "lower": lower,
                "upper": upper,
                "count": len(members),
                "mean_confidence": mean_confidence,
                "accuracy": accuracy,
            }
        )
    return {
        "ece": ece,
        "brier": sum(
            (confidence - label) ** 2
            for confidence, label in zip(confidences, labels)
        ) / len(confidences),
        "nll": -sum(
            label * math.log(max(1e-12, confidence))
            + (1.0 - label) * math.log(max(1e-12, 1.0 - confidence))
            for confidence, label in zip(confidences, labels)
        ) / len(confidences),
        "reliability": reliability,
    }


def risk_coverage_curve(
    rows: Sequence[VisualAssociationResult | VisualQueryResult],
) -> list[dict[str, float | int]]:
    accepted = sorted(
        (row for row in rows if row.accepted),
        key=lambda row: (
            -row.probabilities[row.decision_key],
            row.record_id,
        ),
    )
    curve: list[dict[str, float | int]] = [
        {"accepted": 0, "coverage": 0.0, "risk": 0.0, "threshold": 1.0}
    ]
    errors = 0
    for index, row in enumerate(accepted, start=1):
        correct = (
            row.correct_update
            if isinstance(row, VisualAssociationResult)
            else row.top1_correct
        )
        errors += int(not correct)
        curve.append(
            {
                "accepted": index,
                "coverage": _divide(index, len(rows)),
                "risk": errors / index,
                "threshold": row.probabilities[row.decision_key],
            }
        )
    return curve


def write_visual_results_jsonl(
    path: Path,
    dataset: VisualEvaluationDataset,
) -> None:
    rows: list[dict[str, object]] = []
    for record_type, records in (
        ("property", dataset.properties),
        ("association", dataset.associations),
        ("query", dataset.queries),
    ):
        for row in records:
            payload = asdict(row)
            payload["record_type"] = record_type
            rows.append(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def read_visual_results_jsonl(path: Path) -> VisualEvaluationDataset:
    buckets: dict[str, list[object]] = {
        "property": [],
        "association": [],
        "query": [],
    }
    classes = {
        "property": VisualPropertyResult,
        "association": VisualAssociationResult,
        "query": VisualQueryResult,
    }
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        raw = json.loads(line)
        if not isinstance(raw, dict):
            raise ValueError(f"line {line_number} must contain an object")
        record_type = raw.pop("record_type", None)
        if record_type not in classes:
            raise ValueError(f"line {line_number} has unknown record_type")
        for tuple_field in ("candidate_entity_ids", "ranked_entity_ids"):
            if tuple_field in raw:
                raw[tuple_field] = tuple(raw[tuple_field])
        buckets[record_type].append(classes[record_type](**raw))
    return VisualEvaluationDataset(
        tuple(buckets["property"]),
        tuple(buckets["association"]),
        tuple(buckets["query"]),
    )


def _identity_fields(row: object) -> None:
    for name in ("record_id", "cluster_id", "system", "source", "property_name"):
        _text(getattr(row, name), name)
    _split(getattr(row, "split"))


def _candidate_ids(values: Sequence[str]) -> tuple[str, ...]:
    result = tuple(values)
    if not result or len(result) != len(set(result)):
        raise ValueError("candidate_entity_ids must be nonempty and unique")
    for value in result:
        _text(value, "candidate_entity_id")
        if value == NULL_ENTITY:
            raise ValueError("null sentinel cannot be a real candidate")
    return result


def _distribution(
    probabilities: Mapping[str, float],
    candidates: Sequence[str],
) -> None:
    if set(probabilities) != {*candidates, NULL_ENTITY}:
        raise ValueError("probabilities must cover every candidate and null exactly")
    for value in probabilities.values():
        _probability(value, "association probability")
    if not math.isclose(sum(probabilities.values()), 1.0, abs_tol=1e-8):
        raise ValueError("association probabilities must sum to one")


def _population_hash(rows: Sequence[object]) -> str | None:
    if not rows:
        return None
    identifiers = sorted(f"{row.cluster_id}\0{row.record_id}" for row in rows)
    return hashlib.sha256("\n".join(identifiers).encode("utf-8")).hexdigest()


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _is_number(value: str) -> bool:
    try:
        float(value)
    except ValueError:
        return False
    return True


def _json_value(value: object) -> None:
    try:
        _canonical(value)
    except (TypeError, ValueError) as error:
        raise ValueError("typed values must be JSON serializable") from error


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _split(value: str) -> None:
    if value not in VALID_SPLITS:
        raise ValueError(f"unknown visual evaluation split: {value}")


def _probability(value: float, name: str) -> None:
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1]")


def _divide(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _f1(true_positive: int, predicted: int, expected: int) -> float | None:
    denominator = predicted + expected
    return 2.0 * true_positive / denominator if denominator else None
