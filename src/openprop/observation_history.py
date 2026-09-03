from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from .persistence_data import PersistenceTrainingExample


@dataclass(frozen=True, slots=True)
class ObservationHistoryRecord:
    record_id: str
    entity_id: str
    property_name: str
    subject_type: str
    state_predicate: str
    context_object: str
    scene: str
    observed_at: float
    followup_at: float
    state_changed: bool
    source: str | None = None
    observation_confidence: float = 1.0
    last_confirmed_at: float | None = None

    def __post_init__(self) -> None:
        if not self.record_id.strip() or not self.entity_id.strip():
            raise ValueError("record_id and entity_id cannot be empty")
        if self.followup_at < self.observed_at:
            raise ValueError("followup_at cannot precede observed_at")
        if not 0.0 <= self.observation_confidence <= 1.0:
            raise ValueError("observation_confidence must be between 0 and 1")
        if self.last_confirmed_at is not None:
            if not self.state_changed:
                raise ValueError("last_confirmed_at is only used for detected state changes")
            if not self.observed_at <= self.last_confirmed_at < self.followup_at:
                raise ValueError(
                    "last_confirmed_at must fall between observed_at and followup_at"
                )

    def to_training_example(self) -> PersistenceTrainingExample:
        return PersistenceTrainingExample(
            property_name=self.property_name,
            subject_type=self.subject_type,
            state_predicate=self.state_predicate,
            context_object=self.context_object,
            scene=self.scene,
            duration_seconds=self.followup_at - self.observed_at,
            event_observed=self.state_changed,
            group_id=self.entity_id,
            interval_start_seconds=(
                None
                if self.last_confirmed_at is None
                else self.last_confirmed_at - self.observed_at
            ),
        )


@dataclass(frozen=True, slots=True)
class PersistenceDataSplit:
    train: tuple[PersistenceTrainingExample, ...]
    validation: tuple[PersistenceTrainingExample, ...]
    test: tuple[PersistenceTrainingExample, ...]


def write_history_jsonl(
    path: str | Path,
    records: Iterable[ObservationHistoryRecord],
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(asdict(record), ensure_ascii=False, separators=(",", ":")))
            stream.write("\n")


def read_history_jsonl(path: str | Path) -> tuple[ObservationHistoryRecord, ...]:
    records: list[ObservationHistoryRecord] = []
    seen_ids: set[str] = set()
    with Path(path).open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
                record = ObservationHistoryRecord(**payload)
            except (json.JSONDecodeError, TypeError, ValueError) as error:
                raise ValueError(f"invalid history record at line {line_number}: {error}") from error
            if record.record_id in seen_ids:
                raise ValueError(f"duplicate record_id at line {line_number}: {record.record_id}")
            seen_ids.add(record.record_id)
            records.append(record)
    return tuple(records)


def history_to_examples(
    records: Iterable[ObservationHistoryRecord],
) -> tuple[PersistenceTrainingExample, ...]:
    return tuple(record.to_training_example() for record in records)


def grouped_split(
    examples: Iterable[PersistenceTrainingExample],
    *,
    train_fraction: float = 0.7,
    validation_fraction: float = 0.15,
    seed: int = 23,
) -> PersistenceDataSplit:
    if not 0 < train_fraction < 1 or not 0 < validation_fraction < 1:
        raise ValueError("split fractions must be between zero and one")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("train and validation fractions must leave a test split")
    rows = tuple(examples)
    if not rows:
        raise ValueError("cannot split an empty dataset")

    grouped: dict[str, list[PersistenceTrainingExample]] = {}
    for index, example in enumerate(rows):
        group = example.group_id or f"__ungrouped_{index}"
        grouped.setdefault(group, []).append(example)
    groups = list(grouped)
    if len(groups) < 3:
        raise ValueError("at least three entity groups are required")
    random.Random(seed).shuffle(groups)
    train_end = max(1, int(len(groups) * train_fraction))
    validation_end = max(train_end + 1, int(len(groups) * (train_fraction + validation_fraction)))
    validation_end = min(validation_end, len(groups) - 1)

    def collect(keys: list[str]) -> tuple[PersistenceTrainingExample, ...]:
        return tuple(example for key in keys for example in grouped[key])

    return PersistenceDataSplit(
        collect(groups[:train_end]),
        collect(groups[train_end:validation_end]),
        collect(groups[validation_end:]),
    )
