from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PersistenceTrainingExample:
    property_name: str
    subject_type: str
    state_predicate: str
    context_object: str
    scene: str
    duration_seconds: float
    event_observed: bool
    group_id: str = ""
    interval_start_seconds: float | None = None

    def __post_init__(self) -> None:
        if self.duration_seconds < 0:
            raise ValueError("duration_seconds cannot be negative")
        if self.interval_start_seconds is not None:
            if not self.event_observed:
                raise ValueError("only observed events can be interval censored")
            if not 0 <= self.interval_start_seconds < self.duration_seconds:
                raise ValueError(
                    "interval_start_seconds must be non-negative and precede duration_seconds"
                )

    @property
    def is_interval_censored(self) -> bool:
        """Whether the transition is known only within an observation interval."""

        return self.event_observed and self.interval_start_seconds is not None

    def features(self) -> tuple[str, ...]:
        return (
            self.property_name,
            self.subject_type,
            self.state_predicate,
            self.context_object,
            self.scene,
        )
