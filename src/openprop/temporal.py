from __future__ import annotations

from dataclasses import dataclass

from .models import EntityEvent, Observation, PropertyDefinition


@dataclass(frozen=True, slots=True)
class FreshnessResult:
    freshness: float
    age_seconds: float | None
    time_retention: float
    event_retention: float
    applied_events: tuple[str, ...] = ()

    @property
    def reason(self) -> str:
        if self.age_seconds is None:
            return "freshness=1.000 (no temporal timestamp/policy)"
        events = ",".join(self.applied_events) if self.applied_events else "none"
        return (
            f"freshness={self.freshness:.3f}, age={self.age_seconds:.0f}s, "
            f"time={self.time_retention:.3f}, event={self.event_retention:.3f}, "
            f"events={events}"
        )


def observation_freshness(
    definition: PropertyDefinition,
    observation: Observation,
    events: list[EntityEvent],
    *,
    as_of: float,
) -> FreshnessResult:
    policy = definition.temporal_policy
    if policy is None or observation.timestamp is None:
        return FreshnessResult(1.0, None, 1.0, 1.0)

    age_seconds = max(0.0, as_of - observation.timestamp)
    if policy.half_life_seconds is None:
        time_retention = 1.0
    else:
        time_retention = 2.0 ** (-age_seconds / policy.half_life_seconds)
    time_retention = max(policy.minimum_freshness, time_retention)

    retention_by_event = {
        event_type.casefold(): retention
        for event_type, retention in policy.event_retention.items()
    }
    event_retention = 1.0
    applied_events: list[str] = []
    for event in sorted(events, key=lambda item: item.timestamp):
        if event.timestamp <= observation.timestamp or event.timestamp > as_of:
            continue
        retention = retention_by_event.get(event.event_type.casefold())
        if retention is None:
            continue
        # An uncertain event interpolates between no effect and full effect.
        event_retention *= 1.0 - event.confidence * (1.0 - retention)
        applied_events.append(event.event_type)

    freshness = max(0.0, min(1.0, time_retention * event_retention))
    return FreshnessResult(
        freshness,
        age_seconds,
        time_retention,
        event_retention,
        tuple(applied_events),
    )
