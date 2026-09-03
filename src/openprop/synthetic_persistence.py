from __future__ import annotations

import random

from .persistence_data import PersistenceTrainingExample


def contextual_location_data(
    *,
    samples_per_context: int = 300,
    censor_after_hours: float = 12.0,
    seed: int = 17,
) -> tuple[PersistenceTrainingExample, ...]:
    """Synthetic censored traces with different table/cabinet hazards.

    The rates are intentionally far apart so the experiment tests whether the
    learning path can recover context dependence, not whether these numbers are
    realistic estimates.
    """
    if samples_per_context <= 0 or censor_after_hours <= 0:
        raise ValueError("sample count and censor horizon must be positive")
    rng = random.Random(seed)
    contexts = (
        ("on", "table", 0.50),
        ("inside", "cabinet", 0.05),
    )
    examples: list[PersistenceTrainingExample] = []
    for predicate, context_object, hazard_per_hour in contexts:
        for index in range(samples_per_context):
            transition_time = rng.expovariate(hazard_per_hour)
            censor_time = rng.uniform(0.5, censor_after_hours)
            event_observed = transition_time <= censor_time
            duration_hours = min(transition_time, censor_time)
            examples.append(
                PersistenceTrainingExample(
                    property_name="location",
                    subject_type="cup",
                    state_predicate=predicate,
                    context_object=context_object,
                    scene="kitchen",
                    duration_seconds=duration_hours * 3600.0,
                    event_observed=event_observed,
                    group_id=f"cup-{predicate}-{context_object}-{index:04d}",
                )
            )
    rng.shuffle(examples)
    return tuple(examples)
