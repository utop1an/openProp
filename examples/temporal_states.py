from openprop.comparators import default_comparators
from openprop.matcher import EntityMatcher
from openprop.models import (
    Entity,
    EntityEvent,
    Observation,
    PropertyConstraint,
    PropertyDefinition,
    QueryFrame,
    RelationValue,
    TemporalPolicy,
    ValueType,
)
from openprop.property_registry import PropertyRegistry
from openprop.selectors import MentionBasedSelector


HOUR = 60 * 60
DAY = 24 * HOUR
NOW = 1_000_000.0


def show(title, result):
    evidence = result.evidence[0]
    print(
        f"{title:28} score={result.score:.3f} "
        f"freshness={evidence.freshness:.3f} "
        f"confidence={evidence.effective_confidence:.3f}"
    )


def main() -> None:
    registry = PropertyRegistry()
    registry.register(
        PropertyDefinition(
            "location",
            "current spatial relation",
            ValueType.RELATION,
            temporal_policy=TemporalPolicy(half_life_seconds=2 * HOUR),
        )
    )
    registry.register(
        PropertyDefinition(
            "cleanliness",
            "whether clothing is clean",
            ValueType.CATEGORICAL,
            temporal_policy=TemporalPolicy(
                half_life_seconds=7 * DAY,
                event_retention={"worn": 0.1},
            ),
        )
    )
    matcher = EntityMatcher(registry, default_comparators(), MentionBasedSelector())

    on_table = RelationValue("on", {"object": "table"})
    old_cup = Entity(
        "old_cup",
        {"location": Observation(on_table, timestamp=NOW - 5 * HOUR)},
    )
    location_query = QueryFrame(
        "the cup on the table",
        (PropertyConstraint("location", on_table),),
    )
    show("cup location, 5 hours old", matcher.match(location_query, [old_cup], as_of=NOW)[0])

    clean_observation = Observation("clean", timestamp=NOW - 3 * DAY)
    unworn = Entity("unworn_shirt", {"cleanliness": clean_observation})
    worn = Entity(
        "worn_shirt",
        {"cleanliness": clean_observation},
        [EntityEvent("worn", timestamp=NOW - DAY)],
    )
    clean_query = QueryFrame(
        "the clean shirt",
        (PropertyConstraint("cleanliness", "clean"),),
    )
    clean_results = matcher.match(clean_query, [worn, unworn], as_of=NOW)
    by_id = {result.entity_id: result for result in clean_results}
    show("clean, observed 3 days ago", by_id["unworn_shirt"])
    show("clean, then worn", by_id["worn_shirt"])


if __name__ == "__main__":
    main()
