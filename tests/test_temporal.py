import unittest

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
from openprop.temporal import observation_freshness


HOUR = 60 * 60
DAY = 24 * HOUR
NOW = 1_000_000.0


class TemporalEvidenceTests(unittest.TestCase):
    def matcher(self, definition):
        registry = PropertyRegistry()
        registry.register(definition)
        return EntityMatcher(registry, default_comparators(), MentionBasedSelector())

    def test_five_hour_old_location_has_lower_coverage(self):
        definition = PropertyDefinition(
            "location",
            "current spatial relation",
            ValueType.RELATION,
            temporal_policy=TemporalPolicy(half_life_seconds=2 * HOUR),
        )
        value = RelationValue("on", {"object": "table"})
        current = Entity("current", {"location": Observation(value, timestamp=NOW)})
        stale = Entity("stale", {"location": Observation(value, timestamp=NOW - 5 * HOUR)})
        query = QueryFrame("cup on table", (PropertyConstraint("location", value),))

        results = self.matcher(definition).match(query, [stale, current], as_of=NOW)
        self.assertEqual(results[0].entity_id, "current")
        self.assertAlmostEqual(results[1].coverage, 2 ** (-2.5))
        self.assertAlmostEqual(results[1].match_score, 1.0)
        self.assertAlmostEqual(results[1].evidence[0].age_seconds, 5 * HOUR)

    def test_three_day_old_clean_state_decays(self):
        definition = PropertyDefinition(
            "cleanliness",
            "whether clothing is clean",
            ValueType.CATEGORICAL,
            temporal_policy=TemporalPolicy(
                half_life_seconds=7 * DAY,
                event_retention={"worn": 0.1},
            ),
        )
        observation = Observation("clean", timestamp=NOW - 3 * DAY)
        result = observation_freshness(definition, observation, [], as_of=NOW)
        self.assertAlmostEqual(result.freshness, 2 ** (-3 / 7))

    def test_worn_event_further_invalidates_old_clean_state(self):
        definition = PropertyDefinition(
            "cleanliness",
            "whether clothing is clean",
            ValueType.CATEGORICAL,
            temporal_policy=TemporalPolicy(
                half_life_seconds=7 * DAY,
                event_retention={"worn": 0.1},
            ),
        )
        observation = Observation("clean", timestamp=NOW - 3 * DAY)
        worn = EntityEvent("worn", timestamp=NOW - DAY)
        result = observation_freshness(definition, observation, [worn], as_of=NOW)
        self.assertAlmostEqual(result.freshness, (2 ** (-3 / 7)) * 0.1)
        self.assertEqual(result.applied_events, ("worn",))

    def test_event_before_observation_does_not_invalidate_it(self):
        definition = PropertyDefinition(
            "cleanliness",
            "whether clothing is clean",
            ValueType.CATEGORICAL,
            temporal_policy=TemporalPolicy(event_retention={"worn": 0.1}),
        )
        observation = Observation("clean", timestamp=NOW - DAY)
        old_event = EntityEvent("worn", timestamp=NOW - 2 * DAY)
        result = observation_freshness(definition, observation, [old_event], as_of=NOW)
        self.assertEqual(result.freshness, 1.0)
        self.assertFalse(result.applied_events)

    def test_property_without_policy_preserves_legacy_confidence(self):
        definition = PropertyDefinition("type", "object type", ValueType.SEMANTIC)
        observation = Observation("cup", timestamp=NOW - 365 * DAY)
        result = observation_freshness(definition, observation, [], as_of=NOW)
        self.assertEqual(result.freshness, 1.0)


if __name__ == "__main__":
    unittest.main()
