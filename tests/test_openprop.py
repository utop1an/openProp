import unittest

from openprop import (
    Entity,
    EntityMatcher,
    MentionBasedSelector,
    Observation,
    PropertyConstraint,
    PropertyDefinition,
    PropertyRegistry,
    QueryFrame,
    RelationValue,
    ValueType,
    default_comparators,
)


class OpenPropTests(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = PropertyRegistry()
        self.registry.register(PropertyDefinition("type", "object category", ValueType.SEMANTIC))
        self.registry.register(PropertyDefinition("color", "surface color", ValueType.SEMANTIC))
        self.registry.register(
            PropertyDefinition(
                "location", "spatial relation", ValueType.RELATION, aliases=("spatial relation",)
            )
        )
        self.matcher = EntityMatcher(
            self.registry, default_comparators(), MentionBasedSelector()
        )

    def query(self) -> QueryFrame:
        return QueryFrame(
            "red cup on table",
            (
                PropertyConstraint("type", "cup", 0.95),
                PropertyConstraint("color", "red", 0.93),
                PropertyConstraint(
                    "spatial relation", RelationValue("on", {"object": "table"}), 0.97
                ),
            ),
        )

    def test_complete_match_ranks_first(self) -> None:
        match = Entity(
            "match",
            {
                "type": Observation("cup"),
                "color": Observation("red"),
                "location": Observation(RelationValue("on", {"object": "table"})),
            },
        )
        mismatch = Entity(
            "mismatch",
            {
                "type": Observation("bowl"),
                "color": Observation("blue"),
                "location": Observation(RelationValue("on", {"object": "table"})),
            },
        )
        results = self.matcher.match(self.query(), [mismatch, match])
        self.assertEqual(results[0].entity_id, "match")
        self.assertAlmostEqual(results[0].score, 1.0)

    def test_unknown_is_missing_evidence_not_mismatch(self) -> None:
        entity = Entity(
            "partially-observed",
            {
                "type": Observation("cup"),
                "color": Observation.unknown(),
                "location": Observation(RelationValue("on", {"object": "table"})),
            },
        )
        result = self.matcher.match(self.query(), [entity])[0]
        color = next(item for item in result.evidence if item.property_name == "color")
        self.assertIsNone(color.score)
        self.assertEqual(result.match_score, 1.0)
        self.assertLess(result.coverage, 1.0)
        self.assertLess(result.score, result.match_score)

    def test_registry_reuses_alias_and_adds_new_property(self) -> None:
        alias = self.registry.resolve("spatial relation")
        self.assertEqual(alias.definition.name, "location")
        new_definition = PropertyDefinition("temperature", "surface temperature", ValueType.NUMERIC)
        resolution = self.registry.resolve_or_register(new_definition)
        self.assertTrue(resolution.created)
        self.assertIs(self.registry.get("temperature"), new_definition)

    def test_numeric_comparator_uses_tolerance(self) -> None:
        self.registry.register(PropertyDefinition("temperature", "temperature", ValueType.NUMERIC))
        query = QueryFrame("warm object", (PropertyConstraint("temperature", 40, tolerance=10),))
        entity = Entity("warm", {"temperature": Observation(42)})
        result = self.matcher.match(query, [entity])[0]
        self.assertGreater(result.score, 0.8)


if __name__ == "__main__":
    unittest.main()

