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


def build_registry() -> PropertyRegistry:
    registry = PropertyRegistry()
    registry.register(PropertyDefinition("type", "semantic object category", ValueType.SEMANTIC))
    registry.register(PropertyDefinition("color", "perceived surface color", ValueType.SEMANTIC))
    registry.register(
        PropertyDefinition(
            "location",
            "spatial relation between an entity and another entity",
            ValueType.RELATION,
            aliases=("position relation", "spatial relation"),
        )
    )
    return registry


def main() -> None:
    registry = build_registry()
    entities = [
        Entity(
            "cup_red",
            {
                "type": Observation("cup"),
                "color": Observation("red"),
                "location": Observation(RelationValue("on", {"object": "table"})),
            },
        ),
        Entity(
            "cup_unknown_color",
            {
                "type": Observation("cup"),
                "color": Observation.unknown(source="camera occlusion"),
                "location": Observation(RelationValue("on", {"object": "table"})),
            },
        ),
        Entity(
            "blue_bowl",
            {
                "type": Observation("bowl"),
                "color": Observation("blue"),
                "location": Observation(RelationValue("on", {"object": "table"})),
            },
        ),
    ]
    query = QueryFrame(
        "桌上的红色杯子",
        (
            PropertyConstraint("type", "cup", relevance=0.95),
            PropertyConstraint("color", "red", relevance=0.93),
            PropertyConstraint(
                "spatial relation",
                RelationValue("on", {"object": "table"}),
                relevance=0.97,
            ),
        ),
    )
    matcher = EntityMatcher(registry, default_comparators(), MentionBasedSelector())
    for result in matcher.match(query, entities):
        print(
            f"{result.entity_id:20} score={result.score:.3f} "
            f"match={result.match_score:.3f} coverage={result.coverage:.3f}"
        )


if __name__ == "__main__":
    main()

