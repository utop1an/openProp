import os

from openprop import (
    Entity,
    EntityMatcher,
    LLMQueryParser,
    MentionBasedSelector,
    Observation,
    OpenAIResponsesClient,
    PropertyDefinition,
    PropertyRegistry,
    RelationValue,
    ValueType,
    default_comparators,
)


def main() -> None:
    model = os.environ.get("OPENAI_MODEL")
    if not model:
        raise SystemExit("Set OPENAI_MODEL to a Structured Outputs-capable model.")

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
    registry.register(
        PropertyDefinition("temperature", "estimated surface temperature", ValueType.NUMERIC)
    )

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
            "cup_blue",
            {
                "type": Observation("cup"),
                "color": Observation("blue"),
                "location": Observation(RelationValue("on", {"object": "table"})),
            },
        ),
    ]

    parser = LLMQueryParser(OpenAIResponsesClient(model=model))
    parsed = parser.parse("桌上的红色杯子", registry)
    matcher = EntityMatcher(registry, default_comparators(), MentionBasedSelector())

    print("LLM constraints:")
    for constraint in parsed.frame.constraints:
        print(
            f"  {constraint.property_name}: {constraint.desired_value!r} "
            f"(relevance={constraint.relevance:.2f})"
        )
    print("Ranking:")
    for result in matcher.match(parsed.frame, entities):
        print(f"  {result.entity_id}: {result.score:.3f}")


if __name__ == "__main__":
    main()
