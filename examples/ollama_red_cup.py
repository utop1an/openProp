import os

from openprop import (
    Entity,
    EntityMatcher,
    LLMQueryParser,
    MentionBasedSelector,
    Observation,
    OllamaClient,
    PropertyDefinition,
    PropertyRegistry,
    RelationValue,
    ValueType,
    default_comparators,
)


def main() -> None:
    model = os.environ.get("OLLAMA_MODEL", "gemma3:4b")
    base_url = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")

    registry = PropertyRegistry()
    registry.register(PropertyDefinition("type", "semantic object category", ValueType.SEMANTIC))
    registry.register(PropertyDefinition("color", "perceived surface color", ValueType.SEMANTIC))
    registry.register(
        PropertyDefinition(
            "location",
            "spatial relation between an entity and another entity",
            ValueType.RELATION,
            aliases=("position relation", "spatial relation"),
            metadata={"argument_roles": ["object"]},
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

    parser = LLMQueryParser(OllamaClient(model=model, base_url=base_url))
    parsed = parser.parse("桌上的红色杯子", registry)
    matcher = EntityMatcher(registry, default_comparators(), MentionBasedSelector())

    print(f"Ollama model: {model}")
    print("LLM constraints:")
    for constraint in parsed.frame.constraints:
        print(
            f"  {constraint.property_name}: {constraint.desired_value!r} "
            f"(relevance={constraint.relevance:.2f})"
        )
    if parsed.ignored_properties:
        print(f"Ignored properties: {', '.join(parsed.ignored_properties)}")
    print("Ranking:")
    for result in matcher.match(parsed.frame, entities):
        print(f"  {result.entity_id}: {result.score:.3f}")


if __name__ == "__main__":
    main()
