import json
import unittest
from types import SimpleNamespace

from openprop.llm import LLMError, LLMQueryParser, OpenAIResponsesClient
from openprop.models import PropertyDefinition, RelationValue, ValueType
from openprop.property_registry import PropertyRegistry


def scalar(value):
    return {
        "kind": "scalar",
        "scalar": value,
        "predicate": None,
        "arguments": [],
        "vector": [],
    }


def relation(predicate, **arguments):
    return {
        "kind": "relation",
        "scalar": None,
        "predicate": predicate,
        "arguments": [
            {"role": role, "value": value} for role, value in arguments.items()
        ],
        "vector": [],
    }


def constraint(name, value, *, value_type="semantic", relevance=0.9, description="test"):
    return {
        "property_name": name,
        "description": description,
        "value_type": value_type,
        "known_property": True,
        "relevance": relevance,
        "tolerance": None,
        "value": value,
    }


class FakeJSONClient:
    def __init__(self, response):
        self.response = response
        self.last_call = None

    def generate_json(self, **kwargs):
        self.last_call = kwargs
        return self.response


class FakeResponses:
    def __init__(self, output):
        self.output = output
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(output_text=json.dumps(self.output))


class LLMQueryParserTests(unittest.TestCase):
    def setUp(self):
        self.registry = PropertyRegistry()
        self.registry.register(PropertyDefinition("type", "object category", ValueType.SEMANTIC))
        self.registry.register(PropertyDefinition("color", "surface color", ValueType.SEMANTIC))
        self.registry.register(
            PropertyDefinition(
                "location",
                "spatial relation",
                ValueType.RELATION,
                aliases=("spatial relation",),
            )
        )

    def test_parses_alias_relation_and_weights(self):
        client = FakeJSONClient(
            {
                "constraints": [
                    constraint("type", scalar("cup"), relevance=0.95),
                    constraint("color", scalar("red"), relevance=0.93),
                    constraint(
                        "spatial relation",
                        relation("on", object="table"),
                        value_type="relation",
                        relevance=0.97,
                    ),
                ]
            }
        )
        parsed = LLMQueryParser(client).parse("桌上的红色杯子", self.registry)
        self.assertEqual([item.property_name for item in parsed.frame.constraints], ["type", "color", "location"])
        self.assertEqual(
            parsed.frame.constraints[-1].desired_value,
            RelationValue("on", {"object": "table"}),
        )
        request = json.loads(client.last_call["input_text"])
        self.assertEqual(request["query"], "桌上的红色杯子")
        self.assertEqual(len(request["property_dictionary"]), 3)

    def test_unknown_property_is_ignored_by_default(self):
        response = {
            "constraints": [
                constraint("color", scalar("red")),
                constraint("material", scalar("ceramic"), description="object material"),
            ]
        }
        parsed = LLMQueryParser(FakeJSONClient(response)).parse("red ceramic", self.registry)
        self.assertEqual(parsed.ignored_properties, ("material",))
        self.assertIsNone(self.registry.get("material"))

    def test_unknown_property_can_be_registered_explicitly(self):
        item = constraint("material", scalar("ceramic"), description="object material")
        item["known_property"] = False
        parsed = LLMQueryParser(
            FakeJSONClient({"constraints": [item]}), allow_property_creation=True
        ).parse("ceramic object", self.registry)
        self.assertEqual(parsed.created_properties[0].name, "material")
        self.assertIsNotNone(self.registry.get("material"))

    def test_duplicate_constraint_is_rejected(self):
        response = {
            "constraints": [
                constraint("color", scalar("red")),
                constraint("color", scalar("crimson")),
            ]
        }
        with self.assertRaisesRegex(LLMError, "duplicate"):
            LLMQueryParser(FakeJSONClient(response)).parse("red", self.registry)

    def test_openai_adapter_uses_strict_schema_and_no_storage(self):
        responses = FakeResponses({"constraints": []})
        sdk = SimpleNamespace(responses=responses)
        client = OpenAIResponsesClient(model="test-model", client=sdk)
        result = client.generate_json(
            instructions="instructions",
            input_text="input",
            schema_name="test_schema",
            schema={"type": "object"},
        )
        self.assertEqual(result, {"constraints": []})
        self.assertFalse(responses.kwargs["store"])
        self.assertTrue(responses.kwargs["text"]["format"]["strict"])


if __name__ == "__main__":
    unittest.main()
