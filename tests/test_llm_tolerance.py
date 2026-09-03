import unittest

from openprop.llm import LLMQueryParser
from openprop.models import PropertyDefinition, ValueType
from openprop.property_registry import PropertyRegistry


class FakeClient:
    def generate_json(self, **_):
        empty = {"predicate": None, "arguments": [], "vector": []}
        return {
            "constraints": [
                {
                    "property_name": "color",
                    "description": "color",
                    "value_type": "semantic",
                    "known_property": True,
                    "relevance": 0.9,
                    "tolerance": None,
                    "value": {"kind": "scalar", "scalar": "red", **empty},
                },
                {
                    "property_name": "size",
                    "description": "size",
                    "value_type": "numeric",
                    "known_property": True,
                    "relevance": 0.5,
                    "tolerance": None,
                    "value": {"kind": "scalar", "scalar": "large", **empty},
                },
            ]
        }


class TolerantParserTests(unittest.TestCase):
    def setUp(self):
        self.registry = PropertyRegistry()
        self.registry.register(PropertyDefinition("color", "color", ValueType.SEMANTIC))
        self.registry.register(PropertyDefinition("size", "size", ValueType.NUMERIC))

    def test_skips_invalid_constraint_and_reports_it(self):
        parsed = LLMQueryParser(
            FakeClient(), skip_invalid_constraints=True
        ).parse("large red object", self.registry)
        self.assertEqual([item.property_name for item in parsed.frame.constraints], ["color"])
        self.assertEqual(len(parsed.validation_errors), 1)
        self.assertIn("size", parsed.validation_errors[0])

    def test_strict_mode_still_rejects_invalid_constraint(self):
        with self.assertRaisesRegex(Exception, "number"):
            LLMQueryParser(FakeClient()).parse("large red object", self.registry)


if __name__ == "__main__":
    unittest.main()
