import json
import unittest

from openprop.llm import LLMQueryParser
from openprop.models import PropertyDefinition, ValueType
from openprop.property_registry import PropertyRegistry


class LLMMetadataTests(unittest.TestCase):
    def test_property_metadata_is_sent_to_llm(self):
        registry = PropertyRegistry()
        registry.register(
            PropertyDefinition(
                "location",
                "spatial relation",
                ValueType.RELATION,
                metadata={"argument_roles": ["object"]},
            )
        )
        payload = json.loads(LLMQueryParser._input("on the table", registry))
        self.assertEqual(
            payload["property_dictionary"][0]["metadata"],
            {"argument_roles": ["object"]},
        )


if __name__ == "__main__":
    unittest.main()
