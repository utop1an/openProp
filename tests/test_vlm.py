import json
import unittest
from types import SimpleNamespace

from openprop import (
    EntityMatcher,
    EntityObservationLedger,
    MentionBasedSelector,
    OpenAIResponsesVLMClient,
    ObservationState,
    PropertyConstraint,
    PropertyDefinition,
    PropertyUpdatePolicy,
    PropertyRegistry,
    QueryFrame,
    VLMError,
    VLMPropertyUpdater,
    ValueType,
    VisualFrame,
    default_comparators,
)


def scalar(value):
    return {
        "kind": "scalar",
        "scalar": value,
        "predicate": None,
        "arguments": [],
        "vector": [],
    }


def update(name, value, **overrides):
    item = {
        "frame_id": "f1",
        "entity_id": "cup-1",
        "property_name": name,
        "description": "visual property",
        "value_type": "semantic",
        "known_property": True,
        "state": "observed",
        "confidence": 0.9,
        "value": scalar(value),
    }
    item.update(overrides)
    return item


class FakeVLMClient:
    def __init__(self, response):
        self.response = response
        self.last_call = None

    def generate_json(self, **kwargs):
        self.last_call = kwargs
        return self.response


class FakeResponses:
    def __init__(self, payload):
        self.payload = payload
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(output_text=json.dumps(self.payload))


class VLMPropertyUpdaterTests(unittest.TestCase):
    def setUp(self):
        self.registry = PropertyRegistry()
        self.registry.register(
            PropertyDefinition(
                "color",
                "surface color",
                ValueType.SEMANTIC,
                metadata={"allowed_values": ["red", "blue"]},
                update_policy=PropertyUpdatePolicy(
                    minimum_confidence=0.7,
                    allowed_sources=("camera-1",),
                ),
            )
        )
        self.registry.register(
            PropertyDefinition(
                "identity",
                "stable identity",
                ValueType.ENTITY_REFERENCE,
                update_policy=PropertyUpdatePolicy(allow_visual_updates=False),
            )
        )
        self.frame = VisualFrame(
            "f1", "data:image/png;base64,AA==", 100.0, "camera-1", ("cup-1", "cup-2")
        )

    def test_proposal_uses_trusted_frame_provenance_and_registry_type(self):
        client = FakeVLMClient({"updates": [update("color", "red")]})
        parsed = VLMPropertyUpdater(client).update([self.frame], self.registry)

        proposal = parsed.proposals[0]
        self.assertEqual(proposal.property_name, "color")
        self.assertEqual(proposal.observation.value, "red")
        self.assertEqual(proposal.observation.source, "camera-1")
        self.assertEqual(proposal.observation.timestamp, 100.0)
        request = json.loads(client.last_call["input_text"])
        self.assertNotIn("source", request["visual_history"][0])
        self.assertNotIn("captured_at", request["visual_history"][0])
        self.assertEqual(client.last_call["image_urls"], (self.frame.image_url,))

    def test_policy_filters_low_confidence_and_non_visual_properties(self):
        response = {"updates": [
            update("color", "red", confidence=0.4),
            update(
                "identity",
                "cup-1",
                value_type="entity_reference",
                confidence=0.99,
            ),
        ]}
        parsed = VLMPropertyUpdater(FakeVLMClient(response)).parse_response(
            [self.frame], self.registry, response
        )
        self.assertFalse(parsed.proposals)
        self.assertEqual(parsed.ignored_properties, ("color", "identity"))

    def test_unknown_is_missing_evidence_with_no_value(self):
        response = {"updates": [update(
            "color",
            None,
            state="unknown",
            confidence=0.95,
        )]}
        parsed = VLMPropertyUpdater(FakeVLMClient(response)).parse_response(
            [self.frame], self.registry, response
        )
        observation = parsed.proposals[0].observation
        self.assertIs(observation.state, ObservationState.UNKNOWN)
        self.assertEqual(observation.confidence, 0.0)
        self.assertIsNone(observation.value)

    def test_wrong_entity_type_and_policy_value_are_rejected(self):
        wrong_entity = {"updates": [update("color", "red", entity_id="other")]}
        with self.assertRaisesRegex(VLMError, "not a candidate"):
            VLMPropertyUpdater(FakeVLMClient(wrong_entity)).parse_response(
                [self.frame], self.registry, wrong_entity
            )
        wrong_type = {"updates": [update("color", "red", value_type="numeric")]}
        with self.assertRaisesRegex(VLMError, "expects semantic"):
            VLMPropertyUpdater(FakeVLMClient(wrong_type)).parse_response(
                [self.frame], self.registry, wrong_type
            )
        wrong_value = {"updates": [update("color", "green")]}
        with self.assertRaisesRegex(VLMError, "allowed_values"):
            VLMPropertyUpdater(FakeVLMClient(wrong_value)).parse_response(
                [self.frame], self.registry, wrong_value
            )

    def test_ledger_preserves_history_and_materializes_as_of_snapshots(self):
        first = {"updates": [update("color", "red")]}
        second_frame = VisualFrame(
            "f2", "data:image/png;base64,BB==", 200.0, "camera-1", ("cup-1",)
        )
        second = {"updates": [update("color", "blue", frame_id="f2")]}
        updater = VLMPropertyUpdater(FakeVLMClient(first))
        p1 = updater.parse_response([self.frame], self.registry, first).proposals
        p2 = updater.parse_response([second_frame], self.registry, second).proposals
        ledger = EntityObservationLedger(self.registry)
        ledger.extend((*p1, *p2))

        self.assertEqual(len(ledger.entries(entity_id="cup-1", property_name="color")), 2)
        self.assertEqual(ledger.snapshot("cup-1", as_of=150).properties["color"].value, "red")
        current = ledger.snapshot("cup-1")
        self.assertEqual(current.properties["color"].value, "blue")

        query = QueryFrame("blue cup", (PropertyConstraint("color", "blue"),))
        matcher = EntityMatcher(
            self.registry, default_comparators(), MentionBasedSelector()
        )
        result = matcher.match(query, [current])[0]
        self.assertEqual(result.match_score, 1.0)
        self.assertEqual(result.coverage, 0.9)
        self.assertEqual(result.score, 0.9)

    def test_openai_adapter_sends_images_with_strict_no_store_output(self):
        responses = FakeResponses({"updates": []})
        client = OpenAIResponsesVLMClient(
            model="vision-test",
            client=SimpleNamespace(responses=responses),
        )
        result = client.generate_json(
            instructions="i",
            input_text="t",
            image_urls=("data:image/png;base64,AA==",),
            schema_name="updates",
            schema={"type": "object"},
        )
        self.assertEqual(result, {"updates": []})
        self.assertFalse(responses.kwargs["store"])
        self.assertTrue(responses.kwargs["text"]["format"]["strict"])
        content = responses.kwargs["input"][0]["content"]
        self.assertEqual(content[1]["type"], "input_image")


if __name__ == "__main__":
    unittest.main()
