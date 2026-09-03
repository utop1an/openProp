import unittest

from openprop.association import (
    AssociationAuditLedger,
    AssociationPolicy,
    MultiEntityAssociator,
    VLMPropertyDetector,
    VisualPropertyDetection,
)
from openprop.comparators import default_comparators
from openprop.models import (
    Entity,
    Observation,
    PropertyConstraint,
    PropertyDefinition,
    PropertyUpdatePolicy,
    QueryFrame,
    ValueType,
)
from openprop.property_registry import PropertyRegistry
from openprop.selectors import MentionBasedSelector
from openprop.vlm import EntityObservationLedger, VisualFrame

def scalar(value):
    return {
        "kind": "scalar",
        "scalar": value,
        "predicate": None,
        "arguments": [],
        "vector": [],
    }


class FakeVLMClient:
    def __init__(self, response):
        self.response = response
        self.last_call = None

    def generate_json(self, **kwargs):
        self.last_call = kwargs
        return self.response



class MultiEntityAssociationTests(unittest.TestCase):
    def setUp(self):
        self.registry = PropertyRegistry()
        self.registry.register(
            PropertyDefinition("type", "object category", ValueType.CATEGORICAL)
        )
        self.registry.register(
            PropertyDefinition("color", "surface color", ValueType.CATEGORICAL)
        )
        self.registry.register(
            PropertyDefinition(
                "motion_state",
                "observed motion state",
                ValueType.CATEGORICAL,
                metadata={"allowed_values": ["moved", "stationary"]},
                update_policy=PropertyUpdatePolicy(minimum_confidence=0.5),
            )
        )
        self.entities = [
            Entity(
                "cup-1",
                {
                    "type": Observation("cup", timestamp=10),
                    "color": Observation("red", timestamp=10),
                },
            ),
            Entity(
                "cup-2",
                {
                    "type": Observation("cup", timestamp=10),
                    "color": Observation("red", timestamp=10),
                },
            ),
        ]
        self.query = QueryFrame(
            "the red cup",
            (
                PropertyConstraint("type", "cup", 1.0),
                PropertyConstraint("color", "red", 1.0),
            ),
        )
        self.frame = VisualFrame(
            "frame-1",
            "data:image/png;base64,AA==",
            100.0,
            "camera-1",
            ("cup-1", "cup-2"),
        )
        self.policy = AssociationPolicy(
            acceptance_threshold=0.8,
            margin_threshold=0.2,
            source_reliability={"camera-1": 0.9},
        )
        self.associator = MultiEntityAssociator(
            self.registry,
            default_comparators(),
            MentionBasedSelector(),
            policy=self.policy,
        )

    def detection(
        self,
        detection_id="d1",
        *,
        affinities=None,
        track_affinities=None,
    ):
        return VisualPropertyDetection(
            detection_id,
            self.frame,
            "motion_state",
            "moved",
            0.95,
            0.90,
            affinities if affinities is not None else {"cup-1": 0.95, "cup-2": 0.10},
            track_id="track-7",
            track_affinities=(
                track_affinities if track_affinities is not None else {"cup-1": 0.95, "cup-2": 0.20}
            ),
        )

    def test_strong_single_target_commits_only_top_entity(self):
        hypothesis = self.associator.associate(
            self.detection(), self.query, self.entities
        )
        self.assertTrue(hypothesis.accepted)
        self.assertEqual(hypothesis.accepted_entity_id, "cup-1")
        self.assertGreater(hypothesis.candidates[0].posterior, 0.9)
        expected = (
            0.95
            * 0.90
            * hypothesis.candidates[0].posterior
            * 0.9
        )
        self.assertAlmostEqual(hypothesis.update_confidence, expected)

        ledger = EntityObservationLedger(self.registry)
        proposals = self.associator.commit([hypothesis], ledger)
        self.assertEqual(len(proposals), 1)
        self.assertEqual(proposals[0].entity_id, "cup-1")
        self.assertNotIn("motion_state", ledger.snapshot("cup-2").properties)
        self.assertEqual(
            ledger.snapshot("cup-1").properties["motion_state"].value,
            "moved",
        )

    def test_vlm_detector_emits_unbound_detection_with_complete_affinities(self):
        response = {"detections": [{
            "detection_id": "d1",
            "frame_id": "frame-1",
            "track_id": "track-7",
            "property_name": "motion_state",
            "value_type": "categorical",
            "detection_confidence": 0.95,
            "value_confidence": 0.90,
            "candidate_affinities": [
                {"entity_id": "cup-1", "affinity": 0.95},
                {"entity_id": "cup-2", "affinity": 0.10},
            ],
            "track_affinities": [
                {"entity_id": "cup-1", "affinity": 0.90},
                {"entity_id": "cup-2", "affinity": 0.20},
            ],
            "region": [0.1, 0.2, 0.4, 0.6],
            "value": scalar("moved"),
        }]}
        client = FakeVLMClient(response)
        detections = VLMPropertyDetector(client).detect(
            [self.frame], self.registry
        )
        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0].candidate_affinities["cup-1"], 0.95)
        self.assertEqual(detections[0].region, (0.1, 0.2, 0.4, 0.6))
        self.assertFalse(hasattr(detections[0], "entity_id"))
        self.assertEqual(client.last_call["image_urls"], (self.frame.image_url,))
        self.assertNotIn("camera-1", client.last_call["input_text"])
        self.assertNotIn("100.0", client.last_call["input_text"])

    def test_ambiguous_detection_abstains_instead_of_updating_both(self):
        detection = self.detection(
            affinities={"cup-1": 0.80, "cup-2": 0.78},
            track_affinities={},
        )
        hypothesis = self.associator.associate(
            detection, self.query, self.entities
        )
        self.assertFalse(hypothesis.accepted)
        self.assertIn("threshold", hypothesis.reason)
        self.assertEqual(hypothesis.decision_entity_id, "cup-1")
        self.assertGreater(hypothesis.candidates[0].posterior, 0.45)
        self.assertGreater(hypothesis.candidates[1].posterior, 0.45)

        ledger = EntityObservationLedger(self.registry)
        self.assertEqual(self.associator.commit([hypothesis], ledger), ())
        self.assertEqual(ledger.snapshots(), ())

        audit = AssociationAuditLedger()
        audit.append(hypothesis)
        self.assertEqual(audit.entries(accepted=False), (hypothesis,))

    def test_null_candidate_absorbs_unsupported_association(self):
        detection = self.detection(
            affinities={"cup-1": 0.01, "cup-2": 0.01},
            track_affinities={},
        )
        hypothesis = self.associator.associate(
            detection, self.query, self.entities
        )
        self.assertFalse(hypothesis.accepted)
        self.assertGreater(hypothesis.null_probability, 0.7)

    def test_missing_query_evidence_is_neutral_not_negative(self):
        hypothesis = self.associator.associate(
            self.detection(),
            self.query,
            (Entity("cup-1"), Entity("cup-2")),
        )
        self.assertEqual(hypothesis.candidates[0].query_score, 0.0)
        self.assertEqual(hypothesis.accepted_entity_id, "cup-1")

    def test_batch_conflict_fails_closed(self):
        first = self.detection("d1")
        second = self.detection(
            "d2",
            affinities={"cup-1": 0.90, "cup-2": 0.10},
            track_affinities={"cup-1": 0.90, "cup-2": 0.10},
        )
        hypotheses = self.associator.associate_batch(
            [first, second], self.query, self.entities
        )
        self.assertFalse(any(item.accepted for item in hypotheses))
        self.assertTrue(
            all("conflicting detections" in item.reason for item in hypotheses)
        )

    def test_distinct_detections_can_update_distinct_entities(self):
        first = self.detection("d1")
        second = self.detection(
            "d2",
            affinities={"cup-1": 0.05, "cup-2": 0.98},
            track_affinities={"cup-1": 0.05, "cup-2": 0.98},
        )
        hypotheses = self.associator.associate_batch(
            [first, second], self.query, self.entities
        )
        self.assertEqual(
            {item.accepted_entity_id for item in hypotheses},
            {"cup-1", "cup-2"},
        )
        ledger = EntityObservationLedger(self.registry)
        proposals = self.associator.commit(hypotheses, ledger)
        self.assertEqual(len(proposals), 2)

    def test_association_rejects_snapshot_feedback_and_incomplete_affinities(self):
        future_entities = [
            Entity(
                "cup-1",
                {"type": Observation("cup", timestamp=100)},
            ),
            self.entities[1],
        ]
        with self.assertRaisesRegex(ValueError, "strictly pre-event"):
            self.associator.associate(
                self.detection(), self.query, future_entities
            )
        with self.assertRaisesRegex(ValueError, "cover every frame candidate"):
            VisualPropertyDetection(
                "bad",
                self.frame,
                "motion_state",
                "moved",
                0.9,
                0.9,
                {"cup-1": 0.9},
            )


if __name__ == "__main__":
    unittest.main()
