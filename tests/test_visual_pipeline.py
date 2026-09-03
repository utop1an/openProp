import unittest

from openprop.association import (
    AssociationPolicy,
    MultiEntityAssociator,
    VisualPropertyDetection,
)
from openprop.comparators import default_comparators
from openprop.models import (
    Entity,
    EntityEvent,
    Observation,
    PropertyConstraint,
    PropertyDefinition,
    PropertyUpdatePolicy,
    QueryFrame,
    ValueType,
)
from openprop.property_registry import PropertyRegistry
from openprop.selectors import MentionBasedSelector
from openprop.visual_pipeline import EntityStateStore, VisualUpdateOrchestrator
from openprop.vlm import EntityObservationLedger, PropertyUpdateProposal, VisualFrame


class FakeDetector:
    def __init__(self, detections):
        self.detections = tuple(detections)

    def detect(self, frames, registry):
        return self.detections


class VisualUpdateOrchestratorTests(unittest.TestCase):
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
                "observed motion",
                ValueType.CATEGORICAL,
                metadata={"allowed_values": ("moved", "stationary")},
                update_policy=PropertyUpdatePolicy(minimum_confidence=0.2),
            )
        )
        self.entities = (
            Entity(
                "cup-1",
                {
                    "type": Observation("cup", timestamp=1.0),
                    "color": Observation("red", timestamp=1.0),
                },
                [
                    EntityEvent("seen", 5.0),
                    EntityEvent("future", 10.0),
                ],
            ),
            Entity(
                "cup-2",
                {
                    "type": Observation("cup", timestamp=1.0),
                    "color": Observation("red", timestamp=1.0),
                },
            ),
        )
        self.query = QueryFrame(
            "the red cup",
            (
                PropertyConstraint("type", "cup", relevance=0.5),
                PropertyConstraint("color", "red", relevance=0.5),
            ),
        )

    def associator(self):
        return MultiEntityAssociator(
            self.registry,
            default_comparators(),
            MentionBasedSelector(),
            policy=AssociationPolicy(
                acceptance_threshold=0.75,
                margin_threshold=0.2,
                null_weight=0.05,
            ),
        )

    @staticmethod
    def detection(frame, detection_id, target):
        other = "cup-2" if target == "cup-1" else "cup-1"
        return VisualPropertyDetection(
            detection_id,
            frame,
            "motion_state",
            "moved",
            0.95,
            0.95,
            {target: 0.98, other: 0.02},
            track_affinities={target: 0.95, other: 0.05},
        )

    def test_state_store_materializes_strict_pre_event_properties_and_events(self):
        ledger = EntityObservationLedger(self.registry)
        ledger.append(
            PropertyUpdateProposal(
                "cup-1",
                "motion_state",
                Observation("moved", confidence=0.9, source="camera", timestamp=10.0),
                "same-time",
            )
        )
        state = EntityStateStore(self.registry, self.entities, observations=ledger)
        before = state.snapshot("cup-1", before=10.0)
        self.assertNotIn("motion_state", before.properties)
        self.assertEqual(["seen"], [event.event_type for event in before.events])
        current = state.snapshot("cup-1")
        self.assertEqual("moved", current.properties["motion_state"].value)
        self.assertEqual(["seen", "future"], [event.event_type for event in current.events])

    def test_orchestrator_orders_frames_audits_and_commits(self):
        early = VisualFrame(
            "early",
            "early.png",
            10.0,
            "camera",
            ("cup-1", "cup-2"),
        )
        late = VisualFrame(
            "late",
            "late.png",
            20.0,
            "camera",
            ("cup-1", "cup-2"),
        )
        detections = (
            self.detection(late, "late-detection", "cup-2"),
            self.detection(early, "early-detection", "cup-1"),
        )
        state = EntityStateStore(self.registry, self.entities)
        pipeline = VisualUpdateOrchestrator(
            FakeDetector(detections),
            self.associator(),
            state,
        )
        run = pipeline.run(self.query, (late, early))
        self.assertEqual(
            ("early", "late"),
            tuple(update.frame.frame_id for update in run.frame_updates),
        )
        self.assertEqual(("cup-1", "cup-2"), tuple(p.entity_id for p in run.proposals))
        self.assertEqual(2, len(pipeline.audit.entries()))
        self.assertEqual(2, len(state.observations.entries()))

    def test_orchestrator_rejects_untrusted_frame_substitution(self):
        trusted = VisualFrame(
            "frame",
            "trusted.png",
            10.0,
            "camera",
            ("cup-1", "cup-2"),
        )
        substituted = VisualFrame(
            "frame",
            "other.png",
            10.0,
            "camera",
            ("cup-1", "cup-2"),
        )
        detection = self.detection(substituted, "detection", "cup-1")
        pipeline = VisualUpdateOrchestrator(
            FakeDetector((detection,)),
            self.associator(),
            EntityStateStore(self.registry, self.entities),
        )
        with self.assertRaisesRegex(ValueError, "trusted input"):
            pipeline.run(self.query, (trusted,))


if __name__ == "__main__":
    unittest.main()

