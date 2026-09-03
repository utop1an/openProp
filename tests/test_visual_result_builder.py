import unittest

from openprop.association import AssociationPolicy, MultiEntityAssociator, VisualPropertyDetection
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
from openprop.visual_result_builder import (
    VisualDetectionTruth,
    VisualFrameEvaluationTruth,
    build_visual_detection_results,
)
from openprop.vlm import VisualFrame


class VisualResultBuilderTests(unittest.TestCase):
    def setUp(self):
        self.registry = PropertyRegistry()
        self.registry.register(
            PropertyDefinition("type", "object type", ValueType.CATEGORICAL)
        )
        self.registry.register(
            PropertyDefinition(
                "motion_state",
                "motion",
                ValueType.CATEGORICAL,
                metadata={"allowed_values": ("moved", "stationary")},
                update_policy=PropertyUpdatePolicy(minimum_confidence=0.2),
            )
        )
        self.policy = AssociationPolicy(
            acceptance_threshold=0.5,
            margin_threshold=0.1,
            null_weight=0.01,
            source_reliability={"ai2thor": 0.9},
        )
        self.associator = MultiEntityAssociator(
            self.registry,
            default_comparators(),
            MentionBasedSelector(),
            policy=self.policy,
        )
        self.frame = VisualFrame(
            "frame",
            "data:image/png;base64,AA==",
            10.0,
            "ai2thor",
            ("e1", "e2"),
        )
        self.entities = (
            Entity("e1", {"type": Observation("cup", timestamp=1.0)}),
            Entity("e2", {"type": Observation("cup", timestamp=1.0)}),
        )
        self.query = QueryFrame(
            "the cup",
            (PropertyConstraint("type", "cup", 1.0),),
        )

    def detection(self, detection_id, region, affinities):
        return VisualPropertyDetection(
            detection_id,
            self.frame,
            "motion_state",
            "moved",
            0.95,
            0.9,
            affinities,
            track_affinities={},
            region=region,
        )

    def hypotheses(self):
        detections = (
            self.detection(
                "d1",
                (0.1, 0.1, 0.3, 0.3),
                {"e1": 0.95, "e2": 0.05},
            ),
            self.detection(
                "d2",
                (0.1, 0.1, 0.31, 0.31),
                {"e1": 0.9, "e2": 0.1},
            ),
        )
        return tuple(
            self.associator.associate(item, self.query, self.entities)
            for item in detections
        )

    def context(self, target_two="e2", malformed=False):
        return VisualFrameEvaluationTruth(
            "scene-1",
            "calibration",
            "frame",
            "ai2thor",
            "multi-change",
            1,
            (
                VisualDetectionTruth(
                    "event-1",
                    "motion_state",
                    "moved",
                    "e1",
                    (0.1, 0.1, 0.3, 0.3),
                ),
                VisualDetectionTruth(
                    "event-2",
                    "motion_state",
                    "moved",
                    target_two,
                    (0.6, 0.1, 0.8, 0.3),
                ),
            ),
            malformed,
        )

    def test_builder_matches_by_iou_and_keeps_misses_and_duplicates(self):
        batch = build_visual_detection_results(
            (self.context(),),
            self.hypotheses(),
            associator=self.associator,
            system="openprop",
        )
        self.assertEqual(len(batch.properties), 3)
        matched = next(row for row in batch.properties if row.record_id == "event-1")
        missed = next(row for row in batch.properties if row.record_id == "event-2")
        duplicate = next(row for row in batch.properties if not row.expected)
        self.assertTrue(matched.exact_value_match)
        self.assertFalse(missed.detected)
        self.assertEqual(duplicate.duplicate_count, 1)
        self.assertEqual(len(batch.associations), 2)
        association = next(
            row for row in batch.associations if row.record_id == "event-1"
        )
        self.assertEqual(association.target_entity_id, "e1")
        self.assertEqual(association.decision_entity_id, "e1")
        self.assertAlmostEqual(association.confidence_scale, 0.95 * 0.9 * 0.9)
        false_positive = next(
            row for row in batch.associations if row.target_entity_id is None
        )
        self.assertTrue(false_positive.false_update)

    def test_matching_is_prediction_order_invariant(self):
        hypotheses = self.hypotheses()
        forward = build_visual_detection_results(
            (self.context(),),
            hypotheses,
            associator=self.associator,
            system="openprop",
        )
        reverse = build_visual_detection_results(
            (self.context(),),
            tuple(reversed(hypotheses)),
            associator=self.associator,
            system="openprop",
        )
        forward_rows = {
            row.record_id: (row.expected, row.detected, row.duplicate_count)
            for row in forward.properties
        }
        reverse_rows = {
            row.record_id: (row.expected, row.detected, row.duplicate_count)
            for row in reverse.properties
        }
        self.assertEqual(forward_rows, reverse_rows)

    def test_candidate_miss_preserves_physical_target_truth(self):
        context = VisualFrameEvaluationTruth(
            "scene",
            "test",
            "frame",
            "ai2thor",
            "target-missing",
            1,
            (
                VisualDetectionTruth(
                    "event",
                    "motion_state",
                    "moved",
                    "e3",
                    (0.1, 0.1, 0.3, 0.3),
                ),
            ),
        )
        batch = build_visual_detection_results(
            (context,),
            (self.hypotheses()[0],),
            associator=self.associator,
            system="openprop",
        )
        row = batch.associations[0]
        self.assertEqual(row.target_entity_id, "e3")
        self.assertNotIn("e3", row.probabilities)
        self.assertTrue(row.false_update)

    def test_unlocalized_prediction_cannot_claim_localization_match(self):
        detection = self.detection(
            "unlocalized",
            None,
            {"e1": 0.95, "e2": 0.05},
        )
        hypothesis = self.associator.associate(
            detection, self.query, self.entities
        )
        batch = build_visual_detection_results(
            (self.context(),),
            (hypothesis,),
            associator=self.associator,
            system="openprop",
        )
        self.assertEqual(sum(row.expected and not row.detected for row in batch.properties), 2)
        self.assertEqual(sum(not row.expected for row in batch.properties), 1)

    def test_malformed_context_marks_failures_without_dropping_truth(self):
        batch = build_visual_detection_results(
            (self.context(malformed=True),),
            (),
            associator=self.associator,
            system="openprop",
        )
        self.assertEqual(len(batch.properties), 2)
        self.assertTrue(all(row.malformed for row in batch.properties))
        self.assertTrue(all(not row.detected for row in batch.properties))


if __name__ == "__main__":
    unittest.main()
