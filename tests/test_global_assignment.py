import unittest
from dataclasses import replace

from openprop.association import AssociationPolicy, MultiEntityAssociator, VisualPropertyDetection
from openprop.comparators import default_comparators
from openprop.global_assignment import GlobalOneToOneAssociator
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


class GlobalOneToOneAssociationTests(unittest.TestCase):
    def setUp(self):
        self.registry = PropertyRegistry()
        self.registry.register(
            PropertyDefinition("type", "object category", ValueType.CATEGORICAL)
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
            Entity("cup-1", {"type": Observation("cup", timestamp=1.0)}),
            Entity("cup-2", {"type": Observation("cup", timestamp=1.0)}),
        )
        self.query = QueryFrame(
            "the cups",
            (PropertyConstraint("type", "cup", 1.0),),
        )
        self.frame = VisualFrame(
            "frame",
            "data:image/png;base64,AA==",
            10.0,
            "camera",
            ("cup-1", "cup-2"),
        )
        self.policy = AssociationPolicy(
            acceptance_threshold=0.5,
            margin_threshold=0.1,
            null_weight=0.01,
        )

    def detection(self, detection_id, left, right):
        return VisualPropertyDetection(
            detection_id,
            self.frame,
            "motion_state",
            "moved",
            0.95,
            0.95,
            {"cup-1": left, "cup-2": right},
            track_affinities={},
        )

    def associator(self, cls=GlobalOneToOneAssociator):
        return cls(
            self.registry,
            default_comparators(),
            MentionBasedSelector(),
            policy=self.policy,
        )

    def test_global_constraint_recovers_two_distinct_updates(self):
        detections = (
            self.detection("d1", 0.95, 0.50),
            self.detection("d2", 0.90, 0.85),
        )
        independent = self.associator(MultiEntityAssociator).associate_batch(
            detections,
            self.query,
            self.entities,
        )
        self.assertNotEqual(
            {item.accepted_entity_id for item in independent if item.accepted},
            {"cup-1", "cup-2"},
        )

        global_result = self.associator().associate_batch(
            detections,
            self.query,
            self.entities,
        )
        self.assertEqual(
            {item.detection.detection_id: item.accepted_entity_id for item in global_result},
            {"d1": "cup-1", "d2": "cup-2"},
        )
        self.assertTrue(
            all("global one-to-one" in item.reason for item in global_result)
        )
        ledger = EntityObservationLedger(self.registry)
        proposals = self.associator().commit(global_result, ledger)
        self.assertEqual(len(proposals), 2)

    def test_detection_order_does_not_change_global_assignment(self):
        detections = (
            self.detection("d1", 0.95, 0.50),
            self.detection("d2", 0.90, 0.85),
        )
        associator = self.associator()
        forward = associator.associate_batch(detections, self.query, self.entities)
        reverse = associator.associate_batch(
            tuple(reversed(detections)), self.query, self.entities
        )
        forward_map = {
            item.detection.detection_id: item.accepted_entity_id for item in forward
        }
        reverse_map = {
            item.detection.detection_id: item.accepted_entity_id for item in reverse
        }
        self.assertEqual(forward_map, reverse_map)

    def test_candidate_presentation_order_does_not_change_assignment(self):
        detections = (
            self.detection("d1", 0.95, 0.50),
            self.detection("d2", 0.90, 0.85),
        )
        reversed_frame = VisualFrame(
            self.frame.frame_id,
            self.frame.image_url,
            self.frame.captured_at,
            self.frame.source,
            tuple(reversed(self.frame.candidate_entity_ids)),
        )
        reversed_detections = tuple(
            replace(item, frame=reversed_frame) for item in detections
        )
        associator = self.associator()
        expected = {
            item.detection.detection_id: item.accepted_entity_id
            for item in associator.associate_batch(detections, self.query, self.entities)
        }
        actual = {
            item.detection.detection_id: item.accepted_entity_id
            for item in associator.associate_batch(
                reversed_detections, self.query, self.entities
            )
        }
        self.assertEqual(actual, expected)

    def test_symmetric_global_tie_abstains(self):
        result = self.associator().associate_batch(
            (
                self.detection("d1", 0.9, 0.9),
                self.detection("d2", 0.9, 0.9),
            ),
            self.query,
            self.entities,
        )
        self.assertEqual(
            {item.decision_entity_id for item in result},
            {"cup-1", "cup-2"},
        )
        self.assertFalse(any(item.accepted for item in result))
        self.assertTrue(
            all(
                "threshold" in item.reason or "margin" in item.reason
                for item in result
            )
        )
        for hypothesis in result:
            probabilities = (
                hypothesis.null_probability
                + sum(candidate.posterior for candidate in hypothesis.candidates)
            )
            self.assertAlmostEqual(probabilities, 1.0)

    def test_candidate_limit_fails_closed(self):
        associator = GlobalOneToOneAssociator(
            self.registry,
            default_comparators(),
            MentionBasedSelector(),
            policy=self.policy,
            max_entities=1,
        )
        result = associator.associate_batch(
            (self.detection("d1", 0.95, 0.5),),
            self.query,
            self.entities,
        )
        self.assertFalse(result[0].accepted)
        self.assertIn("candidate limit", result[0].reason)

    def test_solver_prefers_reusable_null_when_all_edges_are_zero(self):
        assignment, marginals = GlobalOneToOneAssociator._solve(
            ((1.0, 0.0, 0.0), (1.0, 0.0, 0.0))
        )
        self.assertEqual(assignment, (0, 0))
        self.assertEqual(marginals, ((1.0, 0.0, 0.0), (1.0, 0.0, 0.0)))


if __name__ == "__main__":
    unittest.main()
