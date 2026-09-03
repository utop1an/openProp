import unittest
from dataclasses import replace

from openprop.visual_calibration import (
    apply_acceptance_policy,
    calibrate_acceptance_policy,
)
from openprop.visual_evaluation import NULL_ENTITY, VisualAssociationResult
from test_visual_evaluation import VisualEvaluationTests


class VisualCalibrationTests(unittest.TestCase):
    def calibration_rows(self):
        fixture = VisualEvaluationTests()
        return tuple(
            replace(
                row,
                cluster_id="calibration-scene",
                split="calibration",
                accepted_entity_id=None,
            )
            for row in fixture.association_rows()
        )

    def test_policy_selection_uses_calibration_safety_then_utility(self):
        policy = calibrate_acceptance_policy(
            self.calibration_rows(),
            acceptance_thresholds=(0.5, 0.75),
            margin_thresholds=(0.1, 0.3),
            max_false_update_rate=0.0,
        )
        self.assertEqual(policy.acceptance_threshold, 0.75)
        self.assertEqual(policy.margin_threshold, 0.3)
        self.assertEqual(policy.correct_updates, 1)
        self.assertEqual(policy.false_updates, 0)
        self.assertEqual(policy.searched_policies, 4)
        self.assertGreater(policy.feasible_policies, 0)

    def test_policy_application_does_not_use_test_truth(self):
        policy = calibrate_acceptance_policy(
            self.calibration_rows(),
            acceptance_thresholds=(0.75,),
            margin_thresholds=(0.3,),
            max_false_update_rate=0.0,
        )
        wrong_test = VisualAssociationResult(
            "wrong-test",
            "test-scene",
            "test",
            "openprop",
            "real-video",
            "motion_state",
            "d",
            "f",
            ("e1", "e2"),
            "e2",
            "e1",
            None,
            {"e1": 0.9, "e2": 0.05, NULL_ENTITY: 0.05},
            "misleading",
            1,
        )
        applied = apply_acceptance_policy((wrong_test,), policy)
        self.assertEqual(applied[0].accepted_entity_id, "e1")
        self.assertTrue(applied[0].false_update)

    def test_test_rows_cannot_enter_policy_search(self):
        test_rows = tuple(
            replace(row, split="test", cluster_id="test-scene")
            for row in self.calibration_rows()
        )
        with self.assertRaisesRegex(ValueError, "calibration rows"):
            calibrate_acceptance_policy(
                test_rows,
                acceptance_thresholds=(0.5,),
                margin_thresholds=(0.1,),
                max_false_update_rate=0.0,
            )

    def test_ineligible_and_malformed_rows_never_become_updates(self):
        rows = self.calibration_rows()
        policy = calibrate_acceptance_policy(
            rows,
            acceptance_thresholds=(0.75,),
            margin_thresholds=(0.3,),
            max_false_update_rate=0.0,
        )
        blocked = (
            replace(
                rows[0],
                split="test",
                cluster_id="blocked-scene",
                eligible=False,
            ),
            replace(
                rows[0],
                record_id="malformed",
                split="test",
                cluster_id="blocked-scene",
                malformed=True,

            ),
        )
        applied = apply_acceptance_policy(blocked, policy)
        self.assertFalse(any(row.accepted for row in applied))

    def test_combined_confidence_minimum_survives_offline_calibration(self):
        rows = self.calibration_rows()
        policy = calibrate_acceptance_policy(
            rows,
            acceptance_thresholds=(0.75,),
            margin_thresholds=(0.3,),
            max_false_update_rate=0.0,
        )
        blocked = replace(
            rows[0],
            split="test",
            cluster_id="combined-confidence",
            confidence_scale=0.2,
            minimum_update_confidence=0.5,
        )
        applied = apply_acceptance_policy((blocked,), policy)
        self.assertFalse(applied[0].accepted)

    def test_impossible_safety_grid_fails_closed(self):
        wrong = replace(
            self.calibration_rows()[0],
            target_entity_id="e2",
            probabilities={"e1": 0.9, "e2": 0.05, NULL_ENTITY: 0.05},
        )
        with self.assertRaisesRegex(ValueError, "no acceptance policy"):
            calibrate_acceptance_policy(
                (wrong,),
                acceptance_thresholds=(0.0,),
                margin_thresholds=(0.0,),
                max_false_update_rate=0.0,
            )

    def test_candidate_count_aware_null_prior_prevents_crowded_false_update(self):
        correct = replace(
            self.calibration_rows()[0],
            probabilities={"e1": 0.8, "e2": 0.1, NULL_ENTITY: 0.1},
        )
        crowded = VisualAssociationResult(
            "crowded-null",
            "calibration-crowded",
            "calibration",
            "openprop",
            "real-video",
            "motion_state",
            "d-crowded",
            "f-crowded",
            ("e1", "e2", "e3", "e4"),
            None,
            "e1",
            None,
            {"e1": 0.4, "e2": 0.1, "e3": 0.1, "e4": 0.1, NULL_ENTITY: 0.3},
            "crowded-no-change",
            4,
        )
        policy = calibrate_acceptance_policy(
            (correct, crowded),
            acceptance_thresholds=(0.4,),
            margin_thresholds=(0.0,),
            null_scales=(0.5,),
            candidate_count_powers=(0.0, 1.0),
            max_false_update_rate=0.0,
        )
        self.assertEqual(policy.null_scale, 0.5)
        self.assertEqual(policy.candidate_count_power, 1.0)
        self.assertEqual(policy.candidate_count_levels, 2)
        self.assertEqual(policy.supported_candidate_counts, (2, 4))
        applied = apply_acceptance_policy(
            (replace(crowded, split="test", cluster_id="test-crowded"),), policy
        )[0]
        self.assertIsNone(applied.decision_entity_id)
        self.assertIsNone(applied.accepted_entity_id)
        self.assertGreater(applied.probabilities[NULL_ENTITY], applied.probabilities["e1"])

    def test_candidate_count_power_requires_multiple_count_levels(self):
        with self.assertRaisesRegex(ValueError, "multiple candidate counts"):
            calibrate_acceptance_policy(
                self.calibration_rows(),
                acceptance_thresholds=(0.5,),
                margin_thresholds=(0.0,),
                null_scales=(1.0,),
                candidate_count_powers=(1.0,),
                max_false_update_rate=1.0,
            )

    def test_unseen_candidate_count_abstains_after_policy_freeze(self):
        policy = calibrate_acceptance_policy(
            self.calibration_rows(),
            acceptance_thresholds=(0.5,),
            margin_thresholds=(0.0,),
            max_false_update_rate=1.0,
        )
        row = self.calibration_rows()[0]
        unseen = VisualAssociationResult(
            "unseen-count", "test-unseen", "test", row.system, row.source,
            row.property_name, "d-unseen", "f-unseen", ("e1", "e2", "e3"),
            "e1", "e1", None,
            {"e1": 0.8, "e2": 0.05, "e3": 0.05, NULL_ENTITY: 0.1},
            row.condition, 2,
        )
        applied = apply_acceptance_policy((unseen,), policy)[0]
        self.assertFalse(applied.accepted)
        self.assertIn("absent from calibration", applied.reason)


if __name__ == "__main__":
    unittest.main()
