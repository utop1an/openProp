import unittest
from dataclasses import replace

from openprop.combined_confidence import (
    apply_combined_confidence_calibration,
    fit_combined_confidence_calibration,
)
from openprop.visual_evaluation import (
    NULL_ENTITY,
    VisualAssociationResult,
    VisualEvaluationDataset,
    aggregate_visual_evaluation,
)


class CombinedConfidenceTests(unittest.TestCase):
    def row(
        self,
        record,
        *,
        probability,
        correct,
        source="camera-a",
        split="calibration",
        accepted=False,
        minimum=0.0,
    ):
        return VisualAssociationResult(
            record,
            f"cluster-{record}",
            split,
            "openprop",
            source,
            "motion_state",
            f"d-{record}",
            f"f-{record}",
            ("e1", "e2"),
            "e1" if correct else "e2",
            "e1",
            "e1" if accepted else None,
            {"e1": probability, "e2": (1 - probability) / 2, NULL_ENTITY: (1 - probability) / 2},
            "confidence-calibration",
            1,
            False,
            "pre-confidence decision",
            True,
            1.0,
            minimum,
        )

    def calibration_rows(self):
        return (
            self.row("a-low-correct", probability=0.2, correct=True),
            self.row("a-mid-wrong", probability=0.6, correct=False),
            self.row("a-high-correct", probability=0.9, correct=True),
            self.row("b-low-wrong", probability=0.2, correct=False, source="camera-b"),
            self.row("b-high-correct", probability=0.9, correct=True, source="camera-b"),
        )

    def test_laplace_isotonic_mapping_is_monotone_and_source_conditioned(self):
        calibration = fit_combined_confidence_calibration(
            self.calibration_rows(), minimum_source_rows=2
        )
        values = calibration.global_model.values
        self.assertTrue(all(left <= right for left, right in zip(values, values[1:])))
        self.assertEqual(calibration.calibration_rows, 5)
        self.assertEqual(set(calibration.source_models), {"camera-a", "camera-b"})
        self.assertNotEqual(
            calibration.predict(0.2, "camera-a"),
            calibration.predict(0.2, "camera-b"),
        )
        self.assertEqual(
            calibration.predict(0.2, "unseen-camera"),
            calibration.global_model.predict(0.2),
        )

    def test_application_is_target_blind_and_can_only_revoke(self):
        calibration = fit_combined_confidence_calibration(
            self.calibration_rows(), minimum_source_rows=30
        )
        accepted = self.row(
            "test-accepted",
            probability=0.6,
            correct=True,
            split="test",
            accepted=True,
            minimum=0.9,
        )
        changed_truth = replace(accepted, target_entity_id="e2")
        left = apply_combined_confidence_calibration((accepted,), calibration)[0]
        right = apply_combined_confidence_calibration((changed_truth,), calibration)[0]
        self.assertEqual(
            left.calibrated_update_confidence,
            right.calibrated_update_confidence,
        )
        self.assertIsNone(left.accepted_entity_id)
        self.assertIn("calibrated combined confidence", left.reason)

        already_rejected = replace(accepted, record_id="test-rejected", accepted_entity_id=None)
        applied = apply_combined_confidence_calibration(
            (already_rejected,), calibration
        )[0]
        self.assertIsNone(applied.accepted_entity_id)

    def test_test_rows_and_empty_eligible_population_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "calibration rows"):
            fit_combined_confidence_calibration(
                (replace(self.calibration_rows()[0], split="test"),)
            )
        blocked = replace(self.calibration_rows()[0], eligible=False)
        with self.assertRaisesRegex(ValueError, "no eligible decisions"):
            fit_combined_confidence_calibration((blocked,))

    def test_report_compares_raw_and_calibrated_combined_confidence(self):
        calibration = fit_combined_confidence_calibration(
            self.calibration_rows(), minimum_source_rows=30
        )
        rows = tuple(
            replace(row, split="test", cluster_id=f"test-{index}")
            for index, row in enumerate(self.calibration_rows())
        )
        applied = apply_combined_confidence_calibration(rows, calibration)
        report = aggregate_visual_evaluation(
            VisualEvaluationDataset(associations=applied), split="test"
        )
        metrics = report["systems"]["openprop"]["association"][
            "combined_update_confidence"
        ]
        self.assertEqual(metrics["total"], 5)
        self.assertIn("brier", metrics["raw"])
        self.assertIn("brier", metrics["calibrated"])


if __name__ == "__main__":
    unittest.main()
