import math
import tempfile
import unittest
from pathlib import Path

from openprop.visual_evaluation import (
    NULL_ENTITY,
    VisualAssociationResult,
    VisualEvaluationDataset,
    VisualPropertyResult,
    VisualQueryResult,
    aggregate_visual_evaluation,
    read_visual_results_jsonl,
    risk_coverage_curve,
    write_visual_results_jsonl,
)


class VisualEvaluationTests(unittest.TestCase):
    def property_rows(self):
        common = {
            "cluster_id": "scene-test",
            "split": "test",
            "system": "openprop",
            "source": "ai2thor",
            "property_name": "open_state",
        }
        return (
            VisualPropertyResult(
                "p1", **common, expected=True, detected=True,
                gold_value="open", predicted_value="open", confidence=0.9
            ),
            VisualPropertyResult(
                "p2", **common, expected=True, detected=False,
                gold_value="closed", predicted_value=None, confidence=0.0
            ),
            VisualPropertyResult(
                "p3", **common, expected=False, detected=True,
                gold_value=None, predicted_value="open", confidence=0.7
            ),
        )

    def association_rows(self):
        common = {
            "cluster_id": "scene-test",
            "split": "test",
            "system": "openprop",
            "source": "ai2thor",
            "property_name": "motion_state",
            "frame_id": "frame",
            "candidate_entity_ids": ("e1", "e2"),
            "condition": "crowded",
            "distractor_count": 1,
        }
        return (
            VisualAssociationResult(
                "a1", **common, detection_id="d1", target_entity_id="e1",
                decision_entity_id="e1", accepted_entity_id="e1",
                probabilities={"e1": 0.8, "e2": 0.1, NULL_ENTITY: 0.1},
            ),
            VisualAssociationResult(
                "a2", **common, detection_id="d2", target_entity_id="e2",
                decision_entity_id="e1", accepted_entity_id=None,
                probabilities={"e1": 0.55, "e2": 0.4, NULL_ENTITY: 0.05},
            ),
            VisualAssociationResult(
                "a3", **common, detection_id="d3", target_entity_id=None,
                decision_entity_id=None, accepted_entity_id=None,
                probabilities={"e1": 0.05, "e2": 0.05, NULL_ENTITY: 0.9},
            ),
            VisualAssociationResult(
                "a4", **common, detection_id="d4", target_entity_id="e1",
                decision_entity_id="e2", accepted_entity_id="e2",
                probabilities={"e1": 0.2, "e2": 0.7, NULL_ENTITY: 0.1},
            ),
        )

    def query_rows(self):
        common = {
            "cluster_id": "scene-test",
            "split": "test",
            "system": "openprop",
            "source": "ai2thor",
            "property_name": "motion_state",
            "horizon_seconds": 60.0,
            "distractor_count": 1,
            "condition": "delayed",
        }
        return (
            VisualQueryResult(
                "q1", **common, candidate_entity_ids=("e1", "e2"),
                target_entity_id="e1", ranked_entity_ids=("e1", "e2"),
                decision_entity_id="e1", accepted_entity_id="e1",
                probabilities={"e1": 0.8, "e2": 0.1, NULL_ENTITY: 0.1},
                latency_seconds=1.0, vlm_calls=1,
            ),
            VisualQueryResult(
                "q2", **common, candidate_entity_ids=("e1",),
                target_entity_id="e2", ranked_entity_ids=("e1",),
                decision_entity_id="e1", accepted_entity_id=None,
                probabilities={"e1": 0.8, NULL_ENTITY: 0.2},
                latency_seconds=2.0, vlm_calls=1,
            ),
            VisualQueryResult(
                "q3", **common, candidate_entity_ids=("e1", "e2"),
                target_entity_id=None, ranked_entity_ids=("e1", "e2"),
                decision_entity_id=None, accepted_entity_id=None,
                probabilities={"e1": 0.05, "e2": 0.05, NULL_ENTITY: 0.9},
                latency_seconds=0.0, vlm_calls=0,
            ),
        )

    def dataset(self):
        return VisualEvaluationDataset(
            self.property_rows(),
            self.association_rows(),
            self.query_rows(),
        )

    def test_aggregation_keeps_all_units_and_failures_in_denominators(self):
        report = aggregate_visual_evaluation(self.dataset(), split="test", ece_bins=5)
        result = report["systems"]["openprop"]
        prop = result["property"]
        self.assertEqual(prop["total"], 3)
        self.assertEqual(prop["detection_f1"], 0.5)
        self.assertEqual(prop["typed_exact_match"], 0.5)

        association = result["association"]
        self.assertEqual(association["total"], 4)
        self.assertEqual(association["coverage"], 0.5)
        self.assertEqual(association["correct_update_rate"], 0.25)
        self.assertEqual(association["false_update_rate"], 0.25)
        self.assertEqual(association["selective_accuracy"], 0.5)
        self.assertEqual(association["false_null_rate"], 1 / 3)

        query = result["query"]
        self.assertEqual(query["total"], 3)
        self.assertEqual(query["candidate_recall"], 0.5)
        self.assertAlmostEqual(query["top1"], 2 / 3)
        self.assertAlmostEqual(query["mrr"], 2 / 3)
        self.assertEqual(query["coverage"], 1 / 3)
        self.assertEqual(query["mean_vlm_calls"], 2 / 3)
        for metric in ("ece", "brier", "nll"):
            self.assertTrue(math.isfinite(query[metric]))
        self.assertIn("association_by_source", result["slices"])
        self.assertIn("query_by_candidate_count", result["slices"])

    def test_risk_coverage_uses_total_population_denominator(self):
        curve = risk_coverage_curve(self.association_rows())
        self.assertEqual(curve[0]["coverage"], 0.0)
        self.assertEqual(curve[-1]["accepted"], 2)
        self.assertEqual(curve[-1]["coverage"], 0.5)
        self.assertEqual(curve[-1]["risk"], 0.5)

    def test_jsonl_round_trip_preserves_typed_values_and_probabilities(self):
        dataset = self.dataset()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "visual-results.jsonl"
            write_visual_results_jsonl(path, dataset)
            restored = read_visual_results_jsonl(path)
        self.assertEqual(restored, dataset)

    def test_cluster_leakage_fails_closed(self):
        calibration = VisualPropertyResult(
            "cal",
            "shared-scene",
            "calibration",
            "openprop",
            "ai2thor",
            "open_state",
            True,
            True,
            "open",
            "open",
            0.9,
        )
        test = VisualPropertyResult(
            "test",
            "shared-scene",
            "test",
            "openprop",
            "ai2thor",
            "open_state",
            True,
            True,
            "open",
            "open",
            0.9,
        )
        with self.assertRaisesRegex(ValueError, "multiple splits"):
            VisualEvaluationDataset((calibration, test))

    def test_incomplete_or_unnormalized_distribution_fails_closed(self):
        common = {
            "record_id": "bad",
            "cluster_id": "scene",
            "split": "test",
            "system": "openprop",
            "source": "ai2thor",
            "property_name": "motion_state",
            "detection_id": "d",
            "frame_id": "f",
            "candidate_entity_ids": ("e1", "e2"),
            "target_entity_id": "e1",
            "decision_entity_id": "e1",
            "accepted_entity_id": "e1",
            "condition": "strong",
            "distractor_count": 1,
        }
        with self.assertRaisesRegex(ValueError, "cover every candidate"):
            VisualAssociationResult(
                **common,
                probabilities={"e1": 0.9, NULL_ENTITY: 0.1},
            )
        with self.assertRaisesRegex(ValueError, "sum to one"):
            VisualAssociationResult(
                **common,
                probabilities={"e1": 0.9, "e2": 0.2, NULL_ENTITY: 0.1},
            )

    def test_accepted_identity_must_equal_decision(self):
        with self.assertRaisesRegex(ValueError, "equal the decision"):
            VisualQueryResult(
                "q",
                "scene",
                "test",
                "openprop",
                "ai2thor",
                "motion_state",
                ("e1", "e2"),
                "e1",
                ("e1", "e2"),
                "e1",
                "e2",
                {"e1": 0.8, "e2": 0.1, NULL_ENTITY: 0.1},
                1.0,
                1,
                "bad",
            )


if __name__ == "__main__":
    unittest.main()
