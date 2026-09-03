import unittest

from openprop.association import AssociationPolicy
from openprop.query_decision import QueryDecisionPolicy
from openprop.visual_replay import replay_visual_case
from openprop.visual_replay_evaluation import evaluate_visual_replay

from test_visual_replay import VisualReplayTests


class VisualReplayEvaluationTests(unittest.TestCase):
    def setUp(self):
        fixture = VisualReplayTests()
        self.input_payload = fixture.input_payload()
        self.case_payload = fixture.case_payload()
        self.response = fixture.response()
        self.common = dict(
            association_policy=AssociationPolicy(
                acceptance_threshold=0.5,
                margin_threshold=0.1,
                null_weight=0.01,
            ),
            query_policy=QueryDecisionPolicy(
                acceptance_threshold=0.5,
                margin_threshold=0.1,
                null_weight=0.01,
            ),
        )

    def truth(self):
        return {
            "schema_version": 1,
            "evaluation_only": True,
            "case_id": "case-1",
            "cluster_id": "FloorPlan1",
            "split": "calibration",
            "source": "ai2thor-rgb",
            "condition": "one-change-one-distractor",
            "distractor_count": 1,
            "frames": [
                {
                    "frame_id": "before",
                    "events": [
                        {
                            "event_id": "event-open-e1",
                            "property_name": "open_state",
                            "gold_value": "open",
                            "target_entity_id": "e1",
                            "region": [0.1, 0.1, 0.3, 0.3],
                        }
                    ],
                }
            ],
            "query": {
                "record_id": "query-case-1",
                "property_name": "open_state",
                "target_entity_id": "e1",
                "horizon_seconds": 1.0,
                "eligible": True,
            },
        }

    def test_truth_is_attached_after_replay_to_all_three_analysis_units(self):
        outcome = replay_visual_case(
            self.input_payload,
            self.case_payload,
            self.response,
            assignment="global",
            **self.common,
        )
        dataset = evaluate_visual_replay(
            outcome, self.truth(), system="openprop-global", vlm_calls=1
        )
        self.assertEqual(len(dataset.properties), 1)
        self.assertEqual(len(dataset.associations), 1)
        self.assertEqual(len(dataset.queries), 1)
        self.assertTrue(dataset.properties[0].exact_value_match)
        self.assertTrue(dataset.associations[0].correct_update)
        self.assertTrue(dataset.queries[0].top1_correct)

    def test_independent_and_global_have_paired_query_population(self):
        datasets = []
        for assignment in ("independent", "global"):
            outcome = replay_visual_case(
                self.input_payload,
                self.case_payload,
                self.response,
                assignment=assignment,
                **self.common,
            )
            datasets.append(
                evaluate_visual_replay(
                    outcome,
                    self.truth(),
                    system=f"openprop-{assignment}",
                    vlm_calls=1,
                )
            )
        left, right = datasets
        self.assertEqual(left.queries[0].record_id, right.queries[0].record_id)
        self.assertEqual(left.queries[0].target_entity_id, right.queries[0].target_entity_id)
        self.assertEqual(
            left.queries[0].candidate_entity_ids,
            right.queries[0].candidate_entity_ids,
        )

    def test_case_and_frame_mismatch_fail_before_metric_construction(self):
        outcome = replay_visual_case(
            self.input_payload,
            self.case_payload,
            self.response,
            assignment="global",
            **self.common,
        )
        wrong_case = self.truth()
        wrong_case["case_id"] = "other"
        with self.assertRaisesRegex(ValueError, "does not match"):
            evaluate_visual_replay(outcome, wrong_case, system="openprop")
        missing_frame = self.truth()
        missing_frame["frames"] = [
            {"frame_id": "other-frame", "events": []}
        ]
        with self.assertRaisesRegex(ValueError, "cover every replay frame"):
            evaluate_visual_replay(outcome, missing_frame, system="openprop")

    def test_malformed_response_retains_truth_and_query_denominators(self):
        outcome = replay_visual_case(
            self.input_payload,
            self.case_payload,
            {"wrong_key": []},
            assignment="global",
            **self.common,
        )
        dataset = evaluate_visual_replay(
            outcome, self.truth(), system="openprop-malformed", vlm_calls=1
        )
        self.assertEqual(len(dataset.properties), 1)
        self.assertFalse(dataset.properties[0].detected)
        self.assertTrue(dataset.properties[0].malformed)
        self.assertEqual(dataset.associations, ())
        self.assertTrue(dataset.queries[0].malformed)


if __name__ == "__main__":
    unittest.main()
