import unittest

from openprop.association import AssociationPolicy
from openprop.query_decision import QueryDecisionPolicy
from openprop.visual_replay import replay_visual_case


class VisualReplayTests(unittest.TestCase):
    def input_payload(self):
        return {
            "schema_version": 1,
            "episode_id": "FloorPlan1.open",
            "frames": [
                {
                    "frame_id": "before",
                    "image_url": "before.png",
                    "captured_at": 0.0,
                    "source": "ai2thor-rgb",
                    "candidate_entity_ids": ["e1", "e2"],
                    "candidate_regions": {
                        "e1": [0.1, 0.1, 0.3, 0.3],
                        "e2": [0.6, 0.1, 0.8, 0.3],
                    },
                }
            ],
        }

    def case_payload(self):
        return {
            "schema_version": 1,
            "case_id": "case-1",
            "query_time": 1.0,
            "query_candidate_entity_ids": ["e1", "e2"],
            "query": {
                "text": "which cup is open",
                "constraints": [
                    {
                        "property_name": "open_state",
                        "desired_value": "open",
                        "relevance": 1.0,
                    }
                ],
            },
            "initial_entities": [
                {"entity_id": "e1", "properties": {}},
                {"entity_id": "e2", "properties": {}},
            ],
        }

    def response(self):
        return {
            "detections": [
                {
                    "detection_id": "d1",
                    "frame_id": "before",
                    "track_id": None,
                    "property_name": "open_state",
                    "value_type": "categorical",
                    "detection_confidence": 0.99,
                    "value_confidence": 0.99,
                    "candidate_affinities": [
                        {"entity_id": "e1", "affinity": 0.99},
                        {"entity_id": "e2", "affinity": 0.01},
                    ],
                    "track_affinities": [],
                    "region": [0.1, 0.1, 0.3, 0.3],
                    "value": {"kind": "scalar", "scalar": "open"},
                }
            ]
        }

    def test_truth_free_replay_updates_ledger_then_answers_query(self):
        outcome = replay_visual_case(
            self.input_payload(),
            self.case_payload(),
            self.response(),
            assignment="global",
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
        self.assertEqual(len(outcome.run.proposals), 1)
        self.assertEqual(outcome.run.proposals[0].entity_id, "e1")
        self.assertEqual(outcome.query_decision.accepted_entity_id, "e1")

    def test_independent_and_global_share_the_same_truth_free_population(self):
        common = dict(
            association_policy=AssociationPolicy(
                acceptance_threshold=0.5, margin_threshold=0.1, null_weight=0.01
            ),
            query_policy=QueryDecisionPolicy(null_weight=0.01),
        )
        independent = replay_visual_case(
            self.input_payload(), self.case_payload(), self.response(),
            assignment="independent", **common
        )
        global_result = replay_visual_case(
            self.input_payload(), self.case_payload(), self.response(),
            assignment="global", **common
        )
        self.assertEqual(independent.query, global_result.query)
        self.assertEqual(
            independent.run.detections[0].frame,
            global_result.run.detections[0].frame,
        )

    def test_truth_fields_and_future_initial_observations_fail_closed(self):
        leaked = self.case_payload()
        leaked["target_entity_id"] = "e1"
        with self.assertRaisesRegex(ValueError, "truth fields"):
            replay_visual_case(
                self.input_payload(), leaked, self.response(), assignment="global"
            )
        future = self.case_payload()
        future["initial_entities"][0]["properties"] = {
            "type": {
                "value": "Cup",
                "state": "observed",
                "timestamp": 0.0,
            }
        }
        with self.assertRaisesRegex(ValueError, "strictly precede"):
            replay_visual_case(
                self.input_payload(), future, self.response(), assignment="global"
            )

    def test_malformed_model_response_is_retained_as_empty_truth_free_run(self):
        outcome = replay_visual_case(
            self.input_payload(),
            self.case_payload(),
            {"wrong_key": []},
            assignment="global",
        )
        self.assertTrue(outcome.malformed_response)
        self.assertIn("detections array", outcome.response_error)
        self.assertEqual(outcome.run.detections, ())
        self.assertEqual(outcome.run.proposals, ())
        self.assertIsNone(outcome.query_decision.accepted_entity_id)


if __name__ == "__main__":
    unittest.main()
