import unittest
from dataclasses import replace

from openprop.models import MatchResult
from openprop.query_decision import (
    QueryDecisionPolicy,
    apply_query_acceptance_policy,
    build_visual_query_result,
    calibrate_query_acceptance_policy,
    decide_query_match,
)
from openprop.visual_evaluation import NULL_ENTITY
from openprop.visual_evaluation import VisualQueryResult


def match(entity_id, score, coverage=1.0):
    return MatchResult(entity_id, score, score, coverage, ())


class QueryDecisionTests(unittest.TestCase):
    def query_row(self, record_id, target, probability, *, split="calibration"):
        remainder = 1.0 - probability
        return VisualQueryResult(
            record_id,
            "cluster-" + record_id,
            split,
            "openprop",
            "ai2thor",
            "open_state",
            ("e1", "e2"),
            target,
            ("e1", "e2"),
            "e1",
            "e1",
            {"e1": probability, "e2": remainder / 2, NULL_ENTITY: remainder / 2},
            60.0,
            1,
            "delayed",
        )

    def test_strong_candidate_is_accepted_with_normalized_null_distribution(self):
        decision = decide_query_match(
            (match("e1", 0.9), match("e2", 0.1)),
            policy=QueryDecisionPolicy(
                acceptance_threshold=0.7,
                margin_threshold=0.5,
                null_weight=0.05,
            ),
        )
        self.assertEqual(decision.decision_entity_id, "e1")
        self.assertEqual(decision.accepted_entity_id, "e1")
        self.assertAlmostEqual(sum(decision.probabilities.values()), 1.0)
        self.assertIn(NULL_ENTITY, decision.probabilities)

    def test_null_wins_when_match_evidence_is_absent(self):
        decision = decide_query_match(
            (match("e1", 0.0, 0.0), match("e2", 0.0, 0.0)),
            policy=QueryDecisionPolicy(null_weight=0.05),
        )
        self.assertIsNone(decision.decision_entity_id)
        self.assertIsNone(decision.accepted_entity_id)
        self.assertEqual(decision.reason, "null has highest probability")

    def test_ambiguous_top_is_retained_as_decision_but_abstained(self):
        decision = decide_query_match(
            (match("e1", 0.9), match("e2", 0.89)),
            policy=QueryDecisionPolicy(
                acceptance_threshold=0.4,
                margin_threshold=0.1,
                null_weight=0.01,
            ),
        )
        self.assertEqual(decision.decision_entity_id, "e1")
        self.assertIsNone(decision.accepted_entity_id)
        self.assertIn("margin", decision.reason)

    def test_candidate_presentation_order_is_invariant(self):
        policy = QueryDecisionPolicy(null_weight=0.1)
        forward = decide_query_match(
            (match("e2", 0.3), match("e1", 0.8)), policy=policy
        )
        reverse = decide_query_match(
            (match("e1", 0.8), match("e2", 0.3)), policy=policy
        )
        self.assertEqual(forward, reverse)

    def test_result_builder_adds_truth_only_after_decision(self):
        decision = decide_query_match((match("e1", 0.8), match("e2", 0.2)))
        row = build_visual_query_result(
            decision,
            record_id="q1",
            cluster_id="scene-1",
            split="test",
            system="openprop",
            source="ai2thor",
            property_name="open_state",
            target_entity_id="e1",
            horizon_seconds=60.0,
            distractor_count=1,
            condition="delayed",
            vlm_calls=1,
        )
        self.assertTrue(row.top1_correct)
        self.assertEqual(row.target_entity_id, "e1")

    def test_invalid_scores_and_duplicate_ids_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "unique"):
            decide_query_match((match("e1", 0.4), match("e1", 0.5)))
        with self.assertRaisesRegex(ValueError, "score"):
            decide_query_match((match("e1", 1.1),))

    def test_query_calibration_selects_safe_threshold_on_calibration_only(self):
        rows = (
            self.query_row("correct", "e1", 0.8),
            self.query_row("false", "e2", 0.6),
        )
        policy = calibrate_query_acceptance_policy(
            rows,
            acceptance_thresholds=(0.5, 0.7),
            margin_thresholds=(0.0,),
            max_false_answer_rate=0.0,
        )
        self.assertEqual(policy.acceptance_threshold, 0.7)
        self.assertEqual(policy.correct_answers, 1)
        self.assertEqual(policy.false_answers, 0)
        with self.assertRaisesRegex(ValueError, "calibration"):
            calibrate_query_acceptance_policy(
                (self.query_row("test", "e1", 0.9, split="test"),),
                acceptance_thresholds=(0.5,),
                margin_thresholds=(0.0,),
                max_false_answer_rate=0.0,
            )

    def test_frozen_query_policy_application_is_target_blind(self):
        calibration = (
            self.query_row("correct", "e1", 0.8),
            self.query_row("false", "e2", 0.6),
        )
        policy = calibrate_query_acceptance_policy(
            calibration,
            acceptance_thresholds=(0.7,),
            margin_thresholds=(0.0,),
            max_false_answer_rate=0.0,
        )
        first = self.query_row("q", "e1", 0.8, split="test")
        second = self.query_row("q", "e2", 0.8, split="test")
        self.assertEqual(
            apply_query_acceptance_policy((first,), policy)[0].accepted_entity_id,
            apply_query_acceptance_policy((second,), policy)[0].accepted_entity_id,
        )

    def test_query_null_prior_can_scale_with_candidate_count(self):
        correct = self.query_row("correct-count", "e1", 0.8)
        crowded = VisualQueryResult(
            "crowded-null",
            "cluster-crowded",
            "calibration",
            "openprop",
            "ai2thor",
            "open_state",
            ("e1", "e2", "e3", "e4"),
            None,
            ("e1", "e2", "e3", "e4"),
            "e1",
            None,
            {"e1": 0.4, "e2": 0.1, "e3": 0.1, "e4": 0.1, NULL_ENTITY: 0.3},
            60.0,
            3,
            "crowded-no-target",
        )
        policy = calibrate_query_acceptance_policy(
            (correct, crowded),
            acceptance_thresholds=(0.4,),
            margin_thresholds=(0.0,),
            null_scales=(0.5,),
            candidate_count_powers=(0.0, 1.0),
            max_false_answer_rate=0.0,
        )
        self.assertEqual(policy.candidate_count_power, 1.0)
        self.assertEqual(policy.candidate_count_levels, 2)
        self.assertEqual(policy.supported_candidate_counts, (2, 4))
        applied = apply_query_acceptance_policy(
            (replace(crowded, split="test", cluster_id="test-crowded"),), policy
        )[0]
        self.assertIsNone(applied.decision_entity_id)
        self.assertTrue(applied.top1_correct)

    def test_query_count_power_requires_multiple_count_levels(self):
        with self.assertRaisesRegex(ValueError, "multiple candidate counts"):
            calibrate_query_acceptance_policy(
                (self.query_row("single-count", "e1", 0.8),),
                acceptance_thresholds=(0.5,),
                margin_thresholds=(0.0,),
                null_scales=(1.0,),
                candidate_count_powers=(1.0,),
                max_false_answer_rate=1.0,
            )

    def test_query_unseen_candidate_count_abstains(self):
        policy = calibrate_query_acceptance_policy(
            (self.query_row("known-count", "e1", 0.8),),
            acceptance_thresholds=(0.5,), margin_thresholds=(0.0,),
            max_false_answer_rate=1.0,
        )
        unseen = VisualQueryResult(
            "unseen-count", "test-unseen", "test", "openprop", "ai2thor",
            "open_state", ("e1", "e2", "e3"), "e1",
            ("e1", "e2", "e3"), "e1", None,
            {"e1": 0.8, "e2": 0.05, "e3": 0.05, NULL_ENTITY: 0.1},
            60.0, 2, "unseen-count",
        )
        applied = apply_query_acceptance_policy((unseen,), policy)[0]
        self.assertFalse(applied.accepted)


if __name__ == "__main__":
    unittest.main()
