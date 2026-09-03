import unittest
from dataclasses import replace

from openprop.association import AssociationPolicy
from openprop.query_decision import QueryDecisionPolicy
from openprop.visual_evaluation import VisualEvaluationDataset
from openprop.visual_matrix import combine_visual_evaluation_datasets
from openprop.visual_replay import replay_visual_case
from openprop.visual_replay_evaluation import evaluate_visual_replay
from test_visual_replay import VisualReplayTests
from test_visual_replay_evaluation import VisualReplayEvaluationTests


class VisualMatrixTests(unittest.TestCase):
    def datasets(self):
        fixture = VisualReplayTests()
        truth = VisualReplayEvaluationTests().truth()
        common = dict(
            association_policy=AssociationPolicy(
                acceptance_threshold=0.5, margin_threshold=0.1, null_weight=0.01
            ),
            query_policy=QueryDecisionPolicy(
                acceptance_threshold=0.5, margin_threshold=0.1, null_weight=0.01
            ),
        )
        result = []
        for assignment in ("independent", "global"):
            outcome = replay_visual_case(
                fixture.input_payload(), fixture.case_payload(), fixture.response(),
                assignment=assignment, **common
            )
            result.append(
                evaluate_visual_replay(
                    outcome, truth, system=f"openprop-{assignment}", vlm_calls=1
                )
            )
        return tuple(result)

    def test_combiner_enforces_and_hashes_paired_core_populations(self):
        combined, report = combine_visual_evaluation_datasets(
            self.datasets(),
            required_systems=("openprop-independent", "openprop-global"),
        )
        self.assertEqual(len(combined.queries), 2)
        self.assertTrue(report["paired_query_population"])
        self.assertTrue(report["paired_expected_property_population"])
        self.assertEqual(len(report["population_hashes"]["query"]), 64)

    def test_query_population_mismatch_fails_closed(self):
        left, right = self.datasets()
        changed_query = replace(right.queries[0], horizon_seconds=99.0)
        changed = VisualEvaluationDataset(
            right.properties, right.associations, (changed_query,)
        )
        with self.assertRaisesRegex(ValueError, "query population is not paired"):
            combine_visual_evaluation_datasets((left, changed))

    def test_required_system_mismatch_and_duplicate_rows_fail_closed(self):
        datasets = self.datasets()
        with self.assertRaisesRegex(ValueError, "required systems"):
            combine_visual_evaluation_datasets(
                datasets, required_systems=("wrong-a", "wrong-b")
            )
        with self.assertRaisesRegex(ValueError, "unique"):
            combine_visual_evaluation_datasets((datasets[0], datasets[0]))


if __name__ == "__main__":
    unittest.main()
