import unittest
from dataclasses import replace

from openprop.visual_evaluation import VisualEvaluationDataset
from openprop.visual_primary_statistics import primary_visual_query_comparisons
from test_visual_statistics import VisualStatisticsTests


class VisualPrimaryStatisticsTests(unittest.TestCase):
    def dataset(self):
        fixture = VisualStatisticsTests()
        queries = []
        for index in range(6):
            cluster = "room-1" if index < 3 else "room-2"
            record = f"case-{index}"
            queries.extend((
                fixture.query(record, cluster, "base-a", False),
                fixture.query(record, cluster, "base-b", index % 2 == 0),
                fixture.query(record, cluster, "openprop", True),
            ))
        return VisualEvaluationDataset(queries=tuple(queries))

    def test_shared_bootstrap_simultaneous_intervals_and_holm_are_deterministic(self):
        first = primary_visual_query_comparisons(
            self.dataset(), main_system="openprop", baselines=("base-a", "base-b"),
            split="test", bootstrap_replicates=200, seed=31,
        )
        second = primary_visual_query_comparisons(
            self.dataset(), main_system="openprop", baselines=("base-a", "base-b"),
            split="test", bootstrap_replicates=200, seed=31,
        )
        self.assertEqual(first, second)
        self.assertEqual(first["population"], 6)
        self.assertTrue(first["bootstrap"]["shared_resamples_across_primary_family"])
        self.assertIn(
            "familywise_simultaneous_95_ci", first["comparisons"]["base-a"]["top1"]
        )
        self.assertGreaterEqual(
            first["comparisons"]["base-a"]["top1"]["holm_adjusted_p"],
            first["comparisons"]["base-a"]["top1"]["mcnemar_exact_p"],
        )

    def test_population_and_truth_drift_fail_closed(self):
        dataset = self.dataset()
        with self.assertRaisesRegex(ValueError, "not exactly paired"):
            primary_visual_query_comparisons(
                VisualEvaluationDataset(queries=dataset.queries[:-1]),
                main_system="openprop", baselines=("base-a", "base-b"), split="test",
                bootstrap_replicates=100,
            )
        changed = replace(dataset.queries[-1], horizon_seconds=999.0)
        with self.assertRaisesRegex(ValueError, "truth/input fields drifted"):
            primary_visual_query_comparisons(
                VisualEvaluationDataset(queries=(*dataset.queries[:-1], changed)),
                main_system="openprop", baselines=("base-a", "base-b"), split="test",
                bootstrap_replicates=100,
            )


if __name__ == "__main__":
    unittest.main()
