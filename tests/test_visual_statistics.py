import unittest
from dataclasses import replace

from openprop.visual_evaluation import (
    NULL_ENTITY,
    VisualAssociationResult,
    VisualEvaluationDataset,
    VisualQueryResult,
)
from openprop.visual_statistics import paired_visual_system_comparison


class VisualStatisticsTests(unittest.TestCase):
    def query(self, record, cluster, system, correct):
        decision = "e1" if correct else "e2"
        return VisualQueryResult(
            record,
            cluster,
            "test",
            system,
            "real-video-rgb",
            "motion_state",
            ("e1", "e2"),
            "e1",
            ("e1", "e2") if correct else ("e2", "e1"),
            decision,
            decision,
            {
                "e1": 0.8 if correct else 0.1,
                "e2": 0.1 if correct else 0.8,
                NULL_ENTITY: 0.1,
            },
            60.0,
            1,
            "moved-crowded",
        )

    def association(self, record, cluster, system, outcome):
        if outcome == "false":
            decision = accepted = "e2"
            probabilities = {"e1": 0.1, "e2": 0.8, NULL_ENTITY: 0.1}
        elif outcome == "correct":
            decision = accepted = "e1"
            probabilities = {"e1": 0.8, "e2": 0.1, NULL_ENTITY: 0.1}
        else:
            decision = accepted = None
            probabilities = {"e1": 0.2, "e2": 0.2, NULL_ENTITY: 0.6}
        return VisualAssociationResult(
            record,
            cluster,
            "test",
            system,
            "real-video-rgb",
            "motion_state",
            f"detection-{record}",
            f"frame-{record}",
            ("e1", "e2"),
            "e1",
            decision,
            accepted,
            probabilities,
            "moved-crowded",
            1,
        )

    def dataset(self):
        queries = []
        associations = []
        for index in range(4):
            cluster = "room-1" if index < 2 else "room-2"
            record = f"case-{index}"
            queries.extend(
                (
                    self.query(record, cluster, "baseline", False),
                    self.query(record, cluster, "openprop", True),
                )
            )
            associations.extend(
                (
                    self.association(record, cluster, "baseline", "false"),
                    self.association(record, cluster, "openprop", "correct"),
                )
            )
        return VisualEvaluationDataset(
            associations=tuple(associations), queries=tuple(queries)
        )

    def test_paired_cluster_bootstrap_and_exact_tests_are_deterministic(self):
        first = paired_visual_system_comparison(
            self.dataset(),
            baseline="baseline",
            system="openprop",
            split="test",
            bootstrap_replicates=200,
            seed=17,
        )
        second = paired_visual_system_comparison(
            self.dataset(),
            baseline="baseline",
            system="openprop",
            split="test",
            bootstrap_replicates=200,
            seed=17,
        )
        self.assertEqual(first, second)
        top1 = first["query"]["metrics"]["top1"]
        self.assertEqual(top1["delta_system_minus_baseline"], 1.0)
        self.assertEqual(top1["cluster_bootstrap_95_ci"], [1.0, 1.0])
        self.assertEqual(top1["mcnemar_exact_p"], 0.125)
        false_update = first["association"]["metrics"]["false_update_rate"]
        self.assertEqual(false_update["delta_system_minus_baseline"], -1.0)
        self.assertEqual(first["query"]["population"], 4)
        self.assertEqual(first["query"]["clusters"], 2)

    def test_population_and_truth_drift_fail_closed(self):
        dataset = self.dataset()
        missing = VisualEvaluationDataset(
            associations=dataset.associations,
            queries=dataset.queries[:-1],
        )
        with self.assertRaisesRegex(ValueError, "not exactly paired"):
            paired_visual_system_comparison(
                missing,
                baseline="baseline",
                system="openprop",
                split="test",
                bootstrap_replicates=100,
            )
        changed = replace(dataset.queries[-1], horizon_seconds=99.0)
        drifted = VisualEvaluationDataset(
            associations=dataset.associations,
            queries=(*dataset.queries[:-1], changed),
        )
        with self.assertRaisesRegex(ValueError, "truth/input fields drifted"):
            paired_visual_system_comparison(
                drifted,
                baseline="baseline",
                system="openprop",
                split="test",
                bootstrap_replicates=100,
            )

    def test_unpaired_association_rows_cannot_be_compared(self):
        dataset = self.dataset()
        unpaired = VisualEvaluationDataset(
            associations=dataset.associations[:-1], queries=dataset.queries
        )
        with self.assertRaisesRegex(ValueError, "association populations"):
            paired_visual_system_comparison(
                unpaired,
                baseline="baseline",
                system="openprop",
                split="test",
                bootstrap_replicates=100,
            )
        report = paired_visual_system_comparison(
            unpaired,
            baseline="baseline",
            system="openprop",
            split="test",
            bootstrap_replicates=100,
            include_association=False,
        )
        self.assertEqual(report["association"]["status"], "not_requested")
        self.assertEqual(report["query"]["population"], 4)


if __name__ == "__main__":
    unittest.main()
