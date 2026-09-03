import unittest
from dataclasses import replace

from openprop.candidate_evaluation import aggregate_candidate_tracking_matrix
from scripts.build_candidate_experiment_artifacts import build_candidate_artifacts
from test_candidate_evaluation import CandidateEvaluationTests
from test_candidate_statistics import CandidateStatisticsTests
from openprop.candidate_statistics import paired_candidate_system_comparison
from openprop.candidate_tracking import CandidateTracker


class CandidateArtifactTests(unittest.TestCase):
    def report(self):
        fixture = CandidateEvaluationTests()
        run = CandidateTracker().track(
            (fixture.frame(0), fixture.frame(1)),
            (
                fixture.proposal(0, (0.1, 0.1, 0.3, 0.3)),
                fixture.proposal(1, (0.1, 0.1, 0.3, 0.3)),
            ),
        )
        first = fixture.evaluate(run, fixture.truth(), cluster="room-a")
        second = replace(first, cluster_id="room-b", system="external-tracker")
        return aggregate_candidate_tracking_matrix((first, second), split="test")

    def test_candidate_table_and_plot_are_deterministic(self):
        first = build_candidate_artifacts(self.report())
        second = build_candidate_artifacts(self.report())
        self.assertEqual(first, second)
        self.assertEqual(
            set(first),
            {
                "candidate_results.md",
                "candidate_results.tex",
                "candidate_plot_data.json",
                "candidate_tracking.png",
            },
        )
        self.assertTrue(first["candidate_tracking.png"].startswith(b"\x89PNG\r\n"))
        markdown = first["candidate_results.md"].decode()
        latex = first["candidate_results.tex"].decode()
        self.assertIn("Every frame, miss, false positive", markdown)
        self.assertIn("Query target recall", markdown)
        self.assertIn("\\toprule", latex)
        self.assertIn("IDSw", latex)

    def test_paired_intervals_are_rendered_in_table_and_plot(self):
        rows = CandidateStatisticsTests().dataset()
        report = aggregate_candidate_tracking_matrix(rows, split="test")
        comparison = paired_candidate_system_comparison(
            rows, baseline="baseline", system="openprop", split="test",
            bootstrap_replicates=100, seed=7,
        )
        artifacts = build_candidate_artifacts(report, comparison)
        markdown = artifacts["candidate_results.md"].decode()
        latex = artifacts["candidate_results.tex"].decode()
        plot_data = artifacts["candidate_plot_data.json"].decode()
        self.assertIn("Paired cluster-bootstrap 95% CI", markdown)
        self.assertIn("candidate_recall", markdown)
        self.assertIn("tab:candidate-paired", latex)
        self.assertIn("paired_comparison", plot_data)
        self.assertTrue(artifacts["candidate_tracking.png"].startswith(b"\x89PNG\r\n"))

    def test_comparison_split_drift_fails_closed(self):
        rows = CandidateStatisticsTests().dataset()
        comparison = paired_candidate_system_comparison(
            rows, baseline="baseline", system="openprop", split="test",
            bootstrap_replicates=100,
        )
        with self.assertRaisesRegex(ValueError, "splits differ"):
            build_candidate_artifacts(
                {**aggregate_candidate_tracking_matrix(rows, split="test"), "split": "calibration"},
                comparison,
            )


if __name__ == "__main__":
    unittest.main()
