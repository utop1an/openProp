import copy
import unittest

from scripts.build_visual_experiment_artifacts import build_artifacts
from test_visual_evaluation import VisualEvaluationTests
from openprop.visual_evaluation import aggregate_visual_evaluation


class VisualExperimentArtifactTests(unittest.TestCase):
    def report(self):
        fixture = VisualEvaluationTests()
        return aggregate_visual_evaluation(
            fixture.dataset(),
            split="test",
            ece_bins=5,
        )

    def test_build_is_deterministic_and_outputs_table_plot_and_data(self):
        report = self.report()
        first = build_artifacts(report)
        second = build_artifacts(report)
        self.assertEqual(first, second)
        self.assertEqual(
            set(first),
            {
                "main_results.md",
                "main_results.tex",
                "confidence_calibration.md",
                "confidence_calibration.tex",
                "plot_data.json",
                "visual_evaluation.png",
                "combined_confidence.png",
            },
        )
        self.assertTrue(first["visual_evaluation.png"].startswith(b"\x89PNG\r\n"))
        self.assertTrue(first["combined_confidence.png"].startswith(b"\x89PNG\r\n"))

    def test_table_is_booktabs_metric_directed_and_denominator_explicit(self):
        artifacts = build_artifacts(self.report())
        latex = artifacts["main_results.tex"].decode("utf-8")
        markdown = artifacts["main_results.md"].decode("utf-8")
        self.assertIn("\\toprule", latex)
        self.assertIn("\\bottomrule", latex)
        self.assertNotIn("{|", latex)
        self.assertIn("\\uparrow", latex)
        self.assertIn("\\downarrow", latex)
        self.assertIn("failures and abstentions remain", markdown)
        self.assertIn("Query N", markdown)

    def test_population_mismatch_fails_before_comparison(self):
        report = self.report()
        systems = report["systems"]
        systems["other"] = copy.deepcopy(systems["openprop"])
        systems["other"]["population_hashes"]["query"] = "different"
        with self.assertRaisesRegex(ValueError, "query populations differ"):
            build_artifacts(report)

    def test_plot_data_contains_predeclared_axes(self):
        artifacts = build_artifacts(self.report())
        payload = artifacts["plot_data.json"].decode("utf-8")
        for field in (
            "query_reliability",
            "query_risk_coverage",
            "query_by_horizon_seconds",
            "association_by_distractors",
            "combined_update_confidence",
        ):
            self.assertIn(field, payload)


if __name__ == "__main__":
    unittest.main()
