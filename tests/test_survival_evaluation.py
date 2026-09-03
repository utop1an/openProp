import math
import unittest

from openprop.persistence_data import PersistenceTrainingExample
from openprop.survival_evaluation import (
    evaluate_survival,
    exponential_example_negative_log_likelihood,
)


class ConstantHazard:
    def __init__(self, hazard):
        self.hazard = hazard

    def hazard_per_hour(self, features):
        return self.hazard


class CustomSurvival:
    def hazard_per_hour(self, features):
        raise AssertionError("exponential fallback should not be used")

    def survival_probability_at_hours(self, features, duration_hours):
        return 0.25

    def example_negative_log_likelihood(self, example):
        return 1.75


class SurvivalEvaluationTests(unittest.TestCase):
    def test_interval_likelihood_uses_probability_between_inspections(self):
        row = PersistenceTrainingExample(
            "location", "cup", "on", "table", "kitchen", 4 * 3600, True,
            interval_start_seconds=2 * 3600,
        )
        expected = -math.log(math.exp(-0.4) - math.exp(-0.8))
        self.assertAlmostEqual(
            expected, exponential_example_negative_log_likelihood(0.2, row)
        )

    def test_interval_straddling_horizon_is_not_scored(self):
        rows = (
            PersistenceTrainingExample(
                "location", "cup", "on", "table", "kitchen", 4 * 3600, True,
                interval_start_seconds=1 * 3600,
            ),
            PersistenceTrainingExample(
                "location", "cup", "on", "table", "kitchen", 3 * 3600, False
            ),
        )
        horizon = evaluate_survival(
            ConstantHazard(0.2), rows, horizons_hours=(2.0,)
        ).horizons[0]
        self.assertEqual(1, horizon.evaluable_examples)

    def test_excludes_censored_before_horizon_from_brier(self):
        rows = (
            PersistenceTrainingExample(
                "location", "cup", "on", "table", "kitchen", 3600, False
            ),
            PersistenceTrainingExample(
                "location", "cup", "on", "table", "kitchen", 3 * 3600, False
            ),
            PersistenceTrainingExample(
                "location", "cup", "on", "table", "kitchen", 1800, True
            ),
        )
        evaluation = evaluate_survival(
            ConstantHazard(0.2), rows, horizons_hours=(2.0,), calibration_bins=5
        )
        horizon = evaluation.horizons[0]
        self.assertEqual(horizon.evaluable_examples, 2)
        prediction = math.exp(-0.4)
        expected_brier = ((prediction - 1) ** 2 + prediction**2) / 2
        self.assertAlmostEqual(horizon.brier_score, expected_brier)

    def test_model_specific_curve_and_likelihood_override_exponential_fallback(self):
        rows = (
            PersistenceTrainingExample(
                "location", "cup", "on", "table", "kitchen", 3 * 3600, False
            ),
            PersistenceTrainingExample(
                "location", "cup", "on", "table", "kitchen", 1800, True
            ),
        )
        evaluation = evaluate_survival(
            CustomSurvival(), rows, horizons_hours=(2.0,)
        )
        self.assertEqual(1.75, evaluation.negative_log_likelihood)
        expected_brier = ((0.25 - 1.0) ** 2 + 0.25**2) / 2
        self.assertEqual(expected_brier, evaluation.horizons[0].brier_score)


if __name__ == "__main__":
    unittest.main()
