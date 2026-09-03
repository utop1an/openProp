from __future__ import annotations

import unittest

from openprop.source_reliability_evaluation import (
    source_filtered_state_one_probability,
    source_grounding_negative_log_likelihood,
    sourced_grounding_test_rows,
)
from openprop.source_reliability_observation import (
    EstimatedSourceReliabilityProcess,
    SourceEmissionParameters,
)


class SourceReliabilityEvaluationTests(unittest.TestCase):
    def test_truth_is_evaluation_only_and_filter_uses_history(self) -> None:
        parameters = (
            SourceEmissionParameters("a", 0.85, 0.45, 0.95, 0.02),
            SourceEmissionParameters("b", 0.45, 0.85, 0.65, 0.18),
        )
        rows = sourced_grounding_test_rows(
            seed=41,
            row_count=30,
            forward_rate_per_hour=0.3,
            return_rate_per_hour=0.45,
            source_parameters=parameters,
        )
        process = EstimatedSourceReliabilityProcess(
            0.3, 0.45, parameters, False, 0, True, (), 0
        )
        history = rows[0].observation_history
        self.assertFalse(hasattr(history, "current_truth"))
        probability = source_filtered_state_one_probability(history, process)
        self.assertGreaterEqual(probability, 0.0)
        self.assertLessEqual(probability, 1.0)
        self.assertGreaterEqual(source_grounding_negative_log_likelihood(rows, process), 0.0)

    def test_filter_rejects_source_mismatch(self) -> None:
        truth = (SourceEmissionParameters("a", 0.7, 0.7, 0.9, 0.05),)
        rows = sourced_grounding_test_rows(
            seed=5,
            row_count=1,
            forward_rate_per_hour=0.3,
            return_rate_per_hour=0.45,
            source_parameters=truth,
        )
        wrong = EstimatedSourceReliabilityProcess(
            0.3,
            0.45,
            (SourceEmissionParameters("wrong", 0.7, 0.7, 0.9, 0.05),),
            False,
            0,
            True,
            (),
            0,
        )
        with self.assertRaisesRegex(ValueError, "source IDs must align"):
            source_filtered_state_one_probability(rows[0].observation_history, wrong)


if __name__ == "__main__":
    unittest.main()
