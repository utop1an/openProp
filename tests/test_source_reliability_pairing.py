from __future__ import annotations

import unittest

from openprop.source_reliability_evaluation import sourced_grounding_test_rows
from openprop.source_reliability_observation import SourceEmissionParameters


class SourceReliabilityPairingTests(unittest.TestCase):
    def test_severity_conditions_share_evaluation_state_paths(self) -> None:
        common = (
            SourceEmissionParameters("a", 0.65, 0.65, 0.80, 0.10),
            SourceEmissionParameters("b", 0.65, 0.65, 0.80, 0.10),
        )
        conflicting = (
            SourceEmissionParameters("a", 0.90, 0.40, 0.95, 0.02),
            SourceEmissionParameters("b", 0.40, 0.90, 0.65, 0.18),
        )
        controls = sourced_grounding_test_rows(
            seed=101,
            row_count=80,
            forward_rate_per_hour=0.3,
            return_rate_per_hour=0.45,
            source_parameters=common,
        )
        stressed = sourced_grounding_test_rows(
            seed=101,
            row_count=80,
            forward_rate_per_hour=0.3,
            return_rate_per_hour=0.45,
            source_parameters=conflicting,
        )
        self.assertEqual(
            tuple(row.current_truth for row in controls),
            tuple(row.current_truth for row in stressed),
        )
        self.assertNotEqual(
            tuple(row.observation_history for row in controls),
            tuple(row.observation_history for row in stressed),
        )


if __name__ == "__main__":
    unittest.main()
