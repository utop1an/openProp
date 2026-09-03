import math
import unittest

from openprop.persistence_data import PersistenceTrainingExample
from openprop.survival_evaluation import exponential_example_negative_log_likelihood
from openprop.synthetic_survival_oracle import SyntheticWeibullOracle


FEATURES = ("location", "cup", "on", "table", "quiet")


def _row(
    duration_hours: float,
    event_observed: bool,
    interval_start_hours: float | None = None,
) -> PersistenceTrainingExample:
    return PersistenceTrainingExample(
        *FEATURES,
        duration_hours * 3600.0,
        event_observed,
        "oracle-test",
        None if interval_start_hours is None else interval_start_hours * 3600.0,
    )


class SyntheticSurvivalOracleTests(unittest.TestCase):
    def test_shape_one_matches_exponential_event_likelihood(self) -> None:
        oracle = SyntheticWeibullOracle({FEATURES: 0.2}, 1.0)
        for row in (_row(3.0, True), _row(3.0, False), _row(3.0, True, 1.0)):
            self.assertAlmostEqual(
                exponential_example_negative_log_likelihood(0.2, row),
                oracle.example_negative_log_likelihood(row),
            )

    def test_nonexponential_survival_and_interval_mass_are_exact(self) -> None:
        oracle = SyntheticWeibullOracle({FEATURES: 0.25}, 1.6)
        self.assertAlmostEqual(
            math.exp(-((0.25 * 4.0) ** 1.6)),
            oracle.survival_probability_at_hours(FEATURES, 4.0),
        )
        row = _row(4.0, True, 2.0)
        expected_mass = math.exp(-((0.25 * 2.0) ** 1.6)) - math.exp(-1.0)
        self.assertAlmostEqual(
            -math.log(expected_mass),
            oracle.example_negative_log_likelihood(row),
        )

    def test_unknown_context_is_not_silently_backed_off(self) -> None:
        oracle = SyntheticWeibullOracle({FEATURES: 0.2}, 1.0)
        with self.assertRaises(KeyError):
            oracle.risk_score(("location", "book", "on", "table", "quiet"))


if __name__ == "__main__":
    unittest.main()
