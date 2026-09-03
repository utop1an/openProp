import math
import unittest

from openprop.informative_observation import (
    ObservationAwareExponentialModel,
    ObservationEpisode,
    informative_observation_data,
)
from openprop.observation_em import fit_observation_process_em


class FalsePositiveObservationTests(unittest.TestCase):
    def test_likelihood_assigns_positive_mass_before_and_after_transition(self) -> None:
        episode = ObservationEpisode("episode", 2.0, ("positive",))
        loss = ObservationAwareExponentialModel._episode_negative_log_likelihood(
            episode,
            0.2,
            0.1,
            0.8,
            0.75,
            0.05,
        )
        survival = math.exp(-0.2 * 2.0)
        expected = survival * 0.1 * 0.05 + (1.0 - survival) * 0.8 * 0.75
        self.assertAlmostEqual(-math.log(expected), loss)

    def test_false_positive_conditions_keep_latent_test_draws_paired(self) -> None:
        clean = informative_observation_data(
            train_samples=30,
            test_samples=20,
            false_positive_rate=0.0,
            seed=101,
        )
        noisy = informative_observation_data(
            train_samples=30,
            test_samples=20,
            false_positive_rate=0.1,
            seed=101,
        )
        self.assertEqual(clean.exact_test, noisy.exact_test)
        self.assertNotEqual(clean.episodes, noisy.episodes)
        self.assertEqual(0.1, noisy.false_positive_rate)

    def test_training_only_em_recovers_specificity_and_removes_hazard_bias(self) -> None:
        dataset = informative_observation_data(
            train_samples=1200,
            test_samples=20,
            false_positive_rate=0.1,
            seed=101,
        )
        misspecified = fit_observation_process_em(dataset.episodes)
        estimated = fit_observation_process_em(
            dataset.episodes,
            estimate_false_positive_rate=True,
        )
        self.assertTrue(estimated.converged)
        self.assertEqual(0.0, misspecified.false_positive_rate)
        self.assertLess(abs(estimated.false_positive_rate - 0.1), 0.02)
        self.assertLess(abs(estimated.hazard_per_hour - 0.25), 0.02)
        self.assertLess(
            abs(estimated.hazard_per_hour - 0.25),
            abs(misspecified.hazard_per_hour - 0.25),
        )
        self.assertTrue(
            all(
                current <= previous + 1e-9
                for previous, current in zip(
                    estimated.average_negative_log_likelihood_history[:-1],
                    estimated.average_negative_log_likelihood_history[1:],
                    strict=True,
                )
            )
        )

    def test_false_positive_parameters_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "false_positive_rate"):
            informative_observation_data(false_positive_rate=1.0)
        with self.assertRaisesRegex(ValueError, "false_positive_rate"):
            ObservationAwareExponentialModel(0.2, 0.2, 0.8, 0.7, -0.1)


if __name__ == "__main__":
    unittest.main()
