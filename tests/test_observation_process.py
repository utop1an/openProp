import math
import unittest

from openprop.informative_observation import (
    ObservationAwareExponentialModel,
    ObservationEpisode,
    informative_observation_data,
)
from openprop.observation_em import fit_observation_process_em
from openprop.observation_process import observation_process_data
from openprop.statistical_persistence import (
    FactorizedExponentialPersistenceModel,
    GlobalExponentialPersistenceModel,
    PerContextExponentialPersistenceModel,
)


class ObservationProcessTests(unittest.TestCase):
    def test_interval_aware_fit_reduces_inspection_frequency_bias(self) -> None:
        dataset = observation_process_data(
            samples_per_schedule=800,
            test_samples_per_schedule=20,
            seed=101,
        )
        naive = PerContextExponentialPersistenceModel.fit(
            dataset.naive_train, prior_exposure_hours=0
        )
        interval = PerContextExponentialPersistenceModel.fit(
            dataset.interval_train, prior_exposure_hours=0
        )
        contexts = {row.features() for row in dataset.interval_train}
        true_hazard = dataset.true_hazard_per_hour
        naive_error = sum(
            abs(naive.hazard_per_hour(context) - true_hazard)
            for context in contexts
        )
        interval_error = sum(
            abs(interval.hazard_per_hour(context) - true_hazard)
            for context in contexts
        )
        self.assertLess(interval_error, naive_error * 0.4)

    def test_factorized_fit_supports_interval_censoring(self) -> None:
        dataset = observation_process_data(
            samples_per_schedule=300,
            test_samples_per_schedule=10,
            seed=101,
        )
        model = FactorizedExponentialPersistenceModel.fit(
            dataset.interval_train, epochs=600
        )
        contexts = {row.features() for row in dataset.interval_train}
        hazards = [model.hazard_per_hour(context) for context in contexts]
        self.assertLess(max(hazards) - min(hazards), 0.02)
        self.assertLess(
            sum(abs(value - dataset.true_hazard_per_hour) for value in hazards)
            / len(hazards),
            0.03,
        )

    def test_observation_aware_likelihood_marginalizes_hidden_state(self) -> None:
        episode = ObservationEpisode("episode-1", 2.0, ("positive",))
        loss = ObservationAwareExponentialModel._episode_negative_log_likelihood(
            episode,
            0.2,
            0.1,
            0.8,
            0.75,
        )
        expected_probability = (1.0 - math.exp(-0.2 * 2.0)) * 0.8 * 0.75
        self.assertAlmostEqual(-math.log(expected_probability), loss)

    def test_informative_training_logs_do_not_store_transition_truth(self) -> None:
        dataset = informative_observation_data(
            train_samples=30,
            test_samples=20,
            seed=101,
        )
        self.assertEqual(
            {"group_id", "opportunity_interval_hours", "results"},
            set(dataset.episodes[0].__dataclass_fields__),
        )
        self.assertTrue(
            {row.group_id for row in dataset.interval_train}.isdisjoint(
                row.group_id for row in dataset.exact_test
            )
        )
        self.assertTrue(
            any("missing" in episode.results for episode in dataset.episodes)
        )
        paired = informative_observation_data(
            train_samples=30,
            test_samples=20,
            pre_transition_inspection_probability=0.35,
            post_transition_inspection_probability=0.35,
            detection_sensitivity=1.0,
            seed=101,
        )
        self.assertEqual(dataset.exact_test, paired.exact_test)

    def test_observation_aware_fit_recovers_hazard_under_informative_missingness(
        self,
    ) -> None:
        dataset = informative_observation_data(
            train_samples=1800,
            test_samples=20,
            seed=101,
        )
        naive = GlobalExponentialPersistenceModel.fit(dataset.naive_train)
        interval = GlobalExponentialPersistenceModel.fit(dataset.interval_train)
        joint = ObservationAwareExponentialModel.fit(
            dataset.episodes,
            pre_transition_inspection_probability=(
                dataset.pre_transition_inspection_probability
            ),
            post_transition_inspection_probability=(
                dataset.post_transition_inspection_probability
            ),
            detection_sensitivity=dataset.detection_sensitivity,
        )
        truth = dataset.true_hazard_per_hour
        self.assertLess(abs(joint.hazard - truth), 0.01)
        self.assertLess(abs(joint.hazard - truth), abs(naive.hazard - truth))
        self.assertLess(abs(joint.hazard - truth), abs(interval.hazard - truth))
    def test_em_estimates_observation_parameters_without_transition_truth(
        self,
    ) -> None:
        dataset = informative_observation_data(
            train_samples=1200,
            test_samples=20,
            pre_transition_inspection_probability=0.15,
            post_transition_inspection_probability=0.75,
            detection_sensitivity=0.65,
            seed=101,
        )
        estimate = fit_observation_process_em(dataset.episodes)
        self.assertTrue(estimate.converged)
        self.assertTrue(
            all(
                current <= previous + 1e-9
                for previous, current in zip(
                    estimate.average_negative_log_likelihood_history[:-1],
                    estimate.average_negative_log_likelihood_history[1:],
                    strict=True,
                )
            )
        )
        fitted = estimate.as_persistence_model()
        recomputed_nll = sum(
            fitted._episode_negative_log_likelihood(
                episode,
                fitted.hazard,
                fitted.pre_transition_inspection_probability,
                fitted.post_transition_inspection_probability,
                fitted.detection_sensitivity,
            )
            for episode in dataset.episodes
        ) / len(dataset.episodes)
        self.assertAlmostEqual(
            estimate.average_negative_log_likelihood_history[-1],
            recomputed_nll,
            places=10,
        )
        self.assertLess(abs(estimate.hazard_per_hour - 0.25), 0.02)
        self.assertLess(
            abs(estimate.pre_transition_inspection_probability - 0.15),
            0.02,
        )
        self.assertLess(
            abs(estimate.post_transition_inspection_probability - 0.75),
            0.02,
        )
        self.assertLess(abs(estimate.detection_sensitivity - 0.65), 0.03)

    def test_em_rejects_logs_without_positive_identification_anchor(self) -> None:
        episodes = (
            ObservationEpisode("missing-1", 1.0, ("missing", "negative")),
            ObservationEpisode("missing-2", 1.0, ("negative", "missing")),
        )
        with self.assertRaisesRegex(ValueError, "positive observation"):
            fit_observation_process_em(episodes)


if __name__ == "__main__":
    unittest.main()
