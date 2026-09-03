import math
import unittest

from openprop.models import (
    Entity,
    EntityEvent,
    Observation,
    PropertyDefinition,
    TemporalPolicy,
    ValueType,
)
from openprop.observation_em import fit_observation_process_em
from openprop.recurrent_observation import (
    ReversibleBinaryPersistenceModel,
    ctmc_transition_probabilities,
    fit_recurrent_observation_em,
    recurrent_exact_negative_log_likelihood,
    recurrent_exact_test_rows,
    recurrent_observation_data,
    recurrent_state_one_probability,
)


class RecurrentObservationTests(unittest.TestCase):
    def test_ctmc_reduces_to_irreversible_survival_when_return_rate_is_zero(self) -> None:
        matrix = ctmc_transition_probabilities(0.3, 0.0, 2.0)
        expected = 1.0 - math.exp(-0.6)
        self.assertAlmostEqual(matrix[0][1], expected, places=12)
        self.assertEqual(matrix[1], (0.0, 1.0))
        self.assertAlmostEqual(
            recurrent_state_one_probability(0.3, 0.0, 2.0), expected, places=12
        )
        for row in matrix:
            self.assertAlmostEqual(sum(row), 1.0, places=12)

    def test_training_logs_hide_latent_paths_and_exact_rows_remain_paired(self) -> None:
        no_return = recurrent_observation_data(
            seed=13,
            episode_count=20,
            forward_rate_per_hour=0.3,
            return_rate_per_hour=0.0,
        )
        recurrent = recurrent_observation_data(
            seed=13,
            episode_count=20,
            forward_rate_per_hour=0.3,
            return_rate_per_hour=0.5,
        )
        self.assertFalse(hasattr(no_return.episodes[0], "latent_states"))
        self.assertEqual(
            [episode.group_id for episode in no_return.episodes],
            [episode.group_id for episode in recurrent.episodes],
        )
        left = recurrent_exact_test_rows(
            seed=29,
            row_count=100,
            forward_rate_per_hour=0.3,
            return_rate_per_hour=0.0,
        )
        right = recurrent_exact_test_rows(
            seed=29,
            row_count=100,
            forward_rate_per_hour=0.3,
            return_rate_per_hour=0.5,
        )
        self.assertEqual(
            [(row.row_id, row.horizon_hours) for row in left],
            [(row.row_id, row.horizon_hours) for row in right],
        )

    def test_joint_em_recovers_recurrent_rates_and_beats_irreversible_fit(self) -> None:
        dataset = recurrent_observation_data(
            seed=17,
            episode_count=600,
            forward_rate_per_hour=0.3,
            return_rate_per_hour=0.45,
            opportunity_interval_hours=0.5,
            followup_hours=8.0,
            inspection_probability_state_0=0.7,
            inspection_probability_state_1=0.75,
            detection_sensitivity=0.9,
            false_positive_rate=0.04,
        )
        recurrent = fit_recurrent_observation_em(
            dataset.episodes,
            max_iterations=120,
            tolerance=1e-6,
        )
        irreversible = fit_observation_process_em(
            dataset.episodes,
            max_iterations=120,
            tolerance=1e-6,
            estimate_false_positive_rate=True,
        )
        self.assertTrue(recurrent.converged)
        self.assertLess(abs(recurrent.forward_rate_per_hour - 0.3), 0.06)
        self.assertLess(abs(recurrent.return_rate_per_hour - 0.45), 0.08)
        self.assertTrue(
            all(
                current <= previous + 1e-9
                for previous, current in zip(
                    recurrent.average_negative_log_likelihood_history,
                    recurrent.average_negative_log_likelihood_history[1:],
                )
            )
        )
        test_rows = recurrent_exact_test_rows(
            seed=99,
            row_count=2000,
            forward_rate_per_hour=0.3,
            return_rate_per_hour=0.45,
        )
        recurrent_nll = recurrent_exact_negative_log_likelihood(
            test_rows,
            forward_rate_per_hour=recurrent.forward_rate_per_hour,
            return_rate_per_hour=recurrent.return_rate_per_hour,
        )
        irreversible_nll = recurrent_exact_negative_log_likelihood(
            test_rows,
            forward_rate_per_hour=irreversible.hazard_per_hour,
            return_rate_per_hour=0.0,
        )
        self.assertLess(recurrent_nll, irreversible_nll - 0.04)

    def test_matcher_adapter_uses_probability_of_same_boolean_state(self) -> None:
        definition = PropertyDefinition(
            "isOpen",
            "whether the object is open",
            ValueType.CATEGORICAL,
            temporal_policy=TemporalPolicy(event_retention={"forced": 0.5}),
        )
        entity = Entity(
            "cabinet-1",
            events=[EntityEvent("forced", timestamp=3600.0)],
        )
        model = ReversibleBinaryPersistenceModel("isOpen", 0.3, 0.45)
        observation = Observation(False, timestamp=0.0)
        result = model.predict(definition, observation, entity, as_of=7200.0)
        expected_same = 1.0 - recurrent_state_one_probability(0.3, 0.45, 2.0)
        self.assertAlmostEqual(result.time_retention, expected_same, places=12)
        self.assertAlmostEqual(result.freshness, expected_same * 0.5, places=12)
        with self.assertRaisesRegex(ValueError, "boolean observation"):
            model.predict(
                definition,
                Observation("closed", timestamp=0.0),
                entity,
                as_of=7200.0,
            )

    def test_invalid_recurrent_parameters_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            ctmc_transition_probabilities(0.3, -0.1, 1.0)
        with self.assertRaisesRegex(ValueError, "must exceed"):
            recurrent_observation_data(
                seed=1,
                episode_count=10,
                forward_rate_per_hour=0.3,
                return_rate_per_hour=0.2,
                detection_sensitivity=0.1,
                false_positive_rate=0.2,
            )
        with self.assertRaisesRegex(ValueError, "common nonempty"):
            fit_recurrent_observation_em(
                [
                    recurrent_observation_data(
                        seed=2,
                        episode_count=1,
                        forward_rate_per_hour=0.3,
                        return_rate_per_hour=0.2,
                    ).episodes[0],
                    recurrent_observation_data(
                        seed=3,
                        episode_count=1,
                        forward_rate_per_hour=0.3,
                        return_rate_per_hour=0.2,
                        followup_hours=6.0,
                    ).episodes[0],
                ]
            )


if __name__ == "__main__":
    unittest.main()
