from __future__ import annotations

import unittest

from openprop.irregular_recurrent_observation import (
    IrregularObservationEpisode,
    collapse_to_mean_grid,
    fit_irregular_recurrent_observation_em,
    irregular_recurrent_observation_data,
)
from openprop.recurrent_observation import (
    fit_recurrent_observation_em,
    recurrent_exact_negative_log_likelihood,
    recurrent_exact_test_rows,
)


class IrregularRecurrentObservationTests(unittest.TestCase):
    def test_bursty_schedules_are_paired_and_preserve_total_followup(self) -> None:
        moderate = irregular_recurrent_observation_data(
            seed=17,
            episode_count=12,
            gap_contrast=0.5,
        )
        severe = irregular_recurrent_observation_data(
            seed=17,
            episode_count=12,
            gap_contrast=0.9,
        )
        self.assertEqual(
            [episode.group_id for episode in moderate.episodes],
            [episode.group_id for episode in severe.episodes],
        )
        for left, right in zip(moderate.episodes, severe.episodes, strict=True):
            self.assertAlmostEqual(sum(left.intervals_hours), 12.0)
            self.assertAlmostEqual(sum(right.intervals_hours), 12.0)
            self.assertEqual(
                [value > 0.75 for value in left.intervals_hours],
                [value > 0.75 for value in right.intervals_hours],
            )
            self.assertEqual(
                {"group_id", "intervals_hours", "results"},
                set(left.__dataclass_fields__),
            )

    def test_irregular_em_reduces_to_regular_grid_fit(self) -> None:
        dataset = irregular_recurrent_observation_data(
            seed=101,
            episode_count=250,
            return_rate_per_hour=0.45,
            gap_contrast=0.0,
        )
        irregular = fit_irregular_recurrent_observation_em(
            dataset.episodes, max_iterations=100, tolerance=1e-6
        )
        regular = fit_recurrent_observation_em(
            collapse_to_mean_grid(dataset.episodes),
            max_iterations=100,
            tolerance=1e-6,
        )
        self.assertAlmostEqual(
            irregular.forward_rate_per_hour,
            regular.forward_rate_per_hour,
            delta=0.015,
        )
        self.assertAlmostEqual(
            irregular.return_rate_per_hour,
            regular.return_rate_per_hour,
            delta=0.025,
        )
        self.assertAlmostEqual(
            irregular.average_negative_log_likelihood_history[-1],
            regular.average_negative_log_likelihood_history[-1],
            delta=2e-4,
        )

    def test_exact_intervals_recover_rates_and_beat_mean_grid(self) -> None:
        dataset = irregular_recurrent_observation_data(
            seed=101,
            episode_count=350,
            return_rate_per_hour=0.45,
            gap_contrast=0.9,
        )
        irregular = fit_irregular_recurrent_observation_em(
            dataset.episodes, max_iterations=100, tolerance=1e-6
        )
        mean_grid = fit_recurrent_observation_em(
            collapse_to_mean_grid(dataset.episodes),
            max_iterations=100,
            tolerance=1e-6,
        )
        self.assertTrue(irregular.converged)
        self.assertTrue(
            all(
                current <= previous + 1e-8
                for previous, current in zip(
                    irregular.average_negative_log_likelihood_history,
                    irregular.average_negative_log_likelihood_history[1:],
                )
            )
        )
        self.assertLess(abs(irregular.forward_rate_per_hour - 0.30), 0.10)
        self.assertLess(abs(irregular.return_rate_per_hour - 0.45), 0.15)
        self.assertGreater(abs(mean_grid.forward_rate_per_hour - 0.30), 0.12)
        rows = recurrent_exact_test_rows(
            seed=501,
            row_count=10_000,
            forward_rate_per_hour=0.30,
            return_rate_per_hour=0.45,
        )
        irregular_nll = recurrent_exact_negative_log_likelihood(
            rows,
            forward_rate_per_hour=irregular.forward_rate_per_hour,
            return_rate_per_hour=irregular.return_rate_per_hour,
        )
        mean_grid_nll = recurrent_exact_negative_log_likelihood(
            rows,
            forward_rate_per_hour=mean_grid.forward_rate_per_hour,
            return_rate_per_hour=mean_grid.return_rate_per_hour,
        )
        self.assertLess(irregular_nll, mean_grid_nll - 0.01)

    def test_invalid_intervals_ids_and_parameters_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "equal nonzero length"):
            IrregularObservationEpisode("x", (1.0,), ())
        with self.assertRaisesRegex(ValueError, "finite and positive"):
            IrregularObservationEpisode("x", (0.0,), ("missing",))
        with self.assertRaisesRegex(ValueError, "gap_contrast"):
            irregular_recurrent_observation_data(
                seed=1, episode_count=1, gap_contrast=1.0
            )
        dataset = irregular_recurrent_observation_data(seed=1, episode_count=2)
        with self.assertRaisesRegex(ValueError, "unique"):
            fit_irregular_recurrent_observation_em(
                (dataset.episodes[0], dataset.episodes[0])
            )


if __name__ == "__main__":
    unittest.main()
