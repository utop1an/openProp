from __future__ import annotations

import math
import unittest

from openprop.source_reliability_observation import (
    SourceEmissionParameters,
    SourceObservationResult,
    SourcedObservationEpisode,
    fit_source_reliability_em,
    sourced_recurrent_observation_data,
)


class SourceReliabilityObservationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.parameters = (
            SourceEmissionParameters("reliable", 0.85, 0.45, 0.94, 0.03),
            SourceEmissionParameters("degraded", 0.45, 0.85, 0.68, 0.17),
        )

    def test_episode_requires_stable_unique_source_provenance(self) -> None:
        with self.assertRaisesRegex(ValueError, "same ordered source IDs"):
            SourcedObservationEpisode(
                "bad",
                0.5,
                (
                    (
                        SourceObservationResult("a", "positive"),
                        SourceObservationResult("b", "negative"),
                    ),
                    (
                        SourceObservationResult("b", "negative"),
                        SourceObservationResult("a", "positive"),
                    ),
                ),
            )

    def test_generator_exposes_no_latent_state_truth(self) -> None:
        dataset = sourced_recurrent_observation_data(
            seed=3,
            episode_count=3,
            forward_rate_per_hour=0.3,
            return_rate_per_hour=0.45,
            source_parameters=self.parameters,
        )
        self.assertEqual(dataset.episodes[0].source_ids, ("reliable", "degraded"))
        self.assertFalse(hasattr(dataset.episodes[0], "current_truth"))
        self.assertFalse(hasattr(dataset.episodes[0], "latent_states"))

    def test_source_specific_fit_improves_likelihood_and_recovers_rates(self) -> None:
        dataset = sourced_recurrent_observation_data(
            seed=11,
            episode_count=260,
            forward_rate_per_hour=0.3,
            return_rate_per_hour=0.45,
            source_parameters=self.parameters,
        )
        source_aware = fit_source_reliability_em(dataset.episodes)
        pooled = fit_source_reliability_em(dataset.episodes, pooled_sources=True)
        self.assertTrue(source_aware.converged)
        self.assertLess(
            source_aware.average_negative_log_likelihood_history[-1],
            pooled.average_negative_log_likelihood_history[-1],
        )
        self.assertLess(abs(source_aware.forward_rate_per_hour - 0.3), 0.08)
        self.assertLess(abs(source_aware.return_rate_per_hour - 0.45), 0.1)
        fitted = {value.source_id: value for value in source_aware.source_parameters}
        self.assertGreater(
            fitted["reliable"].detection_sensitivity,
            fitted["degraded"].detection_sensitivity,
        )
        self.assertLess(
            fitted["reliable"].false_positive_rate,
            fitted["degraded"].false_positive_rate,
        )

    def test_fit_is_invariant_to_episode_and_source_order(self) -> None:
        dataset = sourced_recurrent_observation_data(
            seed=19,
            episode_count=120,
            forward_rate_per_hour=0.3,
            return_rate_per_hour=0.45,
            source_parameters=self.parameters,
        )
        reordered = tuple(
            SourcedObservationEpisode(
                episode.group_id,
                episode.opportunity_interval_hours,
                tuple(tuple(reversed(step)) for step in episode.results_by_step),
            )
            for episode in reversed(dataset.episodes)
        )
        original = fit_source_reliability_em(dataset.episodes)
        changed = fit_source_reliability_em(reordered)
        self.assertTrue(
            math.isclose(
                original.forward_rate_per_hour,
                changed.forward_rate_per_hour,
                rel_tol=0.0,
                abs_tol=1e-10,
            )
        )
        self.assertTrue(
            math.isclose(
                original.return_rate_per_hour,
                changed.return_rate_per_hour,
                rel_tol=0.0,
                abs_tol=1e-10,
            )
        )

    def test_fit_rejects_incompatible_grids_and_unidentified_labels(self) -> None:
        negative_step = (
            SourceObservationResult("a", "negative"),
            SourceObservationResult("b", "missing"),
        )
        rows = (
            SourcedObservationEpisode("one", 0.5, (negative_step,)),
            SourcedObservationEpisode("two", 1.0, (negative_step,)),
        )
        with self.assertRaisesRegex(ValueError, "common nonempty source opportunity grid"):
            fit_source_reliability_em(rows)
        with self.assertRaisesRegex(ValueError, "positive observation"):
            fit_source_reliability_em(rows[:1])


if __name__ == "__main__":
    unittest.main()
