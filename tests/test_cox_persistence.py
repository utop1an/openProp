import unittest

from openprop.compositional_persistence import compositional_location_data
from openprop.cox_persistence import FactorizedCoxPersistenceModel
from openprop.persistence_data import PersistenceTrainingExample
from openprop.semiparametric_evaluation import evaluate_semiparametric_survival


def _row(
    duration_hours: float,
    event_observed: bool,
    group_id: str,
    *,
    interval_start_hours: float | None = None,
) -> PersistenceTrainingExample:
    return PersistenceTrainingExample(
        "location",
        "cup",
        "on",
        "table",
        "quiet",
        duration_hours * 3600.0,
        event_observed,
        group_id,
        None if interval_start_hours is None else interval_start_hours * 3600.0,
    )


class CoxPersistenceTests(unittest.TestCase):
    def test_breslow_baseline_handles_tied_events_and_censoring(self) -> None:
        rows = (
            _row(1.0, True, "event-1a"),
            _row(1.0, True, "event-1b"),
            _row(2.0, True, "event-2"),
        )
        model = FactorizedCoxPersistenceModel.fit(rows, epochs=20)
        self.assertEqual((1.0, 2.0), model.event_times_hours)
        self.assertAlmostEqual(2.0 / 3.0, model.cumulative_baseline_hazards[0])
        self.assertAlmostEqual(5.0 / 3.0, model.cumulative_baseline_hazards[1])
        self.assertAlmostEqual(
            1.0,
            model.survival_probability_at_hours(rows[0].features(), 0.5),
        )

    def test_factorized_cox_improves_partial_likelihood_and_is_order_invariant(
        self,
    ) -> None:
        dataset = compositional_location_data(samples_per_context=30, seed=41)
        forward = FactorizedCoxPersistenceModel.fit(dataset.train, epochs=500)
        reverse = FactorizedCoxPersistenceModel.fit(
            tuple(reversed(dataset.train)),
            epochs=500,
        )
        self.assertLess(forward.final_partial_nll, forward.initial_partial_nll)
        for context in dataset.contexts:
            self.assertAlmostEqual(
                forward.risk_score(context.features()),
                reverse.risk_score(context.features()),
                places=10,
            )
        report = evaluate_semiparametric_survival(forward, dataset.test)
        self.assertGreater(report.concordance_index, 0.65)
        scale = forward.calibrate_baseline(dataset.validation)
        self.assertGreater(scale, 0.0)
        self.assertAlmostEqual(scale, forward.baseline_scale)
        self.assertLess(report.integrated_brier_score, 0.16)

    def test_cox_rejects_interval_censored_events(self) -> None:
        rows = (
            _row(2.0, True, "interval", interval_start_hours=1.0),
            _row(3.0, False, "censored"),
        )
        with self.assertRaisesRegex(ValueError, "does not support interval"):
            FactorizedCoxPersistenceModel.fit(rows)


if __name__ == "__main__":
    unittest.main()
