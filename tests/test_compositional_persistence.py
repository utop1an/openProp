import math
import unittest

from openprop.advanced_survival_evaluation import evaluate_survival_advanced
from openprop.compositional_persistence import (
    compositional_grounding_benchmark,
    compositional_grounding_registry,
    compositional_location_data,
    evaluate_grounding_model,
)
from openprop.persistence import ExponentialPersistenceModel
from openprop.persistence_data import PersistenceTrainingExample
from openprop.statistical_persistence import (
    FactorizedExponentialPersistenceModel,
    FactorizedPiecewiseExponentialPersistenceModel,
    FactorizedWeibullPersistenceModel,
    GlobalExponentialPersistenceModel,
    PerContextExponentialPersistenceModel,
)


class CompositionalPersistenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.dataset = compositional_location_data(samples_per_context=20)

    def test_split_holds_out_combinations_but_not_feature_values(self) -> None:
        train_contexts = {example.features() for example in self.dataset.train}
        test_contexts = {example.features() for example in self.dataset.test}
        self.assertTrue(train_contexts.isdisjoint(test_contexts))
        for column in range(5):
            train_values = {features[column] for features in train_contexts}
            test_values = {features[column] for features in test_contexts}
            self.assertLessEqual(test_values, train_values)
        train_groups = {example.group_id for example in self.dataset.train}
        test_groups = {example.group_id for example in self.dataset.test}
        self.assertTrue(train_groups.isdisjoint(test_groups))

    def test_per_context_baseline_uses_global_backoff_on_ood_tuple(self) -> None:
        model = PerContextExponentialPersistenceModel.fit(self.dataset.train)
        test_features = self.dataset.test[0].features()
        self.assertNotIn(test_features, model.hazards)
        self.assertEqual(model.global_hazard, model.hazard_per_hour(test_features))

    def test_factorized_baseline_composes_held_out_contexts(self) -> None:
        global_model = GlobalExponentialPersistenceModel.fit(self.dataset.train)
        factorized = FactorizedExponentialPersistenceModel.fit(
            self.dataset.train, epochs=600
        )
        factorized.calibrate(self.dataset.validation)
        global_report = evaluate_survival_advanced(global_model, self.dataset.test)
        factorized_report = evaluate_survival_advanced(factorized, self.dataset.test)
        self.assertLess(
            factorized_report.negative_log_likelihood,
            global_report.negative_log_likelihood,
        )
        hazards = {
            round(factorized.hazard_per_hour(example.features()), 8)
            for example in self.dataset.test
        }
        self.assertGreater(len(hazards), 1)

    def test_factorized_subset_ignores_inactive_typed_columns(self) -> None:
        subject_only = FactorizedExponentialPersistenceModel.fit(
            self.dataset.train,
            epochs=300,
            active_feature_indices=(1,),
        )
        reference = self.dataset.test[0].features()
        inactive_changed = (
            reference[0],
            reference[1],
            "different-predicate",
            "different-object",
            "different-scene",
        )
        self.assertEqual(frozenset({1}), subject_only.active_feature_indices)
        self.assertEqual({}, subject_only.effects[0])
        self.assertEqual({}, subject_only.effects[2])
        self.assertEqual({}, subject_only.effects[3])
        self.assertEqual({}, subject_only.effects[4])
        self.assertAlmostEqual(
            subject_only.hazard_per_hour(reference),
            subject_only.hazard_per_hour(inactive_changed),
        )

        intercept_only = FactorizedExponentialPersistenceModel.fit(
            self.dataset.train,
            epochs=50,
            active_feature_indices=(),
        )
        hazards = {
            round(intercept_only.hazard_per_hour(row.features()), 12)
            for row in self.dataset.test
        }
        self.assertEqual(1, len(hazards))
        with self.assertRaisesRegex(ValueError, "0 through 4"):
            FactorizedExponentialPersistenceModel.fit(
                self.dataset.train, active_feature_indices=(5,)
            )

    def test_weibull_model_recovers_nonexponential_shape(self) -> None:
        dataset = compositional_location_data(
            samples_per_context=20, seed=41, weibull_shape=1.6
        )
        exponential = FactorizedExponentialPersistenceModel.fit(
            dataset.train, epochs=600
        )
        exponential.calibrate(dataset.validation)
        weibull = FactorizedWeibullPersistenceModel.fit(
            dataset.train, epochs=800
        )
        weibull.calibrate(dataset.validation)
        self.assertAlmostEqual(1.6, weibull.shape, delta=0.2)
        exponential_nll = evaluate_survival_advanced(
            exponential, dataset.test
        ).negative_log_likelihood
        weibull_nll = evaluate_survival_advanced(
            weibull, dataset.test
        ).negative_log_likelihood
        self.assertLess(weibull_nll, exponential_nll)



    def test_weibull_interval_likelihood_uses_probability_mass(self) -> None:
        row = PersistenceTrainingExample(
            "location",
            "cup",
            "on",
            "table",
            "kitchen",
            4 * 3600,
            True,
            interval_start_seconds=2 * 3600,
        )
        model = FactorizedWeibullPersistenceModel(
            math.log(0.2),
            math.log(2.0),
            ({}, {}, {}, {}, {}),
            frozenset({"location"}),
        )
        expected = -math.log(math.exp(-(0.2 * 2) ** 2) - math.exp(-(0.2 * 4) ** 2))
        self.assertAlmostEqual(expected, model.example_negative_log_likelihood(row))


    def test_piecewise_interval_likelihood_uses_cross_bin_mass(self) -> None:
        row = PersistenceTrainingExample(
            "location",
            "cup",
            "on",
            "table",
            "kitchen",
            3 * 3600,
            True,
            interval_start_seconds=1 * 3600,
        )
        model = FactorizedPiecewiseExponentialPersistenceModel(
            math.log(0.2),
            ({}, {}, {}, {}, {}),
            (2.0,),
            (0.0, math.log(2.0)),
            frozenset({"location"}),
        )
        expected = -math.log(math.exp(-0.2) - math.exp(-0.8))
        self.assertAlmostEqual(expected, model.example_negative_log_likelihood(row))

    def test_split_specific_horizons_create_duration_shift(self) -> None:
        dataset = compositional_location_data(
            samples_per_context=30,
            censor_after_hours_by_split={
                "train": 6.0,
                "validation": 12.0,
                "test": 24.0,
            },
        )
        train_max = max(row.duration_seconds for row in dataset.train) / 3600.0
        validation_max = (
            max(row.duration_seconds for row in dataset.validation) / 3600.0
        )
        test_max = max(row.duration_seconds for row in dataset.test) / 3600.0
        self.assertLessEqual(train_max, 6.0)
        self.assertLessEqual(validation_max, 12.0)
        self.assertLessEqual(test_max, 24.0)
        self.assertGreater(validation_max, train_max)
        self.assertGreater(test_max, validation_max)


    def test_advanced_survival_metrics_are_bounded(self) -> None:
        model = GlobalExponentialPersistenceModel.fit(self.dataset.train)
        report = evaluate_survival_advanced(model, self.dataset.test)
        self.assertGreater(report.negative_log_likelihood, 0.0)
        self.assertGreaterEqual(report.concordance_index, 0.0)
        self.assertLessEqual(report.concordance_index, 1.0)
        self.assertGreaterEqual(report.integrated_brier_score, 0.0)
        self.assertLessEqual(report.integrated_brier_score, 1.0)

    def test_grounding_requires_context_dependent_persistence(self) -> None:
        cases = compositional_grounding_benchmark(repetitions=2)
        registry = compositional_grounding_registry()
        fixed = evaluate_grounding_model(
            "fixed", ExponentialPersistenceModel(), cases, registry
        )
        oracle_hazards = {
            context.features(): context.hazard_per_hour
            for context in self.dataset.contexts
        }
        oracle = PerContextExponentialPersistenceModel(
            oracle_hazards,
            global_hazard=0.12,
            trained_properties=frozenset({"location"}),
        )
        contextual = evaluate_grounding_model("oracle", oracle, cases, registry)
        self.assertEqual(0.0, fixed.top1_accuracy)
        self.assertEqual(1.0, contextual.top1_accuracy)


if __name__ == "__main__":
    unittest.main()
