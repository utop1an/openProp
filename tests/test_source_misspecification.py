from __future__ import annotations

import unittest
from openprop.advanced_survival_evaluation import evaluate_survival_advanced

from openprop.compositional_persistence import _context_dynamics
from openprop.source_misspecification import (
    SOURCE_MISSPECIFICATION_CONDITIONS,
    MonotoneRiskTransform,
    TypedFeaturePermutationRiskModel,
    source_misspecification_models,
)
from openprop.statistical_persistence import FactorizedExponentialPersistenceModel
from openprop.target_adaptation import build_target_calibration_protocol
from openprop.target_adaptation_stress import target_adaptation_stress_data
from openprop.target_interaction_adaptation import (
    fit_hierarchical_typed_interaction_gate,
)


class SourceMisspecificationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.dataset = target_adaptation_stress_data(samples_per_context=32, seed=67)
        cls.source = FactorizedExponentialPersistenceModel.fit(
            cls.dataset.train, epochs=700
        )
        cls.source.calibrate(cls.dataset.validation)
        cls.protocol = build_target_calibration_protocol(
            cls.dataset,  # type: ignore[arg-type]
            max_calibration_per_context=16,
            split_seed=9_000_070,
        )

    def test_monotone_transforms_preserve_strict_source_order(self) -> None:
        contexts = [context.features() for context in _context_dynamics()]
        source_order = sorted(contexts, key=self.source.hazard_per_hour)
        for transform in (
            MonotoneRiskTransform(self.source, scale=2.0),
            MonotoneRiskTransform(self.source, power=0.5),
            MonotoneRiskTransform(self.source, power=1.75),
        ):
            self.assertEqual(
                source_order,
                sorted(contexts, key=transform.hazard_per_hour),
            )

    def test_typed_permutation_is_explicit_and_does_not_mutate_input(self) -> None:
        model = TypedFeaturePermutationRiskModel(
            self.source,
            {1: {"book": "cup", "cup": "tool", "tool": "book"}},
        )
        features = ("location", "book", "stored", "shelf", "quiet")
        expected = ("location", "cup", "stored", "shelf", "quiet")
        self.assertEqual(expected, model.permuted_features(features))
        self.assertEqual(
            self.source.hazard_per_hour(expected),
            model.hazard_per_hour(features),
        )
        self.assertEqual(("location", "book", "stored", "shelf", "quiet"), features)
        with self.assertRaisesRegex(ValueError, "has no value"):
            model.hazard_per_hour(("location", "unknown", "stored", "shelf", "quiet"))

    def test_fixed_condition_registry_matches_models(self) -> None:
        models = source_misspecification_models(self.source)
        self.assertEqual(set(SOURCE_MISSPECIFICATION_CONDITIONS), set(models))
        self.assertIs(self.source, models["correct_source"])

    def test_hierarchy_does_not_call_monotone_miscalibration_a_reversal(self) -> None:
        calibration = self.protocol.calibration_subset("in_distribution", 16)
        for name in ("rate_x2", "risk_compressed", "risk_expanded"):
            model = source_misspecification_models(self.source)[name]
            gate = fit_hierarchical_typed_interaction_gate(
                model,
                calibration,
                split_seed=12_000_070,
                epochs=700,
            )
            self.assertFalse(gate.activated, name)

    def test_generalized_gate_repairs_confirmed_monotone_miscalibration(self) -> None:
        calibration = self.protocol.calibration_subset("in_distribution", 16)
        test_rows = self.protocol.tests["in_distribution"]
        models = source_misspecification_models(self.source)
        correct = fit_hierarchical_typed_interaction_gate(
            self.source,
            calibration,
            split_seed=12_000_070,
            activation_scope="any_predictive_gain",
            epochs=700,
        )
        self.assertFalse(correct.activated)
        for name in ("rate_x2", "risk_compressed", "risk_expanded"):
            deployed = models[name]
            gate = fit_hierarchical_typed_interaction_gate(
                deployed,
                calibration,
                split_seed=12_000_070,
                activation_scope="any_predictive_gain",
                epochs=700,
            )
            self.assertEqual((), gate.selected_partition, name)
            before = evaluate_survival_advanced(
                deployed, test_rows, horizons_hours=(1.0, 4.0, 8.0, 12.0)
            )
            after = evaluate_survival_advanced(
                gate, test_rows, horizons_hours=(1.0, 4.0, 8.0, 12.0)
            )
            self.assertLess(after.negative_log_likelihood, before.negative_log_likelihood)
            self.assertEqual(before.concordance_index, after.concordance_index)

    def test_invalid_transform_and_permutation_fail_closed(self) -> None:
        with self.assertRaises(ValueError):
            MonotoneRiskTransform(self.source, power=0.0)
        with self.assertRaisesRegex(ValueError, "one-to-one"):
            TypedFeaturePermutationRiskModel(
                self.source,
                {1: {"book": "cup", "tool": "cup"}},
            )
        rows = self.protocol.calibration_subset("in_distribution", 16)
        with self.assertRaisesRegex(ValueError, "activation_scope"):
            fit_hierarchical_typed_interaction_gate(
                self.source,
                rows,
                split_seed=1,
                activation_scope="unknown",  # type: ignore[arg-type]
                epochs=10,
            )


if __name__ == "__main__":
    unittest.main()
