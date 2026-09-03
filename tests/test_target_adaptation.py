import unittest

from openprop.advanced_survival_evaluation import concordance_index
from openprop.latent_mechanism_shift import latent_mechanism_shift_data
from openprop.statistical_persistence import FactorizedExponentialPersistenceModel
from openprop.target_adaptation import (
    build_target_calibration_protocol,
    fit_log_risk_affine_adapter,
    select_sign_gated_model,
)


class TargetAdaptationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.dataset = latent_mechanism_shift_data(
            samples_per_context=30,
            seed=41,
        )
        cls.source = FactorizedExponentialPersistenceModel.fit(
            cls.dataset.train,
            epochs=500,
        )
        cls.source.calibrate(cls.dataset.validation)

    def test_protocol_is_paired_nested_and_group_disjoint(self) -> None:
        protocol = build_target_calibration_protocol(
            self.dataset,
            max_calibration_per_context=12,
            split_seed=9001,
        )
        reference_pool = protocol.calibration_pool["in_distribution"]
        reference_test = protocol.tests["in_distribution"]
        self.assertEqual(36, len(reference_pool))
        self.assertEqual(54, len(reference_test))
        for condition in self.dataset.tests:
            self.assertEqual(
                [row.group_id for row in reference_pool],
                [row.group_id for row in protocol.calibration_pool[condition]],
            )
            self.assertEqual(
                [row.group_id for row in reference_test],
                [row.group_id for row in protocol.tests[condition]],
            )
        small = protocol.calibration_subset("typed_factor_reversal", 3)
        large = protocol.calibration_subset("typed_factor_reversal", 8)
        self.assertEqual(9, len(small))
        self.assertEqual(24, len(large))
        self.assertLessEqual(
            {row.group_id for row in small},
            {row.group_id for row in large},
        )
        self.assertFalse(
            {row.group_id for row in reference_pool}
            & {row.group_id for row in reference_test}
        )

    def test_scale_only_preserves_ranking_but_affine_repairs_reversal(self) -> None:
        protocol = build_target_calibration_protocol(
            self.dataset,
            max_calibration_per_context=12,
            split_seed=9001,
        )
        calibration = protocol.calibration_subset("typed_factor_reversal", 12)
        test = protocol.tests["typed_factor_reversal"]
        source_c = concordance_index(self.source, test)
        scale = fit_log_risk_affine_adapter(
            self.source,
            calibration,
            fit_slope=False,
            epochs=500,
        )
        affine = fit_log_risk_affine_adapter(
            self.source,
            calibration,
            fit_slope=True,
            epochs=800,
        )
        self.assertAlmostEqual(source_c, concordance_index(scale, test))
        self.assertIs(select_sign_gated_model(self.source, affine), affine)
        stable_calibration = protocol.calibration_subset("in_distribution", 12)
        stable_scale = fit_log_risk_affine_adapter(
            self.source, stable_calibration, fit_slope=False, epochs=500
        )
        stable_affine = fit_log_risk_affine_adapter(
            self.source, stable_calibration, fit_slope=True, epochs=800
        )
        self.assertGreater(stable_affine.slope, 0.0)
        self.assertIs(
            select_sign_gated_model(self.source, stable_affine), self.source
        )
        self.assertLess(affine.slope, 0.0)
        self.assertGreater(concordance_index(affine, test), 0.75)
        self.assertGreater(concordance_index(affine, test), source_c + 0.5)
        self.assertLess(
            affine.final_negative_log_likelihood,
            affine.initial_negative_log_likelihood,
        )

    def test_protocol_and_optimizer_reject_invalid_inputs(self) -> None:
        with self.assertRaises(ValueError):
            build_target_calibration_protocol(
                self.dataset,
                max_calibration_per_context=30,
                split_seed=1,
            )
        protocol = build_target_calibration_protocol(
            self.dataset,
            max_calibration_per_context=3,
            split_seed=1,
        )
        with self.assertRaises(ValueError):
            protocol.calibration_subset("in_distribution", 0)
        with self.assertRaises(ValueError):
            fit_log_risk_affine_adapter(
                self.source,
                protocol.calibration_subset("in_distribution", 1),
                fit_slope=True,
                epochs=0,
            )


if __name__ == "__main__":
    unittest.main()
