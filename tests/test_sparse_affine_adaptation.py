from __future__ import annotations

import unittest

from openprop.non_affine_misspecification import non_affine_misspecification_models
from openprop.sparse_affine_adaptation import fit_sparse_coverage_affine_gate
from openprop.statistical_persistence import FactorizedExponentialPersistenceModel
from openprop.target_adaptation import build_target_calibration_protocol
from openprop.target_adaptation_stress import target_adaptation_stress_data


class SparseAffineAdaptationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.dataset = target_adaptation_stress_data(samples_per_context=32, seed=67)
        cls.source = FactorizedExponentialPersistenceModel.fit(
            cls.dataset.train, epochs=500
        )
        cls.source.calibrate(cls.dataset.validation)
        protocol = build_target_calibration_protocol(
            cls.dataset,  # type: ignore[arg-type]
            max_calibration_per_context=16,
            split_seed=13_000_070,
        )
        cls.calibration = protocol.calibration_subset("in_distribution", 16)
        cls.models = non_affine_misspecification_models(cls.source)

    def test_correct_source_remains_inactive(self):
        gate = fit_sparse_coverage_affine_gate(
            self.source,
            self.calibration,
            split_seed=15_000_070,
            epochs=300,
        )
        self.assertFalse(gate.activated)
        self.assertFalse(gate.coverage_rejected)
        self.assertEqual(0.0, gate.adapted_context_fraction)

    def test_bump_uses_sparse_typed_repair_and_preserves_other_groups(self):
        deployed = self.models["local_subject_scene_bump"]
        gate = fit_sparse_coverage_affine_gate(
            deployed,
            self.calibration,
            split_seed=15_000_070,
            epochs=500,
        )
        self.assertTrue(gate.activated)
        self.assertLessEqual(gate.adapted_context_fraction, 0.5)
        stable = ("location", "book", "inside", "cabinet", "quiet")
        self.assertEqual(deployed.hazard_per_hour(stable), gate.hazard_per_hour(stable))

    def test_tight_coverage_cap_rejects_candidate_without_changing_predictions(self):
        deployed = self.models["local_subject_scene_bump"]
        gate = fit_sparse_coverage_affine_gate(
            deployed,
            self.calibration,
            split_seed=15_000_070,
            max_adapted_context_fraction=0.1,
            epochs=500,
        )
        self.assertTrue(gate.coverage_rejected)
        self.assertFalse(gate.activated)
        for context in self.dataset.contexts:
            self.assertEqual(
                deployed.hazard_per_hour(context.features()),
                gate.hazard_per_hour(context.features()),
            )

    def test_global_partition_and_invalid_fraction_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "excludes the global"):
            fit_sparse_coverage_affine_gate(
                self.source,
                self.calibration,
                split_seed=1,
                candidate_partitions=((), (1,)),
                epochs=10,
            )
        with self.assertRaisesRegex(ValueError, "fraction"):
            fit_sparse_coverage_affine_gate(
                self.source,
                self.calibration,
                split_seed=1,
                max_adapted_context_fraction=1.0,
                epochs=10,
            )


if __name__ == "__main__":
    unittest.main()
