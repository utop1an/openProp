from __future__ import annotations

import unittest

from openprop.non_affine_misspecification import non_affine_misspecification_models
from openprop.sparse_nonlinear_adaptation import (
    fit_log_risk_basis_adapter,
    fit_sparse_nonlinear_typed_gate,
)
from openprop.statistical_persistence import FactorizedExponentialPersistenceModel
from openprop.target_adaptation import build_target_calibration_protocol
from openprop.target_adaptation_stress import target_adaptation_stress_data


class SparseNonlinearAdaptationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.dataset = target_adaptation_stress_data(samples_per_context=32, seed=67)
        cls.source = FactorizedExponentialPersistenceModel.fit(
            cls.dataset.train, epochs=500
        )
        cls.source.calibrate(cls.dataset.validation)
        cls.protocol = build_target_calibration_protocol(
            cls.dataset,  # type: ignore[arg-type]
            max_calibration_per_context=16,
            split_seed=13_000_070,
        )
        cls.calibration = cls.protocol.calibration_subset("in_distribution", 16)
        cls.models = non_affine_misspecification_models(cls.source)

    def test_basis_families_fit_finite_positive_hazards(self):
        deployed = self.models["local_subject_scene_bump"]
        rows = tuple(
            row for row in self.calibration
            if row.features()[1] == "cup" and row.features()[4] == "busy"
        )
        for family in ("affine", "quadratic", "hinge"):
            adapter = fit_log_risk_basis_adapter(
                deployed, rows, family=family, epochs=300  # type: ignore[arg-type]
            )
            self.assertGreater(adapter.hazard_per_hour(rows[0].features()), 0.0)
            self.assertLessEqual(
                adapter.final_negative_log_likelihood,
                adapter.initial_negative_log_likelihood + 1e-7,
            )

    def test_sparse_gate_localizes_bump_and_preserves_other_groups(self):
        deployed = self.models["local_subject_scene_bump"]
        gate = fit_sparse_nonlinear_typed_gate(
            deployed, self.calibration, split_seed=15_000_070, epochs=500
        )
        self.assertEqual((1, 4), gate.selected_partition)
        self.assertEqual(frozenset({("cup", "busy")}), gate.significant_groups)
        stable = ("location", "book", "inside", "cabinet", "quiet")
        changed = ("location", "cup", "inside", "cabinet", "busy")
        self.assertEqual(deployed.hazard_per_hour(stable), gate.hazard_per_hour(stable))
        self.assertNotEqual(
            deployed.hazard_per_hour(changed), gate.hazard_per_hour(changed)
        )

    def test_correct_source_fails_closed_and_order_is_invariant(self):
        first = fit_sparse_nonlinear_typed_gate(
            self.source, self.calibration, split_seed=15_000_070, epochs=300
        )
        second = fit_sparse_nonlinear_typed_gate(
            self.source,
            tuple(reversed(self.calibration)),
            split_seed=15_000_070,
            candidate_partitions=((1, 4), (4,), (1,)),
            epochs=300,
        )
        self.assertFalse(first.activated)
        self.assertEqual(first.selected_partition, second.selected_partition)
        for context in self.dataset.contexts:
            self.assertAlmostEqual(
                first.hazard_per_hour(context.features()),
                second.hazard_per_hour(context.features()),
                places=10,
            )

    def test_sparse_closure_and_unseen_value_routing_fail_closed(self):
        deployed = self.models["local_subject_scene_bump"]
        gate = fit_sparse_nonlinear_typed_gate(
            deployed,
            self.calibration,
            split_seed=15_000_070,
            max_adapted_context_fraction=0.1,
            epochs=300,
        )
        self.assertFalse(gate.activated)
        ordinary = fit_sparse_nonlinear_typed_gate(
            deployed, self.calibration, split_seed=15_000_070, epochs=300
        )
        unseen = ("location", "novel", "inside", "cabinet", "busy")
        self.assertEqual(
            deployed.hazard_per_hour(unseen), ordinary.hazard_per_hour(unseen)
        )

    def test_invalid_family_partition_and_optimizer_fail_closed(self):
        with self.assertRaises(ValueError):
            fit_log_risk_basis_adapter(
                self.source,
                self.calibration,
                family="unknown",  # type: ignore[arg-type]
            )
        with self.assertRaisesRegex(ValueError, "excludes the global"):
            fit_sparse_nonlinear_typed_gate(
                self.source,
                self.calibration,
                split_seed=1,
                candidate_partitions=((), (1,)),
                epochs=10,
            )
        with self.assertRaisesRegex(ValueError, "fraction"):
            fit_sparse_nonlinear_typed_gate(
                self.source,
                self.calibration,
                split_seed=1,
                max_adapted_context_fraction=1.0,
                epochs=10,
            )


if __name__ == "__main__":
    unittest.main()
