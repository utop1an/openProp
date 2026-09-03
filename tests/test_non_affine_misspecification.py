from __future__ import annotations

import unittest

from openprop.compositional_persistence import _context_dynamics
from openprop.non_affine_misspecification import (
    NON_AFFINE_MISSPECIFICATION_CONDITIONS,
    LocalNonAffineRiskWarp,
    affected_contexts,
    non_affine_misspecification_models,
)
from openprop.statistical_persistence import FactorizedExponentialPersistenceModel
from openprop.target_adaptation_stress import target_adaptation_stress_data


class NonAffineMisspecificationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        dataset = target_adaptation_stress_data(samples_per_context=16, seed=67)
        cls.source = FactorizedExponentialPersistenceModel.fit(
            dataset.train, epochs=500
        )
        cls.source.calibrate(dataset.validation)
        cls.contexts = tuple(context.features() for context in _context_dynamics())

    def test_registry_has_control_and_three_distinct_local_warps(self):
        models = non_affine_misspecification_models(self.source)
        self.assertEqual(set(NON_AFFINE_MISSPECIFICATION_CONDITIONS), set(models))
        self.assertIs(self.source, models["correct_source_control"])
        affected = {
            name: affected_contexts(model, self.source, self.contexts)
            for name, model in models.items()
        }
        self.assertEqual(frozenset(), affected["correct_source_control"])
        self.assertEqual(
            {row for row in self.contexts if row[1] == "cup"},
            affected["local_subject_saturation"],
        )
        busy = {row for row in self.contexts if row[4] == "busy"}
        self.assertTrue(affected["local_scene_fold"])
        self.assertLess(len(affected["local_scene_fold"]), len(busy))
        self.assertTrue(affected["local_scene_fold"].issubset(busy))
        self.assertEqual(
            {row for row in self.contexts if row[1] == "cup" and row[4] == "busy"},
            affected["local_subject_scene_bump"],
        )

    def test_stable_contexts_are_bitwise_unchanged(self):
        for model in non_affine_misspecification_models(self.source).values():
            for features in self.contexts:
                if isinstance(model, LocalNonAffineRiskWarp) and model.applies_to(features):
                    continue
                self.assertEqual(
                    self.source.hazard_per_hour(features),
                    model.hazard_per_hour(features),
                )

    def test_saturation_is_monotone_but_not_affine_in_log_risk(self):
        model = LocalNonAffineRiskWarp(self.source, {1: "cup"}, "tanh_log")
        values = (0.02, 0.06, 0.12, 0.30, 0.60)
        warped = tuple(model.warped_hazard(value) for value in values)
        self.assertEqual(tuple(sorted(warped)), warped)
        slopes = [
            (warped[index + 1] / warped[index]) / (values[index + 1] / values[index])
            for index in range(len(values) - 1)
        ]
        self.assertGreater(max(slopes) - min(slopes), 0.1)

    def test_fold_is_nonmonotone_and_bump_is_smoothly_localized(self):
        fold = LocalNonAffineRiskWarp(self.source, {4: "busy"}, "absolute_log_fold")
        self.assertGreater(fold.warped_hazard(0.03), fold.warped_hazard(0.06))
        self.assertGreater(fold.warped_hazard(0.06), fold.warped_hazard(0.12))
        self.assertLess(fold.warped_hazard(0.12), fold.warped_hazard(0.24))
        bump = LocalNonAffineRiskWarp(
            self.source, {1: "cup", 4: "busy"}, "gaussian_log_bump"
        )
        central_ratio = bump.warped_hazard(0.12) / 0.12
        distant_ratio = bump.warped_hazard(0.60) / 0.60
        self.assertGreater(central_ratio, distant_ratio)

    def test_invalid_routing_and_parameters_fail_closed(self):
        with self.assertRaises(ValueError):
            LocalNonAffineRiskWarp(self.source, {}, "tanh_log")
        with self.assertRaises(ValueError):
            LocalNonAffineRiskWarp(self.source, {5: "bad"}, "tanh_log")
        with self.assertRaises(ValueError):
            LocalNonAffineRiskWarp(
                self.source,
                {1: "cup"},
                "unknown",  # type: ignore[arg-type]
            )
        model = LocalNonAffineRiskWarp(self.source, {1: "cup"}, "tanh_log")
        with self.assertRaisesRegex(ValueError, "five features"):
            model.applies_to(("too", "short"))
        with self.assertRaisesRegex(ValueError, "unique"):
            affected_contexts(model, self.source, (self.contexts[0], self.contexts[0]))


if __name__ == "__main__":
    unittest.main()
