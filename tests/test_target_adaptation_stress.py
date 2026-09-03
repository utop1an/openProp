import unittest

from openprop.advanced_survival_evaluation import concordance_index
from openprop.statistical_persistence import FactorizedExponentialPersistenceModel
from openprop.target_adaptation import build_target_calibration_protocol
from openprop.target_adaptation_stress import (
    STRESS_CONDITIONS,
    corrupt_calibration_event_labels,
    fit_confirmed_feature_grouped_sign_gate,
    fit_feature_grouped_sign_gate,
    target_adaptation_stress_data,
)


class TargetAdaptationStressTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.dataset = target_adaptation_stress_data(
            samples_per_context=30,
            seed=41,
        )
        cls.source = FactorizedExponentialPersistenceModel.fit(
            cls.dataset.train,
            epochs=500,
        )
        cls.source.calibrate(cls.dataset.validation)
        cls.protocol = build_target_calibration_protocol(
            cls.dataset,  # type: ignore[arg-type]
            max_calibration_per_context=12,
            split_seed=7001,
        )

    def test_generator_pairs_all_eighteen_contexts_and_declares_changes(self) -> None:
        self.assertEqual(set(STRESS_CONDITIONS), set(self.dataset.tests))
        self.assertTrue(all(len(rows) == 540 for rows in self.dataset.tests.values()))
        reference = self.dataset.tests["in_distribution"]
        for rows in self.dataset.tests.values():
            self.assertEqual(
                [(row.group_id, row.features()) for row in reference],
                [(row.group_id, row.features()) for row in rows],
            )
        self.assertEqual(0, len(self.dataset.changed_contexts["in_distribution"]))
        self.assertEqual(18, len(self.dataset.changed_contexts["global_reversal"]))
        self.assertEqual(6, len(self.dataset.changed_contexts["subject_cup_reversal"]))
        self.assertEqual(9, len(self.dataset.changed_contexts["scene_busy_reversal"]))
        self.assertEqual(
            9, len(self.dataset.changed_contexts["subject_scene_xor_reversal"])
        )

    def test_declared_subject_gate_repairs_only_the_changed_subject(self) -> None:
        condition = "subject_cup_reversal"
        calibration = self.protocol.calibration_subset(condition, 12)
        test = self.protocol.tests[condition]
        model = fit_feature_grouped_sign_gate(
            self.source,
            calibration,
            feature_index=1,
            epochs=800,
        )
        self.assertEqual(frozenset({"cup"}), model.activated_groups)
        changed = self.dataset.changed_contexts[condition]
        changed_rows = tuple(row for row in test if row.features() in changed)
        stable_rows = tuple(row for row in test if row.features() not in changed)
        self.assertGreater(
            concordance_index(model, changed_rows),
            concordance_index(self.source, changed_rows) + 0.5,
        )
        self.assertEqual(
            concordance_index(self.source, stable_rows),
            concordance_index(model, stable_rows),
        )

    def test_declared_scene_gate_repairs_only_busy_scenes(self) -> None:
        condition = "scene_busy_reversal"
        calibration = self.protocol.calibration_subset(condition, 12)
    def test_confirmation_gate_requires_two_identity_halves_to_agree(self) -> None:
        condition = "scene_busy_reversal"
        calibration = self.protocol.calibration_subset(condition, 12)
        model = fit_confirmed_feature_grouped_sign_gate(
            self.source,
            calibration,
            feature_index=4,
            confirmation_seed=1234,
            epochs=600,
        )
        self.assertEqual(frozenset({"busy"}), model.activated_groups)
        self.assertLess(model.group_confirmation_slopes["busy"][0], 0.0)
        self.assertLess(model.group_confirmation_slopes["busy"][1], 0.0)
        stable = fit_confirmed_feature_grouped_sign_gate(
            self.source,
            self.protocol.calibration_subset("in_distribution", 12),
            feature_index=4,
            confirmation_seed=1234,
            epochs=600,
        )
        self.assertEqual(frozenset(), stable.activated_groups)

        model = fit_feature_grouped_sign_gate(
            self.source,
            calibration,
            feature_index=4,
            epochs=800,
        )
        self.assertEqual(frozenset({"busy"}), model.activated_groups)

    def test_label_noise_is_deterministic_scoped_and_nonmutating(self) -> None:
        rows = self.protocol.calibration_subset("global_reversal", 8)
        first = corrupt_calibration_event_labels(rows, fraction=0.2, seed=99)
        second = corrupt_calibration_event_labels(rows, fraction=0.2, seed=99)
        self.assertEqual(first, second)
        changed = [
            (before, after)
            for before, after in zip(rows, first, strict=True)
            if before.event_observed != after.event_observed
        ]
        self.assertEqual(round(0.2 * len(rows)), len(changed))
        self.assertTrue(
            all(
                before.group_id == after.group_id
                and before.features() == after.features()
                and before.duration_seconds == after.duration_seconds
                for before, after in changed
            )
        )
        self.assertEqual(
            rows,
            self.protocol.calibration_subset("global_reversal", 8),
        )


if __name__ == "__main__":
    unittest.main()
