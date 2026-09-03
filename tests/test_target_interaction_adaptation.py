from __future__ import annotations

import unittest

from openprop.statistical_persistence import FactorizedExponentialPersistenceModel
from openprop.target_adaptation import build_target_calibration_protocol
from openprop.target_adaptation_stress import target_adaptation_stress_data
from openprop.target_interaction_adaptation import (
    DEFAULT_TYPED_PARTITIONS,
    fit_hierarchical_typed_interaction_gate,
)


class TargetInteractionAdaptationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.dataset = target_adaptation_stress_data(samples_per_context=24, seed=53)
        cls.source = FactorizedExponentialPersistenceModel.fit(
            cls.dataset.train,
            epochs=800,
        )
        cls.source.calibrate(cls.dataset.validation)
        cls.protocol = build_target_calibration_protocol(
            cls.dataset,  # type: ignore[arg-type]
            max_calibration_per_context=16,
            split_seed=9_000_056,
        )

    def _fit(self, condition: str, *, partitions=DEFAULT_TYPED_PARTITIONS):
        return fit_hierarchical_typed_interaction_gate(
            self.source,
            self.protocol.calibration_subset(condition, 16),
            split_seed=12_000_056,
            candidate_partitions=partitions,
            epochs=800,
        )

    def test_in_distribution_gate_fails_closed(self) -> None:
        gate = self._fit("in_distribution")
        self.assertFalse(gate.activated)
        self.assertIsNone(gate.selected_partition)
        self.assertEqual(frozenset(), gate.significant_groups)
        self.assertEqual(12, gate.candidate_group_count)
        for row in self.protocol.tests["in_distribution"]:
            self.assertEqual(
                self.source.hazard_per_hour(row.features()),
                gate.hazard_per_hour(row.features()),
            )

    def test_xor_selects_pairwise_partition_and_preserves_stable_cells(self) -> None:
        gate = self._fit("subject_scene_xor_reversal")
        self.assertEqual((1, 4), gate.selected_partition)
        self.assertEqual(
            frozenset({("cup", "quiet"), ("book", "busy"), ("tool", "busy")}),
            gate.significant_groups,
        )
        self.assertTrue(gate.partition_heterogeneity_veto["f4"])
        self.assertFalse(gate.partition_heterogeneity_veto["f1xf4"])
        for subject, scene in gate.significant_groups:
            label = f"f1={subject}|f4={scene}"
            self.assertLessEqual(
                gate.confirmation_p_values[label], gate.bonferroni_threshold
            )
        changed = self.dataset.changed_contexts["subject_scene_xor_reversal"]
        for row in self.protocol.tests["subject_scene_xor_reversal"]:
            source_hazard = self.source.hazard_per_hour(row.features())
            adapted_hazard = gate.hazard_per_hour(row.features())
            if row.features() in changed:
                self.assertNotEqual(source_hazard, adapted_hazard)
            else:
                self.assertEqual(source_hazard, adapted_hazard)

    def test_candidate_order_does_not_change_selection_or_predictions(self) -> None:
        forward = self._fit("subject_scene_xor_reversal")
        reverse = self._fit(
            "subject_scene_xor_reversal",
            partitions=tuple(reversed(DEFAULT_TYPED_PARTITIONS)),
        )
        self.assertEqual(forward.selected_partition, reverse.selected_partition)
        self.assertEqual(forward.significant_groups, reverse.significant_groups)
        self.assertEqual(forward.confirmation_p_values, reverse.confirmation_p_values)
        for row in self.protocol.tests["subject_scene_xor_reversal"]:
            self.assertEqual(
                forward.hazard_per_hour(row.features()),
                reverse.hazard_per_hour(row.features()),
            )

    def test_validation_rejects_duplicate_groups_and_invalid_partitions(self) -> None:
        rows = self.protocol.calibration_subset("in_distribution", 16)
        with self.assertRaisesRegex(ValueError, "unique"):
            fit_hierarchical_typed_interaction_gate(
                self.source,
                rows + (rows[0],),
                split_seed=1,
                epochs=10,
            )
        with self.assertRaisesRegex(ValueError, "increasing"):
            fit_hierarchical_typed_interaction_gate(
                self.source,
                rows,
                split_seed=1,
                candidate_partitions=((4, 1),),
                epochs=10,
            )


if __name__ == "__main__":
    unittest.main()
