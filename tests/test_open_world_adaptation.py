from __future__ import annotations

import math
import unittest

from openprop.advanced_survival_evaluation import evaluate_survival_advanced
from openprop.open_world_adaptation import (
    CALIBRATED_NOVEL_SUBJECT,
    OPEN_WORLD_ADAPTATION_CONDITIONS,
    SOURCE_SEEN,
    TARGET_CALIBRATED_NOVEL,
    TARGET_UNCALIBRATED_NOVEL,
    UNCALIBRATED_NOVEL_SUBJECT,
    open_world_adaptation_data,
)
from openprop.persistence_data import PersistenceTrainingExample
from openprop.statistical_persistence import FactorizedExponentialPersistenceModel
from openprop.target_adaptation import build_target_calibration_protocol
from openprop.target_interaction_adaptation import (
    fit_hierarchical_typed_interaction_gate,
)


class _ConstantRiskModel:
    def __init__(self, hazard: float) -> None:
        self.hazard = hazard

    def hazard_per_hour(self, features: tuple[str, ...]) -> float:
        if len(features) != 5:
            raise ValueError("five typed features required")
        return self.hazard


class OpenWorldAdaptationTests(unittest.TestCase):
    def test_dataset_pairs_conditions_and_keeps_novel_values_typed(self) -> None:
        dataset = open_world_adaptation_data(samples_per_context=8, seed=31)
        self.assertEqual(45, len(dataset.contexts))
        self.assertEqual(set(OPEN_WORLD_ADAPTATION_CONDITIONS), set(dataset.tests))
        reference = tuple(
            (row.group_id, row.features())
            for row in dataset.tests["open_world_control"]
        )
        for rows in dataset.tests.values():
            self.assertEqual(
                reference,
                tuple((row.group_id, row.features()) for row in rows),
            )
        labels = set(dataset.support_by_context.values())
        self.assertEqual(
            {SOURCE_SEEN, TARGET_CALIBRATED_NOVEL, TARGET_UNCALIBRATED_NOVEL},
            labels,
        )
        subjects = {features[1] for features in dataset.support_by_context}
        self.assertIn(CALIBRATED_NOVEL_SUBJECT, subjects)
        self.assertIn(UNCALIBRATED_NOVEL_SUBJECT, subjects)
        self.assertNotIn("unknown", subjects)

    def test_protocol_keeps_predeclared_novel_subject_test_only(self) -> None:
        dataset = open_world_adaptation_data(samples_per_context=8, seed=41)
        protocol = build_target_calibration_protocol(
            dataset,  # type: ignore[arg-type]
            max_calibration_per_context=3,
            split_seed=41 + 37_000_003,
            calibration_contexts=dataset.calibration_contexts,
        )
        calibration = protocol.calibration_subset("open_world_control", 3)
        self.assertEqual(36 * 3, len(calibration))
        self.assertTrue(all(row.subject_type != UNCALIBRATED_NOVEL_SUBJECT for row in calibration))
        plate_test = [
            row
            for row in protocol.tests["open_world_control"]
            if row.subject_type == UNCALIBRATED_NOVEL_SUBJECT
        ]
        self.assertEqual(9 * 8, len(plate_test))
        self.assertEqual(dataset.test_only_contexts, protocol.test_only_contexts)
        with self.assertRaisesRegex(ValueError, "occur"):
            build_target_calibration_protocol(
                dataset,  # type: ignore[arg-type]
                max_calibration_per_context=3,
                split_seed=1,
                calibration_contexts={
                    ("location", "ghost", "on", "table", "quiet")
                },
            )

    def test_general_scope_can_fit_intercept_only_cell(self) -> None:
        source = _ConstantRiskModel(0.1)
        rows = tuple(
            PersistenceTrainingExample(
                property_name="location",
                subject_type="bottle",
                state_predicate="inside",
                context_object="cabinet",
                scene="quiet",
                duration_seconds=0.25 * 3600.0,
                event_observed=True,
                group_id=f"constant-{index:03d}",
            )
            for index in range(60)
        )
        general = fit_hierarchical_typed_interaction_gate(
            source,
            rows,
            split_seed=71,
            candidate_partitions=((),),
            activation_scope="any_predictive_gain",
            epochs=500,
        )
        reversal = fit_hierarchical_typed_interaction_gate(
            source,
            rows,
            split_seed=71,
            candidate_partitions=((),),
            activation_scope="reversal_only",
            epochs=200,
        )
        self.assertEqual((), general.selected_partition)
        self.assertFalse(general.discovery_slope_fitted["global"])
        self.assertTrue(general.discovery_bic_active["global"])
        self.assertGreater(general.hazard_per_hour(rows[0].features()), 1.0)
        unseen = ("location", "plate", "inside", "cabinet", "quiet")
        self.assertTrue(math.isclose(
            general.hazard_per_hour(unseen),
            source.hazard_per_hour(unseen),
            rel_tol=0.0,
            abs_tol=0.0,
        ))
        self.assertFalse(reversal.activated)

    def test_declared_triple_represents_latin_rule_but_unseen_group_falls_back(self) -> None:
        dataset = open_world_adaptation_data(samples_per_context=96, seed=67)
        source = FactorizedExponentialPersistenceModel.fit(
            dataset.train,
            epochs=600,
        )
        protocol = build_target_calibration_protocol(
            dataset,  # type: ignore[arg-type]
            max_calibration_per_context=48,
            split_seed=67 + 37_000_003,
            calibration_contexts=dataset.calibration_contexts,
        )
        condition = "three_way_subject_object_scene_latin"
        calibration = protocol.calibration_subset(condition, 48)
        gate = fit_hierarchical_typed_interaction_gate(
            source,
            calibration,
            split_seed=67 + 41_000_003,
            candidate_partitions=((), (1, 3, 4)),
            activation_scope="any_predictive_gain",
            discovery_complexity="bic",
            epochs=400,
        )
        self.assertEqual((1, 3, 4), gate.selected_partition)
        test = protocol.tests[condition]
        source_nll = evaluate_survival_advanced(
            source, test, horizons_hours=(1.0, 4.0, 8.0, 12.0)
        ).negative_log_likelihood
        gate_nll = evaluate_survival_advanced(
            gate, test, horizons_hours=(1.0, 4.0, 8.0, 12.0)
        ).negative_log_likelihood
        self.assertLess(gate_nll, source_nll - 0.1)

        plate_features = next(iter(dataset.test_only_contexts))
        self.assertTrue(math.isclose(
            gate.hazard_per_hour(plate_features),
            source.hazard_per_hour(plate_features),
            rel_tol=0.0,
            abs_tol=0.0,
        ))

    def test_partition_order_is_limited_to_three(self) -> None:
        rows = tuple(
            PersistenceTrainingExample(
                "location", "cup", "on", "table", "quiet", 3600.0, False,
                f"order-{index:03d}",
            )
            for index in range(6)
        )
        with self.assertRaisesRegex(ValueError, "three-way"):
            fit_hierarchical_typed_interaction_gate(
                _ConstantRiskModel(0.1), rows, split_seed=3,
                candidate_partitions=((0, 1, 3, 4),),
            )


if __name__ == "__main__":
    unittest.main()
