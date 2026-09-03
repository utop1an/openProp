import unittest

from openprop.latent_mechanism_shift import (
    MECHANISM_CONDITIONS,
    latent_mechanism_shift_data,
)


class LatentMechanismShiftTests(unittest.TestCase):
    def test_conditions_share_latent_test_draws_without_feature_leakage(self) -> None:
        dataset = latent_mechanism_shift_data(samples_per_context=20, seed=41)
        self.assertEqual(set(MECHANISM_CONDITIONS), set(dataset.tests))
        controls = dataset.tests["in_distribution"]
        self.assertEqual(60, len(controls))
        for condition, rows in dataset.tests.items():
            self.assertEqual(
                [row.group_id for row in controls],
                [row.group_id for row in rows],
            )
            self.assertEqual(
                [row.features() for row in controls],
                [row.features() for row in rows],
            )
            self.assertTrue(
                all(condition not in row.group_id for row in rows)
            )
        self.assertTrue(
            any(
                control.duration_seconds != shifted.duration_seconds
                or control.event_observed != shifted.event_observed
                for control, shifted in zip(
                    controls,
                    dataset.tests["global_rate_acceleration"],
                    strict=True,
                )
            )
        )

    def test_typed_factor_shift_reverses_true_context_risk_order(self) -> None:
        dataset = latent_mechanism_shift_data(samples_per_context=5, seed=41)
        source = dataset.test_hazards["in_distribution"]
        reversal = dataset.test_hazards["typed_factor_reversal"]
        source_order = sorted(source, key=source.get)
        reversal_order = sorted(reversal, key=reversal.get)
        self.assertEqual(source_order, list(reversed(reversal_order)))

    def test_source_splits_remain_group_disjoint_from_paired_tests(self) -> None:
        dataset = latent_mechanism_shift_data(samples_per_context=10, seed=41)
        train_groups = {row.group_id for row in dataset.train}
        validation_groups = {row.group_id for row in dataset.validation}
        test_groups = {
            row.group_id for row in dataset.tests["in_distribution"]
        }
        self.assertTrue(train_groups.isdisjoint(validation_groups))
        self.assertTrue(train_groups.isdisjoint(test_groups))
        self.assertTrue(validation_groups.isdisjoint(test_groups))


if __name__ == "__main__":
    unittest.main()
