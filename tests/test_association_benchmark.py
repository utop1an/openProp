import unittest

from openprop.association_benchmark import (
    ASSOCIATION_CONDITIONS,
    AssociationBenchmarkCase,
    AssociationBenchmarkSplit,
    association_benchmark_registry,
    association_benchmark_split,
    calibrate_association_policy,
    default_association_benchmark_associator,
    evaluate_association,
)


class AssociationBenchmarkTests(unittest.TestCase):
    def test_split_is_balanced_group_disjoint_and_truth_separate(self):
        split = association_benchmark_split(
            calibration_per_condition=3,
            test_per_condition=4,
        )
        self.assertFalse(
            {case.group_id for case in split.calibration}
            & {case.group_id for case in split.test}
        )
        for rows, expected in ((split.calibration, 3), (split.test, 4)):
            for condition in ASSOCIATION_CONDITIONS:
                self.assertEqual(
                    expected,
                    sum(case.condition == condition for case in rows),
                )
        for case in (*split.calibration, *split.test):
            for entity in case.entities:
                self.assertNotIn("current_truth", entity.properties)
                self.assertNotIn("target", entity.properties)
            matcher_input = (
                case.detection,
                case.query,
                case.entities,
            )
            self.assertNotIn(case.target_entity_id, matcher_input)

    def test_validation_only_calibration_meets_false_update_gate(self):
        split = association_benchmark_split(
            calibration_per_condition=5,
            test_per_condition=8,
        )
        registry = association_benchmark_registry()
        base = default_association_benchmark_associator(registry)
        calibrated = calibrate_association_policy(
            base,
            split.calibration,
            max_false_update_rate=0.0,
        )
        self.assertEqual(30, calibrated.searched_policies)
        self.assertGreater(calibrated.feasible_policies, 0)
        self.assertEqual(0.0, calibrated.validation.false_update_rate)

        associator = default_association_benchmark_associator(
            registry,
            policy=calibrated.policy,
        )
        report = evaluate_association(associator, split.test)
        self.assertEqual(0, report.false_updates)
        self.assertEqual(8, report.correct_updates)
        self.assertEqual(1.0, report.selective_accuracy)
        self.assertEqual(1.0, report.candidate_order_invariance)
        self.assertEqual(1.0, report.query_paraphrase_invariance)
        self.assertEqual(
            8,
            report.by_condition["strong"]["correct_updates"],
        )
        for condition in ("ambiguous", "misleading", "null"):
            self.assertEqual(
                1.0,
                report.by_condition[condition]["abstention_rate"],
            )

    def test_evaluation_metrics_keep_abstentions_in_denominator(self):
        split = association_benchmark_split(
            calibration_per_condition=2,
            test_per_condition=3,
        )
        registry = association_benchmark_registry()
        report = evaluate_association(
            default_association_benchmark_associator(registry),
            split.test,
        )
        self.assertEqual(12, report.total)
        self.assertEqual(report.total, report.accepted + report.abstentions)
        self.assertAlmostEqual(
            report.correct_update_rate,
            report.correct_updates / report.total,
        )
        self.assertAlmostEqual(
            report.target_recall,
            report.correct_updates / report.target_present,
        )

    def test_invalid_protocol_inputs_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "positive"):
            association_benchmark_split(calibration_per_condition=0)
        split = association_benchmark_split(
            calibration_per_condition=1,
            test_per_condition=1,
        )
        with self.assertRaisesRegex(ValueError, "disjoint"):
            AssociationBenchmarkSplit(split.calibration, split.calibration)
        registry = association_benchmark_registry()
        base = default_association_benchmark_associator(registry)
        with self.assertRaisesRegex(ValueError, "cannot be empty"):
            calibrate_association_policy(base, ())
        with self.assertRaisesRegex(ValueError, r"in \[0, 1\]"):
            calibrate_association_policy(
                base,
                split.calibration,
                max_false_update_rate=1.1,
            )
        misleading = tuple(
            case for case in split.calibration if case.condition == "misleading"
        )
        with self.assertRaisesRegex(ValueError, "false-update gate"):
            calibrate_association_policy(
                base,
                misleading,
                acceptance_thresholds=(0.0,),
                margin_thresholds=(0.0,),
                max_false_update_rate=0.0,
            )


    def test_case_rejects_truth_leakage(self):
        split = association_benchmark_split(
            calibration_per_condition=1,
            test_per_condition=1,
        )
        case = split.calibration[0]
        leaking_entities = list(case.entities)
        leaking_entities[0].properties["target"] = leaking_entities[0].properties["type"]
        with self.assertRaisesRegex(ValueError, "truth"):
            AssociationBenchmarkCase(
                case.case_id,
                case.group_id,
                case.condition,
                case.query,
                case.paraphrase_query,
                tuple(leaking_entities),
                case.detection,
                case.target_entity_id,
            )


if __name__ == "__main__":
    unittest.main()
