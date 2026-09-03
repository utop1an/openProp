import unittest

from openprop.compositional_persistence import (
    compositional_grounding_benchmark,
    compositional_grounding_registry,
    compositional_location_data,
)
from openprop.typed_context_ablation import (
    TYPED_CONTEXT_CONDITIONS,
    aggregate_typed_context_runs,
    evaluate_typed_context_seed,
)


def _run(seed: int, offset: float) -> dict[str, object]:
    conditions = {}
    for index, condition in enumerate(TYPED_CONTEXT_CONDITIONS):
        quality = float(index) / 100.0 + offset
        conditions[condition.name] = {
            "negative_log_likelihood": 2.0 - quality,
            "concordance_index": 0.5 + quality,
            "integrated_brier_score": 0.3 - quality / 2.0,
            "grounding_top1": min(1.0, 0.5 + quality),
        }
    return {"seed": seed, "conditions": conditions}


class TypedContextAblationTests(unittest.TestCase):
    def test_condition_matrix_is_semantic_and_complete(self) -> None:
        conditions = {condition.name: condition for condition in TYPED_CONTEXT_CONDITIONS}
        self.assertEqual(8, len(conditions))
        self.assertEqual((), conditions["intercept_only"].active_feature_indices)
        self.assertEqual((2, 3), conditions["relation_only"].active_feature_indices)
        self.assertEqual(
            (1, 2, 3, 4), conditions["full_context"].active_feature_indices
        )
        self.assertEqual(
            ("subject", "relation", "scene"),
            conditions["full_context"].active_groups,
        )

    def test_aggregation_orients_all_metrics_as_full_advantage(self) -> None:
        report = aggregate_typed_context_runs(
            (_run(31, 0.0), _run(41, 0.01), _run(53, -0.01)),
            bootstrap_samples=200,
        )
        comparison = report["paired_full_advantage"]["intercept_only"]
        for metric in (
            "negative_log_likelihood",
            "concordance_index",
            "integrated_brier_score",
            "grounding_top1",
        ):
            self.assertGreater(comparison[metric]["mean_full_advantage"], 0.0)
            self.assertEqual(3, comparison[metric]["wins"])
            self.assertEqual(0, comparison[metric]["losses"])

        simultaneous = report["simultaneous_primary_component_inference"]
        self.assertEqual(3, simultaneous["family_size"])
        self.assertEqual(
            [
                "relation_scene",
                "subject_scene",
                "subject_relation",
            ], simultaneous["family"]
        )
        for condition in simultaneous["family"]:
            lower, upper = report["paired_full_advantage"][condition][
                "negative_log_likelihood"
            ]["simultaneous_bootstrap_95_ci"]
            self.assertGreater(lower, 0.0)
    def test_small_end_to_end_run_keeps_conditions_paired(self) -> None:
        dataset = compositional_location_data(samples_per_context=3, seed=31)
        report = evaluate_typed_context_seed(
            seed=31,
            dataset=dataset,
            grounding_cases=compositional_grounding_benchmark(repetitions=1),
            registry=compositional_grounding_registry(),
            epochs=20,
        )
        self.assertEqual(31, report["seed"])
        self.assertEqual(
            {condition.name for condition in TYPED_CONTEXT_CONDITIONS},
            set(report["conditions"]),
        )
        for condition in TYPED_CONTEXT_CONDITIONS:
            row = report["conditions"][condition.name]
            self.assertEqual(
                list(condition.active_feature_indices),
                row["active_feature_indices"],
            )
            self.assertGreater(row["negative_log_likelihood"], 0.0)
            self.assertGreaterEqual(row["concordance_index"], 0.0)
            self.assertLessEqual(row["concordance_index"], 1.0)
            self.assertGreaterEqual(row["grounding_top1"], 0.0)
            self.assertLessEqual(row["grounding_top1"], 1.0)

    def test_aggregation_rejects_duplicate_or_incomplete_runs(self) -> None:
        run = _run(31, 0.0)
        with self.assertRaisesRegex(ValueError, "unique"):
            aggregate_typed_context_runs((run, run), bootstrap_samples=10)
        incomplete = _run(41, 0.0)
        del incomplete["conditions"]["scene_only"]
        with self.assertRaisesRegex(ValueError, "frozen condition matrix"):
            aggregate_typed_context_runs((incomplete,), bootstrap_samples=10)


if __name__ == "__main__":
    unittest.main()
