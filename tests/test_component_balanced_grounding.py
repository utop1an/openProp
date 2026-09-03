import unittest

from openprop.component_balanced_grounding import (
    GROUNDING_MODEL_CONDITIONS,
    aggregate_component_balanced_runs,
    component_balanced_grounding_benchmark,
    evaluate_component_balanced_seed,
)
from openprop.compositional_persistence import (
    compositional_grounding_registry,
    compositional_location_data,
    evaluate_grounding_model,
)
from openprop.statistical_persistence import PerContextExponentialPersistenceModel


class ComponentBalancedGroundingTests(unittest.TestCase):
    def test_cases_are_balanced_isolated_and_truth_separated(self) -> None:
        cases = component_balanced_grounding_benchmark()
        self.assertEqual(40, len(cases))
        self.assertEqual(20, sum("target-old" in case.tags for case in cases))
        self.assertEqual(20, sum("target-new" in case.tags for case in cases))
        self.assertEqual(
            {"probe-subject", "probe-relation", "probe-scene"},
            {tag for case in cases for tag in case.tags if tag.startswith("probe-")},
        )
        self.assertEqual(len(cases), len({case.case_id for case in cases}))
        for case in cases:
            plausible = [
                entity for entity in case.entities if entity.entity_id != case.entities[0].entity_id
            ]
            matching = [
                entity
                for entity in case.entities
                if entity.properties["color"].value == "red"
            ]
            self.assertEqual(2, len(matching))
            typed_contexts = {
                (
                    entity.properties["type"].value,
                    entity.properties["location"].value.predicate,
                    entity.properties["location"].value.arguments["object"],
                    entity.properties["scene"].value,
                )
                for entity in matching
            }
            self.assertEqual(1, len(typed_contexts))
            for entity in case.entities:
                self.assertNotIn("current_truth", entity.properties)
                self.assertNotIn("target", entity.properties)

    def test_true_hazard_oracle_solves_all_cases_and_order_is_invariant(self) -> None:
        dataset = compositional_location_data(samples_per_context=2)
        hazards = {context.features(): context.hazard_per_hour for context in dataset.contexts}
        oracle = PerContextExponentialPersistenceModel(
            hazards,
            global_hazard=0.12,
            trained_properties=frozenset({"location"}),
        )
        cases = component_balanced_grounding_benchmark()
        registry = compositional_grounding_registry()
        forward = evaluate_grounding_model("oracle", oracle, cases, registry)
        reversed_cases = tuple(
            type(case)(
                case.case_id,
                case.query,
                tuple(reversed(case.entities)),
                case.target_id,
                case.gold_frame,
                case.as_of,
                case.current_truth,
                case.tags,
            )
            for case in cases
        )
        backward = evaluate_grounding_model("oracle", oracle, reversed_cases, registry)
        self.assertEqual(1.0, forward.top1_accuracy)
        self.assertEqual(forward, backward)

    def test_invalid_case_parameters_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "old > new"):
            component_balanced_grounding_benchmark(old_age_hours=1, new_age_hours=1)
        with self.assertRaisesRegex(ValueError, "confidence"):
            component_balanced_grounding_benchmark(new_confidence=1.0)

    def test_aggregation_pairs_each_probe_with_its_axis_ablation(self) -> None:
        def run(seed: int, full: float) -> dict[str, object]:
            conditions = {}
            for name in GROUNDING_MODEL_CONDITIONS:
                conditions[name] = {
                    "top1": full if name == "full_context" else 0.5,
                    "top1_by_probe": {
                        "subject": full if name == "full_context" else 0.5,
                        "relation": full if name == "full_context" else 0.5,
                        "scene": full if name == "full_context" else 0.5,
                    },
                }
            return {"seed": seed, "conditions": conditions}

        report = aggregate_component_balanced_runs(
            (run(1, 0.9), run(2, 1.0)), bootstrap_samples=100
        )
        self.assertEqual("no_subject", report["paired_probe_advantage"]["subject"]["ablation"])
        self.assertEqual("no_relation", report["paired_probe_advantage"]["relation"]["ablation"])
        self.assertEqual("no_scene", report["paired_probe_advantage"]["scene"]["ablation"])
        self.assertGreater(
            report["paired_probe_advantage"]["subject"]["mean_full_advantage"],
            0.0,
        )

        simultaneous = report["simultaneous_probe_inference"]
        self.assertEqual(["subject", "relation", "scene"], simultaneous["family"])
        self.assertEqual(3, simultaneous["family_size"])
        for probe in simultaneous["family"]:
            lower, upper = report["paired_probe_advantage"][probe][
                "simultaneous_bootstrap_95_ci"
            ]
            self.assertGreater(lower, 0.0)
    def test_small_learned_run_covers_the_frozen_model_matrix(self) -> None:
        dataset = compositional_location_data(samples_per_context=3, seed=31)
        report = evaluate_component_balanced_seed(
            seed=31,
            dataset=dataset,
            cases=component_balanced_grounding_benchmark(),
            epochs=20,
        )
        self.assertEqual(31, report["seed"])
        self.assertEqual(set(GROUNDING_MODEL_CONDITIONS), set(report["conditions"]))
        for name, indices in GROUNDING_MODEL_CONDITIONS.items():
            row = report["conditions"][name]
            self.assertEqual(list(indices), row["active_feature_indices"])
            self.assertGreater(row["validation_hazard_scale"], 0.0)
            self.assertGreaterEqual(row["top1"], 0.0)
            self.assertLessEqual(row["top1"], 1.0)
            self.assertEqual(
                {"subject", "relation", "scene"}, set(row["top1_by_probe"])
            )
            self.assertEqual({"old", "new"}, set(row["top1_by_target_age"]))


if __name__ == "__main__":
    unittest.main()
