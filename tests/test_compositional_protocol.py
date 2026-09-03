import unittest
from dataclasses import replace

from openprop.compositional_persistence import (
    compositional_grounding_benchmark,
    compositional_grounding_registry,
    compositional_location_data,
    evaluate_grounding_model,
)
from openprop.statistical_persistence import PerContextExponentialPersistenceModel


class CompositionalProtocolTests(unittest.TestCase):
    def test_current_truth_is_separate_and_not_stored_as_entity_metadata(self) -> None:
        for case in compositional_grounding_benchmark(repetitions=2):
            self.assertEqual(
                {entity.entity_id for entity in case.entities},
                set(case.current_truth),
            )
            self.assertTrue(
                all("current_truth" not in entity.properties for entity in case.entities)
            )

    def test_contextual_result_is_candidate_order_invariant(self) -> None:
        dataset = compositional_location_data(samples_per_context=10)
        oracle = PerContextExponentialPersistenceModel(
            {
                context.features(): context.hazard_per_hour
                for context in dataset.contexts
            },
            global_hazard=0.12,
            trained_properties=frozenset({"location"}),
        )
        cases = compositional_grounding_benchmark(repetitions=2)
        reversed_cases = tuple(
            replace(case, entities=tuple(reversed(case.entities))) for case in cases
        )
        registry = compositional_grounding_registry()
        original = evaluate_grounding_model("oracle", oracle, cases, registry)
        reordered = evaluate_grounding_model(
            "oracle", oracle, reversed_cases, registry
        )
        self.assertEqual(1.0, original.top1_accuracy)
        self.assertEqual(original.top1_accuracy, reordered.top1_accuracy)


if __name__ == "__main__":
    unittest.main()
