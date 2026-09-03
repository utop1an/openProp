import unittest

from openprop.benchmark import core_benchmark, core_registry
from openprop.evaluation import EvaluationStrategy, evaluate


class EvaluationTests(unittest.TestCase):
    def test_core_benchmark_has_thirty_valid_unique_cases(self):
        cases = core_benchmark()
        self.assertEqual(len(cases), 30)
        self.assertEqual(len({case.case_id for case in cases}), 30)
        for case in cases:
            self.assertIn(case.target_id, {entity.entity_id for entity in case.entities})
            self.assertTrue(case.gold_frame.constraints)

    def test_gold_weighted_resolves_every_case(self):
        report = evaluate(
            core_benchmark(),
            core_registry(),
            EvaluationStrategy.GOLD_WEIGHTED,
        )
        self.assertEqual(report.failures, 0)
        self.assertEqual(report.top1_accuracy, 1.0)
        self.assertEqual(report.mean_reciprocal_rank, 1.0)
        self.assertEqual(report.property_f1, 1.0)

    def test_llm_strategy_requires_client(self):
        with self.assertRaisesRegex(ValueError, "llm_client"):
            evaluate(
                core_benchmark()[:1],
                core_registry(),
                EvaluationStrategy.LLM_WEIGHTED,
            )


if __name__ == "__main__":
    unittest.main()
