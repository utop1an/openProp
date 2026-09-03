import unittest

from openprop.benchmark import core_registry
from openprop.evaluation import EvaluationStrategy, evaluate
from openprop.interference import interference_benchmark


class InterferenceBenchmarkTests(unittest.TestCase):
    def test_every_case_contains_irrelevant_attributes(self):
        cases = interference_benchmark()
        self.assertEqual(len(cases), 30)
        for case in cases:
            self.assertTrue(case.distractor_constraints)
            self.assertNotEqual(case.target_id, case.distractor_entity_id)
            self.assertIn("无关背景记录", case.query)

    def test_weighting_resists_interference_better_than_equal_weights(self):
        cases = interference_benchmark()
        weighted = evaluate(cases, core_registry(), EvaluationStrategy.GOLD_WEIGHTED)
        equal = evaluate(cases, core_registry(), EvaluationStrategy.GOLD_EQUAL)
        self.assertEqual(weighted.top1_accuracy, 1.0)
        self.assertGreater(weighted.top1_accuracy, equal.top1_accuracy)
        self.assertLessEqual(equal.top1_accuracy, 0.2)

    def test_rejects_large_distractor_weight(self):
        with self.assertRaises(ValueError):
            interference_benchmark(distractor_weight=0.1)


if __name__ == "__main__":
    unittest.main()
