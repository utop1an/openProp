import unittest

from openprop.language_paraphrases import (
    CLEAN_QUERIES,
    CONTROL_QUERIES,
    RELATION_QUERIES,
    paraphrased_temporal_grounding_benchmark,
)
from openprop.language_temporal_grounding import (
    LanguageTemporalStrategy,
    evaluate_language_temporal_grounding,
)
from openprop.temporal_grounding import (
    temporal_grounding_benchmark,
    temporal_grounding_registry,
)


class LanguageParaphraseBenchmarkTests(unittest.TestCase):
    def test_queries_are_unique_disjoint_and_bilingual(self):
        cases = paraphrased_temporal_grounding_benchmark()
        original = {case.query for case in temporal_grounding_benchmark(repetitions=10)}
        queries = {case.query for case in cases}
        self.assertEqual(40, len(cases))
        self.assertEqual(30, len(queries))
        self.assertFalse(queries & original)
        for bank in (RELATION_QUERIES, CLEAN_QUERIES, CONTROL_QUERIES):
            self.assertEqual(10, len(bank))
            self.assertEqual(10, len(set(bank)))
            self.assertTrue(any("zh" in case.tags and case.query in bank for case in cases))
            self.assertTrue(any("en" in case.tags and case.query in bank for case in cases))

    def test_paraphrases_change_only_language_not_grounding_contract(self):
        original = temporal_grounding_benchmark(repetitions=10)
        paraphrased = paraphrased_temporal_grounding_benchmark()
        for before, after in zip(original, paraphrased, strict=True):
            self.assertEqual(before.case_id, after.case_id)
            self.assertEqual(before.entities, after.entities)
            self.assertEqual(before.target_id, after.target_id)
            self.assertEqual(before.current_truth, after.current_truth)
            self.assertEqual(before.gold_frame.constraints, after.gold_frame.constraints)
            self.assertNotEqual(before.query, after.query)

    def test_gold_temporal_grounding_remains_solved(self):
        report = evaluate_language_temporal_grounding(
            paraphrased_temporal_grounding_benchmark(),
            temporal_grounding_registry(),
            LanguageTemporalStrategy.GOLD,
        )
        self.assertEqual(1.0, report.top1_accuracy)
        self.assertEqual(0, report.failures)


if __name__ == "__main__":
    unittest.main()
