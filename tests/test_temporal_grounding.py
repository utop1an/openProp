import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from openprop.models import ObservationState
from openprop.temporal_grounding import (
    TemporalStrategy,
    evaluate_temporal_grounding,
    temporal_grounding_benchmark,
    temporal_grounding_registry,
    write_temporal_grounding_jsonl,
)


class TemporalGroundingBenchmarkTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cases = temporal_grounding_benchmark(repetitions=3)
        self.registry = temporal_grounding_registry()

    def test_cases_have_separate_current_truth_and_no_target_marker_in_entities(self) -> None:
        self.assertEqual(12, len(self.cases))
        self.assertEqual(len(self.cases), len({case.case_id for case in self.cases}))
        for case in self.cases:
            entity_ids = {entity.entity_id for entity in case.entities}
            self.assertIn(case.target_id, entity_ids)
            self.assertEqual(entity_ids, set(case.current_truth))
            self.assertTrue(all("target" not in entity.properties for entity in case.entities))

    def test_benchmark_contains_missing_stale_event_and_irrelevant_evidence(self) -> None:
        observations = [
            observation
            for case in self.cases
            for entity in case.entities
            for observation in entity.properties.values()
        ]
        self.assertTrue(any(item.state is ObservationState.UNKNOWN for item in observations))
        self.assertTrue(any(entity.events for case in self.cases for entity in case.entities))
        self.assertTrue(any(
            observation.timestamp is not None and observation.timestamp < case.as_of - 3600
            for case in self.cases
            for entity in case.entities
            for observation in entity.properties.values()
        ))
        self.assertTrue(all("irrelevant-properties" in case.tags for case in self.cases))

    def test_fixed_decay_improves_end_to_end_grounding(self) -> None:
        baseline = evaluate_temporal_grounding(
            self.cases, self.registry, TemporalStrategy.NO_DECAY
        )
        temporal = evaluate_temporal_grounding(
            self.cases, self.registry, TemporalStrategy.FIXED_DECAY
        )
        self.assertEqual(0.25, baseline.top1_accuracy)
        self.assertEqual(1.0, temporal.top1_accuracy)
        self.assertGreater(temporal.top1_accuracy, baseline.top1_accuracy)
        reversed_cases = tuple(
            replace(case, entities=tuple(reversed(case.entities))) for case in self.cases
        )
        reordered = evaluate_temporal_grounding(
            reversed_cases, self.registry, TemporalStrategy.NO_DECAY
        )
        self.assertEqual(baseline.top1_accuracy, reordered.top1_accuracy)
        self.assertEqual(1.0, temporal.accuracy_by_tag["static-control"])

    def test_jsonl_exports_observations_and_hidden_truth_separately(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "benchmark.jsonl"
            write_temporal_grounding_jsonl(path, self.cases)
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(self.cases), len(rows))
        self.assertIn("entities", rows[0])
        self.assertIn("current_truth", rows[0])
        self.assertIn("target_id", rows[0])


if __name__ == "__main__":
    unittest.main()

