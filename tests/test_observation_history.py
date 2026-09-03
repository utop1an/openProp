import importlib.util
import tempfile
import unittest
from pathlib import Path

from openprop.observation_history import (
    ObservationHistoryRecord,
    grouped_split,
    history_to_examples,
    read_history_jsonl,
    write_history_jsonl,
)


class ObservationHistoryTests(unittest.TestCase):
    def test_jsonl_round_trip(self):
        record = ObservationHistoryRecord(
            "record-1",
            "cup-1",
            "location",
            "cup",
            "on",
            "table",
            "kitchen",
            10,
            20,
            False,
            source="camera",
            observation_confidence=0.9,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "history.jsonl"
            write_history_jsonl(path, [record])
            self.assertEqual(read_history_jsonl(path), (record,))

    @unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch not installed")
    def test_grouped_split_has_no_entity_leakage(self):
        from openprop.synthetic_persistence import contextual_location_data

        split = grouped_split(contextual_location_data(samples_per_context=30))
        train = {example.group_id for example in split.train}
        validation = {example.group_id for example in split.validation}
        test = {example.group_id for example in split.test}
        self.assertFalse(train & validation)
        self.assertFalse(train & test)
        self.assertFalse(validation & test)

    def test_history_record_becomes_censored_example(self):
        record = ObservationHistoryRecord(
            "record-1",
            "shirt-1",
            "cleanliness",
            "shirt",
            "clean",
            "none",
            "wardrobe",
            100,
            400,
            False,
        )
        example = history_to_examples([record])[0]
        self.assertEqual(example.duration_seconds, 300)
        self.assertFalse(example.event_observed)
        self.assertEqual(example.group_id, "shirt-1")


    def test_detection_interval_is_preserved_for_training(self):
        record = ObservationHistoryRecord(
            "record-2",
            "cup-2",
            "location",
            "cup",
            "on",
            "table",
            "kitchen",
            100,
            500,
            True,
            last_confirmed_at=300,
        )
        example = record.to_training_example()
        self.assertTrue(example.is_interval_censored)
        self.assertEqual(200, example.interval_start_seconds)
        self.assertEqual(400, example.duration_seconds)


if __name__ == "__main__":
    unittest.main()
