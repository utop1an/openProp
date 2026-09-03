import json
import tempfile
import unittest
from pathlib import Path

from openprop.teach_adapter import (
    TeachReplaySnapshot,
    apply_teach_state_diff,
    read_teach_replay,
    reconstruct_teach_snapshots,
    teach_hidden_current_truth,
    teach_visible_observation_history,
)


class TeachAdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.initial_state = {
            "objects": [
                {
                    "objectId": "Mug|1|2|3",
                    "objectType": "Mug",
                    "visible": True,
                    "isDirty": True,
                    "isOpen": False,
                },
                {
                    "objectId": "Plate|1|2|3",
                    "objectType": "Plate",
                    "visible": False,
                    "isDirty": False,
                },
            ],
            "custom_object_metadata": {},
        }

    def test_invisible_gap_becomes_interval_censored_change(self) -> None:
        snapshots = reconstruct_teach_snapshots(
            self.initial_state,
            (
                (0.0, {"objects": {}}),
                (1.0, {"objects": {}}),
                (2.0, {"objects": {"Mug|1|2|3": {"visible": False}}}),
                (
                    4.0,
                    {
                        "objects": {
                            "Mug|1|2|3": {
                                "visible": True,
                                "isDirty": False,
                            }
                        }
                    },
                ),
            ),
        )
        records = teach_visible_observation_history(
            "game-1",
            snapshots,
            scene="Kitchen",
            property_names=("isDirty",),
        )
        mug_records = [
            record for record in records if record.entity_id.endswith("Mug|1|2|3")
        ]
        self.assertEqual(2, len(mug_records))
        changed = next(record for record in mug_records if record.state_changed)
        example = changed.to_training_example()
        self.assertEqual(0.0, changed.observed_at)
        self.assertEqual(1.0, changed.last_confirmed_at)
        self.assertEqual(4.0, changed.followup_at)
        self.assertEqual(1.0, example.interval_start_seconds)
        self.assertEqual(4.0, example.duration_seconds)
        self.assertFalse(
            any(record.entity_id.endswith("Plate|1|2|3") for record in records)
        )

    def test_created_object_replaces_unsliced_base(self) -> None:
        initial = {
            "objects": [
                {
                    "objectId": "Bread|1|2|3",
                    "objectType": "Bread",
                    "visible": True,
                    "isCooked": False,
                }
            ],
            "custom_object_metadata": {},
        }
        state = apply_teach_state_diff(
            initial,
            {
                "objects": {
                    "Bread|1|2|3|BreadSliced_1": {
                        "objectType": "BreadSliced",
                        "isCooked": True,
                    }
                }
            },
        )
        ids = {obj["objectId"] for obj in state["objects"]}
        self.assertNotIn("Bread|1|2|3", ids)
        self.assertIn("Bread|1|2|3|BreadSliced_1", ids)

    def test_final_truth_is_evaluation_only(self) -> None:
        final = TeachReplaySnapshot(
            5.0,
            (
                {
                    "objectId": "Mug|1|2|3",
                    "objectType": "Mug",
                    "visible": False,
                    "isDirty": False,
                },
            ),
            is_final=True,
        )
        with self.assertRaises(ValueError):
            teach_visible_observation_history(
                "game-1",
                (final,),
                scene="Kitchen",
                property_names=("isDirty",),
            )
        truth = teach_hidden_current_truth(final, property_names=("isDirty",))
        self.assertEqual({"isDirty": False}, truth["Mug|1|2|3"])

    def test_reads_numeric_snapshots_and_separate_end_truth(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "statediff.2.5.json").write_text(
                json.dumps({"objects": {"Mug|1|2|3": {"visible": False}}}),
                encoding="utf-8",
            )
            (root / "statediff.1.0.json").write_text(
                json.dumps({"objects": {}}),
                encoding="utf-8",
            )
            (root / "statediff.end.json").write_text(
                json.dumps({"objects": {"Mug|1|2|3": {"isDirty": False}}}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "final_timestamp"):
                read_teach_replay(self.initial_state, root)
            with self.assertRaisesRegex(ValueError, "finite"):
                read_teach_replay(self.initial_state, root, final_timestamp=float("nan"))
            replay = read_teach_replay(
                self.initial_state, root, final_timestamp=3.0
            )
        self.assertEqual([1.0, 2.5], [item.timestamp for item in replay.observations])
        self.assertIsNotNone(replay.final_truth)
        self.assertTrue(replay.final_truth.is_final)
        self.assertEqual(3.0, replay.final_truth.timestamp)


if __name__ == "__main__":
    unittest.main()
