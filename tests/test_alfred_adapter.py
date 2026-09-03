import json
import tempfile
import unittest
from pathlib import Path

from openprop.alfred_adapter import load_alfred_language_dataset
from openprop.models import RelationValue


def _trajectory(task_id, task_type, *, obj="Apple", parent="Fridge", query="Put it away"):
    return {
        "task_id": task_id,
        "task_type": task_type,
        "scene": {"floor_plan": "FloorPlan1"},
        "pddl_params": {"object_target": obj, "parent_target": parent},
        "turk_annotations": {
            "anns": [{"task_desc": query, "high_descs": ["metadata must be ignored"]}]
        },
    }


class AlfredAdapterTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        (self.root / "valid_unseen").mkdir()

    def tearDown(self):
        self.temporary.cleanup()

    def _write(self, name, row):
        target = self.root / "valid_unseen" / name
        target.mkdir()
        (target / "traj_data.json").write_text(json.dumps(row), encoding="utf-8")

    def test_supported_goal_preserves_typed_state_and_inside_relation(self):
        self._write(
            "heat",
            _trajectory(
                "task-1",
                "pick_heat_then_place_in_recep",
                obj="Tomato",
                parent="Microwave",
                query="Heat a tomato and put it in the microwave.",
            ),
        )
        dataset = load_alfred_language_dataset(self.root, splits=("valid_unseen",))
        case = dataset.cases[0]
        values = {item.property_name: item.desired_value for item in case.gold_frame.constraints}
        self.assertEqual("tomato", values["type"])
        self.assertEqual("hot", values["thermal_state"])
        self.assertEqual(RelationValue("inside", {"object": "microwave"}), values["location"])
        self.assertEqual("Heat a tomato and put it in the microwave.", case.query)
        self.assertNotIn("high_descs", case.query)
        self.assertEqual(
            "none; lite release has no frame-level visibility state",
            dataset.audit["protocol"]["observation_claim"],
        )

    def test_surface_receptacle_maps_to_on_and_simple_weights_sum_to_one(self):
        self._write(
            "simple",
            _trajectory(
                "task-2",
                "pick_and_place_simple",
                obj="Mug",
                parent="DiningTable",
            ),
        )
        case = load_alfred_language_dataset(
            self.root, splits=("valid_unseen",)
        ).cases[0]
        values = {item.property_name: item.desired_value for item in case.gold_frame.constraints}
        self.assertEqual(RelationValue("on", {"object": "dining table"}), values["location"])
        self.assertAlmostEqual(1.0, sum(item.relevance for item in case.gold_frame.constraints))

    def test_multi_entity_task_is_excluded_with_explicit_reason(self):
        self._write(
            "excluded",
            _trajectory("task-3", "look_at_obj_in_light", parent="DeskLamp"),
        )
        self._write(
            "supported",
            _trajectory("task-4", "pick_and_place_simple"),
        )
        dataset = load_alfred_language_dataset(self.root, splits=("valid_unseen",))
        self.assertEqual(1, len(dataset.cases))
        self.assertEqual(1, len(dataset.exclusions))
        self.assertEqual("multi_entity_goal", dataset.exclusions[0].reason)

    def test_malformed_and_duplicate_cases_fail_closed(self):
        malformed = _trajectory("task-5", "pick_and_place_simple")
        malformed["pddl_params"].pop("parent_target")
        self._write("malformed", malformed)
        with self.assertRaisesRegex(ValueError, "requires object_target"):
            load_alfred_language_dataset(self.root, splits=("valid_unseen",))

        (self.root / "valid_unseen" / "malformed" / "traj_data.json").unlink()
        self._write("one", _trajectory("duplicate", "pick_and_place_simple"))
        self._write("two", _trajectory("duplicate", "pick_and_place_simple"))
        with self.assertRaisesRegex(ValueError, "duplicate ALFRED language case"):
            load_alfred_language_dataset(self.root, splits=("valid_unseen",))

    def test_missing_requested_split_fails(self):
        with self.assertRaisesRegex(ValueError, "missing ALFRED split"):
            load_alfred_language_dataset(self.root, splits=("valid_seen",))


if __name__ == "__main__":
    unittest.main()
