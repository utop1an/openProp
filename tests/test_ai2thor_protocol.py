import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from openprop.ai2thor_protocol import (
    build_ai2thor_scene_split,
    default_ai2thor_scene_catalog,
    write_ai2thor_scene_split,
)


class AI2ThorProtocolTests(unittest.TestCase):
    def test_split_is_balanced_disjoint_and_model_output_blind(self):
        payload = build_ai2thor_scene_split()
        assignments = payload["assignments"]
        self.assertEqual(len(assignments), 120)
        self.assertEqual(len({row["scene"] for row in assignments}), 120)
        self.assertFalse(payload["selection_uses_model_outputs"])
        for category in ("kitchen", "living_room", "bedroom", "bathroom"):
            counts = {
                split: sum(
                    row["category"] == category and row["split"] == split
                    for row in assignments
                )
                for split in ("development", "calibration", "test")
            }
            self.assertEqual(
                counts, {"development": 18, "calibration": 6, "test": 6}
            )

    def test_catalog_presentation_order_does_not_change_split(self):
        catalog = default_ai2thor_scene_catalog()
        reversed_catalog = {
            category: tuple(reversed(scenes))
            for category, scenes in reversed(tuple(catalog.items()))
        }
        self.assertEqual(
            build_ai2thor_scene_split(scene_catalog=catalog),
            build_ai2thor_scene_split(scene_catalog=reversed_catalog),
        )

    def test_seed_changes_assignment_but_not_population(self):
        first = build_ai2thor_scene_split(seed="seed-a")
        second = build_ai2thor_scene_split(seed="seed-b")
        self.assertNotEqual(first["split_sha256"], second["split_sha256"])
        self.assertEqual(
            {row["scene"] for row in first["assignments"]},
            {row["scene"] for row in second["assignments"]},
        )

    def test_write_and_check_are_byte_deterministic(self):
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "scene_split.json"
            expected = write_ai2thor_scene_split(path)
            self.assertEqual(expected, write_ai2thor_scene_split(path, check=True))
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["assignments"][0]["split"] = "test"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "drifted"):
                write_ai2thor_scene_split(path, check=True)

    def test_malformed_catalog_and_seed_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "seed"):
            build_ai2thor_scene_split(seed="")
        catalog = default_ai2thor_scene_catalog()
        catalog["kitchen"] = catalog["kitchen"][:-1]
        with self.assertRaisesRegex(ValueError, "30 unique"):
            build_ai2thor_scene_split(scene_catalog=catalog)


if __name__ == "__main__":
    unittest.main()
