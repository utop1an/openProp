import unittest

from scripts.build_adt_pilot_selection import assign_splits, stable_key
from scripts.download_adt_pilot_visuals import preview_target


class ADTPilotSelectionTests(unittest.TestCase):
    def test_split_is_order_invariant_and_uses_all_roles(self):
        rows = [
            {"sequence_name": f"Apartment_release_clean_seq{i:03d}_M1", "activity": "clean"}
            for i in range(10)
        ]
        forward = assign_splits(rows, "seed")
        backward = assign_splits(list(reversed(rows)), "seed")
        self.assertEqual(forward, backward)
        self.assertEqual(set(forward.values()), {"train", "calibration", "test"})
        self.assertEqual(stable_key("seed", "case"), stable_key("seed", "case"))

    def test_preview_target_is_confined(self):
        target = preview_target(
            __import__("pathlib").Path("/tmp/adt"), "sequence", "preview.mp4"
        )
        self.assertEqual(target.name, "preview.mp4")
        with self.assertRaises(ValueError):
            preview_target(__import__("pathlib").Path("/tmp/adt"), "sequence", "../x")


if __name__ == "__main__":
    unittest.main()
