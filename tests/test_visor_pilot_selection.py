import json
import tempfile
import unittest
from pathlib import Path

from scripts.build_visor_pilot_selection import eligible, stable_key, video_statistics


class VisorPilotSelectionTests(unittest.TestCase):
    def test_stable_key_is_repeatable(self):
        self.assertEqual(stable_key("seed", "P01_01"), stable_key("seed", "P01_01"))
        self.assertNotEqual(stable_key("seed", "P01_01"), stable_key("seed", "P01_02"))

    def test_statistics_capture_ambiguity_contact_and_reappearance(self):
        frames = []
        for number in range(6):
            annotations = [
                {"id": "cup-a", "name": "cup", "in_contact_object": None},
                {"id": "cup-b", "name": "cup", "in_contact_object": None},
            ]
            if number == 2:
                annotations = [{"id": "cup-b", "name": "cup"}]
            annotations.append(
                {"id": "left", "name": "left hand", "in_contact_object": "cup-a"}
            )
            frames.append(
                {
                    "image": {
                        "name": f"P01_01_frame_{number:010d}.jpg",
                        "subsequence": "P01_01_seq_00001",
                    },
                    "annotations": annotations,
                }
            )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "P01_01.json"
            path.write_text(json.dumps({"video_annotations": frames}), encoding="utf-8")
            row = video_statistics(path, "train")
        self.assertGreater(row["same_name_ambiguous_frame_count"], 0)
        self.assertGreater(row["contact_link_count"], 0)
        self.assertGreater(row["sparse_reappearance_gap_count"], 0)
        self.assertTrue(eligible(row))


if __name__ == "__main__":
    unittest.main()
