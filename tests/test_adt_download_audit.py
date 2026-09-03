import json
import tempfile
import unittest
from pathlib import Path

from scripts.audit_adt_pilot_download import audit_sequence, marker_files


class AdtDownloadAuditTests(unittest.TestCase):
    def make_sequence(self, root: Path, name: str, segmentation: bool = True) -> dict:
        sequence_root = root / name
        sequence_root.mkdir()
        for filename in {
            "instances.json",
            "scene_objects.csv",
            "2d_bounding_box.csv",
            "3d_bounding_box.csv",
            "aria_trajectory.csv",
            "metadata.json",
        }:
            (sequence_root / filename).write_text("x", encoding="utf-8")
        preview = sequence_root / "preview.mp4"
        preview.write_bytes(b"preview")
        if segmentation:
            (sequence_root / "segmentations.vrs").write_bytes(b"mask")
        (sequence_root / ".download_status.json").write_text(
            json.dumps({"main_groundtruth": True, "segmentation": segmentation}),
            encoding="utf-8",
        )
        return {
            "video_main_rgb": {
                "filename": "preview.mp4",
                "file_size_bytes": 7,
                "sha1sum": "1aa787fe0cfb373575fc2c0f6f826e7c6dc9fd41",
            }
        }

    def test_audit_sequence_checks_selected_modalities(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            name = "Apartment_release_clean_seq001_M0001"
            manifest_row = self.make_sequence(root, name)
            row = {
                "sequence_name": name,
                "download": {"segmentation": True},
                "split": "train",
            }
            result = audit_sequence(root, row, {name: manifest_row})
            self.assertTrue(result["preview_ready"])
            self.assertTrue(result["segmentation_ready"])

    def test_marker_excludes_itself(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "payload.bin").write_bytes(b"abc")
            (root / ".openprop-dataset-complete.json").write_text("{}")
            rows = marker_files(root)
            self.assertEqual([row["path"] for row in rows], ["payload.bin"])
            self.assertEqual(rows[0]["bytes"], 3)


if __name__ == "__main__":
    unittest.main()
