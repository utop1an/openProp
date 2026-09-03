import hashlib
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.prepare_visual_datasets import build_status


class VisualDatasetPreparationTests(unittest.TestCase):
    def test_ai2thor_protocol_is_ready_without_claiming_performance(self):
        repository_root = Path(__file__).resolve().parents[1]
        with TemporaryDirectory() as temporary:
            status = build_status(
                repository_root / "data/visual/registry.json",
                Path(temporary),
                repository_root,
            )
        rows = {row["id"]: row for row in status["datasets"]}
        self.assertTrue(rows["ai2thor_ithor"]["ready"])
        self.assertFalse(rows["ego4d_hands_objects"]["ready"])
        self.assertTrue(status["required_now_ready"])
        self.assertFalse(status["real_world_claim_ready"])
        self.assertFalse(status["performance_evidence"])

    def test_completion_marker_verifies_hashes_and_license(self):
        repository_root = Path(__file__).resolve().parents[1]
        with TemporaryDirectory() as temporary:
            data_root = Path(temporary)
            target = data_root / "custom_real_video"
            target.mkdir()
            media = target / "frame.png"
            media.write_bytes(b"frame")
            marker = {
                "schema_version": 1,
                "dataset_id": "custom_real_video",
                "source_release": "pilot-v1",
                "license_accepted_by_user": True,
                "files": [{
                    "path": "frame.png",
                    "bytes": 5,
                    "sha256": hashlib.sha256(b"frame").hexdigest(),
                }],
            }
            marker_path = target / ".openprop-dataset-complete.json"
            marker_path.write_text(json.dumps(marker), encoding="utf-8")
            status = build_status(
                repository_root / "data/visual/registry.json",
                data_root,
                repository_root,
            )
            rows = {row["id"]: row for row in status["datasets"]}
            self.assertTrue(rows["custom_real_video"]["ready"])

            media.write_bytes(b"drift!")
            with self.assertRaisesRegex(ValueError, "byte count drifted"):
                build_status(
                    repository_root / "data/visual/registry.json",
                    data_root,
                    repository_root,
                )


if __name__ == "__main__":
    unittest.main()
