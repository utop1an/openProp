import hashlib
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from openprop.ai2thor_capture import (
    CAPTURE_TRUTH_BOUNDARY,
    prepare_ai2thor_capture_manifest,
    verify_ai2thor_capture_manifest,
)


def write_artifact(root: Path, relative: str, data: bytes) -> dict[str, object]:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return {
        "path": relative,
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def valid_manifest(root: Path) -> Path:
    stage_artifacts = {}
    for stage, is_open in (("before", False), ("after", True)):
        prefix = f"FloorPlan1/open/{stage}"
        image = write_artifact(root, f"{prefix}.png", b"\x89PNG\r\n\x1a\nRGB")
        metadata_payload = {
            "sceneName": "FloorPlan1",
            "screenWidth": 300,
            "screenHeight": 300,
            "objects": [
                {
                    "objectId": "Cabinet|1",
                    "objectType": "Cabinet",
                    "visible": True,
                    "position": {"x": 0.0, "y": 0.5, "z": 1.0},
                    "openable": True,
                    "isOpen": is_open,
                }
            ],
        }
        metadata = write_artifact(
            root,
            f"{prefix}.metadata.json",
            (json.dumps(metadata_payload) + "\n").encode("utf-8"),
        )
        boxes = write_artifact(
            root,
            f"{prefix}.boxes.json",
            b'{"Cabinet|1": [10, 20, 110, 220]}\n',
        )
        stage_artifacts[stage] = {
            "image": image,
            "metadata": metadata,
            "boxes": boxes,
        }
    payload = {
        "schema_version": 2,
        "evaluation_only": True,
        "truth_boundary": CAPTURE_TRUTH_BOUNDARY,
        "ai2thor_version": "5.0.0",
        "host_platform": "test",
        "render_platform": "cloud",
        "scene": "FloorPlan1",
        "width": 300,
        "height": 300,
        "families_requested": ["open"],
        "run_status": "completed",
        "records": [
            {
                "family": "open",
                "scene": "FloorPlan1",
                "status": "captured",
                "object_id": "Cabinet|1",
                "object_type": "Cabinet",
                "action": "OpenObject",
                "arguments": {"objectId": "Cabinet|1"},
                "before_state": False,
                "expected_after_state": True,
                "last_action_success": True,
                "error_message": "",
                "before": stage_artifacts["before"],
                "after": stage_artifacts["after"],
            }
        ],
    }
    path = root / "FloorPlan1.capture-manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


class AI2ThorCaptureBundleTests(unittest.TestCase):
    def test_complete_bundle_verifies_all_artifacts_and_statuses(self):
        with TemporaryDirectory() as temporary:
            report = verify_ai2thor_capture_manifest(valid_manifest(Path(temporary)))
            self.assertEqual(report["artifacts_verified"], 6)
            self.assertEqual(report["status_counts"]["captured"], 1)
            self.assertFalse(report["truth_exposed_to_matcher"])

    def test_preparation_physically_separates_vlm_inputs_and_truth(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            report = prepare_ai2thor_capture_manifest(
                valid_manifest(root), root / "prepared"
            )
            self.assertEqual(report["prepared_episodes"], 1)
            self.assertFalse(report["truth_exposed_to_matcher"])
            episode = report["episodes"][0]
            self.assertEqual(
                episode["candidate_coverage"]["before"],
                {
                    "visible_entities": 1,
                    "anchored_candidates": 1,
                    "coverage": 1.0,
                    "unanchored_visible_entity_ids": [],
                },
            )
            input_path = root / "prepared" / episode["input"]["path"]
            truth_path = root / "prepared" / episode["truth"]["path"]
            inputs = json.loads(input_path.read_text(encoding="utf-8"))
            truth = json.loads(truth_path.read_text(encoding="utf-8"))
            self.assertEqual(
                set(inputs),
                {"schema_version", "episode_id", "capture_manifest_sha256", "frames"},
            )
            self.assertNotIn("objects", json.dumps(inputs))
            self.assertNotIn("transition", inputs)
            self.assertTrue(truth["evaluation_only"])
            changes = truth["transition"]["changes"]["Cabinet|1"]
            self.assertEqual(changes[0]["property_name"], "open_state")
            self.assertEqual((changes[0]["before"], changes[0]["after"]), ("closed", "open"))

    def test_hash_drift_and_path_escape_fail_closed(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = valid_manifest(root)
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            payload["records"][0]["before"]["image"]["sha256"] = "0" * 64
            manifest.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "hash drifted"):
                verify_ai2thor_capture_manifest(manifest)

            manifest = valid_manifest(root)
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            payload["records"][0]["before"]["image"]["path"] = "../outside.png"
            manifest.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "relative and confined"):
                verify_ai2thor_capture_manifest(manifest)

    def test_initialization_failure_is_a_valid_zero_artifact_run(self):
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "FloorPlan1.capture-manifest.json"
            path.write_text(
                json.dumps(
                    {
                        "schema_version": 2,
                        "evaluation_only": True,
                        "truth_boundary": CAPTURE_TRUTH_BOUNDARY,
                        "scene": "FloorPlan1",
                        "families_requested": ["open"],
                        "run_status": "initialization_failed",
                        "initialization_error": {
                            "type": "TimeoutError",
                            "message": "backend timed out",
                        },
                        "records": [],
                    }
                ),
                encoding="utf-8",
            )
            report = verify_ai2thor_capture_manifest(path)
            self.assertEqual(report["run_status"], "initialization_failed")
            self.assertEqual(report["artifacts_verified"], 0)

    def test_completed_run_must_account_for_every_requested_family(self):
        with TemporaryDirectory() as temporary:
            manifest = valid_manifest(Path(temporary))
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            payload["families_requested"].append("toggle")
            manifest.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "every requested family"):
                verify_ai2thor_capture_manifest(manifest)


if __name__ == "__main__":
    unittest.main()
