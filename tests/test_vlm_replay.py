import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from openprop.vlm_replay import (
    read_captured_vlm_response,
    write_captured_vlm_response,
)


class VLMReplayTests(unittest.TestCase):
    def test_response_is_bound_to_exact_truth_free_input(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_path = root / "episode.json"
            input_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "episode_id": "FloorPlan1.open",
                        "capture_manifest_sha256": "a" * 64,
                        "frames": [
                            {
                                "frame_id": "before",
                                "image_url": "before.png",
                                "candidate_entity_ids": ["opaque-1"],
                                "candidate_regions": {"opaque-1": [0.1, 0.1, 0.4, 0.4]},
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            response_path = root / "response.json"
            write_captured_vlm_response(
                response_path,
                input_artifact=input_path,
                provider="provider-a",
                model="model-a",
                system_id="openprop-global",
                request_settings={"temperature": 0},
                response={"detections": []},
            )
            loaded = read_captured_vlm_response(
                response_path, input_artifact=input_path
            )
            self.assertEqual(loaded["input_episode_id"], "FloorPlan1.open")
            input_path.write_text(input_path.read_text() + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "input hash drifted"):
                read_captured_vlm_response(response_path, input_artifact=input_path)

    def test_recursive_truth_fields_are_rejected_before_capture(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_path = root / "unsafe.json"
            input_path.write_text(
                json.dumps(
                    {
                        "episode_id": "unsafe",
                        "frames": [{"frame_id": "x", "nested": {"current_truth": []}}],
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "truth fields"):
                write_captured_vlm_response(
                    root / "response.json",
                    input_artifact=input_path,
                    provider="p",
                    model="m",
                    system_id="s",
                    request_settings={},
                    response={},
                )

    def test_malformed_response_metadata_fails_closed(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_path = root / "input.json"
            input_path.write_text(
                json.dumps({"episode_id": "e", "frames": [{}]}), encoding="utf-8"
            )
            response_path = root / "response.json"
            response_path.write_text(json.dumps({"schema_version": 1}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "provider"):
                read_captured_vlm_response(response_path, input_artifact=input_path)


if __name__ == "__main__":
    unittest.main()
