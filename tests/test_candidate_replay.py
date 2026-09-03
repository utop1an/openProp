import json
import sys
import unittest
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from openprop.candidate_replay import build_tracked_vlm_input, track_candidate_input
from openprop.candidate_tracking import CandidateTrackingPolicy
from openprop.candidate_calibration import candidate_tracking_policy_from_frozen_payload
from openprop.vlm_replay import write_captured_vlm_response
from scripts.track_visual_candidates import main


class CandidateReplayTests(unittest.TestCase):
    def payload(self):
        return {
            "schema_version": 1,
            "episode_id": "real-video-001",
            "frames": [
                {
                    "frame_id": "before",
                    "image_url": "before.png",
                    "captured_at": 0.0,
                    "source": "detector-rgb",
                    "proposals": [],
                },
                {
                    "frame_id": "after",
                    "image_url": "after.png",
                    "captured_at": 1.0,
                    "source": "detector-rgb",
                    "proposals": [
                        {
                            "proposal_id": "p1",
                            "region": [0.1, 0.1, 0.3, 0.3],
                            "confidence": 0.9,
                            "semantic_type": "mug",
                            "external_track_id": "tracker-7",
                        }
                    ],
                },
            ],
        }

    def test_tracked_output_is_truth_free_and_vlm_replay_compatible(self):
        payload = self.payload()
        run = track_candidate_input(payload)
        output = build_tracked_vlm_input(payload, run)
        self.assertEqual(output["frames"][0]["candidate_entity_ids"], [])
        self.assertEqual(output["frames"][1]["candidate_entity_ids"], ["track-000001"])
        encoded = json.dumps(output)
        self.assertNotIn("target_entity_id", encoded)
        self.assertNotIn("external_track_id", encoded)
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_path = root / "tracked.json"
            response_path = root / "response.json"
            input_path.write_text(json.dumps(output), encoding="utf-8")
            captured = write_captured_vlm_response(
                response_path,
                input_artifact=input_path,
                provider="test",
                model="test-vlm",
                system_id="candidate-tracked",
                response={"detections": []},
                request_settings={"temperature": 0},
            )
            self.assertEqual(captured["input_episode_id"], "real-video-001")

    def test_recursive_truth_fields_fail_before_tracking(self):
        payload = self.payload()
        payload["frames"][0]["metadata"] = {"target_entity_id": "secret"}
        with self.assertRaisesRegex(ValueError, "evaluation truth fields"):
            track_candidate_input(payload)

    def test_cli_writes_content_addressed_output_and_audit(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "proposals.json"
            output = root / "tracked.json"
            policy_path = root / "policy.json"
            source.write_text(json.dumps(self.payload()), encoding="utf-8")
            policy_payload = {
                "schema_version": 1,
                "calibration_only_selection": True,
                "test_truth_used_for_selection": False,
                "frozen": {"policy": asdict(CandidateTrackingPolicy())},
            }
            policy_path.write_text(json.dumps(policy_payload), encoding="utf-8")
            argv = [
                "track_visual_candidates.py",
                "--input", str(source),
                "--output", str(output),
                "--policy", str(policy_path),
            ]
            with patch.object(sys, "argv", argv):
                main()
            audit = json.loads(
                Path(str(output) + ".audit.json").read_text(encoding="utf-8")
            )
            self.assertFalse(audit["truth_used_for_tracking"])
            self.assertEqual(len(audit["input_sha256"]), 64)
            self.assertEqual(len(audit["output_sha256"]), 64)
            self.assertEqual(audit["candidate_generation"]["tracks"], 1)
            self.assertEqual(len(audit["frozen_policy"]["sha256"]), 64)
            self.assertEqual(len(audit["per_frame"]), 2)

    def test_frozen_policy_loader_rejects_test_selected_policy(self):
        payload = {
            "schema_version": 1,
            "calibration_only_selection": True,
            "test_truth_used_for_selection": True,
            "frozen": {"policy": asdict(CandidateTrackingPolicy())},
        }
        with self.assertRaisesRegex(ValueError, "no test-truth"):
            candidate_tracking_policy_from_frozen_payload(payload)


if __name__ == "__main__":
    unittest.main()
