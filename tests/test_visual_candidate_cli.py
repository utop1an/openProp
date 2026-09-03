import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from scripts.evaluate_visual_candidates import main
from test_candidate_replay import CandidateReplayTests


class VisualCandidateCLITests(unittest.TestCase):
    def truth(self):
        return {
            "schema_version": 1,
            "evaluation_only": True,
            "episode_id": "real-video-001",
            "cluster_id": "room-1/person-1",
            "split": "test",
            "source": "detector-rgb",
            "frames": [
                {"frame_id": "before", "objects": []},
                {
                    "frame_id": "after",
                    "objects": [
                        {
                            "entity_id": "real-mug",
                            "region": [0.1, 0.1, 0.3, 0.3],
                        }
                    ],
                },
            ],
            "query": {"frame_id": "after", "target_entity_id": "real-mug"},
        }

    def test_cli_freezes_tracking_before_loading_truth_and_keeps_denominators(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "proposals.json"
            truth = root / "truth.json"
            output = root / "evaluation.json"
            source.write_text(
                json.dumps(CandidateReplayTests().payload()), encoding="utf-8"
            )
            truth.write_text(json.dumps(self.truth()), encoding="utf-8")
            argv = [
                "evaluate_visual_candidates.py",
                "--input", str(source),
                "--truth", str(truth),
                "--output", str(output),
            ]
            with patch.object(sys, "argv", argv):
                main()
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(report["summary"]["frames"], 2)
            self.assertEqual(report["summary"]["candidate_recall"], 1.0)
            self.assertEqual(report["summary"]["query_target_recall"], 1.0)
            audit = json.loads(
                Path(str(output) + ".audit.json").read_text(encoding="utf-8")
            )
            self.assertTrue(audit["tracking_frozen_before_truth_load"])
            self.assertFalse(audit["test_truth_used_for_policy_selection"])
            self.assertEqual(audit["denominators"]["frames"], 2)
            self.assertEqual(len(audit["truth_sha256"]), 64)

    def test_truth_episode_drift_fails_closed(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "proposals.json"
            truth = root / "truth.json"
            output = root / "evaluation.json"
            source.write_text(
                json.dumps(CandidateReplayTests().payload()), encoding="utf-8"
            )
            payload = self.truth()
            payload["episode_id"] = "wrong"
            truth.write_text(json.dumps(payload), encoding="utf-8")
            argv = [
                "evaluate_visual_candidates.py", "--input", str(source),
                "--truth", str(truth), "--output", str(output),
            ]
            with patch.object(sys, "argv", argv):
                with self.assertRaisesRegex(ValueError, "does not match"):
                    main()


if __name__ == "__main__":
    unittest.main()
