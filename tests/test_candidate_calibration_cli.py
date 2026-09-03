import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from scripts.calibrate_visual_candidates import main
from test_candidate_calibration import CandidateCalibrationTests


class CandidateCalibrationCLITests(unittest.TestCase):
    def truth_payload(self, case):
        return {
            "schema_version": 1,
            "evaluation_only": True,
            "episode_id": case.input_payload["episode_id"],
            "cluster_id": case.cluster_id,
            "split": case.split,
            "source": case.source,
            "frames": [
                {
                    "frame_id": frame.frame_id,
                    "objects": [
                        {"entity_id": item.entity_id, "region": list(item.region)}
                        for item in frame.objects
                    ],
                }
                for frame in case.truth
            ],
            "query": {
                "frame_id": case.query_frame_id,
                "target_entity_id": case.query_target_entity_id,
            },
        }

    def test_cli_hashes_calibration_artifacts_and_freezes_policy(self):
        case = CandidateCalibrationTests().case()
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_path = root / "input.json"
            truth_path = root / "truth.json"
            output = root / "policy.json"
            input_path.write_text(json.dumps(case.input_payload), encoding="utf-8")
            truth_path.write_text(json.dumps(self.truth_payload(case)), encoding="utf-8")
            argv = [
                "calibrate_visual_candidates.py",
                "--input", str(input_path),
                "--truth", str(truth_path),
                "--minimum-proposal-confidences", "0.25", "0.5",
                "--minimum-link-scores", "0.35", "0.8",
                "--max-missed-frames", "0", "2",
                "--minimum-candidate-recall", "1.0",
                "--maximum-identity-switch-rate", "0.0",
                "--output", str(output),
            ]
            with patch.object(sys, "argv", argv):
                main()
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertTrue(report["calibration_only_selection"])
            self.assertFalse(report["test_truth_used_for_selection"])
            self.assertEqual(len(report["artifacts"][0]["input_sha256"]), 64)
            self.assertEqual(len(report["artifacts"][0]["truth_sha256"]), 64)
            self.assertEqual(
                report["frozen"]["policy"]["minimum_proposal_confidence"], 0.25
            )
            self.assertEqual(report["frozen"]["policy"]["minimum_link_score"], 0.35)

    def test_input_truth_count_mismatch_fails_closed(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "input.json"
            source.write_text("{}", encoding="utf-8")
            argv = [
                "calibrate_visual_candidates.py",
                "--input", str(source),
                "--truth", str(root / "a.json"), str(root / "b.json"),
                "--output", str(root / "out.json"),
            ]
            with patch.object(sys, "argv", argv):
                with self.assertRaisesRegex(ValueError, "counts must match"):
                    main()


if __name__ == "__main__":
    unittest.main()
