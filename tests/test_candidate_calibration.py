import unittest
from dataclasses import replace

from openprop.candidate_calibration import (
    CandidateCalibrationCase,
    calibrate_candidate_tracking_policy,
)
from openprop.candidate_evaluation import CandidateFrameTruth, CandidateTruthObject


class CandidateCalibrationTests(unittest.TestCase):
    def case(self, *, split="calibration"):
        payload = {
            "schema_version": 1,
            "episode_id": "calibration-1",
            "frames": [
                {
                    "frame_id": "f0",
                    "image_url": "f0.png",
                    "captured_at": 0.0,
                    "source": "detector",
                    "proposals": [
                        {
                            "proposal_id": "target-before",
                            "region": [0.05, 0.1, 0.2, 0.3],
                            "confidence": 0.9,
                            "semantic_type": "mug",
                            "external_track_id": "ext-target",
                        }
                    ],
                },
                {
                    "frame_id": "f1",
                    "image_url": "f1.png",
                    "captured_at": 1.0,
                    "source": "detector",
                    "proposals": [
                        {
                            "proposal_id": "target-after",
                            "region": [0.7, 0.6, 0.9, 0.9],
                            "confidence": 0.4,
                            "semantic_type": "mug",
                            "external_track_id": "ext-target",
                        },
                        {
                            "proposal_id": "false-positive",
                            "region": [0.3, 0.3, 0.4, 0.4],
                            "confidence": 0.3,
                            "semantic_type": "mug",
                            "external_track_id": "ext-fp",
                        },
                    ],
                },
            ],
        }
        truth = (
            CandidateFrameTruth(
                "f0", (CandidateTruthObject("target", (0.05, 0.1, 0.2, 0.3)),)
            ),
            CandidateFrameTruth(
                "f1", (CandidateTruthObject("target", (0.7, 0.6, 0.9, 0.9)),)
            ),
        )
        return CandidateCalibrationCase(
            payload,
            truth,
            "room-calibration-1",
            split,
            "detector",
            "f1",
            "target",
        )

    def test_calibration_selects_recall_safe_and_identity_stable_policy(self):
        policy = calibrate_candidate_tracking_policy(
            (self.case(),),
            minimum_proposal_confidences=(0.25, 0.5),
            minimum_link_scores=(0.35, 0.8),
            max_missed_frames=(0, 2),
            minimum_candidate_recall=1.0,
            maximum_identity_switch_rate=0.0,
        )
        self.assertEqual(policy.policy.minimum_proposal_confidence, 0.25)
        self.assertEqual(policy.policy.minimum_link_score, 0.35)
        self.assertEqual(policy.candidate_recall, 1.0)
        self.assertEqual(policy.identity_switch_rate, 0.0)
        self.assertEqual(policy.searched_policies, 8)
        self.assertGreater(policy.feasible_policies, 0)

    def test_test_cases_and_impossible_gate_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "calibration cases"):
            calibrate_candidate_tracking_policy(
                (replace(self.case(), split="test"),),
                minimum_proposal_confidences=(0.25,),
                minimum_link_scores=(0.35,),
                max_missed_frames=(2,),
                minimum_candidate_recall=1.0,
                maximum_identity_switch_rate=0.0,
            )
        with self.assertRaisesRegex(ValueError, "no candidate tracking policy"):
            calibrate_candidate_tracking_policy(
                (self.case(),),
                minimum_proposal_confidences=(0.95,),
                minimum_link_scores=(0.35,),
                max_missed_frames=(2,),
                minimum_candidate_recall=1.0,
                maximum_identity_switch_rate=0.0,
            )


if __name__ == "__main__":
    unittest.main()
