import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from openprop.visual_evaluation import (
    read_visual_results_jsonl,
    write_visual_results_jsonl,
)
from scripts.calibrate_visual_pipeline import main
from test_visual_matrix import VisualMatrixTests


class VisualCalibrationPipelineTests(unittest.TestCase):
    def test_cli_freezes_all_three_stages_with_one_audit(self):
        dataset = VisualMatrixTests().datasets()[1]
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "raw.jsonl"
            output = root / "calibrated.jsonl"
            policy = root / "calibration-policy.json"
            write_visual_results_jsonl(source, dataset)
            argv = [
                "calibrate_visual_pipeline.py",
                "--input", str(source),
                "--system", "openprop-global",
                "--output", str(output),
                "--policy-output", str(policy),
                "--minimum-source-rows", "1",
            ]
            with patch.object(sys, "argv", argv):
                main()
            restored = read_visual_results_jsonl(output)
            self.assertEqual(len(restored.queries), 1)
            self.assertEqual(len(restored.associations), 1)
            self.assertIsNotNone(
                restored.associations[0].calibrated_update_confidence
            )
            audit = json.loads(policy.read_text(encoding="utf-8"))
            self.assertEqual(
                audit["execution_order"],
                [
                    "association_null_and_admission",
                    "combined_update_confidence",
                    "final_query_null_and_admission",
                ],
            )
            self.assertTrue(audit["calibration_only_selection"])
            self.assertFalse(audit["test_truth_used_for_selection"])
            self.assertEqual(len(audit["output_sha256"]), 64)
            self.assertEqual(
                audit["policies"]["association"]["supported_candidate_counts"],
                [2],
            )

    def test_cli_rejects_source_output_collision(self):
        dataset = VisualMatrixTests().datasets()[1]
        with TemporaryDirectory() as temporary:
            source = Path(temporary) / "raw.jsonl"
            write_visual_results_jsonl(source, dataset)
            argv = [
                "calibrate_visual_pipeline.py",
                "--input", str(source),
                "--system", "openprop-global",
                "--output", str(source),
                "--policy-output", str(Path(temporary) / "policy.json"),
            ]
            with patch.object(sys, "argv", argv):
                with self.assertRaisesRegex(ValueError, "must differ"):
                    main()


if __name__ == "__main__":
    unittest.main()
