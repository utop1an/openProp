import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from openprop.visual_evaluation import read_visual_results_jsonl
from openprop.vlm_replay import write_captured_vlm_response
from scripts.evaluate_visual_case import main
from test_visual_replay import VisualReplayTests
from test_visual_replay_evaluation import VisualReplayEvaluationTests


class VisualCaseCLITests(unittest.TestCase):
    def test_file_level_case_produces_metric_ready_jsonl(self):
        fixture = VisualReplayTests()
        truth_fixture = VisualReplayEvaluationTests()
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_path = root / "input.json"
            case_path = root / "case.json"
            truth_path = root / "truth.json"
            response_path = root / "response.json"
            output_path = root / "results.jsonl"
            input_path.write_text(json.dumps(fixture.input_payload()), encoding="utf-8")
            case_path.write_text(json.dumps(fixture.case_payload()), encoding="utf-8")
            truth_path.write_text(json.dumps(truth_fixture.truth()), encoding="utf-8")
            write_captured_vlm_response(
                response_path,
                input_artifact=input_path,
                provider="fixture-provider",
                model="fixture-model",
                system_id="fixture-system",
                request_settings={"temperature": 0},
                response=fixture.response(),
            )
            argv = [
                "evaluate_visual_case.py",
                "--input", str(input_path),
                "--case", str(case_path),
                "--response", str(response_path),
                "--truth", str(truth_path),
                "--assignment", "global",
                "--system", "openprop-global",
                "--output", str(output_path),
                "--association-threshold", "0.5",
                "--association-margin", "0.1",
                "--association-null-weight", "0.01",
                "--query-threshold", "0.5",
                "--query-margin", "0.1",
                "--query-null-weight", "0.01",
            ]
            with patch.object(sys, "argv", argv):
                main()
            dataset = read_visual_results_jsonl(output_path)
            self.assertEqual(len(dataset.properties), 1)
            self.assertEqual(len(dataset.associations), 1)
            self.assertEqual(len(dataset.queries), 1)
            self.assertEqual(dataset.queries[0].system, "openprop-global")
            audit = json.loads(
                output_path.with_suffix(".jsonl.audit.json").read_text(encoding="utf-8")
            )
            self.assertTrue(audit["decision_frozen_before_truth_load"])
            self.assertFalse(audit["test_truth_used_for_policy_selection"])
            self.assertEqual(
                audit["denominators"],
                {"property": 1, "association": 1, "query": 1},
            )
            self.assertEqual(len(audit["artifacts"]["output"]["sha256"]), 64)

    def test_output_cannot_overwrite_any_source_artifact(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            shared = root / "same.json"
            shared.write_text("{}", encoding="utf-8")
            argv = [
                "evaluate_visual_case.py",
                "--input", str(shared),
                "--case", str(shared),
                "--response", str(shared),
                "--truth", str(shared),
                "--assignment", "global",
                "--system", "openprop",
                "--output", str(shared),
            ]
            with patch.object(sys, "argv", argv):
                with self.assertRaisesRegex(ValueError, "paths must differ"):
                    main()


if __name__ == "__main__":
    unittest.main()
