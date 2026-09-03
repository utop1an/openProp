import json
import sys
import unittest
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from scripts.compare_candidate_systems import main
from test_candidate_statistics import CandidateStatisticsTests


class CandidateStatisticsCLITests(unittest.TestCase):
    def test_cli_hashes_inputs_and_writes_paired_report(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = []
            for index, evaluation in enumerate(CandidateStatisticsTests().dataset()):
                path = root / f"candidate-{index}.json"
                path.write_text(
                    json.dumps(
                        {
                            "schema_version": 1,
                            "episode_id": evaluation.record_id,
                            "system": evaluation.system,
                            "evaluation": asdict(evaluation),
                        }
                    ),
                    encoding="utf-8",
                )
                paths.append(path)
            output = root / "comparison.json"
            argv = [
                "compare_candidate_systems.py", "--input", *(str(path) for path in paths),
                "--baseline", "baseline", "--system", "openprop", "--split", "test",
                "--bootstrap-replicates", "100", "--seed", "13", "--output", str(output),
            ]
            with patch.object(sys, "argv", argv):
                main()
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(report["population"]["episodes"], 4)
            self.assertEqual(len(report["input_sha256"]), 8)
            self.assertEqual(
                report["metrics"]["candidate_recall"]["delta_system_minus_baseline"],
                1.0,
            )

    def test_output_cannot_overwrite_an_input(self):
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "candidate.json"
            evaluation = CandidateStatisticsTests().dataset()[0]
            path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "evaluation": asdict(evaluation),
                    }
                ),
                encoding="utf-8",
            )
            argv = [
                "compare_candidate_systems.py", "--input", str(path),
                "--baseline", "baseline", "--system", "openprop", "--split", "test",
                "--output", str(path),
            ]
            with patch.object(sys, "argv", argv):
                with self.assertRaisesRegex(ValueError, "must be distinct"):
                    main()


if __name__ == "__main__":
    unittest.main()
