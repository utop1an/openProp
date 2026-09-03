import json
import sys
import unittest
from dataclasses import asdict, replace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from openprop.candidate_tracking import CandidateTracker
from scripts.aggregate_visual_candidates import main
from test_candidate_evaluation import CandidateEvaluationTests


class CandidateAggregateCLITests(unittest.TestCase):
    def evaluations(self):
        fixture = CandidateEvaluationTests()
        run = CandidateTracker().track(
            (fixture.frame(0), fixture.frame(1)),
            (
                fixture.proposal(0, (0.1, 0.1, 0.3, 0.3)),
                fixture.proposal(1, (0.1, 0.1, 0.3, 0.3)),
            ),
        )
        first = fixture.evaluate(run, fixture.truth(), cluster="room-a")
        return first, replace(first, cluster_id="room-b", system="external-tracker")

    def test_cli_aggregates_systems_and_hashes_inputs(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            inputs = []
            for index, evaluation in enumerate(self.evaluations()):
                path = root / f"result-{index}.json"
                path.write_text(
                    json.dumps(
                        {
                            "schema_version": 1,
                            "episode_id": f"episode-{index}",
                            "system": evaluation.system,
                            "evaluation": asdict(evaluation),
                        }
                    ),
                    encoding="utf-8",
                )
                inputs.append(path)
            output = root / "aggregate.json"
            argv = [
                "aggregate_visual_candidates.py", "--input",
                *(str(path) for path in inputs),
                "--split", "test", "--output", str(output),
            ]
            with patch.object(sys, "argv", argv):
                main()
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(set(report["systems"]), {"tracker", "external-tracker"})
            self.assertEqual(len(report["input_sha256"]), 2)

    def test_duplicate_episode_system_fails_closed(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            evaluation = self.evaluations()[0]
            inputs = []
            for index in range(2):
                path = root / f"duplicate-{index}.json"
                path.write_text(
                    json.dumps(
                        {
                            "schema_version": 1,
                            "episode_id": "same",
                            "system": evaluation.system,
                            "evaluation": asdict(evaluation),
                        }
                    ), encoding="utf-8",
                )
                inputs.append(path)
            argv = [
                "aggregate_visual_candidates.py", "--input",
                *(str(path) for path in inputs), "--split", "test",
                "--output", str(root / "out.json"),
            ]
            with patch.object(sys, "argv", argv):
                with self.assertRaisesRegex(ValueError, "duplicated"):
                    main()


if __name__ == "__main__":
    unittest.main()
