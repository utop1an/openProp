import json
import tempfile
import unittest
from pathlib import Path

from openprop.teach_audit import audit_teach_sessions, read_teach_audit_manifest


class TeachAuditTests(unittest.TestCase):
    def _fixture(self, root: Path, episode_id: str, floorplan: str) -> dict[str, object]:
        session = root / episode_id
        states = session / "states"
        states.mkdir(parents=True)
        initial = {
            "objects": [
                {
                    "objectId": "Mug|1|2|3",
                    "objectType": "Mug",
                    "visible": True,
                    "isDirty": True,
                },
                {
                    "objectId": "Plate|1|2|3",
                    "objectType": "Plate",
                    "visible": False,
                    "isDirty": False,
                },
            ],
            "custom_object_metadata": {},
        }
        (session / "initial.json").write_text(json.dumps(initial), encoding="utf-8")
        diffs = {
            "statediff.0.json": {"objects": {}},
            "statediff.1.json": {"objects": {"Mug|1|2|3": {"visible": False}}},
            "statediff.3.json": {
                "objects": {"Mug|1|2|3": {"visible": True, "isDirty": False}}
            },
            "statediff.end.json": {
                "objects": {"Mug|1|2|3": {"isDirty": False}}
            },
        }
        for name, payload in diffs.items():
            (states / name).write_text(json.dumps(payload), encoding="utf-8")
        return {
            "episode_id": episode_id,
            "floorplan": floorplan,
            "initial_state": f"{episode_id}/initial.json",
            "state_directory": f"{episode_id}/states",
            "final_timestamp": 4.0,
        }

    def test_report_counts_censoring_without_truth_leak(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [
                self._fixture(root, "game-1", "FloorPlan1"),
                self._fixture(root, "game-2", "FloorPlan2"),
                self._fixture(root, "game-3", "FloorPlan3"),
            ]
            manifest = root / "manifest.jsonl"
            manifest.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            report = audit_teach_sessions(
                read_teach_audit_manifest(manifest), property_names=("isDirty",)
            )
        self.assertEqual(3, report["totals"]["sessions"])
        self.assertEqual(9, report["totals"]["snapshots"])
        self.assertEqual(3, report["censoring"]["interval_censored_event"])
        self.assertEqual(3, report["censoring"]["right_censored"])
        self.assertEqual(0, report["censoring"]["exact_event"])
        self.assertEqual(3, report["property_transitions"]["isDirty"])
        self.assertEqual([], report["warnings"])
        self.assertTrue(report["floorplan_split"]["feasible"])
        self.assertEqual(0, report["gold_grounding"]["cases"])
        self.assertFalse(report["feasibility_gate"]["layer_b_ready"])
        self.assertFalse(report["feasibility_gate"]["main_claim_ready"])
        self.assertIn("grounding_cases", report["feasibility_gate"]["failed_checks"])
        self.assertEqual(
            "evaluation-only query, target, and audit labels; never matcher entities",
            report["protocol"]["final_truth_use"],
        )

    def test_manifest_rejects_duplicate_episode_ids(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            row = self._fixture(root, "game-1", "FloorPlan1")
            manifest = root / "manifest.jsonl"
            manifest.write_text(
                json.dumps(row) + "\n" + json.dumps(row) + "\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "duplicate episode_id"):
                read_teach_audit_manifest(manifest)

    def test_manifest_rejects_nonfinite_final_timestamp(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            row = self._fixture(root, "game-1", "FloorPlan1")
            row["final_timestamp"] = "nan"
            manifest = root / "manifest.jsonl"
            manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(
                ValueError, "final_timestamp must be finite and nonnegative"
            ):
                read_teach_audit_manifest(manifest)



if __name__ == "__main__":
    unittest.main()
