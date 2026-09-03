import json
import tempfile
import unittest
from pathlib import Path

from scripts.audit_alfred_longitudinal_feasibility import (
    build_report,
    entity_lineage,
)


class AlfredLongitudinalFeasibilityTests(unittest.TestCase):
    def test_sliced_child_id_preserves_entity_lineage(self) -> None:
        base = "Apple|-00.12|+00.84|+01.22"
        self.assertEqual(
            entity_lineage(base), entity_lineage(base + "|AppleSliced_1")
        )

    def test_report_counts_distinct_entities_not_lifecycle_ids(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trial = root / "valid_unseen" / "task" / "trial"
            trial.mkdir(parents=True)
            first = "Apple|+00.00|+00.00|+00.00"
            second = "Apple|+01.00|+00.00|+00.00"
            payload = {
                "task_id": "trial",
                "task_type": "pick_two_obj_and_place",
                "scene": {
                    "object_poses": [
                        {"objectName": "Apple_a"},
                        {"objectName": "Apple_b"},
                    ]
                },
                "turk_annotations": {"anns": [{"high_descs": ["put the apple down"]}]},
                "plan": {
                    "low_actions": [
                        {"high_idx": 0, "api_action": {"action": "SliceObject", "objectId": first}},
                        {"high_idx": 0, "api_action": {"action": "PickupObject", "objectId": first + "|AppleSliced_1"}},
                        {"high_idx": 0, "api_action": {"action": "PickupObject", "objectId": second}},
                        {"high_idx": 0, "api_action": {"action": "PutObject", "objectId": second}},
                    ]
                },
            }
            (trial / "traj_data.json").write_text(json.dumps(payload), encoding="utf-8")
            report = build_report(root, ["valid_unseen"])
        self.assertEqual(1, report["eligible_multi_history_cases"])
        self.assertEqual(2, report["cases"][0]["candidate_count"])


if __name__ == "__main__":
    unittest.main()
