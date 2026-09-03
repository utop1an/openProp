import json
import tempfile
import unittest
from pathlib import Path

from openprop.teach_feasibility import (
    TeachFeasibilityCriteria,
    assign_teach_floorplan_splits,
    evaluate_teach_feasibility,
    read_teach_dialogue_alignment_audit,
)


class TeachFeasibilityTests(unittest.TestCase):
    def test_floorplan_assignment_is_deterministic_disjoint_and_nonempty(self):
        counts = {"Floor1": 5, "Floor2": 4, "Floor3": 3, "Floor4": 2}
        first = assign_teach_floorplan_splits(counts, seed=7)
        second = assign_teach_floorplan_splits(counts, seed=7)
        self.assertEqual(first, second)
        self.assertTrue(first["feasible"])
        sets = [set(item["floorplans"]) for item in first["splits"].values()]
        self.assertTrue(sets[0].isdisjoint(sets[1]))
        self.assertTrue(sets[0].isdisjoint(sets[2]))
        self.assertTrue(sets[1].isdisjoint(sets[2]))
        self.assertEqual(14, sum(item["sessions"] for item in first["splits"].values()))

    def test_fewer_than_three_floorplans_fails_closed(self):
        result = assign_teach_floorplan_splits({"Floor1": 2, "Floor2": 2})
        self.assertFalse(result["feasible"])
        self.assertEqual({}, result["splits"])

    def test_gate_reports_layer_b_but_refuses_main_claim_without_dialogue_audit(self):
        report = {
            "totals": {
                "sessions": 3,
                "floorplans": 3,
                "snapshots": 9,
                "unique_visible_entities": 6,
                "history_records": 6,
            },
            "censoring": {"interval_censored_event": 3},
            "property_transitions": {"isDirty": 3},
            "floorplan_split": {"feasible": True},
            "gold_grounding": {
                "cases": 6,
                "temporal_discriminative_cases": 3,
                "candidate_size_min": 2,
                "target_ties_in_final_truth": 0,
            },
        }
        criteria = TeachFeasibilityCriteria(
            profile="unit",
            min_sessions=3,
            min_floorplans=3,
            min_snapshots=9,
            min_visible_entities=6,
            min_history_records=6,
            min_interval_events=3,
            min_transition_properties=1,
            min_grounding_cases=6,
            min_temporal_discriminative_cases=3,
            min_candidate_size=2,
            min_dialogue_alignments=1,
            min_manual_alignment_labels=1,
            min_manual_alignment_precision=0.9,
        )
        result = evaluate_teach_feasibility(report, criteria=criteria)
        self.assertTrue(result["layer_a_ready"])
        self.assertTrue(result["layer_b_ready"])
        self.assertFalse(result["layer_c_ready"])
        self.assertFalse(result["main_claim_ready"])
        self.assertIn("manual_alignment_precision", result["failed_checks"])
        passed = evaluate_teach_feasibility(
            report,
            criteria=criteria,
            dialogue_alignment={
                "aligned_cases": 3,
                "manually_labeled_cases": 3,
                "manual_precision": 1.0,
                "validated_against_automatic": True,
            },
        )
        self.assertTrue(passed["main_claim_ready"])

    def test_invalid_criteria_and_split_fractions_are_rejected(self):
        with self.assertRaises(ValueError):
            TeachFeasibilityCriteria(min_manual_alignment_precision=1.1)
        with self.assertRaises(ValueError):
            assign_teach_floorplan_splits(
                {"A": 1, "B": 1, "C": 1},
                train_fraction=0.9,
                validation_fraction=0.2,
            )

    def test_dialogue_audit_derives_precision_from_unique_frozen_labels(self):
        payload = {
            "alignment_policy_id": "next-successful-object-v1",
            "frozen_manifest_sha256": "a" * 64,
            "aligned_cases": 3,
            "labels": [
                {"case_id": "a", "is_correct": True},
                {"case_id": "b", "is_correct": False},
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "audit.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            result = read_teach_dialogue_alignment_audit(path)
        self.assertEqual(2, result["manually_labeled_cases"])
        self.assertEqual(1, result["correct_alignments"])
        self.assertEqual(0.5, result["manual_precision"])
        self.assertFalse(result["validated_against_automatic"])

    def test_dialogue_audit_rejects_unfrozen_or_duplicate_labels(self):
        payload = {
            "alignment_policy_id": "v1",
            "frozen_manifest_sha256": "bad",
            "aligned_cases": 2,
            "labels": [],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "audit.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "64-character"):
                read_teach_dialogue_alignment_audit(path)
            payload["frozen_manifest_sha256"] = "b" * 64
            payload["labels"] = [
                {"case_id": "same", "is_correct": True},
                {"case_id": "same", "is_correct": True},
            ]
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unique"):
                read_teach_dialogue_alignment_audit(path)

    def test_dialogue_audit_binds_labels_to_current_automatic_cases(self):
        payload = {
            "alignment_policy_id": "next-successful-object-v1",
            "frozen_manifest_sha256": "c" * 64,
            "aligned_cases": 2,
            "labels": [{"case_id": "case-1", "is_correct": True}],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "audit.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            result = read_teach_dialogue_alignment_audit(
                path,
                expected_manifest_sha256="c" * 64,
                expected_policy_id="next-successful-object-v1",
                expected_aligned_case_ids=["case-1", "case-2"],
                expected_aligned_cases=2,
            )
            self.assertTrue(result["validated_against_automatic"])
            for key, value, message in (
                ("frozen_manifest_sha256", "d" * 64, "frozen manifest"),
                ("alignment_policy_id", "other", "alignment policy"),
                ("aligned_cases", 3, "automatic alignment output"),
            ):
                forged = dict(payload)
                forged[key] = value
                path.write_text(json.dumps(forged), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, message):
                    read_teach_dialogue_alignment_audit(
                        path,
                        expected_manifest_sha256="c" * 64,
                        expected_policy_id="next-successful-object-v1",
                        expected_aligned_case_ids=["case-1", "case-2"],
                        expected_aligned_cases=2,
                    )
            forged = dict(payload)
            forged["labels"] = [{"case_id": "unknown", "is_correct": True}]
            path.write_text(json.dumps(forged), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "not an automatic"):
                read_teach_dialogue_alignment_audit(
                    path,
                    expected_manifest_sha256="c" * 64,
                    expected_policy_id="next-successful-object-v1",
                    expected_aligned_case_ids=["case-1", "case-2"],
                    expected_aligned_cases=2,
                )

if __name__ == "__main__":
    unittest.main()
