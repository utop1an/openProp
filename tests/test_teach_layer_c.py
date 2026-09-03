import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from openprop.language_temporal_grounding import (
    LanguageTemporalStrategy,
    RawLanguageResponse,
    evaluate_language_temporal_grounding,
)
from openprop.teach_audit import TeachAuditSession
from openprop.teach_dialogue_alignment import TEACH_DIALOGUE_ALIGNMENT_POLICY_ID
from openprop.teach_grounding import teach_grounding_registry
from openprop.teach_layer_c import prepare_teach_layer_c_cases, validate_teach_layer_c_gate
from openprop.temporal_grounding import NoDecayPersistenceModel


class TeachLayerCTests(unittest.TestCase):
    def _fixture(self, root: Path):
        states = root / "states"
        states.mkdir()
        initial = root / "initial.json"
        initial.write_text(
            json.dumps(
                {
                    "objects": [
                        {"objectId": "Mug|1", "objectType": "Mug", "visible": True, "isDirty": True},
                        {"objectId": "Mug|2", "objectType": "Mug", "visible": True, "isDirty": False},
                        {"objectId": "Plate|1", "objectType": "Plate", "visible": False, "isDirty": False},
                    ],
                    "custom_object_metadata": {},
                }
            ),
            encoding="utf-8",
        )
        diffs = {
            "statediff.0.json": {"objects": {}},
            "statediff.1.json": {"objects": {"Mug|1": {"visible": False}}},
            "statediff.2.json": {
                "objects": {"Mug|2": {"visible": True, "isPickedUp": True}}
            },
            "statediff.3.json": {"objects": {"Plate|1": {"visible": True}}},
            "statediff.end.json": {
                "objects": {"Mug|1": {"isDirty": False}, "Mug|2": {"isDirty": True}}
            },
        }
        for name, payload in diffs.items():
            (states / name).write_text(json.dumps(payload), encoding="utf-8")
        session = TeachAuditSession("ep-1", "FloorPlan1", initial, states, 4.0)
        digest = "a" * 64
        report = {
            "alignment_policy_id": TEACH_DIALOGUE_ALIGNMENT_POLICY_ID,
            "frozen_manifest_sha256": digest,
            "cases": [
                {
                    "case_id": "case-mug",
                    "episode_id": "ep-1",
                    "action_time": 2.0,
                    "target_object_id": "Mug|2",
                    "target_object_type": "Mug",
                    "commander_text": "Pick up the mug.",
                },
                {
                    "case_id": "case-plate",
                    "episode_id": "ep-1",
                    "action_time": 3.0,
                    "target_object_id": "Plate|1",
                    "target_object_type": "Plate",
                    "commander_text": "Move the plate.",
                },
            ],
        }
        report["aligned_cases"] = len(report["cases"])
        report["case_ids"] = [row["case_id"] for row in report["cases"]]
        return session, report, digest

    @staticmethod
    def _feasibility(report):
        return {
            "dialogue_alignment_auto": {
                "alignment_policy_id": report["alignment_policy_id"],
                "frozen_manifest_sha256": report["frozen_manifest_sha256"],
                "aligned_cases": report["aligned_cases"],
                "case_ids": list(report["case_ids"]),
                "cases": json.loads(json.dumps(report["cases"])),
            },
            "feasibility_gate": {
                "main_claim_ready": True,
                "checks": {
                    name: {"passed": True}
                    for name in (
                        "dialogue_audit_bound_to_automatic",
                        "dialogue_alignments",
                        "manual_alignment_labels",
                        "manual_alignment_precision",
                    )
                },
            },
        }

    def test_pre_action_cases_retain_coverage_failures_without_truth_leakage(self):
        with tempfile.TemporaryDirectory() as directory:
            session, report, digest = self._fixture(Path(directory))
            prepared = prepare_teach_layer_c_cases(
                [session], report, expected_manifest_sha256=digest, property_names=("isDirty", "isPickedUp")
            )
        self.assertEqual(2, len(prepared.cases))
        mug, plate = prepared.cases
        self.assertIn("same-type-ambiguity", mug.tags)
        self.assertIn("target-unobserved-before-action", plate.tags)
        self.assertEqual(1, prepared.audit["tag_counts"]["input-coverage-failure"])
        self.assertFalse(prepared.audit["action_result_used_as_matcher_evidence"])
        self.assertFalse(prepared.audit["final_truth_used"])
        target = next(entity for entity in mug.entities if entity.entity_id.endswith("Mug|2"))
        self.assertNotIn("isPickedUp", target.properties)
        self.assertTrue(
            all(
                observation.timestamp < mug.as_of
                for entity in mug.entities
                for observation in entity.properties.values()
            )
        )
        self.assertFalse(
            any(
                "selected_by_recorded_interaction" in entity.properties
                for case in prepared.cases
                for entity in case.entities
            )
        )

    def test_oracle_and_predicted_reports_use_all_case_denominators(self):
        with tempfile.TemporaryDirectory() as directory:
            session, alignment, digest = self._fixture(Path(directory))
            prepared = prepare_teach_layer_c_cases(
                [session], alignment, expected_manifest_sha256=digest, property_names=("isDirty", "isPickedUp")
            )
        registry = teach_grounding_registry(("isDirty", "isPickedUp"))
        no_decay = NoDecayPersistenceModel()
        oracle = evaluate_language_temporal_grounding(
            prepared.cases,
            registry,
            LanguageTemporalStrategy.GOLD,
            persistence_model=no_decay,
        )
        responses = {
            "Pick up the mug.": RawLanguageResponse(
                "Pick up the mug.",
                {"constraints": [{"property_name": "type", "value": {"kind": "scalar", "scalar": "Mug"}, "relevance": 1.0}]},
                0.01,
            )
        }
        predicted = evaluate_language_temporal_grounding(
            prepared.cases,
            registry,
            LanguageTemporalStrategy.LLM_STRICT,
            responses=responses,
            persistence_model=no_decay,
        )
        self.assertEqual(2, oracle.cases)
        self.assertEqual(2, predicted.cases)
        self.assertEqual(1, predicted.completed)
        self.assertEqual(1, predicted.failures)
        self.assertEqual(0.5, predicted.parse_success_rate)
        self.assertLessEqual(predicted.top1_accuracy, predicted.conditional_top1_accuracy)

    def test_candidate_order_does_not_change_oracle_ranking(self):
        with tempfile.TemporaryDirectory() as directory:
            session, alignment, digest = self._fixture(Path(directory))
            prepared = prepare_teach_layer_c_cases(
                [session], alignment, expected_manifest_sha256=digest, property_names=("isDirty",)
            )
        registry = teach_grounding_registry(("isDirty",))
        reversed_cases = tuple(
            replace(case, entities=tuple(reversed(case.entities))) for case in prepared.cases
        )
        first = evaluate_language_temporal_grounding(
            prepared.cases, registry, LanguageTemporalStrategy.GOLD, persistence_model=NoDecayPersistenceModel()
        )
        second = evaluate_language_temporal_grounding(
            reversed_cases, registry, LanguageTemporalStrategy.GOLD, persistence_model=NoDecayPersistenceModel()
        )
        self.assertEqual(
            [(row.predicted_id, row.rank) for row in first.results],
            [(row.predicted_id, row.rank) for row in second.results],
        )

    def test_manifest_policy_episode_and_action_time_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            session, report, digest = self._fixture(Path(directory))
            with self.assertRaisesRegex(ValueError, "manifest"):
                prepare_teach_layer_c_cases([session], report, expected_manifest_sha256="b" * 64)
            wrong_policy = dict(report, alignment_policy_id="changed")
            with self.assertRaisesRegex(ValueError, "policy"):
                prepare_teach_layer_c_cases([session], wrong_policy, expected_manifest_sha256=digest)
            unknown = dict(
                report,
                cases=[dict(report["cases"][0], episode_id="missing"), report["cases"][1]],
            )
            with self.assertRaisesRegex(ValueError, "unknown episode"):
                prepare_teach_layer_c_cases([session], unknown, expected_manifest_sha256=digest)
            missing_time = dict(
                report,
                cases=[dict(report["cases"][0], action_time=2.5), report["cases"][1]],
            )
            with self.assertRaisesRegex(ValueError, "no replay snapshot"):
                prepare_teach_layer_c_cases([session], missing_time, expected_manifest_sha256=digest)

    def test_performance_gate_is_bound_to_manual_and_automatic_audits(self):
        with tempfile.TemporaryDirectory() as directory:
            _, report, digest = self._fixture(Path(directory))
        feasibility = self._feasibility(report)
        validate_teach_layer_c_gate(
            report, feasibility, expected_manifest_sha256=digest
        )
        not_ready = dict(feasibility)
        not_ready["feasibility_gate"] = dict(
            feasibility["feasibility_gate"], main_claim_ready=False
        )
        with self.assertRaisesRegex(ValueError, "has not passed"):
            validate_teach_layer_c_gate(
                report, not_ready, expected_manifest_sha256=digest
            )
        changed = json.loads(json.dumps(feasibility))
        changed["dialogue_alignment_auto"]["case_ids"].reverse()
        with self.assertRaisesRegex(ValueError, "case IDs"):
            validate_teach_layer_c_gate(
                report, changed, expected_manifest_sha256=digest
            )
        changed_content = json.loads(json.dumps(feasibility))
        changed_content["dialogue_alignment_auto"]["cases"][0]["commander_text"] = "changed"
        with self.assertRaisesRegex(ValueError, "case contents"):
            validate_teach_layer_c_gate(
                report, changed_content, expected_manifest_sha256=digest
            )
        failed_manual = json.loads(json.dumps(feasibility))
        failed_manual["feasibility_gate"]["checks"]["manual_alignment_precision"]["passed"] = False
        with self.assertRaisesRegex(ValueError, "manual alignment"):
            validate_teach_layer_c_gate(
                report, failed_manual, expected_manifest_sha256=digest
            )


if __name__ == "__main__":
    unittest.main()

