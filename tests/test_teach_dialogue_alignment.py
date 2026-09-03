import json
import tempfile
import unittest
from pathlib import Path

from openprop.teach_audit import TeachAuditSession, read_teach_manifest
from openprop.teach_dialogue_alignment import (
    TEACH_DIALOGUE_ALIGNMENT_POLICY_ID,
    align_teach_game_dialogue,
    audit_teach_dialogue_alignments,
    build_teach_dialogue_alignment_label_template,
    freeze_teach_dialogue_alignment_sample,
    teach_manifest_sha256,
)


def _game_payload():
    interactions = [
        {"action_id": 100, "time_start": 0, "agent_id": 0, "utterance": "Pick up the mug."},
        {"action_id": 100, "time_start": 1, "agent_id": 1, "utterance": "okay"},
        {"action_id": 200, "time_start": 2, "success": 0, "oid": "Mug|1"},
        {"action_id": 200, "time_start": 3, "success": 1, "oid": "Mug|1"},
        {"action_id": 100, "time_start": 4, "agent_id": 0, "utterance": "put it there"},
        {"action_id": 201, "time_start": 5, "success": 1, "oid": "Plate|1"},
        {"action_id": 100, "time_start": 6, "agent_id": 0, "utterance": "Clean the mug and plate"},
        {"action_id": 208, "time_start": 7, "success": True, "oid": "Mug|1"},
        {"action_id": 100, "time_start": 8, "agent_id": 1, "utterance": "I will open the cabinet"},
        {"action_id": 202, "time_start": 9, "success": 1, "oid": "Cabinet|1"},
        {"action_id": 100, "time_start": 10, "agent_id": 0, "utterance": "open the cabinet"},
        {"action_id": 202, "time_start": 11, "success": 1, "oid": "Cabinet|1"},
        {"action_id": 200, "time_start": 12, "success": 1, "oid": "Mug|2"},
    ]
    return {
        "definitions": {
            "agents": [
                {"agent_id": 0, "agent_name": "Commander"},
                {"agent_id": 1, "agent_name": "Driver"},
            ],
            "actions": [
                {"action_id": 100, "action_name": "Text", "action_type": "Keyboard"},
                {"action_id": 200, "action_name": "Pickup", "action_type": "ObjectInteraction"},
                {"action_id": 201, "action_name": "Place", "action_type": "ObjectInteraction"},
                {"action_id": 202, "action_name": "Open", "action_type": "ObjectInteraction"},
                {"action_id": 208, "action_name": "Clean", "action_type": "ObjectInteraction"},
            ],
        },
        "tasks": [{"episodes": [{"episode_id": "ep-1", "interactions": interactions}]}],
    }


class TeachDialogueAlignmentTests(unittest.TestCase):
    def test_policy_accepts_only_unambiguous_commander_segments(self):
        result = align_teach_game_dialogue(
            _game_payload(),
            episode_id="ep-1",
            known_object_types=("Mug", "Plate", "Cabinet"),
        )
        self.assertEqual(6, result["successful_object_interactions"])
        self.assertEqual(2, result["aligned_cases"])
        self.assertEqual(
            {
                "ambiguous_object_type": 1,
                "no_commander_utterance": 1,
                "no_dialogue_segment": 1,
                "target_type_not_mentioned": 1,
            },
            result["rejection_counts"],
        )
        self.assertEqual([3, 11], [row["interaction_index"] for row in result["cases"]])
        self.assertEqual(2, len(result["cases"][0]["dialogue"]))

    def test_recorded_order_and_episode_identity_are_validated(self):
        payload = _game_payload()
        payload["tasks"][0]["episodes"][0]["interactions"][1]["time_start"] = -1
        with self.assertRaisesRegex(ValueError, "ordered"):
            align_teach_game_dialogue(payload, episode_id="ep-1")
        payload = _game_payload()
        payload["tasks"].append({"episodes": [{"episode_id": "ep-1", "interactions": []}]})
        with self.assertRaisesRegex(ValueError, "exactly one"):
            align_teach_game_dialogue(payload, episode_id="ep-1")

    def test_sample_and_label_template_are_frozen_and_order_invariant(self):
        cases = [{"case_id": f"case-{index}", "payload": index} for index in range(8)]
        first = freeze_teach_dialogue_alignment_sample(cases, sample_size=4, seed=7)
        second = freeze_teach_dialogue_alignment_sample(list(reversed(cases)), sample_size=4, seed=7)
        self.assertEqual(first, second)
        template = build_teach_dialogue_alignment_label_template(
            {
                "alignment_policy_id": TEACH_DIALOGUE_ALIGNMENT_POLICY_ID,
                "frozen_manifest_sha256": "a" * 64,
                "aligned_cases": len(cases),
                "cases": cases,
            },
            sample_size=4,
            seed=7,
        )
        self.assertEqual(4, template["sample_size"])
        self.assertEqual("a" * 64, template["frozen_manifest_sha256"])
        self.assertTrue(all(row["is_correct"] is None for row in template["labels"]))

    def test_manifest_linked_audit_preserves_provenance_and_game_path(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            initial = root / "initial.json"
            game = root / "game.json"
            states = root / "states"
            states.mkdir()
            initial.write_text(
                json.dumps({"objects": [{"objectType": name} for name in ("Mug", "Plate", "Cabinet")]}),
                encoding="utf-8",
            )
            game.write_text(json.dumps(_game_payload()), encoding="utf-8")
            manifest = root / "manifest.jsonl"
            manifest.write_text(
                json.dumps(
                    {
                        "episode_id": "ep-1",
                        "floorplan": "FloorPlan1",
                        "initial_state": initial.name,
                        "state_directory": states.name,
                        "final_timestamp": 12,
                        "game_file": game.name,
                    }
                ) + "\n",
                encoding="utf-8",
            )
            sessions = read_teach_manifest(manifest)
            self.assertEqual(game, sessions[0].game_file)
            digest = teach_manifest_sha256(manifest)
            result = audit_teach_dialogue_alignments(
                sessions, frozen_manifest_sha256=digest
            )
        self.assertEqual(digest, result["frozen_manifest_sha256"])
        self.assertEqual(1, result["sessions_with_game_file"])
        self.assertEqual(0, result["sessions_missing_game_file"])
        self.assertEqual(2, result["aligned_cases"])
        self.assertEqual(len(result["case_ids"]), len(set(result["case_ids"])))

    def test_missing_game_file_is_reported_not_silently_ignored(self):
        session = TeachAuditSession(
            episode_id="ep-1",
            floorplan="FloorPlan1",
            initial_state=Path("initial.json"),
            state_directory=Path("states"),
            final_timestamp=1.0,
        )
        result = audit_teach_dialogue_alignments(
            [session], frozen_manifest_sha256="b" * 64
        )
        self.assertEqual(1, result["sessions_missing_game_file"])
        self.assertEqual(["ep-1"], result["missing_game_episode_ids"])


if __name__ == "__main__":
    unittest.main()
