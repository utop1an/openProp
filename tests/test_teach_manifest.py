from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from openprop.teach_audit import audit_teach_sessions, read_teach_audit_manifest
from openprop.teach_dialogue_alignment import audit_teach_dialogue_alignments
from openprop.teach_manifest import prepare_official_teach_manifest


def _game_payload(
    episode_id: str,
    floorplan: str,
    *,
    interaction_times: tuple[tuple[float, float], ...] = ((0.0, 0.5), (2.0, 1.0)),
    extra_episode: bool = False,
) -> dict[str, object]:
    initial = {
        "time_start": 0.0,
        "objects": [
            {
                "objectId": "Mug|1|2|3",
                "objectType": "Mug",
                "visible": True,
                "isDirty": True,
            },
            {
                "objectId": "Mug|4|5|6",
                "objectType": "Mug",
                "visible": True,
                "isDirty": False,
            },
        ],
        "custom_object_metadata": {},
    }
    interactions = [
        {
            "agent_id": 0,
            "action_id": 1,
            "time_start": start,
            "duration": duration,
            "success": 1,
            "utterance": "use the mug" if index == 0 else "done",
        }
        for index, (start, duration) in enumerate(interaction_times)
    ]
    episode = {
        "episode_id": episode_id,
        "world": floorplan,
        "world_type": "Kitchen",
        "commander_embodied": "True",
        "initial_state": initial,
        "interactions": interactions,
    }
    episodes = [episode]
    if extra_episode:
        episodes.append({**episode, "episode_id": episode_id + "-extra"})
    return {
        "version": "2.0",
        "task_type": "game",
        "definitions": {
            "agents": [
                {"agent_id": 0, "agent_name": "Commander"},
                {"agent_id": 1, "agent_name": "Driver"},
            ],
            "actions": [
                {"action_id": 1, "action_name": "Text", "action_type": "Keyboard"}
            ],
        },
        "tasks": [{"episodes": episodes}],
    }


def _official_fixture(
    root: Path,
    split: str,
    game_id: str,
    episode_id: str,
    floorplan: str,
    *,
    state_times: tuple[float, ...] = (0.0, 2.0),
    extra_episode: bool = False,
) -> None:
    games = root / "games" / split
    states = root / "images" / split / game_id
    games.mkdir(parents=True, exist_ok=True)
    states.mkdir(parents=True, exist_ok=True)
    (games / f"{game_id}.game.json").write_text(
        json.dumps(
            _game_payload(
                episode_id,
                floorplan,
                extra_episode=extra_episode,
            )
        ),
        encoding="utf-8",
    )
    for timestamp in state_times:
        changes: dict[str, object] = {"objects": {}}
        if timestamp == 2.0:
            changes = {
                "objects": {
                    "Mug|1|2|3": {"visible": True, "isDirty": False}
                }
            }
        (states / f"statediff.{timestamp}.json").write_text(
            json.dumps(changes), encoding="utf-8"
        )
    (states / "statediff.end.json").write_text(
        json.dumps(
            {"objects": {"Mug|1|2|3": {"visible": True, "isDirty": False}}}
        ),
        encoding="utf-8",
    )


class TeachManifestPreparationTests(unittest.TestCase):
    def _prepare(self, root: Path, splits=None):
        manifest = root / "prepared" / "manifest.jsonl"
        report = prepare_official_teach_manifest(
            games_root=root / "games",
            states_root=root / "images",
            output_manifest=manifest,
            initial_state_directory=root / "prepared" / "initial",
            splits=splits,
        )
        return manifest, report

    def test_prepares_deterministic_portable_manifest_and_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _official_fixture(root, "train", "game-b", "episode-b", "FloorPlan2")
            _official_fixture(root, "train", "game-a", "episode-a", "FloorPlan1")
            manifest, report = self._prepare(root)
            first = manifest.read_bytes()
            sessions = read_teach_audit_manifest(manifest)
            self.assertTrue(all(row.initial_state.is_file() for row in sessions))
            _, second_report = self._prepare(root)
            second = manifest.read_bytes()
            rows = [json.loads(line) for line in first.decode().splitlines()]
        self.assertEqual(first, second)
        self.assertEqual(2, report["sessions"])
        self.assertEqual(2, report["floorplans"])
        self.assertEqual(report["manifest_sha256"], second_report["manifest_sha256"])
        self.assertEqual(["game-a", "game-b"], [row["official_game_id"] for row in rows])
        self.assertEqual(3.0, rows[0]["final_timestamp"])
        self.assertEqual(2, rows[0]["numeric_state_files"])
        self.assertTrue(all(len(row["game_file_sha256"]) == 64 for row in rows))
        self.assertEqual(("episode-a", "episode-b"), tuple(row.episode_id for row in sessions))

    def test_missing_or_mismatched_replay_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _official_fixture(
                root,
                "train",
                "game-a",
                "episode-a",
                "FloorPlan1",
                state_times=(0.0,),
            )
            with self.assertRaisesRegex(ValueError, "timestamps do not match"):
                self._prepare(root)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            games = root / "games" / "train"
            images = root / "images" / "train"
            games.mkdir(parents=True)
            images.mkdir(parents=True)
            (games / "game-a.game.json").write_text(
                json.dumps(_game_payload("episode-a", "FloorPlan1")),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "missing TEACh replay"):
                self._prepare(root)

    def test_multiple_episodes_and_duplicate_episode_ids_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _official_fixture(
                root,
                "train",
                "game-a",
                "episode-a",
                "FloorPlan1",
                extra_episode=True,
            )
            with self.assertRaisesRegex(ValueError, "exactly one episode"):
                self._prepare(root)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _official_fixture(root, "train", "game-a", "same", "FloorPlan1")
            _official_fixture(root, "train", "game-b", "same", "FloorPlan2")
            with self.assertRaisesRegex(ValueError, "duplicate TEACh episode_id"):
                self._prepare(root)

    def test_prepared_manifest_runs_through_audit_and_dialogue_boundaries(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index in range(3):
                _official_fixture(
                    root,
                    "train",
                    f"game-{index}",
                    f"episode-{index}",
                    f"FloorPlan{index + 1}",
                )
            manifest, report = self._prepare(root, splits=("train",))
            sessions = read_teach_audit_manifest(manifest)
            audit = audit_teach_sessions(sessions, property_names=("isDirty",))
            dialogue = audit_teach_dialogue_alignments(
                sessions,
                frozen_manifest_sha256=report["manifest_sha256"],
            )
        self.assertEqual(3, audit["totals"]["sessions"])
        self.assertEqual(6, audit["totals"]["snapshots"])
        self.assertEqual(3, audit["censoring"]["interval_censored_event"])
        self.assertEqual(3, dialogue["sessions_with_game_file"])
        self.assertEqual([], dialogue["missing_game_episode_ids"])


if __name__ == "__main__":
    unittest.main()
