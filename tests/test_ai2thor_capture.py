import json
import sys
import unittest
from argparse import Namespace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from scripts.capture_ai2thor_pilot import (
    _jsonable,
    action_succeeded,
    audit_intervention_changes,
    build_manifest,
    file_artifact,
    main,
    choose_intervention,
    intervention_for_object,
    move_receptacle_intervention,
    wait_for_scene_settled,
)


def object_row(
    object_id,
    object_type,
    *,
    visible=True,
    openable=False,
    is_open=False,
    toggleable=False,
    is_toggled=False,
    dirtyable=False,
    is_dirty=False,
    fillable=False,
    is_filled=False,
    cookable=False,
    is_cooked=False,
    sliceable=False,
    is_sliced=False,
    breakable=False,
    is_broken=False,
    pickupable=False,
    moveable=False,
    receptacle=False,
    parent_receptacles=None,
    position=None,
):
    return {
        "objectId": object_id,
        "objectType": object_type,
        "visible": visible,
        "openable": openable,
        "isOpen": is_open,
        "toggleable": toggleable,
        "isToggled": is_toggled,
        "dirtyable": dirtyable,
        "isDirty": is_dirty,
        "canFillWithLiquid": fillable,
        "isFilledWithLiquid": is_filled,
        "cookable": cookable,
        "isCooked": is_cooked,
        "sliceable": sliceable,
        "isSliced": is_sliced,
        "breakable": breakable,
        "isBroken": is_broken,
        "pickupable": pickupable,
        "moveable": moveable,
        "receptacle": receptacle,
        "parentReceptacles": parent_receptacles,
        "position": position or {"x": 0.0, "y": 0.0, "z": 0.0},
    }


class FakeEvent:
    def __init__(self, metadata):
        self.metadata = metadata


class FakeController:
    def __init__(self, metadata_sequence, *, action_return=None):
        self._metadata_sequence = list(metadata_sequence)
        self._action_return = action_return
        self.last_event = FakeEvent(self._metadata_sequence[0])

    def step(self, **kwargs):
        if kwargs["action"] == "GetSpawnCoordinatesAboveReceptacle":
            metadata = dict(self.last_event.metadata)
            metadata.update(
                lastActionSuccess=True,
                errorMessage="",
                actionReturn=self._action_return,
            )
        else:
            metadata = self._metadata_sequence.pop(1)
        self.last_event = FakeEvent(metadata)
        return self.last_event


class AI2ThorCaptureTests(unittest.TestCase):
    def test_selection_is_visible_and_deterministic(self):
        metadata = {
            "objects": [
                object_row(
                    "z", "Cabinet", visible=False, openable=True, is_open=False
                ),
                object_row("b", "Drawer", openable=True, is_open=False),
                object_row("a", "Cabinet", openable=True, is_open=False),
            ]
        }
        selected = choose_intervention(metadata, "open", visible_only=True)
        self.assertEqual(selected.object_id, "a")
        self.assertEqual(selected.action, "OpenObject")
        self.assertEqual(selected.expected_after_state, True)

    def test_interventions_reverse_existing_state(self):
        close = intervention_for_object(
            object_row("cab", "Cabinet", openable=True, is_open=True),
            "open",
        )
        toggle_off = intervention_for_object(
            object_row("lamp", "Lamp", toggleable=True, is_toggled=True),
            "toggle",
        )
        clean = intervention_for_object(
            object_row("plate", "Plate", dirtyable=True, is_dirty=True),
            "dirty",
        )
        empty = intervention_for_object(
            object_row("cup", "Cup", fillable=True, is_filled=True),
            "fill",
        )
        self.assertEqual(close.action, "CloseObject")
        self.assertEqual(toggle_off.action, "ToggleObjectOff")
        self.assertEqual(clean.action, "CleanObject")
        self.assertEqual(empty.action, "EmptyLiquidFromObject")

    def test_fill_action_declares_liquid_and_force(self):
        fill = intervention_for_object(
            object_row("cup", "Cup", fillable=True, is_filled=False),
            "fill",
        )
        self.assertEqual(fill.action, "FillObjectWithLiquid")
        self.assertEqual(
            fill.arguments,
            {"objectId": "cup", "forceAction": True, "fillLiquid": "water"},
        )

    def test_irreversible_actions_require_false_initial_state(self):
        cook = intervention_for_object(
            object_row("potato", "Potato", cookable=True), "cook"
        )
        sliced = intervention_for_object(
            object_row("apple", "Apple", sliceable=True), "slice"
        )
        broken = intervention_for_object(
            object_row("vase", "Vase", breakable=True), "break"
        )
        self.assertEqual((cook.action, cook.state_field), ("CookObject", "isCooked"))
        self.assertEqual((sliced.action, sliced.state_field), ("SliceObject", "isSliced"))
        self.assertEqual((broken.action, broken.state_field), ("BreakObject", "isBroken"))
        self.assertIsNone(
            intervention_for_object(
                object_row("potato", "Potato", cookable=True, is_cooked=True),
                "cook",
            )
        )

    def test_move_receptacle_is_deterministic_and_avoids_current_parent(self):
        metadata = {
            "objects": [
                object_row(
                    "apple",
                    "Apple",
                    pickupable=True,
                    parent_receptacles=["plate"],
                ),
                object_row("plate", "Plate", receptacle=True),
                object_row("counter", "CounterTop", receptacle=True),
            ]
        }
        controller = FakeController(
            [metadata],
            action_return=[{"x": 2.0, "y": 1.0, "z": 0.0}, {"x": 1.0, "y": 1.0, "z": 0.0}],
        )
        intervention = move_receptacle_intervention(controller)
        self.assertEqual(intervention.action, "PlaceObjectAtPoint")
        self.assertEqual(intervention.expected_after_state, "counter")
        self.assertEqual(
            intervention.arguments["position"], {"x": 1.0, "y": 1.0, "z": 0.0}
        )

    def test_settling_requires_consecutive_stable_steps(self):
        def metadata(x):
            return {
                "lastActionSuccess": True,
                "errorMessage": "",
                "objects": [object_row("sponge", "Sponge", position={"x": x, "y": 0.0, "z": 0.0})],
            }

        controller = FakeController([metadata(0.0), metadata(0.1), metadata(0.1), metadata(0.1)])
        report = wait_for_scene_settled(
            controller,
            max_steps=3,
            stable_steps=2,
            position_tolerance_metres=0.005,
        )
        self.assertTrue(report["settled"])
        self.assertEqual(report["steps"], 3)
        self.assertEqual(report["step_audits"][0]["moved_object_ids"], ["sponge"])

    def test_change_audit_separates_target_and_incidental_motion(self):
        before = {
            "objects": [
                object_row("cab", "Cabinet", openable=True, is_open=False),
                object_row("sponge", "Sponge", position={"x": 0.0, "y": 1.0, "z": 0.0}),
            ]
        }
        after = {
            "objects": [
                object_row("cab", "Cabinet", openable=True, is_open=True),
                object_row("sponge", "Sponge", position={"x": 0.0, "y": 0.8, "z": 0.0}),
            ]
        }
        intervention = intervention_for_object(before["objects"][0], "open")
        audit = audit_intervention_changes(
            before,
            after,
            intervention,
            position_tolerance_metres=0.005,
        )
        self.assertTrue(audit["intended_change_observed"])
        self.assertEqual(audit["target_changes"][0]["field"], "isOpen")
        self.assertEqual(audit["non_target_changes"][0]["object_id"], "sponge")
        self.assertFalse(audit["non_target_changes_are_causal_labels"])

    def test_ineligible_and_malformed_rows_fail_closed(self):
        self.assertIsNone(
            intervention_for_object(object_row("cup", "Cup"), "open")
        )
        with self.assertRaisesRegex(ValueError, "objects array"):
            choose_intervention({}, "open", visible_only=True)
        with self.assertRaisesRegex(ValueError, "unknown intervention"):
            intervention_for_object(object_row("cup", "Cup"), "teleport")

    def test_action_result_and_json_conversion_are_explicit(self):
        self.assertEqual(
            action_succeeded(
                {"lastActionSuccess": False, "errorMessage": "not reachable"}
            ),
            (False, "not reachable"),
        )
        self.assertEqual(
            _jsonable({"box": (1, 2, 3, 4)}),
            {"box": [1, 2, 3, 4]},
        )
        with self.assertRaisesRegex(ValueError, "lastActionSuccess"):
            action_succeeded({"lastActionSuccess": "yes"})

    def test_initialization_failure_is_preserved_in_manifest(self):
        args = Namespace(
            platform="cloud",
            scene="FloorPlan1",
            width=300,
            height=300,
            families=("open",),
        )
        manifest = build_manifest(
            args,
            records=(),
            initialization_error=TimeoutError("backend timed out"),
        )
        self.assertEqual(manifest["run_status"], "initialization_failed")
        self.assertEqual(manifest["records"], [])
        self.assertEqual(
            manifest["initialization_error"],
            {"type": "TimeoutError", "message": "backend timed out"},
        )
        self.assertTrue(manifest["evaluation_only"])

    def test_file_artifact_is_relative_content_addressed_and_confined(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifact = root / "FloorPlan1" / "open" / "before.png"
            artifact.parent.mkdir(parents=True)
            artifact.write_bytes(b"png-bytes")
            reference = file_artifact(artifact, bundle_root=root)
            self.assertEqual(reference["path"], "FloorPlan1/open/before.png")
            self.assertEqual(reference["bytes"], 9)
            self.assertEqual(len(reference["sha256"]), 64)
            outside = root.parent / "outside-openprop-artifact"
            with self.assertRaisesRegex(ValueError, "bundle root"):
                file_artifact(outside, bundle_root=root)

    def test_cli_writes_failure_manifest_before_reraising(self):
        with TemporaryDirectory() as temporary:
            output = Path(temporary) / "capture"
            argv = [
                "capture_ai2thor_pilot.py",
                "--scene",
                "FloorPlan1",
                "--families",
                "open",
                "--output-dir",
                str(output),
            ]
            with patch.object(sys, "argv", argv), patch(
                "scripts.capture_ai2thor_pilot.make_controller",
                side_effect=TimeoutError("backend timed out"),
            ):
                with self.assertRaisesRegex(TimeoutError, "backend timed out"):
                    main()
            manifest = json.loads(
                (output / "FloorPlan1.capture-manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(manifest["run_status"], "initialization_failed")
            self.assertEqual(manifest["families_requested"], ["open"])
            self.assertEqual(manifest["records"], [])


if __name__ == "__main__":
    unittest.main()
