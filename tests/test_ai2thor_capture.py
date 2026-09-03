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
    build_manifest,
    file_artifact,
    main,
    choose_intervention,
    intervention_for_object,
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
    }


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

    def test_ineligible_and_malformed_rows_fail_closed(self):
        self.assertIsNone(
            intervention_for_object(object_row("cup", "Cup"), "open")
        )
        with self.assertRaisesRegex(ValueError, "objects array"):
            choose_intervention({}, "open", visible_only=True)
        with self.assertRaisesRegex(ValueError, "unknown intervention"):
            intervention_for_object(object_row("cup", "Cup"), "cook")

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
