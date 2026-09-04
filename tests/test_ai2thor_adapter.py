import unittest

from openprop.ai2thor_adapter import (
    ai2thor_property_registry,
    derive_ai2thor_transition,
    extract_ai2thor_frame,
    normalize_ai2thor_regions,
)


def object_row(
    object_id,
    object_type,
    *,
    visible=True,
    x=0.0,
    dirtyable=False,
    is_dirty=False,
):
    return {
        "objectId": object_id,
        "objectType": object_type,
        "visible": visible,
        "position": {"x": x, "y": 0.8, "z": 1.0},
        "parentReceptacles": None,
        "dirtyable": dirtyable,
        "isDirty": is_dirty,
        "pickupable": True,
        "isPickedUp": False,
        "ObjectTemperature": "RoomTemp",
    }


class AI2ThorAdapterTests(unittest.TestCase):
    def test_frame_keeps_truth_outside_vlm_input_and_derives_changes(self):
        before = extract_ai2thor_frame(
            {
                "sceneName": "FloorPlan1",
                "objects": (
                    object_row("Cup|1", "Cup", dirtyable=True),
                    object_row("Bowl|1", "Bowl"),
                ),
            },
            frame_id="before",
            image_url="before.png",
            captured_at=1.0,
        )
        after = extract_ai2thor_frame(
            {
                "sceneName": "FloorPlan1",
                "objects": (
                    object_row(
                        "Cup|1",
                        "Cup",
                        x=0.2,
                        dirtyable=True,
                        is_dirty=True,
                    ),
                    object_row("Bowl|1", "Bowl"),
                ),
            },
            frame_id="after",
            image_url="after.png",
            captured_at=2.0,
        )
        self.assertEqual(("Bowl|1", "Cup|1"), before.frame.candidate_entity_ids)
        self.assertFalse(hasattr(before.frame, "current_truth"))

        transition = derive_ai2thor_transition(
            before,
            after,
            action="DirtyObject+PlaceObjectAtPoint",
        )
        changes = {
            item.property_name
            for item in transition.changes["Cup|1"]
        }
        self.assertEqual(
            {"cleanliness", "position", "motion_state"},
            changes,
        )
        self.assertEqual(("Cup|1",), transition.changed_entity_ids)

    def test_invisible_objects_cannot_be_vlm_candidates(self):
        metadata = {
            "sceneName": "FloorPlan1",
            "objects": [object_row("Cup|1", "Cup", visible=False)],
        }
        with self.assertRaisesRegex(ValueError, "not visible"):
            extract_ai2thor_frame(
                metadata,
                frame_id="frame",
                image_url="frame.png",
                captured_at=1.0,
                candidate_entity_ids=("Cup|1",),
            )

    def test_registry_preserves_typed_and_nonvisual_truth_properties(self):
        registry = ai2thor_property_registry()
        self.assertEqual("categorical", registry.get("cleanliness").value_type.value)
        position = registry.get("position")
        self.assertEqual("vector", position.value_type.value)
        self.assertFalse(position.update_policy.allow_visual_updates)
        self.assertIsNotNone(registry.get("motion_state").temporal_policy)
    def test_instance_boxes_become_normalized_identity_anchors(self):
        regions = normalize_ai2thor_regions(
            {
                "Cup|1": (10, 20, 50, 80),
                "Bowl|1": (60, 10, 90, 40),
            },
            screen_width=100,
            screen_height=100,
            candidate_entity_ids=("Cup|1", "Bowl|1"),
        )
        self.assertEqual((0.1, 0.2, 0.5, 0.8), regions["Cup|1"])
        with self.assertRaisesRegex(ValueError, "missing instance detection"):
            normalize_ai2thor_regions(
                {},
                screen_width=100,
                screen_height=100,
                candidate_entity_ids=("Cup|1",),
            )

    def test_default_candidates_require_visibility_and_a_2d_anchor(self):
        frame = extract_ai2thor_frame(
            {
                "sceneName": "FloorPlan1",
                "screenWidth": 100,
                "screenHeight": 100,
                "objects": [
                    object_row("Cup|1", "Cup"),
                    object_row("CounterTop|1", "CounterTop"),
                    object_row("Wall|1", "Wall", visible=False),
                ],
            },
            frame_id="frame",
            image_url="frame.png",
            captured_at=1.0,
            instance_detections_2d={
                "Cup|1": (10, 20, 50, 80),
                "Wall|1": (0, 0, 100, 100),
            },
        )
        self.assertEqual(("Cup|1",), frame.frame.candidate_entity_ids)
        self.assertEqual({"Cup|1"}, set(frame.frame.candidate_regions))
        self.assertEqual(3, len(frame.current_truth))

    def test_default_candidates_omit_invalid_boxes_but_explicit_candidates_fail(self):
        metadata = {
            "sceneName": "FloorPlan1",
            "screenWidth": 100,
            "screenHeight": 100,
            "objects": [object_row("Cup|1", "Cup")],
        }
        boxes = {"Cup|1": (-1, 20, 50, 80)}
        frame = extract_ai2thor_frame(
            metadata,
            frame_id="frame",
            image_url="frame.png",
            captured_at=1.0,
            instance_detections_2d=boxes,
        )
        self.assertEqual((), frame.frame.candidate_entity_ids)
        self.assertEqual({}, frame.frame.candidate_regions)
        self.assertEqual(1, len(frame.current_truth))
        with self.assertRaisesRegex(ValueError, "outside the image"):
            extract_ai2thor_frame(
                metadata,
                frame_id="frame",
                image_url="frame.png",
                captured_at=1.0,
                candidate_entity_ids=("Cup|1",),
                instance_detections_2d=boxes,
            )


if __name__ == "__main__":
    unittest.main()
