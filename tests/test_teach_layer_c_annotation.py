import copy
import unittest

from openprop.models import Entity, Observation, PropertyConstraint, QueryFrame
from openprop.teach_layer_c import TeachLayerCPrepared
from openprop.teach_layer_c_annotation import (
    apply_teach_layer_c_annotation_resolution,
    build_teach_layer_c_annotation_template,
    resolve_teach_layer_c_annotations,
)
from openprop.temporal_grounding import TemporalGroundingCase


class TeachLayerCAnnotationTests(unittest.TestCase):
    def setUp(self):
        self.properties = ("isDirty", "isOpen")
        cases = (
            TemporalGroundingCase(
                "dirty",
                "Pick up the dirty mug.",
                (Entity("ep:Mug|1", {"type": Observation("Mug")}),),
                "ep:Mug|1",
                QueryFrame(
                    "Pick up the dirty mug.",
                    (PropertyConstraint("type", "Mug", 1.0),),
                ),
                2.0,
                {"ep:Mug|1": {"selected_by_recorded_interaction": True}},
            ),
            TemporalGroundingCase(
                "plain",
                "Move the plate.",
                (Entity("ep:Plate|1", {"type": Observation("Plate")}),),
                "ep:Plate|1",
                QueryFrame(
                    "Move the plate.",
                    (PropertyConstraint("type", "Plate", 1.0),),
                ),
                3.0,
                {"ep:Plate|1": {"selected_by_recorded_interaction": True}},
            ),
        )
        self.prepared = TeachLayerCPrepared(
            cases,
            {"frozen_manifest_sha256": "a" * 64, "oracle_frame": "type only"},
        )

    def _annotation(self, annotator_id, *, dirty=True):
        payload = build_teach_layer_c_annotation_template(
            self.prepared,
            annotator_id=annotator_id,
            property_names=self.properties,
        )
        dirty_label = payload["labels"][0]
        if dirty:
            start = self.prepared.cases[0].query.index("dirty")
            dirty_label.update(
                {
                    "status": "explicit_attributes",
                    "constraints": [
                        {
                            "property_name": "isDirty",
                            "desired_value": True,
                            "span_start": start,
                            "span_end": start + len("dirty"),
                            "evidence_span": "dirty",
                        }
                    ],
                }
            )
        else:
            dirty_label["status"] = "type_only"
        payload["labels"][1]["status"] = "type_only"
        return payload

    def test_template_is_target_candidate_time_and_model_blind(self):
        template = build_teach_layer_c_annotation_template(
            self.prepared,
            annotator_id="ann-a",
            property_names=self.properties,
        )
        serialized = str(template["cases"])
        for forbidden in (
            "target_id",
            "Mug|1",
            "entities",
            "timestamp",
            "model",
            "selected_by_recorded_interaction",
        ):
            self.assertNotIn(forbidden, serialized)
        self.assertTrue(all(label["status"] is None for label in template["labels"]))

    def test_fixed_type_and_non_discriminative_scene_are_not_annotatable(self):
        for reserved in ("type", "scene"):
            with self.subTest(reserved=reserved), self.assertRaisesRegex(
                ValueError, "cannot include"
            ):
                build_teach_layer_c_annotation_template(
                    self.prepared,
                    annotator_id="ann-a",
                    property_names=(*self.properties, reserved),
                )

    def test_three_identical_annotations_create_normalized_rich_frames(self):
        annotations = [self._annotation(f"ann-{index}") for index in range(3)]
        resolution = resolve_teach_layer_c_annotations(
            self.prepared,
            annotations,
            property_names=self.properties,
        )
        dirty = resolution.frames["dirty"]
        self.assertEqual(("type", "isDirty"), tuple(item.property_name for item in dirty.constraints))
        self.assertEqual((0.5, 0.5), tuple(item.relevance for item in dirty.constraints))
        self.assertEqual(1.0, resolution.audit["pairwise_semantic_agreement"])
        applied = apply_teach_layer_c_annotation_resolution(self.prepared, resolution)
        self.assertIn("independent-rich-text-oracle", applied.cases[0].tags)
        self.assertEqual(
            "independently annotated explicit text attributes",
            applied.audit["oracle_frame"],
        )

    def test_majority_is_deterministic_but_threshold_remains_explicit(self):
        annotations = [
            self._annotation("ann-a"),
            self._annotation("ann-b"),
            self._annotation("ann-c", dirty=False),
        ]
        resolution = resolve_teach_layer_c_annotations(
            self.prepared,
            annotations,
            property_names=self.properties,
            min_pairwise_agreement=0.60,
        )
        self.assertEqual(2 / 3, resolution.audit["pairwise_semantic_agreement"])
        self.assertEqual(1, resolution.audit["majority_resolved_cases"])
        with self.assertRaisesRegex(ValueError, "below"):
            resolve_teach_layer_c_annotations(
                self.prepared,
                annotations,
                property_names=self.properties,
            )

    def test_span_population_blinding_and_annotator_identity_fail_closed(self):
        valid = [self._annotation(f"ann-{index}") for index in range(3)]
        broken_span = copy.deepcopy(valid)
        broken_span[0]["labels"][0]["constraints"][0]["evidence_span"] = "clean"
        with self.assertRaisesRegex(ValueError, "span"):
            resolve_teach_layer_c_annotations(
                self.prepared, broken_span, property_names=self.properties
            )
        leaked = copy.deepcopy(valid)
        leaked[0]["cases"][0]["target_id"] = "ep:Mug|1"
        with self.assertRaisesRegex(ValueError, "blind views"):
            resolve_teach_layer_c_annotations(
                self.prepared, leaked, property_names=self.properties
            )
        duplicate = copy.deepcopy(valid)
        duplicate[2]["annotator_id"] = duplicate[1]["annotator_id"]
        with self.assertRaisesRegex(ValueError, "distinct"):
            resolve_teach_layer_c_annotations(
                self.prepared, duplicate, property_names=self.properties
            )
        incomplete = copy.deepcopy(valid)
        incomplete[0]["labels"].pop()
        with self.assertRaisesRegex(ValueError, "complete"):
            resolve_teach_layer_c_annotations(
                self.prepared, incomplete, property_names=self.properties
            )

    def test_uncertain_majority_cannot_become_gold(self):
        annotations = [self._annotation(f"ann-{index}") for index in range(3)]
        for payload in annotations:
            payload["labels"][0]["status"] = "uncertain"
            payload["labels"][0]["constraints"] = []
        with self.assertRaisesRegex(ValueError, "unresolved"):
            resolve_teach_layer_c_annotations(
                self.prepared, annotations, property_names=self.properties
            )


if __name__ == "__main__":
    unittest.main()

