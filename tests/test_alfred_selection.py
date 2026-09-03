import unittest

from openprop.alfred_ontology import AlfredTrainingOntology
from openprop.alfred_selection import (
    SelectionFusionPolicy,
    extract_alfred_selection_evidence,
    fuse_alfred_selection,
)
from openprop.models import PropertyConstraint, QueryFrame, RelationValue


class AlfredSelectionTests(unittest.TestCase):
    def setUp(self):
        self.ontology = AlfredTrainingOntology(
            frozenset(
                {
                    "apple",
                    "baseball bat",
                    "bowl",
                    "pepper shaker",
                    "salt shaker",
                }
            ),
            frozenset({"bowl", "drawer", "dining table", "fridge"}),
        )

    def test_extracts_unique_type_and_preposition_bound_destination(self):
        result = extract_alfred_selection_evidence(
            "Place the bat in the drawer.", self.ontology
        )
        values = {item.property_name: item.desired_value for item in result.evidence}
        self.assertEqual("baseball bat", values["type"])
        self.assertEqual(
            RelationValue("inside", {"object": "drawer"}), values["location"]
        )
        location = next(item for item in result.evidence if item.property_name == "location")
        self.assertEqual("drawer", location.source_text)
        self.assertEqual("preposition_bound_receptacle_mention", location.rule)

    def test_ambiguous_type_and_unbound_receptacle_do_not_create_evidence(self):
        result = extract_alfred_selection_evidence(
            "The shaker and drawer are nearby.", self.ontology
        )
        self.assertEqual((), result.evidence)

    def test_destination_span_is_not_reused_as_target_type(self):
        result = extract_alfred_selection_evidence(
            "Put the apple in the bowl.", self.ontology
        )
        values = {item.property_name: item.desired_value for item in result.evidence}
        self.assertEqual("apple", values["type"])
        self.assertEqual(
            RelationValue("inside", {"object": "bowl"}), values["location"]
        )

    def test_conflicting_thermal_cues_fail_closed(self):
        result = extract_alfred_selection_evidence(
            "Warm then cool the apple in the fridge.", self.ontology
        )
        names = {item.property_name for item in result.evidence}
        self.assertNotIn("thermal_state", names)
        self.assertIn("type", names)

    def test_fusion_adds_only_span_supported_properties_and_gates_states(self):
        query = "Put the bat in the drawer."
        frame = QueryFrame(
            query,
            (
                PropertyConstraint(
                    "location", RelationValue("on", {"object": "drawer"}), 0.8
                ),
                PropertyConstraint("cleanliness", "clean", 0.2),
            ),
        )
        evidence = extract_alfred_selection_evidence(query, self.ontology)
        fused = fuse_alfred_selection(
            frame,
            evidence,
            policy=SelectionFusionPolicy(gate_unsupported_states=True),
        )
        values = {item.property_name: item.desired_value for item in fused.frame.constraints}
        self.assertEqual({"location", "type"}, set(values))
        self.assertEqual("baseball bat", values["type"])
        self.assertEqual(RelationValue("on", {"object": "drawer"}), values["location"])
        self.assertEqual(2, len(fused.actions))

    def test_add_only_policy_preserves_unsupported_model_state(self):
        query = "Put the apple in the bowl."
        frame = QueryFrame(
            query, (PropertyConstraint("cleanliness", "clean", 1.0),)
        )
        result = fuse_alfred_selection(
            frame,
            extract_alfred_selection_evidence(query, self.ontology),
            policy=SelectionFusionPolicy(gate_unsupported_states=False),
        )
        self.assertIn("cleanliness", {item.property_name for item in result.frame.constraints})

    def test_conflict_gate_requires_positive_state_evidence(self):
        query = "Cool the apple and put it in the bowl."
        frame = QueryFrame(
            query,
            (
                PropertyConstraint("cleanliness", "clean", 0.2),
                PropertyConstraint("thermal_state", "cold", 0.3),
            ),
        )
        evidence = extract_alfred_selection_evidence(query, self.ontology)
        result = fuse_alfred_selection(
            frame,
            evidence,
            policy=SelectionFusionPolicy(
                add_missing=False, gate_conflicting_states=True
            ),
        )
        self.assertEqual(
            {"thermal_state"},
            {item.property_name for item in result.frame.constraints},
        )
        no_cue_query = "Put the apple in the bowl."
        no_cue_frame = QueryFrame(no_cue_query, frame.constraints)
        preserved = fuse_alfred_selection(
            no_cue_frame,
            extract_alfred_selection_evidence(no_cue_query, self.ontology),
            policy=SelectionFusionPolicy(add_missing=False, gate_conflicting_states=True),
        )
        self.assertEqual(frame.constraints, preserved.frame.constraints)

    def test_fusion_rejects_evidence_from_a_different_query(self):
        evidence = extract_alfred_selection_evidence(
            "Put the apple in the bowl.", self.ontology
        )
        with self.assertRaisesRegex(ValueError, "does not match"):
            fuse_alfred_selection(QueryFrame("different", ()), evidence)

    def test_explicit_evidence_can_replace_conflicting_typed_values(self):
        query = "Put the apple in the fridge."
        evidence = extract_alfred_selection_evidence(query, self.ontology)
        frame = QueryFrame(
            query,
            (
                PropertyConstraint("type", "mug", 0.6),
                PropertyConstraint(
                    "location",
                    RelationValue("inside", {"object": "sink basin"}),
                    0.4,
                ),
            ),
        )
        result = fuse_alfred_selection(
            frame,
            evidence,
            policy=SelectionFusionPolicy(override_conflicting_values=True),
        )
        by_name = {item.property_name: item for item in result.frame.constraints}
        self.assertEqual("apple", by_name["type"].desired_value)
        self.assertEqual(
            RelationValue("inside", {"object": "fridge"}),
            by_name["location"].desired_value,
        )
        self.assertEqual(0.6, by_name["type"].relevance)
        self.assertEqual(
            2, sum(action.startswith("replaced") for action in result.actions)
        )

    def test_value_override_is_opt_in(self):
        query = "Put the apple in the fridge."
        evidence = extract_alfred_selection_evidence(query, self.ontology)
        frame = QueryFrame(query, (PropertyConstraint("type", "mug", 1.0),))
        result = fuse_alfred_selection(frame, evidence)
        self.assertEqual("mug", result.frame.constraints[0].desired_value)


if __name__ == "__main__":
    unittest.main()
