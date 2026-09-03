import json
import tempfile
import unittest
from pathlib import Path

from openprop.alfred_adapter import AlfredLanguageCase
from openprop.alfred_language_evaluation import (
    AlfredLanguageStrategy,
    alfred_language_registry,
    evaluate_alfred_language,
)
from openprop.alfred_ontology import (
    AlfredTrainingOntology,
    OntologyNormalisationPolicy,
    fit_alfred_training_ontology,
    normalise_alfred_goal_frame,
    resolve_canonical_label,
)
from openprop.language_temporal_grounding import RawLanguageResponse
from openprop.models import PropertyConstraint, QueryFrame, RelationValue


def _row(task_id, obj, parent, annotation):
    return {
        "task_id": task_id,
        "task_type": "pick_and_place_simple",
        "scene": {"floor_plan": "FloorPlan1"},
        "pddl_params": {"object_target": obj, "parent_target": parent},
        "turk_annotations": {"anns": [{"task_desc": annotation}]},
    }


def _write(root, split, name, row):
    target = root / split / name
    target.mkdir(parents=True)
    (target / "traj_data.json").write_text(json.dumps(row), encoding="utf-8")


def _raw_constraint(name, value, relevance=0.5):
    relation = isinstance(value, RelationValue)
    return {
        "property_name": name,
        "description": name,
        "value_type": "relation" if relation else "semantic",
        "known_property": True,
        "relevance": relevance,
        "tolerance": None,
        "value": {
            "kind": "relation" if relation else "scalar",
            "scalar": None if relation else value,
            "predicate": value.predicate if relation else None,
            "arguments": (
                [{"role": role, "value": item} for role, item in value.arguments.items()]
                if relation
                else []
            ),
            "vector": [],
        },
    }


class AlfredOntologyTests(unittest.TestCase):
    def test_fit_reads_train_pddl_labels_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write(
                root,
                "train",
                "one",
                _row("train-1", "BaseballBat", "Drawer", "move the secret banana"),
            )
            _write(
                root,
                "valid_unseen",
                "one",
                _row("valid-1", "Apple", "Fridge", "put the apple away"),
            )
            ontology = fit_alfred_training_ontology(root)
        self.assertEqual(frozenset({"baseball bat"}), ontology.object_labels)
        self.assertEqual(frozenset({"drawer"}), ontology.receptacle_labels)
        self.assertNotIn("secret banana", ontology.object_labels)
        self.assertNotIn("apple", ontology.object_labels)
        self.assertFalse(ontology.audit()["annotation_text_used_for_fit"])

    def test_unique_token_alias_resolves_but_ambiguous_head_does_not(self):
        labels = frozenset({"baseball bat", "salt shaker", "pepper shaker"})
        self.assertEqual("baseball bat", resolve_canonical_label("bat", labels))
        self.assertEqual("salt shaker", resolve_canonical_label("salt shaker", labels))
        self.assertIsNone(resolve_canonical_label("shaker", labels))

    def test_normalisation_preserves_types_and_repairs_container_semantics(self):
        ontology = AlfredTrainingOntology(
            frozenset({"baseball bat"}), frozenset({"drawer"})
        )
        frame = QueryFrame(
            "Put the bat in a drawer after warming it.",
            (
                PropertyConstraint("type", "bat", 0.3),
                PropertyConstraint(
                    "location", RelationValue("on", {"object": "drawer"}), 0.4
                ),
                PropertyConstraint("thermal_state", "warmed", 0.3),
            ),
        )
        result = normalise_alfred_goal_frame(frame, ontology)
        values = {
            item.property_name: item.desired_value for item in result.frame.constraints
        }
        self.assertEqual("baseball bat", values["type"])
        self.assertEqual(
            RelationValue("inside", {"object": "drawer"}), values["location"]
        )
        self.assertEqual("hot", values["thermal_state"])
        self.assertEqual(3, len(result.actions))
        self.assertEqual([0.3, 0.4, 0.3], [item.relevance for item in result.frame.constraints])

    def test_component_policy_disables_only_requested_transformations(self):
        ontology = AlfredTrainingOntology(
            frozenset({"baseball bat"}), frozenset({"drawer"})
        )
        frame = QueryFrame(
            "Put the bat in a drawer.",
            (
                PropertyConstraint("type", "bat", 0.5),
                PropertyConstraint(
                    "location", RelationValue("on", {"object": "drawer"}), 0.5
                ),
            ),
        )
        no_predicate = normalise_alfred_goal_frame(
            frame,
            ontology,
            policy=OntologyNormalisationPolicy(relation_predicates=False),
        )
        values = {item.property_name: item.desired_value for item in no_predicate.frame.constraints}
        self.assertEqual("baseball bat", values["type"])
        self.assertEqual(RelationValue("on", {"object": "drawer"}), values["location"])
        disabled = normalise_alfred_goal_frame(
            frame, ontology, policy=OntologyNormalisationPolicy(False, False, False, False)
        )
        self.assertEqual(frame, disabled.frame)
        self.assertEqual((), disabled.actions)

    def test_exact_canonical_frame_is_idempotent_and_wrong_unknown_is_preserved(self):
        ontology = AlfredTrainingOntology(
            frozenset({"baseball bat"}), frozenset({"drawer", "dining table"})
        )
        exact = QueryFrame(
            "Put the baseball bat in the drawer.",
            (
                PropertyConstraint("type", "baseball bat", 0.5),
                PropertyConstraint(
                    "location", RelationValue("inside", {"object": "drawer"}), 0.5
                ),
            ),
        )
        first = normalise_alfred_goal_frame(exact, ontology)
        second = normalise_alfred_goal_frame(first.frame, ontology)
        self.assertEqual(exact, first.frame)
        self.assertEqual((), first.actions)
        self.assertEqual(exact, second.frame)
        unknown = QueryFrame(
            "Put it away.",
            (PropertyConstraint("location", RelationValue("near", {"object": "garage"}), 1.0),),
        )
        self.assertEqual(unknown, normalise_alfred_goal_frame(unknown, ontology).frame)

    def test_ontology_strategy_requires_train_ontology_and_improves_frozen_frame(self):
        query = "Put the bat in the drawer."
        case = AlfredLanguageCase(
            "case-1",
            "task-1",
            "valid_unseen",
            "pick_and_place_simple",
            "FloorPlan1",
            query,
            QueryFrame(
                query,
                (
                    PropertyConstraint("type", "baseball bat", 0.5),
                    PropertyConstraint(
                        "location",
                        RelationValue("inside", {"object": "drawer"}),
                        0.5,
                    ),
                ),
            ),
            0,
            "valid_unseen/case/traj_data.json",
        )
        responses = {
            query: RawLanguageResponse(
                query,
                {
                    "constraints": [
                        _raw_constraint("type", "bat"),
                        _raw_constraint(
                            "location", RelationValue("on", {"object": "drawer"})
                        ),
                    ]
                },
                0.01,
            )
        }
        registry = alfred_language_registry()
        with self.assertRaisesRegex(ValueError, "training ontology"):
            evaluate_alfred_language(
                (case,),
                registry,
                AlfredLanguageStrategy.LLM_ONTOLOGY_NORMALIZED,
                responses=responses,
            )
        report = evaluate_alfred_language(
            (case,),
            registry,
            AlfredLanguageStrategy.LLM_ONTOLOGY_NORMALIZED,
            responses=responses,
            ontology=AlfredTrainingOntology(
                frozenset({"baseball bat"}), frozenset({"drawer"})
            ),
        )
        self.assertEqual(1.0, report.exact_frame_accuracy)
        self.assertEqual(1.0, report.normalisation_rate)


if __name__ == "__main__":
    unittest.main()
