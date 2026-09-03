import unittest
from dataclasses import replace

from openprop.alfred_adapter import AlfredLanguageCase
from openprop.alfred_language_evaluation import (
    AlfredLanguageStrategy,
    alfred_language_registry,
    evaluate_alfred_language,
    select_stratified_cases,
)
from openprop.alfred_ontology import AlfredTrainingOntology
from openprop.language_temporal_grounding import RawLanguageResponse
from openprop.models import PropertyConstraint, QueryFrame, RelationValue


def _case(case_id="case-1", task_type="pick_and_place_simple", annotation_index=0):
    query = "Put the apple in the fridge."
    return AlfredLanguageCase(
        case_id,
        case_id,
        "valid_unseen",
        task_type,
        "FloorPlan1",
        query,
        QueryFrame(
            query,
            (
                PropertyConstraint("type", "apple", 0.5),
                PropertyConstraint(
                    "location", RelationValue("inside", {"object": "fridge"}), 0.5
                ),
            ),
        ),
        annotation_index,
        "valid_unseen/case/traj_data.json",
    )


def _constraint(name, value, relevance=0.5):
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


class AlfredLanguageEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.case = _case()
        self.registry = alfred_language_registry()

    def test_gold_and_exact_replay_score_one(self):
        raw = {
            "constraints": [
                _constraint("type", "apple"),
                _constraint("location", RelationValue("inside", {"object": "fridge"})),
            ]
        }
        responses = {
            self.case.query: RawLanguageResponse(self.case.query, raw, 0.01)
        }
        gold = evaluate_alfred_language(
            (self.case,), self.registry, AlfredLanguageStrategy.GOLD
        )
        replay = evaluate_alfred_language(
            (self.case,),
            self.registry,
            AlfredLanguageStrategy.LLM_STRICT,
            responses=responses,
        )
        self.assertEqual(1.0, gold.exact_frame_accuracy)
        self.assertEqual(1.0, replay.exact_frame_accuracy)

    def test_failures_remain_in_all_case_denominator(self):
        second = replace(self.case, case_id="case-2")
        responses = {
            self.case.query: RawLanguageResponse(self.case.query, None, 0.01, "timeout")
        }
        report = evaluate_alfred_language(
            (self.case, second),
            self.registry,
            AlfredLanguageStrategy.LLM_TOLERANT,
            responses=responses,
        )
        self.assertEqual(2, report.failures)
        self.assertEqual(0.0, report.parse_success_rate)
        self.assertEqual(0.0, report.exact_frame_accuracy)

    def test_schema_repair_uses_schema_only_and_improves_value_recall(self):
        raw = {
            "constraints": [
                _constraint("type", "apple"),
                _constraint("location", RelationValue("inside", {"object": "wrong"})),
            ]
        }
        raw["constraints"][1]["value"] = {
            "kind": "relation",
            "predicate": "inside",
            "scalar": "fridge",
            "arguments": [{"role": "object", "value": "wrong"}],
            "vector": [],
        }
        responses = {
            self.case.query: RawLanguageResponse(self.case.query, raw, 0.01)
        }
        tolerant = evaluate_alfred_language(
            (self.case,),
            self.registry,
            AlfredLanguageStrategy.LLM_TOLERANT,
            responses=responses,
        )
        repaired = evaluate_alfred_language(
            (self.case,),
            self.registry,
            AlfredLanguageStrategy.LLM_SCHEMA_REPAIRED,
            responses=responses,
        )
        self.assertEqual(0.5, tolerant.value_recall)
        self.assertEqual(1.0, repaired.value_recall)
        self.assertEqual(1.0, repaired.repair_rate)

    def test_evidence_fusion_adds_span_supported_properties_and_removes_conflict(self):
        query = "Cool the apple and put it in the fridge."
        case = replace(
            self.case,
            query=query,
            task_type="pick_cool_then_place_in_recep",
            gold_frame=QueryFrame(
                query,
                (
                    PropertyConstraint("type", "apple", 0.35),
                    PropertyConstraint("thermal_state", "cold", 0.35),
                    PropertyConstraint(
                        "location",
                        RelationValue("inside", {"object": "fridge"}),
                        0.3,
                    ),
                ),
            ),
        )
        raw = {
            "constraints": [
                _constraint(
                    "location", RelationValue("inside", {"object": "fridge"}), 0.5
                ),
                _constraint("cleanliness", "clean", 0.5),
            ]
        }
        report = evaluate_alfred_language(
            (case,),
            self.registry,
            AlfredLanguageStrategy.LLM_EVIDENCE_FUSED,
            responses={query: RawLanguageResponse(query, raw, 0.01)},
            ontology=AlfredTrainingOntology(
                frozenset({"apple"}), frozenset({"fridge"})
            ),
        )
        self.assertEqual(1.0, report.exact_frame_accuracy)
        self.assertEqual(1.0, report.selection_action_rate)
        self.assertEqual(
            {"type", "thermal_state", "location"},
            set(report.results[0].selected_properties),
        )
        self.assertTrue(
            any("removed conflicting cleanliness" in item for item in report.results[0].selection_actions)
        )

    def test_evidence_fusion_does_not_invent_type_without_query_span(self):
        query = "Acquire an odd item and place it where it is useful."
        case = replace(
            self.case,
            query=query,
            gold_frame=QueryFrame(
                query,
                (
                    PropertyConstraint("type", "salt shaker", 0.5),
                    PropertyConstraint(
                        "location",
                        RelationValue("inside", {"object": "cabinet"}),
                        0.5,
                    ),
                ),
            ),
        )
        raw = {
            "constraints": [
                _constraint(
                    "location", RelationValue("inside", {"object": "cabinet"}), 1.0
                )
            ]
        }
        report = evaluate_alfred_language(
            (case,),
            self.registry,
            AlfredLanguageStrategy.LLM_EVIDENCE_FUSED,
            responses={query: RawLanguageResponse(query, raw, 0.01)},
            ontology=AlfredTrainingOntology(
                frozenset({"salt shaker"}), frozenset({"cabinet"})
            ),
        )
        self.assertNotIn("type", report.results[0].selected_properties)
        self.assertEqual(0.0, report.exact_frame_accuracy)

    def test_stratified_selection_is_deterministic_and_uses_annotation_zero(self):
        cases = []
        for task in ("a", "b"):
            for index in range(3):
                cases.append(
                    replace(
                        self.case,
                        case_id=f"{task}-{index}",
                        task_id=f"{task}-{index}",
                        task_type=task,
                    )
                )
                cases.append(
                    replace(
                        self.case,
                        case_id=f"{task}-{index}-ann1",
                        task_id=f"{task}-{index}",
                        task_type=task,
                        annotation_index=1,
                    )
                )
        selected = select_stratified_cases(cases, trajectories_per_task=2)
        self.assertEqual(["a-0", "a-1", "b-0", "b-1"], [item.case_id for item in selected])
        self.assertTrue(all(item.annotation_index == 0 for item in selected))
        offset = select_stratified_cases(
            cases, trajectories_per_task=2, trajectory_offset=1
        )
        self.assertEqual(
            ["a-1", "a-2", "b-1", "b-2"],
            [item.case_id for item in offset],
        )
        with self.assertRaisesRegex(ValueError, "required"):
            select_stratified_cases(cases, trajectories_per_task=2, trajectory_offset=2)


if __name__ == "__main__":
    unittest.main()
