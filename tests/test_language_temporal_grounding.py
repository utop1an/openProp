import unittest
from dataclasses import replace

from openprop.language_temporal_grounding import (
    LanguageTemporalStrategy,
    RawLanguageResponse,
    collect_language_responses,
    evaluate_language_temporal_grounding,
)
from openprop.models import RelationValue
from openprop.schema_repair import repair_redundant_relation_fields
from openprop.temporal_grounding import (
    temporal_grounding_benchmark,
    temporal_grounding_registry,
)


def _raw_constraint(constraint):
    value = constraint.desired_value
    if isinstance(value, RelationValue):
        encoded = {
            "kind": "relation",
            "scalar": None,
            "predicate": value.predicate,
            "arguments": [
                {"role": role, "value": argument}
                for role, argument in value.arguments.items()
            ],
            "vector": [],
        }
        value_type = "relation"
    else:
        encoded = {
            "kind": "scalar",
            "scalar": value,
            "predicate": None,
            "arguments": [],
            "vector": [],
        }
        value_type = "semantic"
    return {
        "property_name": constraint.property_name,
        "description": constraint.property_name,
        "value_type": value_type,
        "known_property": True,
        "relevance": constraint.relevance,
        "tolerance": constraint.tolerance,
        "value": encoded,
    }


def _responses(cases, *, corrupt_relation=False):
    by_query = {}
    for case in cases:
        constraints = [_raw_constraint(item) for item in case.gold_frame.constraints]
        if corrupt_relation and "location" in {
            item.property_name for item in case.gold_frame.constraints
        }:
            constraints.append(
                {
                    "property_name": "temperature",
                    "description": "temperature",
                    "value_type": "numeric",
                    "known_property": True,
                    "relevance": 0.2,
                    "tolerance": None,
                    "value": {
                        "kind": "scalar",
                        "scalar": "warm",
                        "predicate": None,
                        "arguments": [],
                        "vector": [],
                    },
                }
            )
        by_query[case.query] = RawLanguageResponse(
            case.query, {"constraints": constraints}, 0.01
        )
    return by_query


class _CountingClient:
    def __init__(self, response):
        self.response = response
        self.calls = 0

    def generate_json(self, **_):
        self.calls += 1
        return self.response


class LanguageTemporalGroundingTests(unittest.TestCase):
    def setUp(self):
        self.cases = temporal_grounding_benchmark(repetitions=1)
        self.registry = temporal_grounding_registry()

    def test_strict_and_tolerant_replay_the_same_raw_response(self):
        responses = _responses(self.cases, corrupt_relation=True)
        strict = evaluate_language_temporal_grounding(
            self.cases,
            self.registry,
            LanguageTemporalStrategy.LLM_STRICT,
            responses=responses,
        )
        tolerant = evaluate_language_temporal_grounding(
            self.cases,
            self.registry,
            LanguageTemporalStrategy.LLM_TOLERANT,
            responses=responses,
        )
        self.assertEqual(2, strict.failures)
        self.assertEqual(0.5, strict.top1_accuracy)
        self.assertEqual(0, tolerant.failures)
        self.assertEqual(1.0, tolerant.top1_accuracy)
        self.assertEqual(0.5, tolerant.validation_error_rate)

    def test_failures_remain_in_primary_metric_denominator(self):
        responses = dict(_responses(self.cases))
        query = self.cases[0].query
        responses[query] = RawLanguageResponse(query, None, 0.01, "timeout")
        report = evaluate_language_temporal_grounding(
            self.cases,
            self.registry,
            LanguageTemporalStrategy.LLM_TOLERANT,
            responses=responses,
        )
        self.assertEqual(2, report.failures)
        self.assertEqual(0.5, report.parse_success_rate)
        self.assertEqual(0.5, report.top1_accuracy)
        self.assertEqual(1.0, report.conditional_top1_accuracy)

    def test_candidate_order_does_not_change_results(self):
        responses = _responses(self.cases)
        forward = evaluate_language_temporal_grounding(
            self.cases,
            self.registry,
            LanguageTemporalStrategy.LLM_TOLERANT,
            responses=responses,
        )
        reversed_cases = tuple(
            replace(case, entities=tuple(reversed(case.entities)))
            for case in self.cases
        )
        reverse = evaluate_language_temporal_grounding(
            reversed_cases,
            self.registry,
            LanguageTemporalStrategy.LLM_TOLERANT,
            responses=responses,
        )
        self.assertEqual(
            [item.rank for item in forward.results],
            [item.rank for item in reverse.results],
        )

    def test_collection_requests_each_distinct_query_once(self):
        response = {"constraints": [_raw_constraint(self.cases[0].gold_frame.constraints[0])]}
        client = _CountingClient(response)
        captured = collect_language_responses(self.cases, self.registry, client)
        self.assertEqual(len({case.query for case in self.cases}), client.calls)
        self.assertEqual(client.calls, len(captured))

    def test_schema_repair_corrects_both_observed_relation_permutations(self):
        def response(predicate, scalar, argument):
            return {
                "constraints": [
                    {
                        "property_name": "location",
                        "value": {
                            "kind": "relation",
                            "predicate": predicate,
                            "scalar": scalar,
                            "arguments": [{"role": "object", "value": argument}],
                            "vector": [],
                        },
                    }
                ]
            }

        english = repair_redundant_relation_fields(
            response("table", "on", "cup"), self.registry
        )
        chinese = repair_redundant_relation_fields(
            response("on", "table", "cup"), self.registry
        )
        for repaired in (english, chinese):
            value = repaired.response["constraints"][0]["value"]
            self.assertEqual("on", value["predicate"])
            self.assertEqual([{"role": "object", "value": "table"}], value["arguments"])
            self.assertIsNone(value["scalar"])
            self.assertEqual(1, len(repaired.actions))

    def test_schema_repair_improves_grounding_without_candidate_access(self):
        responses = dict(_responses(self.cases))
        query = self.cases[0].query
        raw = responses[query].response
        location = next(
            item for item in raw["constraints"] if item["property_name"] == "location"
        )
        location["value"] = {
            "kind": "relation",
            "predicate": "on",
            "scalar": "table",
            "arguments": [{"role": "object", "value": "cup"}],
            "vector": [],
        }
        tolerant = evaluate_language_temporal_grounding(
            self.cases,
            self.registry,
            LanguageTemporalStrategy.LLM_TOLERANT,
            responses=responses,
        )
        repaired = evaluate_language_temporal_grounding(
            self.cases,
            self.registry,
            LanguageTemporalStrategy.LLM_SCHEMA_REPAIRED,
            responses=responses,
        )
        self.assertEqual(0.5, tolerant.top1_accuracy)
        self.assertEqual(1.0, repaired.top1_accuracy)
        self.assertEqual(0.5, repaired.repair_rate)


if __name__ == "__main__":
    unittest.main()
