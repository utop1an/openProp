import unittest

from openprop.teach_adapter import TeachReplay, TeachReplaySnapshot
from openprop.teach_grounding import (
    audit_teach_grounding_cases,
    build_teach_gold_grounding_cases,
)


class TeachGroundingTests(unittest.TestCase):
    def _replay(self) -> TeachReplay:
        observations = (
            TeachReplaySnapshot(
                0.0,
                (
                    {
                        "objectId": "Mug|1",
                        "objectType": "Mug",
                        "visible": True,
                        "isDirty": True,
                    },
                    {
                        "objectId": "Mug|2",
                        "objectType": "Mug",
                        "visible": True,
                        "isDirty": False,
                    },
                ),
            ),
            TeachReplaySnapshot(
                2.0,
                (
                    {
                        "objectId": "Mug|1",
                        "objectType": "Mug",
                        "visible": False,
                        "isDirty": True,
                    },
                    {
                        "objectId": "Mug|2",
                        "objectType": "Mug",
                        "visible": True,
                        "isDirty": False,
                    },
                ),
            ),
        )
        final = TeachReplaySnapshot(
            5.0,
            (
                {
                    "objectId": "Mug|1",
                    "objectType": "Mug",
                    "visible": False,
                    "isDirty": False,
                },
                {
                    "objectId": "Mug|2",
                    "objectType": "Mug",
                    "visible": False,
                    "isDirty": True,
                },
            ),
            is_final=True,
        )
        return TeachReplay(observations, final)

    def test_builds_unique_typed_cases_from_observations_and_separate_truth(self):
        cases = build_teach_gold_grounding_cases(
            "game-1", "FloorPlan1", self._replay(), property_names=("isDirty",)
        )
        self.assertEqual(2, len(cases))
        for case in cases:
            self.assertEqual(2, len(case.entities))
            self.assertEqual({"type", "isDirty"}, set(case.current_truth[case.target_id]))
            self.assertIsInstance(case.gold_frame.constraints[1].desired_value, bool)
            self.assertEqual(5.0, case.as_of)
            self.assertIn("temporal-challenge", case.tags)
            self.assertNotIn("primary-evaluable", case.tags)
            for entity in case.entities:
                self.assertNotIn("current_truth", entity.properties)
                raw_id = entity.entity_id.split(":", 1)[1]
                self.assertNotEqual(
                    case.current_truth[entity.entity_id]["isDirty"],
                    entity.properties["isDirty"].value,
                )
                self.assertIn(raw_id, {"Mug|1", "Mug|2"})

    def test_requires_final_truth_and_two_candidates(self):
        with self.assertRaisesRegex(ValueError, "final truth"):
            build_teach_gold_grounding_cases(
                "game-1",
                "FloorPlan1",
                TeachReplay(self._replay().observations, None),
                property_names=("isDirty",),
            )
        replay = self._replay()
        one = TeachReplay(
            tuple(
                TeachReplaySnapshot(item.timestamp, (item.objects[0],))
                for item in replay.observations
            ),
            TeachReplaySnapshot(5.0, (replay.final_truth.objects[0],), is_final=True),
        )
        self.assertEqual(
            (),
            build_teach_gold_grounding_cases(
                "game-1", "FloorPlan1", one, property_names=("isDirty",)
            ),
        )

    def test_audit_reports_temporal_cases_and_candidate_sizes(self):
        cases = build_teach_gold_grounding_cases(
            "game-1", "FloorPlan1", self._replay(), property_names=("isDirty",)
        )
        report = audit_teach_grounding_cases(cases)
        self.assertEqual(2, report["cases"])
        self.assertEqual(2, report["temporal_challenge_cases"])
        self.assertEqual(0, report["temporal_discriminative_cases"])
        self.assertEqual(2, report["unobservable_target_cases"])
        self.assertEqual(2, report["candidate_size_min"])
        self.assertEqual(0, report["target_ties_in_final_truth"])
        self.assertEqual("visible replay snapshots only", report["matcher_input_source"])

    def test_marks_only_newer_positive_target_evidence_as_temporally_discriminative(self):
        observations = (
            TeachReplaySnapshot(
                0.0,
                (
                    {"objectId": "Mug|1", "objectType": "Mug", "visible": True, "isDirty": True},
                    {"objectId": "Mug|2", "objectType": "Mug", "visible": True, "isDirty": True},
                ),
            ),
            TeachReplaySnapshot(
                3.0,
                (
                    {"objectId": "Mug|1", "objectType": "Mug", "visible": True, "isDirty": True},
                    {"objectId": "Mug|2", "objectType": "Mug", "visible": False, "isDirty": True},
                ),
            ),
        )
        final = TeachReplaySnapshot(
            5.0,
            (
                {"objectId": "Mug|1", "objectType": "Mug", "visible": False, "isDirty": True},
                {"objectId": "Mug|2", "objectType": "Mug", "visible": False, "isDirty": False},
            ),
            is_final=True,
        )
        cases = build_teach_gold_grounding_cases(
            "game-2",
            "FloorPlan2",
            TeachReplay(observations, final),
            property_names=("isDirty",),
        )
        positive = next(
            case
            for case in cases
            if case.gold_frame.constraints[-1].desired_value is True
        )
        negative = next(
            case
            for case in cases
            if case.gold_frame.constraints[-1].desired_value is False
        )
        self.assertIn("temporal-discriminative", positive.tags)
        self.assertIn("primary-evaluable", positive.tags)
        self.assertIn("unobservable-target-state", negative.tags)
        self.assertNotIn("primary-evaluable", negative.tags)


if __name__ == "__main__":
    unittest.main()
