import json
import tempfile
import unittest
from pathlib import Path

from openprop.teach_audit import TeachAuditSession, audit_teach_sessions
from openprop.teach_experiment import (
    TEACH_FACTORIZED_ABLATIONS,
    prepare_teach_layer_b_experiment,
    run_teach_layer_b_experiment,
)
from openprop.teach_feasibility import TeachFeasibilityCriteria


class TeachExperimentTests(unittest.TestCase):
    def _session(self, root: Path, index: int) -> TeachAuditSession:
        episode_id = f"game-{index}"
        session = root / episode_id
        states = session / "states"
        states.mkdir(parents=True)
        initial = {
            "objects": [
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
                    "isDirty": True,
                },
            ],
            "custom_object_metadata": {},
        }
        initial_path = session / "initial.json"
        initial_path.write_text(json.dumps(initial), encoding="utf-8")
        diffs = {
            "statediff.0.json": {"objects": {}},
            "statediff.1.json": {
                "objects": {
                    "Mug|1": {"visible": False},
                    "Mug|2": {"visible": False},
                }
            },
            "statediff.2.json": {
                "objects": {
                    "Mug|1": {"visible": True, "isDirty": False},
                    "Mug|2": {"visible": False},
                }
            },
            "statediff.3.json": {
                "objects": {
                    "Mug|1": {"visible": False, "isDirty": False},
                    "Mug|2": {"visible": True, "isDirty": False},
                }
            },
            "statediff.end.json": {
                "objects": {
                    "Mug|1": {"visible": False, "isDirty": True},
                    "Mug|2": {"visible": False, "isDirty": False},
                }
            },
        }
        for name, payload in diffs.items():
            (states / name).write_text(json.dumps(payload), encoding="utf-8")
        return TeachAuditSession(
            episode_id,
            f"FloorPlan{index}",
            initial_path,
            states,
            4.0,
        )

    @staticmethod
    def _criteria() -> TeachFeasibilityCriteria:
        return TeachFeasibilityCriteria(
            profile="unit-layer-b",
            min_sessions=3,
            min_floorplans=3,
            min_snapshots=12,
            min_visible_entities=6,
            min_history_records=12,
            min_interval_events=6,
            min_transition_properties=1,
            min_grounding_cases=6,
            min_temporal_discriminative_cases=3,
            min_candidate_size=2,
            min_dialogue_alignments=1,
            min_manual_alignment_labels=1,
            min_manual_alignment_precision=1.0,
        )

    def test_preparation_is_floorplan_disjoint_and_keeps_unobservable_coverage(self):
        with tempfile.TemporaryDirectory() as directory:
            sessions = tuple(self._session(Path(directory), index) for index in range(1, 4))
            prepared = prepare_teach_layer_b_experiment(
                sessions, property_names=("isDirty",), split_seed=11
            )
        partitions = (prepared.train, prepared.validation, prepared.test)
        self.assertTrue(all(len(item.floorplans) == 1 for item in partitions))
        self.assertTrue(set(prepared.train.floorplans).isdisjoint(prepared.validation.floorplans))
        self.assertTrue(set(prepared.train.floorplans).isdisjoint(prepared.test.floorplans))
        self.assertTrue(set(prepared.validation.floorplans).isdisjoint(prepared.test.floorplans))
        for partition in partitions:
            self.assertEqual(2, len(partition.all_cases))
            self.assertEqual(1, len(partition.primary_cases))
            self.assertIn("temporal-discriminative", partition.primary_cases[0].tags)
            self.assertGreater(len(partition.examples), 0)

    def test_runner_fits_selects_and_tests_without_cross_split_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            sessions = tuple(self._session(Path(directory), index) for index in range(1, 4))
            audit = audit_teach_sessions(
                sessions,
                property_names=("isDirty",),
                criteria=self._criteria(),
            )
            prepared = prepare_teach_layer_b_experiment(
                sessions, property_names=("isDirty",), split_seed=11
            )
            report = run_teach_layer_b_experiment(
                prepared,
                audit["feasibility_gate"],
                half_life_grid_hours=(0.25, 1.0, 4.0),
                factorized_epochs=30,
            )
        self.assertTrue(audit["feasibility_gate"]["layer_b_ready"])
        self.assertEqual("validation interval-aware NLL", report["protocol"]["fixed_selection_metric"])
        self.assertEqual("single final evaluation; no model selection", report["protocol"]["test_use"])
        self.assertEqual(0.0, report["test"]["no_decay"]["grounding"]["top1_accuracy"])
        self.assertEqual(
            1.0,
            report["test"]["validation_selected_fixed"]["grounding"]["top1_accuracy"],
        )
        self.assertEqual(1, report["split"]["test"]["primary_grounding_cases"])
        self.assertEqual(2, report["split"]["test"]["all_grounding_cases"])
        expected_factorized = {name for name, _ in TEACH_FACTORIZED_ABLATIONS}
        self.assertTrue(
            {
                "no_decay",
                "validation_selected_fixed",
                "train_global_exponential",
                "train_per_context_exponential",
                *expected_factorized,
            }.issubset(report["test"])
        )
        self.assertEqual(
            expected_factorized,
            set(report["validation"]["factorized_hazard_scales"]),
        )
        self.assertTrue(
            all(
                scale > 0
                for scale in report["validation"]["factorized_hazard_scales"].values()
            )
        )
        matrix = report["protocol"]["frozen_model_matrix"]
        self.assertEqual(
            ["property"],
            matrix["train_factorized_property_only"]["active_features"],
        )
        self.assertEqual(
            ["property", "subject_type", "observed_state", "context_object", "scene"],
            matrix["train_factorized_exponential"]["active_features"],
        )
        self.assertEqual(
            1.0,
            matrix["train_per_context_exponential"]["global_prior_exposure_hours"],
        )
        support = report["survival_feature_support"]["test"]
        self.assertFalse(support["uses_outcomes"])
        self.assertFalse(support["uses_current_truth"])
        self.assertEqual(0.0, support["exact_context"]["row_coverage"])
        self.assertEqual(1.0, support["features"]["property"]["row_coverage"])
        self.assertEqual(0.0, support["features"]["scene"]["row_coverage"])
        self.assertEqual(
            [prepared.test.floorplans[0].casefold()],
            support["features"]["scene"]["unseen_values"],
        )
        self.assertEqual(
            "feature values and exact context membership only; no event outcomes or current truth",
            report["protocol"]["support_audit_use"],
        )

    def test_runner_refuses_unqualified_data(self):
        with tempfile.TemporaryDirectory() as directory:
            sessions = tuple(self._session(Path(directory), index) for index in range(1, 4))
            prepared = prepare_teach_layer_b_experiment(
                sessions, property_names=("isDirty",)
            )
        with self.assertRaisesRegex(ValueError, "has not passed"):
            run_teach_layer_b_experiment(
                prepared,
                {"layer_b_ready": False},
                factorized_epochs=1,
            )
        with self.assertRaisesRegex(ValueError, "factorized_epochs"):
            run_teach_layer_b_experiment(
                prepared,
                {"layer_b_ready": True},
                factorized_epochs=0,
            )


if __name__ == "__main__":
    unittest.main()
