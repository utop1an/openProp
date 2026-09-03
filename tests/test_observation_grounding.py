import unittest

from openprop.compositional_persistence import (
    compositional_grounding_registry,
    evaluate_grounding_model,
)
from openprop.observation_grounding import (
    CONDITIONS,
    aggregate_observation_grounding_runs,
    evaluate_observation_grounding_seed,
    observation_grounding_benchmark,
    observation_grounding_oracle_model,
    scene_conditioned_observation_data,
)


class ObservationGroundingTests(unittest.TestCase):
    def test_cases_balance_target_scene_and_keep_truth_separate(self) -> None:
        cases = observation_grounding_benchmark()
        self.assertEqual(40, len(cases))
        self.assertEqual(20, sum("target-frequent-scene" in case.tags for case in cases))
        self.assertEqual(20, sum("target-sparse-scene" in case.tags for case in cases))
        self.assertEqual(len(cases), len({case.case_id for case in cases}))
        for case in cases:
            self.assertNotIn("scene", {row.property_name for row in case.gold_frame.constraints})
            matching = [
                entity
                for entity in case.entities
                if entity.properties["color"].value == "red"
            ]
            self.assertEqual(2, len(matching))
            self.assertEqual(
                {"frequent-scene", "sparse-scene"},
                {entity.properties["scene"].value for entity in matching},
            )
            for entity in case.entities:
                self.assertNotIn("current_truth", entity.properties)
                self.assertNotIn("target", entity.properties)

    def test_oracle_solves_cases_and_candidate_order_is_invariant(self) -> None:
        dataset = scene_conditioned_observation_data(
            samples_per_scene=5,
            test_samples_per_scene=2,
            seed=101,
        )
        cases = observation_grounding_benchmark()
        report = evaluate_observation_grounding_seed(
            seed=101,
            dataset=dataset,
            cases=cases,
        )
        self.assertEqual(1.0, report["conditions"]["oracle"]["top1"])

        oracle = observation_grounding_oracle_model(dataset.true_hazard_per_hour)
        registry = compositional_grounding_registry()
        forward = evaluate_grounding_model("oracle", oracle, cases, registry)
        reversed_cases = tuple(
            type(case)(
                case.case_id,
                case.query,
                tuple(reversed(case.entities)),
                case.target_id,
                case.gold_frame,
                case.as_of,
                case.current_truth,
                case.tags,
            )
            for case in cases
        )
        backward = evaluate_grounding_model("oracle", oracle, reversed_cases, registry)
        self.assertEqual(forward, backward)

    def test_invalid_case_parameters_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "positive"):
            observation_grounding_benchmark(repetitions_per_target_scene=0)
        with self.assertRaisesRegex(ValueError, "distractor > target"):
            observation_grounding_benchmark(target_age_hours=4, distractor_age_hours=3)

    def test_aggregation_is_paired_and_oriented_as_interval_advantage(self) -> None:
        def run(seed: int, naive: float, interval: float) -> dict[str, object]:
            conditions = {}
            for name, value in (("naive", naive), ("interval_aware", interval), ("oracle", 1.0)):
                conditions[name] = {
                    "top1": value,
                    "mean_reciprocal_rank": value,
                    "worst_target_scene_top1": value,
                    "target_scene_gap": 1.0 - value,
                    "top1_by_target_scene": {
                        "frequent-scene": value,
                        "sparse-scene": 1.0,
                    },
                }
            return {"seed": seed, "conditions": conditions}

        report = aggregate_observation_grounding_runs(
            (run(1, 0.5, 1.0), run(2, 0.6, 0.9)),
            bootstrap_samples=100,
        )
        self.assertEqual(list(CONDITIONS), list(report["aggregate"]))
        self.assertGreater(
            report["paired_interval_advantage"]["top1"]["mean_advantage"], 0.0
        )
        self.assertGreater(
            report["paired_interval_advantage"]["target_scene_gap"]["mean_advantage"],
            0.0,
        )

    def test_small_end_to_end_run_exposes_directional_naive_bias(self) -> None:
        dataset = scene_conditioned_observation_data(
            samples_per_scene=300,
            test_samples_per_scene=2,
            seed=101,
        )
        report = evaluate_observation_grounding_seed(
            seed=101,
            dataset=dataset,
            cases=observation_grounding_benchmark(repetitions_per_target_scene=2),
        )
        naive = report["conditions"]["naive"]
        interval = report["conditions"]["interval_aware"]
        self.assertGreater(interval["top1"], naive["top1"])
        self.assertGreater(naive["target_scene_gap"], interval["target_scene_gap"])
        self.assertEqual(1.0, interval["top1"])


if __name__ == "__main__":
    unittest.main()
