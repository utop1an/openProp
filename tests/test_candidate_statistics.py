import unittest
from dataclasses import replace

from openprop.candidate_evaluation import evaluate_candidate_tracking
from openprop.candidate_statistics import paired_candidate_system_comparison
from openprop.candidate_tracking import CandidateTracker, RegionProposal
from test_candidate_evaluation import CandidateEvaluationTests


class CandidateStatisticsTests(unittest.TestCase):
    def dataset(self):
        fixture = CandidateEvaluationTests()
        frames = (fixture.frame(0), fixture.frame(1))
        truth = fixture.truth()
        baseline_run = CandidateTracker().track(
            frames,
            (
                RegionProposal("fp0", "f0", (0.6, 0.6, 0.8, 0.8), 0.9),
                RegionProposal("fp1", "f1", (0.6, 0.6, 0.8, 0.8), 0.9),
            ),
        )
        system_run = CandidateTracker().track(
            frames,
            (
                fixture.proposal(0, (0.1, 0.1, 0.3, 0.3)),
                fixture.proposal(1, (0.1, 0.1, 0.3, 0.3)),
            ),
        )
        rows = []
        for index in range(4):
            cluster = "room-1" if index < 2 else "room-2"
            common = dict(
                truth=truth,
                cluster_id=cluster,
                record_id=f"episode-{index}",
                split="test",
                source="camera",
                query_frame_id="f1",
                query_target_entity_id="truth-a",
            )
            rows.append(
                evaluate_candidate_tracking(
                    baseline_run, system="baseline", **common
                )
            )
            rows.append(
                evaluate_candidate_tracking(
                    system_run, system="openprop", **common
                )
            )
        return tuple(rows)

    def test_paired_pooled_cluster_bootstrap_is_deterministic(self):
        first = paired_candidate_system_comparison(
            self.dataset(),
            baseline="baseline",
            system="openprop",
            split="test",
            bootstrap_replicates=200,
            seed=19,
        )
        second = paired_candidate_system_comparison(
            self.dataset(),
            baseline="baseline",
            system="openprop",
            split="test",
            bootstrap_replicates=200,
            seed=19,
        )
        self.assertEqual(first, second)
        recall = first["metrics"]["candidate_recall"]
        self.assertEqual(recall["delta_system_minus_baseline"], 1.0)
        self.assertEqual(recall["cluster_bootstrap_95_ci"], [1.0, 1.0])
        self.assertEqual(first["population"]["episodes"], 4)
        self.assertEqual(first["population"]["clusters"], 2)
        self.assertEqual(
            first["bootstrap"]["aggregation"],
            "resample clusters then pool metric numerators and denominators",
        )

    def test_unpaired_population_and_truth_or_query_drift_fail_closed(self):
        rows = self.dataset()
        with self.assertRaisesRegex(ValueError, "not exactly paired"):
            paired_candidate_system_comparison(
                rows[:-1],
                baseline="baseline",
                system="openprop",
                split="test",
                bootstrap_replicates=100,
            )
        changed_hash = replace(rows[-1], truth_population_sha256="0" * 64)
        with self.assertRaisesRegex(ValueError, "truth/query fields drifted"):
            paired_candidate_system_comparison(
                (*rows[:-1], changed_hash),
                baseline="baseline",
                system="openprop",
                split="test",
                bootstrap_replicates=100,
            )
        changed_query = replace(rows[-1], query_target_entity_id="truth-other")
        with self.assertRaisesRegex(ValueError, "truth/query fields drifted"):
            paired_candidate_system_comparison(
                (*rows[:-1], changed_query),
                baseline="baseline",
                system="openprop",
                split="test",
                bootstrap_replicates=100,
            )


if __name__ == "__main__":
    unittest.main()
