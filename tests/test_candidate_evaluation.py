import unittest

from openprop.candidate_evaluation import (
    CandidateFrameTruth,
    CandidateTruthObject,
    aggregate_candidate_tracking,
    aggregate_candidate_tracking_matrix,
    evaluate_candidate_tracking,
)
from openprop.candidate_tracking import (
    CandidateSourceFrame,
    CandidateTracker,
    RegionProposal,
)


class CandidateEvaluationTests(unittest.TestCase):
    def frame(self, index):
        return CandidateSourceFrame(f"f{index}", f"f{index}.png", index, "camera")

    def proposal(self, index, region, *, external="object-a", suffix=""):
        return RegionProposal(
            f"p{index}{suffix}", f"f{index}", region, 0.9, "mug", external
        )

    def truth(self, count=2):
        return tuple(
            CandidateFrameTruth(
                f"f{index}",
                (CandidateTruthObject("truth-a", (0.1, 0.1, 0.3, 0.3)),),
            )
            for index in range(count)
        )

    def evaluate(self, run, truth, *, cluster="room-a", split="test", query_frame="f1"):
        return evaluate_candidate_tracking(
            run,
            truth,
            cluster_id=cluster,
            record_id=f"record-{cluster}",
            split=split,
            system="tracker",
            source="camera",
            query_frame_id=query_frame,
            query_target_entity_id="truth-a",
        )

    def test_candidate_recall_purity_and_target_recall_with_stable_track(self):
        frames = (self.frame(0), self.frame(1))
        proposals = (
            self.proposal(0, (0.1, 0.1, 0.3, 0.3)),
            self.proposal(1, (0.1, 0.1, 0.3, 0.3)),
        )
        evaluation = self.evaluate(CandidateTracker().track(frames, proposals), self.truth())
        report = aggregate_candidate_tracking((evaluation,), split="test")
        self.assertEqual(report["candidate_recall"], 1.0)
        self.assertEqual(report["candidate_precision"], 1.0)
        self.assertEqual(report["track_purity"], 1.0)
        self.assertEqual(report["query_target_recall"], 1.0)
        self.assertEqual(report["identity_switches"], 0)

    def test_identity_switch_is_counted_when_tracker_restarts_identity(self):
        frames = (self.frame(0), self.frame(1))
        proposals = (
            self.proposal(0, (0.1, 0.1, 0.3, 0.3), external=None),
            self.proposal(1, (0.1, 0.1, 0.3, 0.3), external="new-external"),
        )
        # Force an external-ID conflict while the boxes still overlap.
        proposals = (
            RegionProposal("p0", "f0", (0.1, 0.1, 0.3, 0.3), 0.9, "mug", "old"),
            RegionProposal("p1", "f1", (0.1, 0.1, 0.3, 0.3), 0.9, "mug", "new"),
        )
        evaluation = self.evaluate(CandidateTracker().track(frames, proposals), self.truth())
        self.assertEqual(evaluation.identity_switches, 1)

    def test_gap_then_reacquisition_counts_fragmentation(self):
        frames = (self.frame(0), self.frame(1), self.frame(2))
        proposals = (
            self.proposal(0, (0.1, 0.1, 0.3, 0.3)),
            self.proposal(2, (0.1, 0.1, 0.3, 0.3)),
        )
        truth = self.truth(3)
        evaluation = self.evaluate(
            CandidateTracker().track(frames, proposals),
            truth,
            query_frame="f2",
        )
        self.assertEqual(evaluation.fragmentations, 1)
        report = aggregate_candidate_tracking((evaluation,), split="test")
        self.assertEqual(report["candidate_recall"], 2 / 3)
        self.assertEqual(report["misses"], 1)

    def test_false_positive_and_every_frame_remain_in_denominator(self):
        frames = (self.frame(0), self.frame(1))
        proposals = (
            self.proposal(0, (0.1, 0.1, 0.3, 0.3)),
            self.proposal(0, (0.6, 0.6, 0.8, 0.8), external="other", suffix="-fp"),
        )
        evaluation = self.evaluate(CandidateTracker().track(frames, proposals), self.truth())
        report = aggregate_candidate_tracking((evaluation,), split="test")
        self.assertEqual(report["frames"], 2)
        self.assertEqual(report["false_positives"], 1)
        self.assertEqual(report["candidate_precision"], 0.5)
        self.assertTrue(report["all_frames_retained"])
        self.assertEqual(report["query_target_recall"], 0.0)

    def test_truth_coverage_and_cluster_split_leakage_fail_closed(self):
        run = CandidateTracker().track(
            (self.frame(0), self.frame(1)),
            (self.proposal(0, (0.1, 0.1, 0.3, 0.3)),),
        )
        with self.assertRaisesRegex(ValueError, "cover every"):
            self.evaluate(run, self.truth(1))
        first = self.evaluate(run, self.truth(), cluster="same", split="test")
        second = self.evaluate(run, self.truth(), cluster="same", split="calibration")
        with self.assertRaisesRegex(ValueError, "leaks"):
            aggregate_candidate_tracking((first, second), split="test")

    def test_multi_system_matrix_keeps_candidate_pipelines_separate(self):
        run = CandidateTracker().track(
            (self.frame(0), self.frame(1)),
            (
                self.proposal(0, (0.1, 0.1, 0.3, 0.3)),
                self.proposal(1, (0.1, 0.1, 0.3, 0.3)),
            ),
        )
        first = self.evaluate(run, self.truth(), cluster="room-a")
        second = self.evaluate(run, self.truth(), cluster="room-b")
        second = type(second)(
            second.cluster_id,
            second.record_id,
            second.split,
            "external-tracker",
            second.source,
            second.truth_population_sha256,
            second.query_frame_id,
            second.query_target_entity_id,
            second.iou_threshold,
            second.frames,
            second.identity_switches,
            second.fragmentations,
            second.purity_correct,
            second.purity_total,
            second.query_target_trials,
            second.query_target_hits,
        )
        matrix = aggregate_candidate_tracking_matrix((first, second), split="test")
        self.assertEqual(set(matrix["systems"]), {"tracker", "external-tracker"})


if __name__ == "__main__":
    unittest.main()
