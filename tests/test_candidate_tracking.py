import unittest

from openprop.ai2thor_adapter import ai2thor_property_registry
from openprop.association import MultiEntityAssociator, VLMPropertyDetector
from openprop.candidate_tracking import (
    CandidateAwareVisualOrchestrator,
    CandidateSourceFrame,
    CandidateTracker,
    CandidateTrackingPolicy,
    RegionProposal,
)
from openprop.comparators import default_comparators
from openprop.models import PropertyConstraint, QueryFrame
from openprop.selectors import MentionBasedSelector
from openprop.visual_pipeline import EntityStateStore, VisualUpdateOrchestrator
from openprop.vlm import VisualFrame


class ReplayOnlyClient:
    def generate_json(self, **kwargs):
        raise AssertionError("candidate-aware replay must not call a model")


class CandidateTrackingTests(unittest.TestCase):
    def frames(self, count=2):
        return tuple(
            CandidateSourceFrame(f"f{index}", f"f{index}.png", float(index), "camera")
            for index in range(count)
        )

    def proposal(
        self, proposal_id, frame_id, region, *, external=None, confidence=0.9, kind="mug"
    ):
        return RegionProposal(
            proposal_id, frame_id, region, confidence, kind, external
        )

    def test_external_track_preserves_identity_across_large_motion(self):
        proposals = (
            self.proposal("p0", "f0", (0.05, 0.1, 0.2, 0.3), external="ext-7"),
            self.proposal("p1", "f1", (0.75, 0.6, 0.9, 0.9), external="ext-7"),
        )
        run = CandidateTracker().track(self.frames(), proposals)
        self.assertEqual(
            run.frames[0].candidates[0].track_id,
            run.frames[1].candidates[0].track_id,
        )
        self.assertEqual(run.frames[1].candidates[0].status, "linked")
        self.assertGreater(run.frames[1].candidates[0].link_score, 0.35)

    def test_large_motion_without_continuity_evidence_creates_new_track(self):
        proposals = (
            self.proposal("p0", "f0", (0.05, 0.1, 0.2, 0.3)),
            self.proposal("p1", "f1", (0.75, 0.6, 0.9, 0.9)),
        )
        run = CandidateTracker().track(self.frames(), proposals)
        self.assertNotEqual(
            run.frames[0].candidates[0].track_id,
            run.frames[1].candidates[0].track_id,
        )
        self.assertEqual(run.frames[1].candidates[0].status, "new")

    def test_one_to_one_linking_prevents_track_collision(self):
        proposals = (
            self.proposal("first", "f0", (0.1, 0.1, 0.3, 0.3), external="same"),
            self.proposal("left", "f1", (0.1, 0.1, 0.3, 0.3), external="same"),
            self.proposal("right", "f1", (0.12, 0.1, 0.32, 0.3), external="same"),
        )
        run = CandidateTracker().track(self.frames(), proposals)
        second_ids = [item.track_id for item in run.frames[1].candidates]
        self.assertEqual(len(second_ids), len(set(second_ids)))
        self.assertEqual(sum(item.status == "linked" for item in run.frames[1].candidates), 1)
        self.assertEqual(sum(item.status == "new" for item in run.frames[1].candidates), 1)

    def test_empty_occlusion_frame_and_track_expiry_are_explicit(self):
        frames = self.frames(5)
        proposals = (
            self.proposal("p0", "f0", (0.1, 0.1, 0.3, 0.3), external="x"),
            self.proposal("p4", "f4", (0.1, 0.1, 0.3, 0.3), external="x"),
        )
        run = CandidateTracker(
            CandidateTrackingPolicy(max_missed_frames=2)
        ).track(frames, proposals)
        self.assertEqual(run.frames[1].visual_frame().candidate_entity_ids, ())
        self.assertTrue(run.frames[4].expired_track_ids)
        self.assertEqual(run.frames[4].candidates[0].status, "new")

    def test_low_confidence_and_capacity_rejections_remain_auditable(self):
        policy = CandidateTrackingPolicy(
            minimum_proposal_confidence=0.5,
            max_proposals_per_frame=1,
        )
        proposals = (
            self.proposal("high", "f0", (0.1, 0.1, 0.3, 0.3), confidence=0.9),
            self.proposal("second", "f0", (0.4, 0.1, 0.6, 0.3), confidence=0.8),
            self.proposal("low", "f0", (0.7, 0.1, 0.9, 0.3), confidence=0.2),
        )
        frame = CandidateTracker(policy).track(self.frames(1), proposals).frames[0]
        self.assertTrue(frame.capacity_exceeded)
        self.assertEqual(frame.rejected_proposal_ids, ("low", "second"))
        self.assertEqual(len(frame.candidates), 1)

    def test_candidate_aware_replay_creates_blank_open_world_entities(self):
        registry = ai2thor_property_registry()
        state = EntityStateStore(registry, ())
        associator = MultiEntityAssociator(
            registry, default_comparators(), MentionBasedSelector()
        )
        visual = VisualUpdateOrchestrator(
            VLMPropertyDetector(ReplayOnlyClient()), associator, state
        )
        pipeline = CandidateAwareVisualOrchestrator(CandidateTracker(), visual)
        query = QueryFrame(
            "the moved mug",
            (PropertyConstraint("motion_state", "moved", relevance=1.0),),
        )
        run = pipeline.replay(
            query,
            self.frames(1),
            (self.proposal("p0", "f0", (0.1, 0.1, 0.3, 0.3)),),
            {"detections": []},
        )
        self.assertEqual(run.created_entity_ids, ("track-000001",))
        self.assertEqual(state.entity_ids(), ("track-000001",))
        self.assertEqual(run.visual.detections, ())
        self.assertEqual(state.snapshot("track-000001").properties, {})

    def test_visual_frame_may_represent_a_true_empty_candidate_frame(self):
        frame = VisualFrame("empty", "empty.png", 0.0, "camera", ())
        self.assertEqual(frame.candidate_regions, {})


if __name__ == "__main__":
    unittest.main()
