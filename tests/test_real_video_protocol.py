import hashlib
import json
import unittest
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory

from openprop.real_video_protocol import (
    REAL_VIDEO_PROTOCOL,
    REAL_VIDEO_TRUTH_BOUNDARY,
    prepare_real_video_manifest,
    verify_real_video_manifest,
)
from openprop.visual_replay import replay_visual_case


class RealVideoProtocolTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.root = Path(self.temporary.name)
        media = self.root / "media"
        media.mkdir()
        self.before = media / "before.png"
        self.after = media / "after.png"
        self.before.write_bytes(b"\x89PNG\r\n\x1a\n-before")
        self.after.write_bytes(b"\x89PNG\r\n\x1a\n-after")

    def tearDown(self):
        self.temporary.cleanup()

    def artifact(self, path):
        data = path.read_bytes()
        return {
            "path": path.relative_to(self.root).as_posix(),
            "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
        }

    def episode(self, episode_id="room-a-001", split="calibration"):
        candidates = [
            {"entity_id": "entity-001", "region": [0.1, 0.1, 0.3, 0.4]},
            {"entity_id": "entity-002", "region": [0.6, 0.1, 0.8, 0.4]},
        ]
        return {
            "episode_id": episode_id,
            "cluster_id": "home-01/person-01",
            "split": split,
            "source": "real-video-rgb",
            "condition": "one-move-one-distractor-fixed-camera",
            "distractor_count": 1,
            "frames": [
                {
                    "frame_id": f"{episode_id}.before",
                    "captured_at": 10.0,
                    "image": self.artifact(self.before),
                    "candidates": candidates,
                },
                {
                    "frame_id": f"{episode_id}.after",
                    "captured_at": 20.0,
                    "image": self.artifact(self.after),
                    "candidates": candidates,
                },
            ],
            "initial_entities": [
                {
                    "entity_id": entity_id,
                    "properties": {
                        "type": {
                            "value": "mug",
                            "state": "observed",
                            "confidence": 1.0,
                            "source": "manual-initialization",
                            "timestamp": 0.0,
                        },
                        "motion_state": {
                            "value": "stationary",
                            "state": "observed",
                            "confidence": 1.0,
                            "source": "manual-initialization",
                            "timestamp": 0.0,
                        },
                    },
                }
                for entity_id in ("entity-001", "entity-002")
            ],
            "query_time": 20.0,
            "query": {
                "text": "Which mug was moved?",
                "constraints": [
                    {
                        "property_name": "type",
                        "desired_value": "mug",
                        "relevance": 0.3,
                    },
                    {
                        "property_name": "motion_state",
                        "desired_value": "moved",
                        "relevance": 1.0,
                    },
                ],
            },
            "query_candidate_entity_ids": ["entity-001", "entity-002"],
            "annotations": {
                "frames": [
                    {"frame_id": f"{episode_id}.before", "events": []},
                    {
                        "frame_id": f"{episode_id}.after",
                        "events": [
                            {
                                "event_id": f"{episode_id}.moved",
                                "property_name": "motion_state",
                                "gold_value": "moved",
                                "target_entity_id": "entity-001",
                                "region": [0.1, 0.1, 0.3, 0.4],
                            }
                        ],
                    },
                ],
                "query": {
                    "record_id": f"query.{episode_id}",
                    "property_name": "motion_state",
                    "target_entity_id": "entity-001",
                    "horizon_seconds": 0.0,
                    "eligible": True,
                },
            },
        }

    def manifest(self):
        return {
            "schema_version": 1,
            "protocol": REAL_VIDEO_PROTOCOL,
            "evaluation_only": True,
            "truth_boundary": REAL_VIDEO_TRUTH_BOUNDARY,
            "collection_id": "openprop-custom-pilot-v1",
            "source_policy": {
                "source_kind": "self_recorded",
                "source_name": "OpenProp consented tabletop pilot",
                "license": "research-only",
                "redistribution": "derived annotations only",
                "consent_basis": "written participant consent",
            },
            "annotation_protocol": {
                "tier": "final",
                "annotator_count": 3,
                "adjudicated": True,
                "candidate_source": "manual-box-and-identity-track",
                "guideline_version": "openprop-real-video-annotation-v1",
                "agreement_metric": "pairwise-entity-state-agreement",
                "agreement_value": 0.9,
                "minimum_agreement": 0.8,
            },
            "episodes": [self.episode()],
        }

    def write_manifest(self, payload=None):
        path = self.root / "manifest.json"
        path.write_text(
            json.dumps(payload or self.manifest(), indent=2) + "\n",
            encoding="utf-8",
        )
        return path

    def test_prepare_separates_truth_and_outputs_replay_compatible_case(self):
        manifest = self.write_manifest()
        verification = verify_real_video_manifest(manifest)
        self.assertEqual(verification["episodes"], 1)
        self.assertEqual(verification["events"], 1)
        self.assertFalse(verification["truth_exposed_to_matcher"])

        output = self.root / "prepared"
        report = prepare_real_video_manifest(manifest, output)
        item = report["episodes"][0]
        input_payload = json.loads((output / item["input"]["path"]).read_text())
        case_payload = json.loads((output / item["case"]["path"]).read_text())
        truth_payload = json.loads((output / item["truth"]["path"]).read_text())
        encoded_input = json.dumps(input_payload)
        encoded_case = json.dumps(case_payload)
        self.assertNotIn("target_entity_id", encoded_input)
        self.assertNotIn("annotations", encoded_input)
        self.assertNotIn("target_entity_id", encoded_case)
        self.assertTrue(truth_payload["evaluation_only"])

        outcome = replay_visual_case(
            input_payload,
            case_payload,
            {"malformed": "retained as a model failure"},
            assignment="global",
        )
        self.assertEqual(outcome.case_id, "room-a-001")
        self.assertTrue(outcome.malformed_response)
        self.assertIsNone(outcome.query_decision.accepted_entity_id)

    def test_media_hash_drift_fails_closed(self):
        manifest = self.write_manifest()
        self.after.write_bytes(self.after.read_bytes() + b"tamper")
        with self.assertRaisesRegex(ValueError, "byte count drifted"):
            verify_real_video_manifest(manifest)

    def test_room_person_cluster_cannot_cross_splits(self):
        payload = self.manifest()
        second = self.episode("room-a-002", "test")
        payload["episodes"].append(second)
        with self.assertRaisesRegex(ValueError, "cluster leaks"):
            verify_real_video_manifest(self.write_manifest(payload))

    def test_final_annotation_quality_gate_is_enforced(self):
        payload = self.manifest()
        payload["annotation_protocol"]["annotator_count"] = 1
        with self.assertRaisesRegex(ValueError, "three annotators"):
            verify_real_video_manifest(self.write_manifest(payload))
        payload["annotation_protocol"]["annotator_count"] = 3
        payload["annotation_protocol"]["adjudicated"] = False
        with self.assertRaisesRegex(ValueError, "adjudication"):
            verify_real_video_manifest(self.write_manifest(payload))
        payload = self.manifest()
        payload["annotation_protocol"]["agreement_value"] = 0.79
        with self.assertRaisesRegex(ValueError, "below the frozen gate"):
            verify_real_video_manifest(self.write_manifest(payload))

    def test_truth_frame_coverage_and_candidate_identity_are_checked(self):
        payload = self.manifest()
        payload["episodes"][0]["annotations"]["frames"].pop()
        with self.assertRaisesRegex(ValueError, "cover every frame"):
            verify_real_video_manifest(self.write_manifest(payload))
        payload = self.manifest()
        payload["episodes"][0]["annotations"]["frames"][1]["events"][0][
            "target_entity_id"
        ] = "unknown-object"
        with self.assertRaisesRegex(ValueError, "known candidate"):
            verify_real_video_manifest(self.write_manifest(payload))

    def test_final_query_target_may_be_absent_for_target_missing_slice(self):
        payload = deepcopy(self.manifest())
        payload["episodes"][0]["annotations"]["query"][
            "target_entity_id"
        ] = "entity-not-visible-at-query"
        report = verify_real_video_manifest(self.write_manifest(payload))
        self.assertEqual(report["episodes"], 1)


if __name__ == "__main__":
    unittest.main()
