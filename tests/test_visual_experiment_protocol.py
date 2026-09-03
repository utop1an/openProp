import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from openprop.visual_experiment_protocol import (
    AI2THOR_SPLIT_SHA256,
    build_visual_experiment_protocol,
    validate_visual_experiment_protocol,
    write_visual_experiment_protocol,
)


class VisualExperimentProtocolTests(unittest.TestCase):
    def test_protocol_freezes_claim_boundaries_and_primary_matrix(self):
        payload = build_visual_experiment_protocol()
        self.assertEqual(
            payload["ai2thor_scene_split"]["split_sha256"], AI2THOR_SPLIT_SHA256
        )
        self.assertEqual(len(payload["random_seeds"]), 5)
        self.assertGreaterEqual(len(payload["primary_comparisons"]), 5)
        public = {
            row["id"]: row for row in payload["evidence_tiers"]
        }
        self.assertFalse(public["ego4d_hands_objects"]["end_to_end_query_claim_eligible"])
        self.assertFalse(public["licensed_web_video"]["end_to_end_query_claim_eligible"])
        self.assertTrue(public["custom_real_video"]["end_to_end_query_claim_eligible"])
        self.assertEqual(
            payload["vlm_freeze_rule"]["current_status"], "exact_models_not_yet_locked"
        )
        self.assertEqual(payload["schema_version"], 2)
        self.assertTrue(
            payload["language_parser_freeze_rule"][
                "reuse_identical_parse_across_visual_systems"
            ]
        )
        self.assertIn(
            "oracle_typed_constraint",
            payload["language_parser_freeze_rule"]["baselines"],
        )
        self.assertTrue(
            payload["model_factorization"][
                "main_system_vlm_does_not_rank_final_entities"
            ]
        )
        systems = {row["id"] for row in payload["systems"]}
        self.assertIn("openprop_rule_parser", systems)
        self.assertIn("openprop_oracle_parser", systems)
        self.assertEqual(
            payload["implementation_readiness"]["captured_real_vlm_responses"],
            "not_yet_available",
        )

    def test_hash_and_unknown_system_drift_fail_closed(self):
        payload = build_visual_experiment_protocol()
        drifted = json.loads(json.dumps(payload))
        drifted["random_seeds"][0] = 1
        with self.assertRaisesRegex(ValueError, "hash drifted"):
            validate_visual_experiment_protocol(drifted)
        unknown = json.loads(json.dumps(payload))
        unknown["primary_comparisons"][0]["baseline"] = "unknown"
        unknown.pop("protocol_sha256")
        import hashlib
        canonical = json.dumps(
            unknown, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        unknown["protocol_sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
        with self.assertRaisesRegex(ValueError, "unknown system"):
            validate_visual_experiment_protocol(unknown)

    def test_write_and_check_are_byte_deterministic(self):
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "protocol.json"
            first = write_visual_experiment_protocol(path)
            second = write_visual_experiment_protocol(path, check=True)
            self.assertEqual(first, second)
            path.write_text(path.read_text(encoding="utf-8") + " ", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "drifted"):
                write_visual_experiment_protocol(path, check=True)


if __name__ == "__main__":
    unittest.main()
