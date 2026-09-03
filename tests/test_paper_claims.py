import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from openprop.paper_claims import verify_claim_manifest


class PaperClaimVerificationTests(unittest.TestCase):
    def _fixture(self, root: Path, *, expected: float = 0.75) -> Path:
        artifact = root / "artifacts" / "result.json"
        artifact.parent.mkdir(parents=True)
        artifact.write_text(json.dumps({"metrics": {"score": 0.75}}), encoding="utf-8")
        digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
        manifest = root / "paper" / "claims.json"
        manifest.parent.mkdir(parents=True)
        manifest.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "claims": [
                        {
                            "id": "C1",
                            "claim": "a scoped result",
                            "scope": "fixture only",
                            "status": "supported_synthetic",
                            "evidence": [
                                {
                                    "artifact": "artifacts/result.json",
                                    "sha256": digest,
                                    "checks": [
                                        {
                                            "pointer": "/metrics/score",
                                            "expected": expected,
                                            "absolute_tolerance": 1e-9,
                                        }
                                    ],
                                }
                            ],
                        },
                        {
                            "id": "N1",
                            "claim": "an unsupported result",
                            "scope": "fixture only",
                            "status": "unsupported",
                            "evidence": [],
                        },
                    ],
                }
            ),
            encoding="utf-8",
        )
        return manifest

    def test_verifies_hash_pointer_metric_and_status_counts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest = self._fixture(Path(directory))
            report = verify_claim_manifest(manifest)
        self.assertTrue(report["verified"])
        self.assertEqual(2, report["claims"])
        self.assertEqual(1, report["artifacts"])
        self.assertEqual(1, report["metric_checks"])
        self.assertEqual(1, report["status_counts"]["unsupported"])

    def test_metric_drift_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest = self._fixture(Path(directory), expected=0.9)
            with self.assertRaisesRegex(ValueError, "metric mismatch"):
                verify_claim_manifest(manifest)

    def test_hash_tampering_and_path_escape_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self._fixture(root)
            artifact = root / "artifacts" / "result.json"
            artifact.write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                verify_claim_manifest(manifest)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self._fixture(root)
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            payload["claims"][0]["evidence"][0]["artifact"] = "../outside.json"
            manifest.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "escapes repository root"):
                verify_claim_manifest(manifest)


if __name__ == "__main__":
    unittest.main()
