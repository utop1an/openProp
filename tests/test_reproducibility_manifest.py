import copy
import json
import tempfile
import unittest
from pathlib import Path

from openprop.reproducibility import (
    build_reproducibility_manifest,
    verify_reproducibility_manifest,
)


class ReproducibilityManifestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = Path(__file__).resolve().parents[1]
        cls.manifest = cls.root / "paper" / "reproducibility_manifest.json"

    def _write_payload(self, payload: dict) -> Path:
        temporary = tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", encoding="utf-8", delete=False
        )
        with temporary:
            json.dump(payload, temporary, indent=2)
        self.addCleanup(Path(temporary.name).unlink, missing_ok=True)
        return Path(temporary.name)

    def test_checked_in_manifest_matches_current_snapshot_and_runtime(self) -> None:
        report = verify_reproducibility_manifest(
            self.manifest,
            repository_root=self.root,
            require_runtime_match=True,
        )
        self.assertTrue(report["verified"])
        self.assertGreater(report["source_files"], 100)
        self.assertEqual(report["experiments"], 11)
        self.assertGreaterEqual(report["experiment_outputs"], 13)
        self.assertEqual(report["external_audits"], 1)
        self.assertEqual(report["external_audit_outputs"], 1)

    def test_build_is_deterministic_and_matches_checked_in_manifest(self) -> None:
        first = build_reproducibility_manifest(self.root)
        second = build_reproducibility_manifest(self.root)
        checked_in = json.loads(self.manifest.read_text(encoding="utf-8"))
        self.assertEqual(first, second)
        self.assertEqual(first, checked_in)

    def test_paper_package_exposes_fail_closed_release_contract(self) -> None:
        manuscript = (self.root / "paper" / "manuscript.md").read_text(encoding="utf-8")
        paper_readme = (self.root / "paper" / "README.md").read_text(encoding="utf-8")
        hierarchy = (self.root / "paper" / "claim-hierarchy.md").read_text(encoding="utf-8")
        for required in (
            "paper/reproducibility_manifest.json",
            "clean Git revision",
            "content hash alone is not a submission",
        ):
            with self.subTest(required=required):
                self.assertIn(required, manuscript)
        self.assertIn("--require-runtime-match", paper_readme)
        self.assertIn("build_paper_tables.py --check", paper_readme)
        self.assertIn("--require-git-revision", hierarchy)

    def test_source_hash_and_path_tampering_fail_closed(self) -> None:
        payload = json.loads(self.manifest.read_text(encoding="utf-8"))
        tampered = copy.deepcopy(payload)
        tampered["source_snapshot"]["files"][0]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "source file hash drifted"):
            verify_reproducibility_manifest(
                self._write_payload(tampered), repository_root=self.root
            )
        escaped = copy.deepcopy(payload)
        escaped["source_snapshot"]["files"][0]["path"] = "../outside"
        with self.assertRaisesRegex(ValueError, "file inventory drifted"):
            verify_reproducibility_manifest(
                self._write_payload(escaped), repository_root=self.root
            )

    def test_output_hash_runtime_and_revision_gates_fail_closed(self) -> None:
        payload = json.loads(self.manifest.read_text(encoding="utf-8"))
        output_tampered = copy.deepcopy(payload)
        output_tampered["experiments"][0]["outputs"][0]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "experiment output drifted"):
            verify_reproducibility_manifest(
                self._write_payload(output_tampered), repository_root=self.root
            )
        runtime_tampered = copy.deepcopy(payload)
        runtime_tampered["environment"]["python"]["version"] = "0.0.0"
        with self.assertRaisesRegex(ValueError, "runtime does not match"):
            verify_reproducibility_manifest(
                self._write_payload(runtime_tampered),
                repository_root=self.root,
                require_runtime_match=True,
            )
        self.assertFalse(payload["release_gates"]["clean_git_revision_bound"])
        with self.assertRaisesRegex(ValueError, "clean bound git revision"):
            verify_reproducibility_manifest(
                self.manifest,
                repository_root=self.root,
                require_release_revision=True,
            )

    def test_external_access_audit_hash_fails_closed(self) -> None:
        payload = json.loads(self.manifest.read_text(encoding="utf-8"))
        tampered = copy.deepcopy(payload)
        tampered["external_audits"][0]["output"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "external audit drifted"):
            verify_reproducibility_manifest(
                self._write_payload(tampered), repository_root=self.root
            )


if __name__ == "__main__":
    unittest.main()

