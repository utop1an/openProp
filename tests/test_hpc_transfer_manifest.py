import hashlib
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.build_hpc_transfer_manifest import DEFAULT_FILES, build_manifest


class HPCTransferManifestTests(unittest.TestCase):
    def test_v3_bundle_names_the_versioned_ubuntu_image(self):
        self.assertIn("openprop-ai2thor-v3.sif", DEFAULT_FILES)
        self.assertIn("openprop-ai2thor-v3.packages.txt", DEFAULT_FILES)

    def test_manifest_hashes_every_required_artifact(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            for index, name in enumerate(DEFAULT_FILES):
                (root / name).write_bytes(f"artifact-{index}".encode())
            payload = build_manifest(root)
            self.assertFalse(payload["performance_evidence"])
            self.assertEqual(len(payload["files"]), len(DEFAULT_FILES))
            first = payload["files"][0]
            self.assertEqual(
                first["sha256"], hashlib.sha256(b"artifact-0").hexdigest()
            )

    def test_missing_artifact_fails_closed(self):
        with TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(FileNotFoundError, "missing HPC transfer artifact"):
                build_manifest(Path(temporary))


if __name__ == "__main__":
    unittest.main()
