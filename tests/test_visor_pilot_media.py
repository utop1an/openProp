import tempfile
import unittest
import zipfile
from pathlib import Path

from scripts.download_visor_pilot_media import media_path, verify_zip


class VisorPilotMediaTests(unittest.TestCase):
    def test_media_path_is_confined(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = media_path(root, "train", "P01", "P01_01")
            self.assertEqual(target.name, "P01_01.zip")
            with self.assertRaises(ValueError):
                media_path(root, "train", "P01", "../escape")

    def test_zip_verification_rejects_path_escape(self):
        with tempfile.TemporaryDirectory() as directory:
            safe = Path(directory) / "safe.zip"
            with zipfile.ZipFile(safe, "w") as archive:
                archive.writestr("P01_01/frame.jpg", b"image")
            self.assertEqual(verify_zip(safe), 1)
            unsafe = Path(directory) / "unsafe.zip"
            with zipfile.ZipFile(unsafe, "w") as archive:
                archive.writestr("../frame.jpg", b"image")
            with self.assertRaises(ValueError):
                verify_zip(unsafe)


if __name__ == "__main__":
    unittest.main()
