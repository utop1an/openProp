import json
import tempfile
import unittest
from pathlib import Path

from scripts.download_adt_screening import build_plan, select_sequences


class ADTScreeningTests(unittest.TestCase):
    def test_selection_is_activity_bounded_and_url_free(self):
        manifest = {
            "sequences": {
                "Apartment_release_meal_seq002_M1": {
                    "main_groundtruth": {
                        "file_size_bytes": 20,
                        "download_url": "secret-two",
                    }
                },
                "Apartment_release_clean_seq001_M1": {
                    "main_groundtruth": {
                        "file_size_bytes": 10,
                        "download_url": "secret-one",
                    }
                },
                "Apartment_release_work_seq003_M1": {
                    "main_groundtruth": {"file_size_bytes": 30}
                },
            }
        }
        selected = select_sequences(manifest, ("clean", "meal"))
        self.assertEqual(
            [row["sequence_name"] for row in selected],
            [
                "Apartment_release_clean_seq001_M1",
                "Apartment_release_meal_seq002_M1",
            ],
        )
        self.assertNotIn("download_url", json.dumps(selected))

    def test_plan_hashes_manifest_without_copying_credentials(self):
        manifest = {
            "sequences": {
                "Apartment_release_clean_seq001_M1": {
                    "main_groundtruth": {
                        "file_size_bytes": 10,
                        "download_url": "secret-one",
                    }
                }
            }
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "access.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            plan = build_plan(path, manifest, ("clean",))
        encoded = json.dumps(plan)
        self.assertEqual(plan["declared_download_bytes"], 10)
        self.assertFalse(plan["contains_download_urls"])
        self.assertNotIn("secret-one", encoded)


if __name__ == "__main__":
    unittest.main()
