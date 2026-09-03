import unittest

from scripts.download_visor_screening import annotation_names


class VisorScreeningDownloadTests(unittest.TestCase):
    def test_annotation_inventory_is_unique_and_sorted(self):
        page = b'P03_14.json P01_01.json P03_14.json'
        self.assertEqual(annotation_names(page), ["P01_01.json", "P03_14.json"])

    def test_empty_inventory_fails_closed(self):
        with self.assertRaises(ValueError):
            annotation_names(b"no resources")


if __name__ == "__main__":
    unittest.main()
