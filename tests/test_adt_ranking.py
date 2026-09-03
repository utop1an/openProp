import unittest

from scripts.rank_adt_sequences import euclidean, max_displacement


class ADTRankingTests(unittest.TestCase):
    def test_max_displacement_is_timestamp_ordered(self):
        points = [
            (30, (0.0, 0.0, 0.0)),
            (10, (1.0, 1.0, 1.0)),
            (20, (2.0, 1.0, 1.0)),
        ]
        self.assertAlmostEqual(max_displacement(points), 3**0.5)

    def test_euclidean_uses_three_typed_coordinates(self):
        self.assertEqual(euclidean((0.0, 0.0, 0.0), (1.0, 2.0, 2.0)), 3.0)


if __name__ == "__main__":
    unittest.main()
