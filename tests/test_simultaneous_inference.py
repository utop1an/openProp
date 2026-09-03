import unittest

from openprop.simultaneous_inference import paired_bootstrap_simultaneous_intervals


class SimultaneousInferenceTests(unittest.TestCase):
    def test_shared_family_is_deterministic_and_contains_each_mean(self) -> None:
        family = {
            "subject": (0.01, 0.03, 0.02, 0.04),
            "relation": (0.10, 0.15, 0.12, 0.17),
            "scene": (0.30, 0.45, 0.35, 0.50),
        }
        first = paired_bootstrap_simultaneous_intervals(
            family, samples=500, seed=17
        )
        second = paired_bootstrap_simultaneous_intervals(
            family, samples=500, seed=17
        )
        self.assertEqual(first, second)
        self.assertEqual(3, first["family_size"])
        self.assertTrue(first["shared_resample_indices"])
        for name, values in family.items():
            lower, upper = first["intervals"][name]
            mean = sum(values) / len(values)
            self.assertLessEqual(lower, mean)
            self.assertGreaterEqual(upper, mean)

    def test_constant_comparison_has_exact_interval(self) -> None:
        report = paired_bootstrap_simultaneous_intervals(
            {"constant": (0.5, 0.5, 0.5), "variable": (0.1, 0.2, 0.4)},
            samples=200,
            seed=3,
        )
        self.assertEqual([0.5, 0.5], report["intervals"]["constant"])

    def test_invalid_family_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "same nonzero"):
            paired_bootstrap_simultaneous_intervals(
                {"a": (1.0,), "b": (1.0, 2.0)}, samples=10
            )
        with self.assertRaisesRegex(ValueError, "confidence"):
            paired_bootstrap_simultaneous_intervals(
                {"a": (1.0, 2.0)}, samples=10, confidence=1.0
            )


if __name__ == "__main__":
    unittest.main()
