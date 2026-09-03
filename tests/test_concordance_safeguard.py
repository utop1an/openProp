from __future__ import annotations

import math
import unittest
from dataclasses import dataclass

from openprop.concordance_safeguard import apply_concordance_safeguard
from openprop.persistence_data import PersistenceTrainingExample


@dataclass(frozen=True)
class _RiskTable:
    values: dict[str, float]
    activated: bool = True

    def hazard_per_hour(self, features: tuple[str, ...]) -> float:
        return self.values[features[1]]


def _calibration() -> tuple[PersistenceTrainingExample, ...]:
    return (
        PersistenceTrainingExample(
            "location", "fast", "inside", "drawer", "busy", 100.0, True, "a"
        ),
        PersistenceTrainingExample(
            "location", "middle", "inside", "drawer", "busy", 200.0, True, "b"
        ),
        PersistenceTrainingExample(
            "location", "slow", "inside", "drawer", "busy", 400.0, False, "c"
        ),
    )


class ConcordanceSafeguardTests(unittest.TestCase):
    def test_rejects_candidate_that_regresses_calibration_ranking(self):
        source = _RiskTable({"fast": 3.0, "middle": 2.0, "slow": 1.0})
        reversed_model = _RiskTable({"fast": 1.0, "middle": 2.0, "slow": 3.0})
        guarded = apply_concordance_safeguard(
            source, reversed_model, _calibration()
        )
        self.assertFalse(guarded.accepted)
        self.assertLess(guarded.concordance_delta, 0.0)
        self.assertEqual(
            source.hazard_per_hour(("location", "fast", "inside", "drawer", "busy")),
            guarded.hazard_per_hour(("location", "fast", "inside", "drawer", "busy")),
        )

    def test_accepts_noninferior_active_candidate(self):
        source = _RiskTable({"fast": 3.0, "middle": 2.0, "slow": 1.0})
        scaled = _RiskTable({"fast": 6.0, "middle": 4.0, "slow": 2.0})
        guarded = apply_concordance_safeguard(source, scaled, _calibration())
        self.assertTrue(guarded.accepted)
        self.assertEqual(0.0, guarded.concordance_delta)
        self.assertEqual(6.0, guarded.hazard_per_hour(("x", "fast", "x", "x", "x")))

    def test_inactive_candidate_and_stricter_margin_fail_closed(self):
        source = _RiskTable({"fast": 3.0, "middle": 2.0, "slow": 1.0})
        inactive = _RiskTable(
            {"fast": 6.0, "middle": 4.0, "slow": 2.0}, activated=False
        )
        self.assertFalse(
            apply_concordance_safeguard(source, inactive, _calibration()).accepted
        )
        self.assertFalse(
            apply_concordance_safeguard(
                source,
                _RiskTable({"fast": 6.0, "middle": 4.0, "slow": 2.0}),
                _calibration(),
                minimum_concordance_delta=1e-3,
            ).accepted
        )

    def test_invalid_inputs_are_rejected(self):
        source = _RiskTable({"fast": 3.0, "middle": 2.0, "slow": 1.0})
        with self.assertRaisesRegex(ValueError, "requires calibration"):
            apply_concordance_safeguard(source, source, ())
        with self.assertRaisesRegex(ValueError, "finite"):
            apply_concordance_safeguard(
                source, source, _calibration(), minimum_concordance_delta=math.nan
            )


if __name__ == "__main__":
    unittest.main()
