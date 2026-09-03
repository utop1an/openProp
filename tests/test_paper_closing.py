from __future__ import annotations

from pathlib import Path
import unittest


class PaperClosingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.manuscript = Path("paper/manuscript.md").read_text(encoding="utf-8")
        self.normalized = " ".join(self.manuscript.split())
        self.audit = Path("paper/closing-section-audit.md").read_text(
            encoding="utf-8"
        )

    def test_manuscript_ends_with_scientific_discussion_and_conclusion(self) -> None:
        headings = (
            "## 7. Discussion",
            "## 8. Limitations and broader impact",
            "## 9. Conclusion",
        )
        positions = [self.manuscript.index(heading) for heading in headings]
        self.assertEqual(sorted(positions), positions)
        self.assertNotIn("## 8. Reproducibility checklist", self.manuscript)
        self.assertTrue(self.manuscript.rstrip().endswith("integrated effectiveness claim."))

    def test_closing_preserves_scope_and_negative_results(self) -> None:
        required = (
            "factorized statistical model, not the neural parameterization",
            "ALFRED result stops at typed-frame prediction",
            "does not establish naturalistic longitudinal effectiveness",
            "evaluation-only current truth",
            "correlated sources",
            "socially sensitive properties",
            "official floorplan-disjoint longitudinal evaluation",
        )
        for statement in required:
            self.assertIn(statement, self.normalized)

    def test_stale_table_reference_is_removed(self) -> None:
        self.assertNotIn("Table 4 therefore records", self.manuscript)
        self.assertIn("Table 7 therefore records", self.manuscript)

    def test_audit_maps_every_claim_and_keeps_release_checks_elsewhere(self) -> None:
        for claim_id in (
            "C1_TYPED_COMPOSITION",
            "C2_TYPED_COMPONENTS",
            "C3_DECISION_UTILITY",
            "C4_INTERVAL_CENSORING",
            "C5_EXTERNAL_LANGUAGE",
            "C6_FALSE_POSITIVE_OBSERVATIONS",
            "C7_RECURRENT_OBSERVATIONS",
            "C8_IRREGULAR_OBSERVATIONS",
            "N1_NEURAL_NECESSITY",
            "N2_REAL_WORLD_GROUNDING",
            "N3_GENERAL_ADAPTATION_SAFETY",
        ):
            self.assertIn(f"`{claim_id}`", self.audit)
        reproducibility = Path("paper/reproducibility.md").read_text(encoding="utf-8")
        self.assertIn("Release gate", reproducibility)
        self.assertIn("Five-dimension self-review", self.audit)


if __name__ == "__main__":
    unittest.main()
