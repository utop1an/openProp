import json
import tempfile
import unittest
from pathlib import Path

from scripts.build_paper_tables import (
    _embedded_table_content,
    build_paper_tables,
)


class PaperTableTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parents[1]
        self.manifest = self.root / "paper" / "claims.json"

    def test_generation_is_deterministic_and_matches_checked_in_outputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            temporary_root = Path(temporary)
            first = build_paper_tables(
                self.manifest,
                temporary_root / "first",
                repository_root=self.root,
            )
            second = build_paper_tables(
                self.manifest,
                temporary_root / "second",
                repository_root=self.root,
            )
            self.assertEqual(
                [path.read_bytes() for path in first],
                [path.read_bytes() for path in second],
            )
            checked_in = self.root / "paper" / "tables"
            for generated in first:
                self.assertEqual(
                    generated.read_bytes(),
                    (checked_in / generated.name).read_bytes(),
                )

    def test_unbound_table_pointer_fails_closed(self):
        payload = json.loads(self.manifest.read_text(encoding="utf-8"))
        claim = next(row for row in payload["claims"] if row["id"] == "C1_TYPED_COMPOSITION")
        checks = claim["evidence"][0]["checks"]
        claim["evidence"][0]["checks"] = [
            row
            for row in checks
            if row["pointer"]
            != "/aggregate/global/concordance_index/standard_deviation"
        ]
        with tempfile.TemporaryDirectory() as temporary:
            temporary_root = Path(temporary)
            altered = temporary_root / "claims.json"
            altered.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "not claim-bound"):
                build_paper_tables(
                    altered,
                    temporary_root / "output",
                    repository_root=self.root,
                )

    def test_tables_keep_scope_directions_and_negative_results_visible(self):
        table_root = self.root / "paper" / "tables"
        main = (table_root / "controlled_compositional_results.md").read_text(
            encoding="utf-8"
        )
        boundary = (table_root / "claim_boundaries.md").read_text(encoding="utf-8")
        for required in (
            "NLL ↓",
            "C-index ↑",
            "IBS ↓",
            "Grounding Top-1 ↑",
            "mean and standard deviation",
            "synthetic mechanism validation",
        ):
            self.assertIn(required, main)
        for required in (
            "Neural persistence is necessary",
            "Semi-real longitudinal effectiveness",
            "Calibration-only adaptation is generally safe",
            "Contradicted",
            "Pending",
            "-0.363",
        ):
            self.assertIn(required, boundary)

    def test_manuscript_embeds_exact_generated_tables(self):
        manuscript = (self.root / "paper" / "manuscript.md").read_text(
            encoding="utf-8"
        )
        table_root = self.root / "paper" / "tables"
        for block, filename in (
            ("controlled-compositional", "controlled_compositional_results.md"),
            ("typed-component-ablation", "typed_component_ablation.md"),
            ("controlled-decision-utility", "controlled_decision_utility.md"),
            ("observation-process-bias", "observation_process_bias.md"),
            ("observation-grounding-decisions", "observation_grounding_decisions.md"),
            ("external-language-results", "external_language_results.md"),
            ("claim-boundaries", "claim_boundaries.md"),
        ):
            expected = (table_root / filename).read_text(encoding="utf-8").strip()
            self.assertEqual(expected, _embedded_table_content(manuscript, block))

    def test_latex_uses_minimal_rules_and_ascii_safe_source(self):
        table_root = self.root / "paper" / "tables"
        for filename in (
            "controlled_compositional_results.tex",
            "typed_component_ablation.tex",
            "controlled_decision_utility.tex",
            "observation_process_bias.tex",
            "observation_grounding_decisions.tex",
            "external_language_results.tex",
            "claim_boundaries.tex",
        ):
            source = (table_root / filename).read_text(encoding="utf-8")
            source.encode("ascii")
            self.assertIn("\\toprule", source)
            self.assertIn("\\midrule", source)
            self.assertIn("\\bottomrule", source)
            self.assertNotIn("\\hline", source)
            self.assertLess(source.index("\\caption{"), source.index("\\label{"))

    def test_component_tables_separate_calibration_from_decision_utility(self):
        table_root = self.root / "paper" / "tables"
        calibration = (table_root / "typed_component_ablation.md").read_text(
            encoding="utf-8"
        )
        decision = (table_root / "controlled_decision_utility.md").read_text(
            encoding="utf-8"
        )
        for required in (
            "NLL(ablation) minus NLL(full)",
            "predeclared primary metric",
            "synthetic mechanism validation",
            "family-wise simultaneous 95% bootstrap intervals",
            "0.052 [0.038, 0.066]",
            "0.402 [0.364, 0.441]",
        ):
            self.assertIn(required, calibration)
        for required in (
            "40 analytic confidence-age crossover cases",
            "ten untouched confirmation seeds",
            "synthetic controlled decision evidence",
            "family-wise simultaneous 95% bootstrap intervals",
            "0.347 [0.329, 0.364]",
            "0.500 [0.500, 0.500]",
        ):
            self.assertIn(required, decision)

    def test_secondary_tables_lock_protocols_uncertainty_and_scope(self):
        table_root = self.root / "paper" / "tables"
        observation = (table_root / "observation_process_bias.md").read_text(
            encoding="utf-8"
        )
        grounding = (table_root / "observation_grounding_decisions.md").read_text(
            encoding="utf-8"
        )
        language = (table_root / "external_language_results.md").read_text(
            encoding="utf-8"
        )
        for required in (
            "Hazard MAE ↓",
            "Schedule gap ↓",
            "Exact test NLL ↓",
            "true hazard 0.25/h",
            "inspection every 0.5 or 4.0 h",
            "600 train and 400 exact-time test samples",
            "0.056 ± 0.004",
            "0.004 ± 0.004",
            "not evidence for arbitrary missingness",
        ):
            self.assertIn(required, observation)
        for required in (
            "Overall Top-1 ↑",
            "Worst-scene Top-1 ↑",
            "Target-scene gap ↓",
            "40 fixed cases balance target scene (20/20)",
            "600 training histories per scene",
            "10 untouched confirmation seeds",
            "Scene affects persistence context but is not queried",
            "0.550 ± 0.150",
            "**1.000 ± 0.000**",
            "**+0.450 [0.350, 0.500]**",
            "**+0.900 [0.700, 1.000]**",
            "**−0.900 [−1.000, −0.700]**",
            "**9/1/0**",
            "not natural prevalence or real-world grounding",
        ):
            self.assertIn(required, grounding)
        for required in (
            "Property F1 ↑",
            "Value recall ↑",
            "Exact frame ↑",
            "fixed train-only BM25",
            "No validation labels",
            "Valid-seen (487)",
            "Valid-unseen (458)",
            "**0.736**",
            "**+0.124 [0.071, 0.177]**",
            "frozen 40-case valid-unseen confirmation sample",
            "Gemma 3 4B",
            "Llama 3.2",
            "language-to-frame parsing only",
            "not visual, temporal, or end-to-end grounding",
        ):
            self.assertIn(required, language)

    def test_markdown_table_numbers_follow_manuscript_order(self):
        table_root = self.root / "paper" / "tables"
        ordered = (
            "controlled_compositional_results.md",
            "typed_component_ablation.md",
            "controlled_decision_utility.md",
            "observation_process_bias.md",
            "observation_grounding_decisions.md",
            "external_language_results.md",
            "claim_boundaries.md",
        )
        for number, filename in enumerate(ordered, start=1):
            source = (table_root / filename).read_text(encoding="utf-8")
            self.assertIn(f"**Table {number}.", source)


if __name__ == "__main__":
    unittest.main()
