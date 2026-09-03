from __future__ import annotations

import json
from pathlib import Path
import unittest


class PaperExperimentalProtocolTests(unittest.TestCase):
    def setUp(self) -> None:
        manuscript = Path("paper/manuscript.md").read_text(encoding="utf-8")
        self.section = manuscript.split("## 4. Experimental protocol", 1)[1].split(
            "## 5. Results", 1
        )[0]
        self.normalized_section = " ".join(self.section.split())
        self.audit = Path("paper/experimental-protocol-audit.md").read_text(
            encoding="utf-8"
        )

    def test_protocol_replaces_outline_placeholders(self) -> None:
        for placeholder in (
            "Document entity-grouped splits",
            "Use global decay, no-decay/static matching",
        ):
            self.assertNotIn(placeholder, self.section)
        for heading in (
            "### 4.1 Evaluation questions",
            "### 4.2 Controlled typed persistence and decisions",
            "### 4.3 Observation-process intervention",
            "### 4.4 External language protocol",
            "### 4.5 Baselines, metrics, and uncertainty",
            "### 4.6 Pending official longitudinal protocol",
        ):
            self.assertIn(heading, self.section)

    def test_protocol_names_populations_analysis_units_and_denominators(self) -> None:
        required = (
            "18 typed contexts",
            "960/240/240",
            "40 fixed analytic cases",
            "20 old-target and 20 new-target",
            "600 training histories per schedule",
            "11,974",
            "945 supported validation descriptions",
            "20,000 deterministic resamples",
            "task ID",
            "language remains templated",
            "\\{0.25,1,4,16,64,256\\}",
            "All failures remain in their predeclared denominators",
            "floorplan-disjoint",
        )
        for statement in required:
            self.assertIn(statement, self.normalized_section)

    def test_every_claim_has_an_audited_protocol_and_scope(self) -> None:
        claims = json.loads(Path("paper/claims.json").read_text(encoding="utf-8"))[
            "claims"
        ]
        for claim in claims:
            self.assertIn(f"`{claim['id']}`", self.audit)
        for boundary in (
            "Synthetic mechanism validation only",
            "Language-to-frame parsing only",
            "Pending external evidence",
            "Contradicted as a general safety claim",
        ):
            self.assertIn(boundary, self.audit)

    def test_protocol_keeps_truth_and_external_evidence_boundaries_explicit(self) -> None:
        for statement in (
            "`current_truth` is evaluation-only",
            "contains no entity observations",
            "No official TEACh metric appears",
            "not a submission-ready result",
        ):
            self.assertIn(statement, self.normalized_section)


if __name__ == "__main__":
    unittest.main()
