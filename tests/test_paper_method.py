from __future__ import annotations

import math
from pathlib import Path
import unittest

from openprop import (
    Entity,
    EntityMatcher,
    MentionBasedSelector,
    Observation,
    PropertyConstraint,
    PropertyDefinition,
    PropertyRegistry,
    QueryFrame,
    ValueType,
    default_comparators,
)
from openprop.models import ComparisonResult


class PaperMethodContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = PropertyRegistry()
        self.registry.register(PropertyDefinition("type", "object type", ValueType.CATEGORICAL))

    def matcher(self, *, coverage_power: float = 1.0, comparators=None) -> EntityMatcher:
        return EntityMatcher(
            self.registry,
            comparators or default_comparators(),
            MentionBasedSelector(),
            coverage_power=coverage_power,
        )

    def test_coverage_power_must_be_finite_and_nonnegative(self) -> None:
        for value in (-1.0, math.inf, math.nan):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.matcher(coverage_power=value)

    def test_custom_comparator_cannot_escape_unit_interval(self) -> None:
        comparators = default_comparators()
        comparators.register(
            ValueType.CATEGORICAL.value,
            lambda *_: ComparisonResult(1.01, "invalid custom score"),
        )
        query = QueryFrame("the cup", (PropertyConstraint("type", "cup"),))
        entity = Entity("cup", {"type": Observation("cup")})
        with self.assertRaisesRegex(ValueError, "comparison score"):
            self.matcher(comparators=comparators).match(query, [entity], as_of=0.0)

    def test_manuscript_freezes_the_executable_method_contract(self) -> None:
        manuscript = Path("paper/manuscript.md").read_text(encoding="utf-8")
        forbidden_placeholders = (
            "Define entities, typed properties",
            "Write the scoring equation",
            "Explain controlled schema growth",
            "Describe query parsing and positive-evidence",
        )
        for placeholder in forbidden_placeholders:
            self.assertNotIn(placeholder, manuscript)
        required = (
            "a_{ik}=w_k\\mathbf{1}[z_{ik}=\\mathrm{obs}]c_{ik}f_{ik}(T)",
            "s_i=M_i C_i^{\\gamma}",
            "O(|\\mathcal{E}||Q|)",
            "right-censored",
            "interval-censored",
            "evaluation-only",
            "exact ties use lexical",
        )
        for statement in required:
            self.assertIn(statement, manuscript)

    def test_method_audit_maps_symbols_to_implementation(self) -> None:
        audit = Path("paper/method-implementation-audit.md").read_text(encoding="utf-8")
        for implementation in (
            "src/openprop/matcher.py",
            "src/openprop/comparators.py",
            "src/openprop/statistical_persistence.py",
            "src/openprop/observation_history.py",
            "src/openprop/llm.py",
        ):
            self.assertIn(implementation, audit)
        self.assertIn("Unknown and not-applicable", audit)
        self.assertIn("current_truth", audit)


if __name__ == "__main__":
    unittest.main()
