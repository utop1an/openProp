import unittest
from pathlib import Path


class PaperRelatedWorkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = Path(__file__).resolve().parents[1]
        cls.manuscript = (cls.root / "paper" / "manuscript.md").read_text(
            encoding="utf-8"
        )
        cls.audit = (cls.root / "paper" / "related-work-audit.md").read_text(
            encoding="utf-8"
        )

    def test_related_work_placeholder_is_removed(self) -> None:
        self.assertNotIn(
            "Organize by problem boundary rather than a list of papers",
            self.manuscript,
        )
        self.assertNotIn("For every subsection", self.manuscript)

    def test_direct_and_concurrent_neighbors_are_cited(self) -> None:
        required_urls = (
            "https://proceedings.mlr.press/v229/chang23b.html",
            "https://proceedings.mlr.press/v202/kurenkov23a.html",
            "https://arxiv.org/abs/2411.04999",
            "https://arxiv.org/abs/2608.04933",
            "https://arxiv.org/abs/2605.29879",
        )
        for url in required_urls:
            with self.subTest(url=url):
                self.assertIn(url, self.manuscript)
                self.assertIn(url, self.audit)

    def test_novelty_boundary_is_explicit(self) -> None:
        normalized = " ".join(self.manuscript.split())
        required_phrases = (
            "given a candidate set and timestamped historical observations",
            "not a general survival estimator",
            "complementary evidence-scoring layer",
        )
        for phrase in required_phrases:
            with self.subTest(phrase=phrase):
                self.assertIn(phrase, normalized)

    def test_audit_records_experimental_implications(self) -> None:
        self.assertIn("| Experimental implication |", self.audit)
        self.assertIn("no apples-to-oranges leaderboard comparison", self.audit)
        self.assertIn("The executable baseline family", self.audit)


if __name__ == "__main__":
    unittest.main()
