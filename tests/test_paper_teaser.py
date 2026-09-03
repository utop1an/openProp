import hashlib
import tempfile
import unittest
from pathlib import Path

from scripts.build_paper_teaser import build_teaser


class PaperTeaserTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(__file__).resolve().parents[1]

    def test_build_is_byte_deterministic_and_matches_checked_in_assets(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            temporary_root = Path(temporary)
            first = build_teaser(temporary_root / "first")
            second = build_teaser(temporary_root / "second")
            self.assertEqual(
                [path.read_bytes() for path in first],
                [path.read_bytes() for path in second],
            )
            checked_in = self.root / "paper" / "figures"
            for generated in first:
                self.assertEqual(
                    hashlib.sha256(generated.read_bytes()).hexdigest(),
                    hashlib.sha256((checked_in / generated.name).read_bytes()).hexdigest(),
                )

    def test_teaser_keeps_task_result_and_scope_boundaries_visible(self) -> None:
        svg = (self.root / "paper" / "figures" / "openprop_teaser.svg").read_text(
            encoding="utf-8"
        )
        for required in (
            "Same language match, different evidence validity",
            "Detected-time naïve",
            "Interval-aware OpenProp",
            "+0.450  [0.350, 0.500]",
            "Synthetic mechanism evidence",
            "current_truth",
            "No real-world effectiveness claim",
        ):
            with self.subTest(required=required):
                self.assertIn(required, svg)

    def test_manuscript_embeds_teaser_before_introduction(self) -> None:
        manuscript = (self.root / "paper" / "manuscript.md").read_text(encoding="utf-8")
        teaser = "![OpenProp current-evidence grounding teaser](figures/openprop_teaser.svg)"
        self.assertIn(teaser, manuscript)
        self.assertLess(manuscript.index(teaser), manuscript.index("## 1. Introduction"))
        self.assertIn("synthetic controlled confirmation", " ".join(manuscript.split()))
        self.assertIn("**Figure 1: Equal typed match does not imply equal current evidence.**", manuscript)
        self.assertIn("**Figure 2: OpenProp resolves language", manuscript)
        self.assertNotIn("current target", (self.root / "paper" / "figures" / "openprop_teaser.svg").read_text(encoding="utf-8"))
        normalized = " ".join(manuscript.split())
        self.assertIn(
            "These results validate the mechanisms and evaluation boundary; integrated semi-real longitudinal evidence remains a required TEACh experiment.", normalized
        )


if __name__ == "__main__":
    unittest.main()
