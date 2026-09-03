import importlib.util
import tempfile
import unittest
from pathlib import Path


@unittest.skipUnless(
    importlib.util.find_spec("matplotlib"),
    "paper figure requires the optional paper dependency",
)
class PaperPipelineFigureTests(unittest.TestCase):
    def test_build_is_deterministic_and_encodes_method_boundaries(self):
        from scripts.build_paper_pipeline_figure import build_figure

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first_svg, first_png = build_figure(root / "first")
            second_svg, second_png = build_figure(root / "second")

            self.assertEqual(first_svg.read_bytes(), second_svg.read_bytes())
            self.assertEqual(first_png.read_bytes(), second_png.read_bytes())
            svg = first_svg.read_text(encoding="utf-8")
            for required in (
                "OpenProp decision boundary",
                "language → typed QueryFrame",
                "unknown → missing",
                "not a mismatch",
                "interval-censored changes",
                "right-censored unchanged histories",
                "current_truth",
                "never matcher input",
                "not benchmark results",
            ):
                self.assertIn(required, svg)

    def test_checked_in_outputs_exist_and_are_nontrivial(self):
        root = Path(__file__).resolve().parents[1]
        svg = root / "paper" / "figures" / "openprop_task_pipeline.svg"
        png = root / "paper" / "figures" / "openprop_task_pipeline.png"
        self.assertGreater(svg.stat().st_size, 20_000)
        self.assertGreater(png.stat().st_size, 100_000)
        self.assertEqual(b"\x89PNG\r\n\x1a\n", png.read_bytes()[:8])


if __name__ == "__main__":
    unittest.main()
