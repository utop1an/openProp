from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


NAVY = "#183B56"
BLUE = "#2F6B9A"
PALE_BLUE = "#E8F2F8"
TEAL = "#238B7E"
PALE_TEAL = "#E5F4F1"
GOLD = "#D89B2B"
PALE_GOLD = "#FFF3D6"
RED = "#B44B52"
PALE_RED = "#FBEAEC"
INK = "#24323D"
MID = "#61717D"
LIGHT = "#F5F7F8"
WHITE = "#FFFFFF"


def _box(
    ax,
    x: float,
    y: float,
    width: float,
    height: float,
    *,
    face: str = WHITE,
    edge: str = "#CBD5DC",
    linewidth: float = 1.2,
    radius: float = 0.012,
    linestyle: str = "-",
    zorder: int = 2,
) -> FancyBboxPatch:
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle=f"round,pad=0.006,rounding_size={radius}",
        facecolor=face,
        edgecolor=edge,
        linewidth=linewidth,
        linestyle=linestyle,
        zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def _label(ax, x: float, y: float, text: str, **kwargs) -> None:
    defaults = {
        "ha": "left",
        "va": "center",
        "color": INK,
        "fontsize": 9.0,
        "zorder": 6,
    }
    defaults.update(kwargs)
    ax.text(x, y, text, **defaults)


def _arrow(
    ax,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = MID,
    linewidth: float = 1.4,
    linestyle: str = "-",
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=12,
            linewidth=linewidth,
            color=color,
            linestyle=linestyle,
            shrinkA=0,
            shrinkB=0,
            zorder=5,
        )
    )


def _panel_title(ax, x: float, letter: str, title: str, subtitle: str) -> None:
    _label(ax, x, 0.825, f"({letter})", fontsize=12.0, fontweight="bold", color=NAVY)
    _label(ax, x + 0.026, 0.825, title, fontsize=11.2, fontweight="bold", color=NAVY)
    _label(ax, x + 0.026, 0.785, subtitle, fontsize=7.8, color=MID)


def _pill(ax, x: float, y: float, text: str, *, face: str, edge: str) -> None:
    width = 0.0105 * len(text) + 0.020
    _box(ax, x, y - 0.018, width, 0.036, face=face, edge=edge, radius=0.018)
    _label(ax, x + width / 2, y, text, fontsize=7.6, ha="center", color=INK)


def _metric_bar(
    ax,
    x: float,
    y: float,
    width: float,
    value: float,
    *,
    color: str,
    label: str,
) -> None:
    _label(ax, x, y + 0.025, label, fontsize=7.8, color=MID)
    ax.add_patch(
        Rectangle((x, y - 0.008), width, 0.020, facecolor="#E1E7EA", edgecolor="none", zorder=3)
    )
    ax.add_patch(
        Rectangle((x, y - 0.008), width * value, 0.020, facecolor=color, edgecolor="none", zorder=4)
    )
    _label(ax, x + width + 0.010, y + 0.002, f"{value:.2f}", fontsize=8.2, fontweight="bold", color=color)


def build_teaser(output_dir: Path) -> tuple[Path, Path]:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.unicode_minus": False,
            "svg.fonttype": "none",
            "svg.hashsalt": "openprop-teaser-v1",
        }
    )
    fig = plt.figure(figsize=(14.2, 4.8), facecolor=WHITE)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    _label(
        ax,
        0.020,
        0.950,
        "Same language match, different evidence validity",
        fontsize=15.0,
        fontweight="bold",
        color=NAVY,
    )
    _label(
        ax,
        0.020,
        0.902,
        "OpenProp separates typed comparison from persistence and observation-process bias.",
        fontsize=9.3,
        color=MID,
    )
    ax.plot([0.020, 0.980], [0.865, 0.865], color="#D8E0E5", linewidth=1.0)

    # (a) One query, two semantically compatible candidates.
    _panel_title(ax, 0.020, "a", "Ambiguous historical evidence", "Both candidates satisfy the typed query.")
    _box(ax, 0.020, 0.675, 0.285, 0.075, face=PALE_GOLD, edge=GOLD)
    _label(ax, 0.036, 0.724, "QUERY", fontsize=7.2, fontweight="bold", color="#815B12")
    _label(ax, 0.036, 0.692, '“the mug beside the kettle”', fontsize=10.2, fontstyle="italic")

    for y, letter, age, schedule, face, edge, status in (
        (0.455, "mug A", "observed 3 h ago", "inspected every 0.5 h", PALE_TEAL, TEAL, "candidate"),
        (0.235, "mug B", "observed 4 h ago", "inspected every 4.0 h", PALE_BLUE, BLUE, "candidate"),
    ):
        _box(ax, 0.020, y, 0.285, 0.175, face=face, edge=edge, linewidth=1.4)
        _label(ax, 0.037, y + 0.137, letter, fontsize=10.3, fontweight="bold", color=edge)
        _label(ax, 0.287, y + 0.137, status, fontsize=7.3, ha="right", color=MID)
        _pill(ax, 0.037, y + 0.090, "type = Mug", face=WHITE, edge=edge)
        _pill(ax, 0.132, y + 0.090, "beside(kettle)", face=WHITE, edge=edge)
        _label(ax, 0.037, y + 0.042, age, fontsize=8.2)
        _label(ax, 0.168, y + 0.042, schedule, fontsize=7.8, color=MID)

    _label(ax, 0.020, 0.175, "Scene/schedule is persistence context, not a query cue.", fontsize=7.5, color=RED)

    # (b) Explicit decomposition and rank flip.
    _panel_title(ax, 0.350, "b", "OpenProp decision boundary", "Match is fixed; evidence validity changes the rank.")
    _box(ax, 0.350, 0.665, 0.325, 0.085, face=LIGHT, edge="#AAB8C1")
    _label(ax, 0.368, 0.722, "Typed match", fontsize=8.3, fontweight="bold")
    _label(ax, 0.655, 0.722, "A = 1.00     B = 1.00", fontsize=8.4, ha="right", family="DejaVu Sans Mono")
    _label(ax, 0.368, 0.688, "semantic parser → deterministic relation comparator", fontsize=7.7, color=MID)

    _box(ax, 0.350, 0.405, 0.325, 0.205, face=PALE_RED, edge=RED)
    _label(ax, 0.368, 0.574, "Detected-time naïve", fontsize=9.4, fontweight="bold", color=RED)
    _label(ax, 0.653, 0.574, "ranks B first", fontsize=8.4, ha="right", fontweight="bold", color=RED)
    _label(ax, 0.368, 0.535, "Treats discovery time as transition time", fontsize=8.0)
    _label(ax, 0.368, 0.500, "→ sparse inspection looks more persistent", fontsize=8.0, color=RED)
    _label(ax, 0.368, 0.455, "freshness(A, 3 h)  <  freshness(B, 4 h)", fontsize=8.1, family="DejaVu Sans Mono")
    _label(ax, 0.368, 0.424, "ranking error from observation frequency", fontsize=7.4, color=MID)

    _arrow(ax, (0.512, 0.397), (0.512, 0.350), color=TEAL, linewidth=1.7)
    _box(ax, 0.350, 0.105, 0.325, 0.235, face=PALE_TEAL, edge=TEAL, linewidth=1.7)
    _label(ax, 0.368, 0.303, "Interval-aware OpenProp", fontsize=9.5, fontweight="bold", color=TEAL)
    _label(ax, 0.653, 0.303, "ranks A first", fontsize=8.5, ha="right", fontweight="bold", color=TEAL)
    _label(ax, 0.368, 0.263, "Keeps the change inside its inspection interval", fontsize=8.0)
    _label(ax, 0.368, 0.228, "→ equal latent hazards stay equal", fontsize=8.0, color=TEAL)
    _label(ax, 0.368, 0.183, "freshness(A, 3 h)  >  freshness(B, 4 h)", fontsize=8.1, family="DejaVu Sans Mono")
    _label(ax, 0.368, 0.142, "audit = typed match × confidence × persistence × coverage", fontsize=7.4, color=MID)

    # (c) Controlled evidence with explicit scope and truth barrier.
    _panel_title(ax, 0.720, "c", "Controlled ranking confirmation", "Synthetic mechanism evidence; 10 untouched seeds.")
    _box(ax, 0.720, 0.425, 0.260, 0.325, face=WHITE, edge=NAVY, linewidth=1.5)
    _label(ax, 0.740, 0.710, "Overall Top-1", fontsize=9.1, fontweight="bold")
    _metric_bar(ax, 0.740, 0.650, 0.150, 0.55, color=RED, label="detected-time naïve")
    _metric_bar(ax, 0.740, 0.570, 0.150, 1.00, color=TEAL, label="interval-aware")
    _box(ax, 0.740, 0.463, 0.220, 0.060, face=PALE_TEAL, edge=TEAL, radius=0.010)
    _label(ax, 0.850, 0.493, "+0.450  [0.350, 0.500]", fontsize=9.2, ha="center", fontweight="bold", color=TEAL)
    _label(ax, 0.850, 0.445, "paired seed-bootstrap 95% CI", fontsize=7.0, ha="center", color=MID)

    _box(ax, 0.720, 0.235, 0.260, 0.140, face=LIGHT, edge="#AAB8C1")
    _label(ax, 0.740, 0.337, "Worst-scene Top-1", fontsize=7.7, color=MID)
    _label(ax, 0.960, 0.337, "0.10 → 1.00", fontsize=9.0, ha="right", fontweight="bold", color=TEAL)
    _label(ax, 0.740, 0.296, "Target-scene gap", fontsize=7.7, color=MID)
    _label(ax, 0.960, 0.296, "0.90 → 0.00", fontsize=9.0, ha="right", fontweight="bold", color=TEAL)
    _label(ax, 0.740, 0.254, "40 balanced analytic cases · 9/1/0 W/T/L", fontsize=7.2, color=MID)

    _box(ax, 0.720, 0.105, 0.260, 0.080, face=PALE_RED, edge=RED, linestyle="--")
    _label(ax, 0.740, 0.153, "current_truth = mug A", fontsize=8.1, family="DejaVu Sans Mono", fontweight="bold", color=RED)
    _label(ax, 0.960, 0.153, "evaluation only", fontsize=7.7, ha="right", color=RED)
    _label(ax, 0.740, 0.121, "never enters parsing, persistence fitting, or ranking", fontsize=7.2, color=RED)

    _label(
        ax,
        0.020,
        0.040,
        "OpenProp does not replace perception or mapping: an upstream memory supplies candidates and timestamped observations.",
        fontsize=7.7,
        color=MID,
    )
    _label(
        ax,
        0.980,
        0.040,
        "No real-world effectiveness claim",
        fontsize=7.7,
        ha="right",
        color=RED,
        fontweight="bold",
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    svg_path = output_dir / "openprop_teaser.svg"
    png_path = output_dir / "openprop_teaser.png"
    fig.savefig(svg_path, format="svg", bbox_inches=None, metadata={"Date": None})
    fig.savefig(
        png_path,
        format="png",
        dpi=220,
        bbox_inches=None,
        metadata={"Software": "OpenProp reproducible paper teaser"},
    )
    plt.close(fig)
    return svg_path, png_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the OpenProp paper teaser.")
    parser.add_argument("--output-dir", type=Path, default=Path("paper/figures"))
    args = parser.parse_args()
    svg_path, png_path = build_teaser(args.output_dir)
    print(f"svg: {svg_path}")
    print(f"png: {png_path}")


if __name__ == "__main__":
    main()

