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


def _arrow(
    ax,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = MID,
    linewidth: float = 1.35,
    style: str = "-|>",
    connectionstyle: str = "arc3,rad=0",
    linestyle: str = "-",
    zorder: int = 4,
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle=style,
            mutation_scale=11,
            linewidth=linewidth,
            color=color,
            connectionstyle=connectionstyle,
            linestyle=linestyle,
            shrinkA=0,
            shrinkB=0,
            zorder=zorder,
        )
    )


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


def _panel_label(ax, x: float, y: float, letter: str, title: str) -> None:
    _label(ax, x, y, f"({letter})", fontsize=12.5, fontweight="bold", color=NAVY)
    _label(ax, x + 0.026, y, title, fontsize=11.4, fontweight="bold", color=NAVY)


def _score_bar(ax, x: float, y: float, width: float, value: float, color: str) -> None:
    ax.add_patch(Rectangle((x, y), width, 0.013, facecolor="#E4E9EC", edgecolor="none", zorder=5))
    ax.add_patch(
        Rectangle((x, y), width * value, 0.013, facecolor=color, edgecolor="none", zorder=6)
    )


def build_figure(output_dir: Path) -> tuple[Path, Path]:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.unicode_minus": False,
            "svg.fonttype": "none",
            "svg.hashsalt": "openprop-main-pipeline-v1",
        }
    )
    fig = plt.figure(figsize=(14.2, 7.2), facecolor=WHITE)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Runtime band and panel headings.
    _label(ax, 0.018, 0.965, "RUNTIME DECISION PATH", fontsize=8.6, fontweight="bold", color=MID)
    ax.plot([0.018, 0.982], [0.945, 0.945], color="#D8E0E5", linewidth=1.0, zorder=1)
    _panel_label(ax, 0.020, 0.915, "a", "Stale, typed memory")
    _panel_label(ax, 0.285, 0.915, "b", "OpenProp decision boundary")
    _panel_label(ax, 0.790, 0.915, "c", "Ranked, auditable output")

    # Panel (a): upstream language and memory, explicitly outside OpenProp.
    _box(ax, 0.020, 0.770, 0.235, 0.095, face=PALE_GOLD, edge=GOLD)
    _label(ax, 0.034, 0.840, "Language query", fontsize=8.2, fontweight="bold", color="#815B12")
    _label(
        ax,
        0.034,
        0.802,
        '“the mug still beside the kettle”',
        fontsize=10.0,
        fontstyle="italic",
    )

    _box(ax, 0.020, 0.460, 0.235, 0.270, face=LIGHT, edge="#BFCBD2")
    _label(ax, 0.034, 0.700, "Dynamic memory / scene graph", fontsize=9.4, fontweight="bold")
    _label(ax, 0.034, 0.674, "supplies observations", fontsize=7.5, color=MID)
    _label(ax, 0.034, 0.653, "perception and mapping stay upstream", fontsize=7.5, color=MID)
    ax.plot([0.055, 0.220], [0.585, 0.585], color="#9DABB4", linewidth=1.2, zorder=3)
    for x, time_text in ((0.062, "t₁"), (0.132, "t₂"), (0.211, "tq")):
        ax.scatter([x], [0.585], s=30, color=BLUE if time_text != "tq" else GOLD, zorder=5)
        _label(ax, x, 0.560, time_text, fontsize=8.0, ha="center", color=MID)
    _box(ax, 0.036, 0.596, 0.088, 0.039, face=PALE_BLUE, edge=BLUE, radius=0.008)
    _label(ax, 0.080, 0.616, "beside(mug, kettle)", fontsize=7.1, ha="center")
    _box(ax, 0.106, 0.500, 0.104, 0.045, face=PALE_TEAL, edge=TEAL, radius=0.008)
    _label(ax, 0.158, 0.523, "isDirty(mug)=false", fontsize=7.1, ha="center")
    _label(ax, 0.034, 0.482, "Observations retain type, confidence,", fontsize=7.4, color=MID)
    _label(ax, 0.034, 0.463, "source, and timestamp.", fontsize=7.4, color=MID)

    _box(ax, 0.020, 0.330, 0.235, 0.090, face=WHITE, edge="#BFCBD2")
    _label(ax, 0.034, 0.390, "Candidate entities at query time", fontsize=8.7, fontweight="bold")
    _label(ax, 0.034, 0.360, "mug-1   mug-2   bowl-1   …", fontsize=9.7, family="DejaVu Sans Mono")
    _label(ax, 0.034, 0.337, "partial evidence; different ages", fontsize=7.6, color=MID)

    # Arrows entering the OpenProp boundary.
    _arrow(ax, (0.255, 0.817), (0.296, 0.817), color=GOLD)
    _arrow(ax, (0.255, 0.530), (0.296, 0.530), color=BLUE)
    _arrow(ax, (0.255, 0.373), (0.296, 0.373), color=BLUE)

    # Panel (b): the method boundary.
    _box(ax, 0.290, 0.300, 0.475, 0.565, face="#FBFDFE", edge=BLUE, linewidth=2.0, radius=0.018)
    _label(ax, 0.305, 0.842, "semantic step", fontsize=7.7, fontweight="bold", color=BLUE)
    _box(ax, 0.305, 0.735, 0.195, 0.090, face=PALE_BLUE, edge=BLUE)
    _label(ax, 0.402, 0.798, "Query parser", fontsize=9.5, fontweight="bold", ha="center")
    _label(ax, 0.402, 0.770, "language → typed QueryFrame", fontsize=8.0, ha="center")
    _label(ax, 0.402, 0.746, "no candidate access", fontsize=7.4, ha="center", color=MID)
    _arrow(ax, (0.500, 0.780), (0.530, 0.780), color=BLUE)
    _box(ax, 0.530, 0.720, 0.215, 0.120, face=WHITE, edge=BLUE)
    _label(ax, 0.545, 0.812, "Typed constraints", fontsize=8.5, fontweight="bold")
    _label(ax, 0.545, 0.782, "type = Mug", fontsize=8.2, family="DejaVu Sans Mono")
    _label(ax, 0.545, 0.756, "beside(arg = kettle)", fontsize=8.2, family="DejaVu Sans Mono")
    _label(ax, 0.545, 0.732, "relevance rₖ retained", fontsize=7.6, color=MID)

    _label(ax, 0.305, 0.688, "deterministic scoring step", fontsize=7.7, fontweight="bold", color=TEAL)
    _box(ax, 0.305, 0.555, 0.128, 0.105, face=PALE_TEAL, edge=TEAL)
    _label(ax, 0.369, 0.633, "Typed comparator", fontsize=8.5, fontweight="bold", ha="center")
    _label(ax, 0.369, 0.606, "semantic · numeric", fontsize=7.7, ha="center")
    _label(ax, 0.369, 0.581, "relation · identity", fontsize=7.7, ha="center")
    _label(ax, 0.369, 0.559, "match mᵢₖ", fontsize=7.4, ha="center", color=MID)

    _box(ax, 0.455, 0.555, 0.135, 0.105, face=PALE_GOLD, edge=GOLD)
    _label(ax, 0.522, 0.633, "Persistence", fontsize=8.5, fontweight="bold", ha="center")
    _label(ax, 0.522, 0.606, "age + typed context", fontsize=7.7, ha="center")
    _label(ax, 0.522, 0.581, "+ invalidating events", fontsize=7.7, ha="center")
    _label(ax, 0.522, 0.559, "freshness fᵢₖ", fontsize=7.4, ha="center", color=MID)

    _box(ax, 0.612, 0.555, 0.133, 0.105, face=LIGHT, edge="#AAB8C1")
    _label(ax, 0.678, 0.633, "Evidence state", fontsize=8.5, fontweight="bold", ha="center")
    _label(ax, 0.678, 0.605, "observed → score", fontsize=7.7, ha="center")
    _label(ax, 0.678, 0.580, "unknown → missing", fontsize=7.7, ha="center")
    _label(ax, 0.678, 0.558, "not a mismatch", fontsize=7.4, ha="center", color=RED)

    _arrow(ax, (0.638, 0.720), (0.638, 0.675), color=BLUE)
    _arrow(ax, (0.410, 0.530), (0.410, 0.500), color=TEAL)
    _arrow(ax, (0.522, 0.530), (0.522, 0.500), color=GOLD)
    _arrow(ax, (0.678, 0.530), (0.678, 0.500), color=MID)

    _box(ax, 0.320, 0.340, 0.410, 0.155, face=WHITE, edge=NAVY, linewidth=1.5)
    _label(ax, 0.525, 0.467, "Evidence-aware entity score", fontsize=9.3, fontweight="bold", ha="center", color=NAVY)
    _label(ax, 0.340, 0.432, "aᵢₖ = rₖ · confidenceᵢₖ · fᵢₖ", fontsize=8.7, family="DejaVu Sans Mono")
    _label(ax, 0.340, 0.402, "matchᵢ = Σ aᵢₖmᵢₖ / Σ aᵢₖ", fontsize=8.7, family="DejaVu Sans Mono")
    _label(ax, 0.340, 0.372, "coverageᵢ = Σ aᵢₖ / Σ rₖ", fontsize=8.7, family="DejaVu Sans Mono")
    _label(ax, 0.548, 0.397, r"$score_i = match_i \cdot coverage_i^\gamma$", fontsize=9.0, fontweight="bold")

    # Panel (c): ranked result plus per-property audit.
    _box(ax, 0.790, 0.625, 0.190, 0.235, face=WHITE, edge=NAVY, linewidth=1.5)
    _label(ax, 0.805, 0.832, "Deterministic ranking", fontsize=9.3, fontweight="bold", color=NAVY)
    _label(ax, 0.963, 0.832, "illustrative", fontsize=7.1, ha="right", color=MID, fontstyle="italic")
    for y, rank, name, score, color in (
        (0.785, "1", "mug-1", 0.88, TEAL),
        (0.735, "2", "mug-2", 0.54, GOLD),
        (0.685, "3", "bowl-1", 0.16, MID),
    ):
        _label(ax, 0.807, y, rank, fontsize=8.0, fontweight="bold", color=color)
        _label(ax, 0.826, y, name, fontsize=8.5, family="DejaVu Sans Mono")
        _score_bar(ax, 0.883, y - 0.007, 0.072, score, color)
        _label(ax, 0.963, y, f"{score:.2f}", fontsize=7.2, ha="right", color=MID)
    _label(ax, 0.805, 0.646, "ties break by entity ID", fontsize=7.5, color=MID)
    _label(ax, 0.963, 0.646, "not benchmark results", fontsize=7.0, ha="right", color=RED)
    _arrow(ax, (0.765, 0.455), (0.790, 0.705), color=NAVY, connectionstyle="arc3,rad=-0.12")

    _box(ax, 0.790, 0.330, 0.190, 0.250, face=LIGHT, edge="#AAB8C1")
    _label(ax, 0.805, 0.552, "Per-property audit", fontsize=9.3, fontweight="bold")
    _label(ax, 0.805, 0.520, "why mug-1 ranked first", fontsize=7.5, color=MID)
    _label(ax, 0.805, 0.480, "type", fontsize=8.0, fontweight="bold")
    _label(ax, 0.850, 0.480, "match 1.00 · fresh 1.00", fontsize=7.4)
    _label(ax, 0.805, 0.445, "beside", fontsize=8.0, fontweight="bold")
    _label(ax, 0.850, 0.445, "match 1.00 · fresh 0.76", fontsize=7.4)
    _label(ax, 0.805, 0.410, "owner", fontsize=8.0, fontweight="bold")
    _label(ax, 0.850, 0.410, "unknown · no negative vote", fontsize=7.4, color=RED)
    ax.plot([0.805, 0.963], [0.382, 0.382], color="#CDD6DB", linewidth=0.9)
    _label(ax, 0.805, 0.358, "match 1.00   coverage 0.88", fontsize=7.9, family="DejaVu Sans Mono")

    # Bottom band: training data path and the evaluation-only truth barrier.
    ax.plot([0.018, 0.982], [0.260, 0.260], color="#D8E0E5", linewidth=1.0, zorder=1)
    _label(ax, 0.018, 0.235, "TRAINING AND EVALUATION BOUNDARIES", fontsize=8.6, fontweight="bold", color=MID)

    _box(ax, 0.020, 0.075, 0.175, 0.105, face=PALE_BLUE, edge=BLUE)
    _label(ax, 0.107, 0.151, "Observation histories", fontsize=8.7, fontweight="bold", ha="center")
    _label(ax, 0.107, 0.122, "timestamps · provenance · events", fontsize=7.6, ha="center")
    _label(ax, 0.107, 0.094, "stored outside property values", fontsize=7.4, ha="center", color=MID)
    _arrow(ax, (0.195, 0.128), (0.230, 0.128), color=BLUE)
    _box(ax, 0.230, 0.075, 0.190, 0.105, face=PALE_TEAL, edge=TEAL)
    _label(ax, 0.325, 0.151, "Survival records", fontsize=8.7, fontweight="bold", ha="center")
    _label(ax, 0.325, 0.122, "interval-censored changes", fontsize=7.6, ha="center")
    _label(ax, 0.325, 0.094, "right-censored unchanged histories", fontsize=7.4, ha="center", color=MID)
    _arrow(ax, (0.420, 0.128), (0.455, 0.128), color=TEAL)
    _box(ax, 0.455, 0.075, 0.170, 0.105, face=PALE_GOLD, edge=GOLD)
    _label(ax, 0.540, 0.151, "Persistence training", fontsize=8.7, fontweight="bold", ha="center")
    _label(ax, 0.540, 0.122, "entity-grouped splits", fontsize=7.6, ha="center")
    _label(ax, 0.540, 0.094, "validation-only selection", fontsize=7.4, ha="center", color=MID)
    _arrow(ax, (0.540, 0.180), (0.540, 0.294), color=GOLD, linestyle="--")
    _label(ax, 0.550, 0.233, "trained model", fontsize=7.2, color="#815B12")

    _box(ax, 0.690, 0.075, 0.125, 0.105, face=PALE_RED, edge=RED, linestyle="--")
    _label(ax, 0.752, 0.151, "current_truth", fontsize=8.8, fontweight="bold", ha="center", family="DejaVu Sans Mono", color=RED)
    _label(ax, 0.752, 0.121, "evaluation only", fontsize=7.8, ha="center", color=RED)
    _label(ax, 0.752, 0.094, "never matcher input", fontsize=7.4, ha="center", color=RED)
    _box(ax, 0.855, 0.075, 0.125, 0.105, face=LIGHT, edge="#AAB8C1")
    _label(ax, 0.917, 0.151, "Metrics", fontsize=8.8, fontweight="bold", ha="center")
    _label(ax, 0.917, 0.121, "Top-k · MRR", fontsize=7.7, ha="center")
    _label(ax, 0.917, 0.094, "calibration · failures", fontsize=7.4, ha="center", color=MID)
    _arrow(ax, (0.815, 0.128), (0.855, 0.128), color=RED, linestyle="--")
    ax.plot([0.980, 0.988, 0.988], [0.742, 0.742, 0.128], color=NAVY, linewidth=1.2, zorder=3)
    _arrow(ax, (0.988, 0.128), (0.980, 0.128), color=NAVY)

    _label(
        ax,
        0.655,
        0.035,
        "Dashed red path is evaluation-only; no reverse path enters OpenProp.",
        fontsize=7.4,
        color=RED,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    svg_path = output_dir / "openprop_task_pipeline.svg"
    png_path = output_dir / "openprop_task_pipeline.png"
    fig.savefig(svg_path, format="svg", bbox_inches=None, metadata={"Date": None})
    fig.savefig(
        png_path,
        format="png",
        dpi=220,
        bbox_inches=None,
        metadata={"Software": "OpenProp reproducible paper figure"},
    )
    plt.close(fig)
    return svg_path, png_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the OpenProp task and method pipeline paper figure."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("paper/figures"),
    )
    args = parser.parse_args()
    svg_path, png_path = build_figure(args.output_dir)
    print(f"svg: {svg_path}")
    print(f"png: {png_path}")


if __name__ == "__main__":
    main()
