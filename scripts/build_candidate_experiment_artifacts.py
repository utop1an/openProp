from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
from typing import Mapping


def build_candidate_artifacts(
    report: Mapping[str, object],
    comparison: Mapping[str, object] | None = None,
) -> dict[str, bytes]:
    systems = report.get("systems")
    if not isinstance(systems, Mapping) or not systems:
        raise ValueError("candidate report must contain systems")
    if comparison is not None:
        _validate_comparison(report, systems, comparison)
    markdown = _markdown(report, systems, comparison)
    latex = _latex(report, systems, comparison)
    plot_data = _plot_data(report, systems, comparison)
    return {
        "candidate_results.md": markdown.encode("utf-8"),
        "candidate_results.tex": latex.encode("utf-8"),
        "candidate_plot_data.json": (
            json.dumps(plot_data, indent=2, sort_keys=True) + "\n"
        ).encode("utf-8"),
        "candidate_tracking.png": _plot(plot_data),
    }


def _markdown(
    report: Mapping[str, object],
    systems: Mapping[str, object],
    comparison: Mapping[str, object] | None,
) -> str:
    lines = [
        f"# Candidate generation and tracking ({report.get('split')})", "",
        "Every frame, miss, false positive, rejected proposal, and capacity failure remains in the denominator.",
        "",
        "| System | Frames | Recall ↑ | Precision ↑ | Query target recall ↑ | Purity ↑ | ID switches ↓ | Fragments ↓ | Capacity failures ↓ |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name in sorted(systems):
        row = systems[name]
        assert isinstance(row, Mapping)
        lines.append("| " + " | ".join((
            str(name), str(row.get("frames", 0)), _format(row.get("candidate_recall")),
            _format(row.get("candidate_precision")), _format(row.get("query_target_recall")),
            _format(row.get("track_purity")), str(row.get("identity_switches", 0)),
            str(row.get("fragmentations", 0)), str(row.get("capacity_exceeded_frames", 0)),
        )) + " |")
    if comparison is not None:
        metrics = comparison["metrics"]
        assert isinstance(metrics, Mapping)
        lines.extend((
            "",
            f"## Paired inference: {comparison.get('system')} minus {comparison.get('baseline')}",
            "",
            "Point estimates pool numerators and denominators. Intervals resample paired clusters; p-values are exact episode-level sign tests.",
            "",
            "| Metric | Baseline | System | Delta | Paired cluster-bootstrap 95% CI | Sign p |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ))
        for name, row in metrics.items():
            if not isinstance(row, Mapping):
                continue
            if row.get("status") != "available":
                lines.append(f"| {name} | -- | -- | -- | unavailable | -- |")
                continue
            interval = row.get("cluster_bootstrap_95_ci")
            assert isinstance(interval, list) and len(interval) == 2
            lines.append(
                "| " + " | ".join((
                    str(name), _format(row.get("baseline")), _format(row.get("system")),
                    _signed(row.get("delta_system_minus_baseline")),
                    f"[{_format(interval[0])}, {_format(interval[1])}]",
                    _format(row.get("paired_episode_sign_exact_p")),
                )) + " |"
            )
    return "\n".join(lines) + "\n"


def _latex(
    report: Mapping[str, object],
    systems: Mapping[str, object],
    comparison: Mapping[str, object] | None,
) -> str:
    lines = [
        "\\begin{table}[t]", "\\centering",
        f"\\caption{{Candidate generation and identity tracking on the {_tex(str(report.get('split')))} split. All frames and failures remain in the denominator.}}",
        "\\label{tab:candidate-tracking}", "\\begin{tabular}{lrrrrrrrr}",
        "\\toprule",
        "System & Frames & Recall $\\uparrow$ & Prec. $\\uparrow$ & Target $\\uparrow$ & Purity $\\uparrow$ & IDSw $\\downarrow$ & Frag. $\\downarrow$ & Cap. $\\downarrow$ \\\\",
        "\\midrule",
    ]
    for name in sorted(systems):
        row = systems[name]
        assert isinstance(row, Mapping)
        lines.append(" & ".join((
            _tex(str(name)), str(row.get("frames", 0)), _format(row.get("candidate_recall")),
            _format(row.get("candidate_precision")), _format(row.get("query_target_recall")),
            _format(row.get("track_purity")), str(row.get("identity_switches", 0)),
            str(row.get("fragmentations", 0)), str(row.get("capacity_exceeded_frames", 0)),
        )) + " \\\\")
    lines.extend(("\\bottomrule", "\\end{tabular}", "\\end{table}", ""))
    if comparison is not None:
        metrics = comparison["metrics"]
        assert isinstance(metrics, Mapping)
        lines.extend((
            "\\begin{table}[t]", "\\centering",
            "\\caption{Exactly paired candidate-system differences. Point estimates pool counts; intervals resample clusters.}",
            "\\label{tab:candidate-paired}", "\\begin{tabular}{lrrrr}",
            "\\toprule", "Metric & Baseline & System & $\\Delta$ & 95\\% CI \\\\",
            "\\midrule",
        ))
        for name, row in metrics.items():
            if not isinstance(row, Mapping) or row.get("status") != "available":
                continue
            interval = row.get("cluster_bootstrap_95_ci")
            assert isinstance(interval, list) and len(interval) == 2
            lines.append(" & ".join((
                _tex(str(name)), _format(row.get("baseline")), _format(row.get("system")),
                _signed(row.get("delta_system_minus_baseline")),
                f"[{_format(interval[0])}, {_format(interval[1])}]",
            )) + " \\\\")
        lines.extend(("\\bottomrule", "\\end{tabular}", "\\end{table}", ""))
    return "\n".join(lines)


def _plot_data(
    report: Mapping[str, object],
    systems: Mapping[str, object],
    comparison: Mapping[str, object] | None,
) -> dict[str, object]:
    output = {
        "schema_version": 1,
        "split": report.get("split"),
        "systems": {
            str(name): {
                key: row.get(key)
                for key in (
                    "candidate_recall", "candidate_precision", "query_target_recall",
                    "track_purity", "identity_switches", "fragmentations", "frames",
                )
            }
            for name, row in sorted(systems.items())
            if isinstance(row, Mapping)
        },
    }
    if comparison is not None:
        output["paired_comparison"] = comparison
    return output


def _plot(plot_data: Mapping[str, object]) -> bytes:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    systems = plot_data["systems"]
    assert isinstance(systems, Mapping)
    names = list(systems)
    paired = plot_data.get("paired_comparison")
    columns = 3 if isinstance(paired, Mapping) else 2
    figure, axes = plt.subplots(
        1, columns, figsize=(10.0 if columns == 3 else 7.0, 3.0), constrained_layout=True
    )
    width = 0.25
    for offset, metric in enumerate(("candidate_recall", "candidate_precision", "query_target_recall")):
        axes[0].bar(
            [index + (offset - 1) * width for index in range(len(names))],
            [float(systems[name].get(metric) or 0.0) for name in names],
            width=width, label=metric.replace("_", " "),
        )
    axes[0].set_xticks(range(len(names)), names, rotation=20, ha="right")
    axes[0].set_ylim(0, 1)
    axes[0].set_title("Candidate coverage")
    axes[0].legend(fontsize=7)
    for offset, metric in enumerate(("identity_switches", "fragmentations")):
        axes[1].bar(
            [index + (offset - 0.5) * 0.35 for index in range(len(names))],
            [float(systems[name].get(metric) or 0.0) for name in names],
            width=0.35, label=metric.replace("_", " "),
        )
    axes[1].set_xticks(range(len(names)), names, rotation=20, ha="right")
    axes[1].set_title("Identity failures")
    axes[1].legend(fontsize=7)
    if isinstance(paired, Mapping):
        metrics = paired.get("metrics")
        assert isinstance(metrics, Mapping)
        chosen = (
            "candidate_recall", "candidate_precision", "query_target_recall", "track_purity"
        )
        available = [
            name for name in chosen
            if isinstance(metrics.get(name), Mapping)
            and metrics[name].get("status") == "available"
        ]
        values = [float(metrics[name]["delta_system_minus_baseline"]) for name in available]
        lowers = []
        uppers = []
        for name, value in zip(available, values):
            interval = metrics[name]["cluster_bootstrap_95_ci"]
            lowers.append(value - float(interval[0]))
            uppers.append(float(interval[1]) - value)
        axes[2].bar(range(len(available)), values, color="#4C78A8")
        if available:
            axes[2].errorbar(
                range(len(available)), values, yerr=[lowers, uppers], fmt="none",
                ecolor="black", capsize=3, linewidth=1,
            )
        axes[2].axhline(0.0, color="black", linewidth=0.8)
        axes[2].set_xticks(
            range(len(available)), [name.replace("_", " ") for name in available],
            rotation=25, ha="right",
        )
        axes[2].set_title("Paired delta (95% CI)")
    for axis in axes:
        axis.grid(axis="y", alpha=0.2)
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", bbox_inches="tight", dpi=160)
    plt.close(figure)
    return buffer.getvalue()


def _format(value: object) -> str:
    return "--" if value is None else f"{float(value):.3f}"


def _signed(value: object) -> str:
    return "--" if value is None else f"{float(value):+.3f}"


def _validate_comparison(
    report: Mapping[str, object],
    systems: Mapping[str, object],
    comparison: Mapping[str, object],
) -> None:
    if comparison.get("split") != report.get("split"):
        raise ValueError("candidate comparison/report splits differ")
    names = {comparison.get("baseline"), comparison.get("system")}
    if None in names or not names.issubset(set(systems)):
        raise ValueError("candidate comparison systems are absent from aggregate report")
    if not isinstance(comparison.get("metrics"), Mapping):
        raise ValueError("candidate comparison must contain metrics")


def _tex(value: str) -> str:
    return value.replace("_", "\\_").replace("%", "\\%")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build candidate tracking paper artifacts.")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--comparison", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    comparison = (
        json.loads(args.comparison.read_text(encoding="utf-8"))
        if args.comparison is not None else None
    )
    artifacts = build_candidate_artifacts(report, comparison)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, data in artifacts.items():
        (args.output_dir / name).write_bytes(data)
    print(f"artifacts={len(artifacts)} output: {args.output_dir}")


if __name__ == "__main__":
    main()
