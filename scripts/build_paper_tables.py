from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any, Mapping

from openprop.paper_claims import verify_claim_manifest


MAIN_CLAIM_ID = "C1_TYPED_COMPOSITION"
BOUNDARY_CLAIM_IDS = (
    "N1_NEURAL_NECESSITY",
    "N2_REAL_WORLD_GROUNDING",
    "N3_GENERAL_ADAPTATION_SAFETY",
)
COMPOSITIONAL_ARTIFACT = "artifacts/compositional_persistence_multiseed_results.json"
COMPONENT_ABLATION_ARTIFACT = "artifacts/typed_context_component_ablation.json"
DECISION_UTILITY_ARTIFACT = "artifacts/component_balanced_grounding_confirmation.json"
OBSERVATION_PROCESS_ARTIFACT = "artifacts/observation_process_results.json"
OBSERVATION_GROUNDING_ARTIFACT = "artifacts/observation_grounding_confirmation.json"
ALFRED_RETRIEVAL_BASELINE_ARTIFACT = "artifacts/alfred_retrieval_baseline.json"
ALFRED_RETRIEVAL_COMPARISON_ARTIFACT = "artifacts/alfred_retrieval_comparison.json"
ALFRED_RETRIEVAL_VS_LLM_ARTIFACT = "artifacts/alfred_retrieval_vs_llm.json"
REPEATED_EVIDENCE_ARTIFACT = "artifacts/repeated_evidence_adaptation_development.json"
MODELS = (
    ("global", "Global exponential"),
    ("factorized", "Factorized exponential"),
    ("neural", "Neural compositional"),
)
METRICS = (
    ("negative_log_likelihood", "NLL ↓", "NLL $\\downarrow$"),
    ("concordance_index", "C-index ↑", "C-index $\\uparrow$"),
    ("integrated_brier_score", "IBS ↓", "IBS $\\downarrow$"),
    ("grounding_top1", "Grounding Top-1 ↑", "Grounding Top-1 $\\uparrow$"),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_pointer(payload: Any, pointer: str) -> Any:
    if not pointer.startswith("/"):
        raise ValueError(f"table JSON pointer must start with '/': {pointer!r}")
    current = payload
    for raw in pointer[1:].split("/"):
        token = raw.replace("~1", "/").replace("~0", "~")
        if isinstance(current, list):
            current = current[int(token)]
        elif isinstance(current, Mapping) and token in current:
            current = current[token]
        else:
            raise ValueError(f"table JSON pointer is missing: {pointer!r}")
    return current


def _claim_map(manifest: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    claims = manifest.get("claims")
    if not isinstance(claims, list):
        raise ValueError("paper table generation requires a claim list")
    return {str(claim["id"]): claim for claim in claims}


def _evidence(
    claim: Mapping[str, Any],
    artifact: str,
) -> Mapping[str, Any]:
    rows = [row for row in claim.get("evidence", []) if row.get("artifact") == artifact]
    if len(rows) != 1:
        raise ValueError(
            f"claim {claim.get('id')} must bind exactly one {artifact} evidence row"
        )
    return rows[0]


def _bound_pointers(evidence: Mapping[str, Any]) -> set[str]:
    return {str(check.get("pointer", "")) for check in evidence.get("checks", [])}


def _require_bound(
    evidence: Mapping[str, Any],
    payload: Any,
    pointer: str,
) -> Any:
    if pointer not in _bound_pointers(evidence):
        raise ValueError(
            f"paper table pointer is not claim-bound for {evidence.get('artifact')}: "
            f"{pointer}"
        )
    return _json_pointer(payload, pointer)


def _mean_std(
    evidence: Mapping[str, Any],
    payload: Any,
    model: str,
    metric: str,
) -> tuple[float, float, tuple[str, str]]:
    mean_pointer = f"/aggregate/{model}/{metric}/mean"
    std_pointer = f"/aggregate/{model}/{metric}/standard_deviation"
    mean = _require_bound(evidence, payload, mean_pointer)
    std = _require_bound(evidence, payload, std_pointer)
    if not isinstance(mean, (int, float)) or not isinstance(std, (int, float)):
        raise ValueError("paper result-table values must be numeric")
    return float(mean), float(std), (mean_pointer, std_pointer)


def _format_markdown(mean: float, std: float, *, best: bool) -> str:
    value = f"{mean:.3f} ± {std:.3f}"
    return f"**{value}**" if best else value


def _format_latex(mean: float, std: float, *, best: bool) -> str:
    value = f"{mean:.3f} $\\pm$ {std:.3f}"
    return f"\\textbf{{{value}}}" if best else value


def _escape_latex(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    return "".join(replacements.get(character, character) for character in value)


def _main_tables(
    claim: Mapping[str, Any],
    evidence: Mapping[str, Any],
    payload: Mapping[str, Any],
) -> tuple[str, str, list[str]]:
    rows: list[tuple[str, list[tuple[float, float]]]] = []
    pointers: list[str] = []
    for model, label in MODELS:
        values: list[tuple[float, float]] = []
        for metric, _, _ in METRICS:
            mean, std, used = _mean_std(evidence, payload, model, metric)
            values.append((mean, std))
            pointers.extend(used)
        rows.append((label, values))

    caption = (
        "Controlled compositional persistence on held-out complete context tuples. "
        "Values report mean and standard deviation across five fixed seeds; all factor "
        "values are seen during training, and every model uses the same test stream. "
        "Bold marks the best mean. This is synthetic mechanism validation, not "
        "real-world evidence."
    )
    markdown_lines = [
        "<!-- Generated by scripts/build_paper_tables.py; do not edit. -->",
        f"**Table 1. {caption}**",
        "",
        "| Model | " + " | ".join(metric[1] for metric in METRICS) + " |",
        "|---|" + "---:|" * len(METRICS),
    ]
    for label, values in rows:
        best = label == "Factorized exponential"
        markdown_lines.append(
            "| "
            + label
            + " | "
            + " | ".join(
                _format_markdown(mean, std, best=best) for mean, std in values
            )
            + " |"
        )
    markdown_lines.extend(
        [
            "",
            f"*Claim boundary:* {claim['scope']}",
            "",
        ]
    )

    latex_lines = [
        "% Generated by scripts/build_paper_tables.py; do not edit.",
        r"\begin{table*}[t]",
        r"\caption{" + _escape_latex(caption) + "}",
        r"\label{tab:controlled-compositional}",
        r"\centering",
        r"\small",
        r"\begin{tabular}{@{}lcccc@{}}",
        r"\toprule",
        "Model & " + " & ".join(metric[2] for metric in METRICS) + r" \\",
        r"\midrule",
    ]
    for label, values in rows:
        best = label == "Factorized exponential"
        latex_lines.append(
            _escape_latex(label)
            + " & "
            + " & ".join(
                _format_latex(mean, std, best=best) for mean, std in values
            )
            + r" \\"
        )
    latex_lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table*}",
            "",
        ]
    )
    return "\n".join(markdown_lines), "\n".join(latex_lines), pointers


def _component_ablation_tables(
    claim: Mapping[str, Any],
    evidence: Mapping[str, Any],
    payload: Mapping[str, Any],
) -> tuple[str, str, list[str]]:
    pointers = [
        "/seeds",
        "/protocol/delta_orientation",
        "/protocol/selection",
        "/protocol/test_policy",
    ]
    for pointer in pointers:
        _require_bound(evidence, payload, pointer)
    rows: list[tuple[str, str, float, float, float, int]] = []
    for omitted, retained, condition in (
        ("Subject", "relation + scene", "relation_scene"),
        ("Relation", "subject + scene", "subject_scene"),
        ("Scene", "subject + relation", "subject_relation"),
    ):
        base = f"/paired_full_advantage/{condition}/negative_log_likelihood"
        used = [
            f"{base}/mean_full_advantage",
            f"{base}/simultaneous_bootstrap_95_ci/0",
            f"{base}/simultaneous_bootstrap_95_ci/1",
            f"{base}/wins",
        ]
        values = [_require_bound(evidence, payload, pointer) for pointer in used]
        pointers.extend(used)
        rows.append(
            (
                omitted,
                retained,
                float(values[0]),
                float(values[1]),
                float(values[2]),
                int(values[3]),
            )
        )

    caption = (
        "Typed-factor ablation on held-out context tuples. Delta NLL is "
        "NLL(ablation) minus NLL(full), so positive values favor the full model. "
        "Intervals are paired, family-wise simultaneous 95% bootstrap intervals "
        "across the three predeclared factor comparisons over ten fixed seeds on "
        "identical test rows. NLL was the predeclared primary metric. This is "
        "synthetic mechanism validation."
    )
    markdown = [
        "<!-- Generated by scripts/build_paper_tables.py; do not edit. -->",
        f"**Table 2. {caption}**",
        "",
        "| Omitted typed factor | Retained factors | ΔNLL ↑ [simultaneous 95% CI] | Wins / 10 |",
        "|---|---|---:|---:|",
    ]
    for omitted, retained, delta, lower, upper, wins in rows:
        markdown.append(
            f"| {omitted} | {retained} | **{delta:.3f} [{lower:.3f}, {upper:.3f}]** | {wins}/10 |"
        )
    markdown.extend(["", f"*Claim boundary:* {claim['scope']}", ""])

    latex = [
        "% Generated by scripts/build_paper_tables.py; do not edit.",
        r"\begin{table*}[t]",
        r"\caption{" + _escape_latex(caption) + "}",
        r"\label{tab:typed-component-ablation}",
        r"\centering",
        r"\small",
        r"\begin{tabular}{@{}llcc@{}}",
        r"\toprule",
        r"Omitted factor & Retained factors & $\Delta$NLL $\uparrow$ [simultaneous 95\% CI] & Wins / 10 \\",
        r"\midrule",
    ]
    for omitted, retained, delta, lower, upper, wins in rows:
        latex.append(
            f"{_escape_latex(omitted)} & {_escape_latex(retained)} & "
            + r"\textbf{"
            + f"{delta:.3f} [{lower:.3f}, {upper:.3f}]"
            + r"} & "
            + f"{wins}/10"
            + r" \\"
        )
    latex.extend(
        [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
    )
    return "\n".join(markdown), "\n".join(latex), pointers


def _decision_utility_tables(
    claim: Mapping[str, Any],
    evidence: Mapping[str, Any],
    payload: Mapping[str, Any],
) -> tuple[str, str, list[str]]:
    pointers = [
        "/phase",
        "/seeds",
        "/cases/total",
        "/cases/target_old",
        "/cases/target_new",
        "/protocol/axis_isolation",
        "/protocol/case_design",
        "/protocol/confirmation_policy",
        "/protocol/truth_boundary",
    ]
    for pointer in pointers:
        _require_bound(evidence, payload, pointer)
    rows: list[
        tuple[str, float, float, float, float, float, float, float, int]
    ] = []
    for factor, ablation in (
        ("subject", "no_subject"),
        ("relation", "no_relation"),
        ("scene", "no_scene"),
    ):
        used = [
            f"/aggregate/full_context/top1_by_probe/{factor}/mean",
            f"/aggregate/full_context/top1_by_probe/{factor}/standard_deviation",
            f"/aggregate/{ablation}/top1_by_probe/{factor}/mean",
            f"/aggregate/{ablation}/top1_by_probe/{factor}/standard_deviation",
            f"/paired_probe_advantage/{factor}/mean_full_advantage",
            f"/paired_probe_advantage/{factor}/simultaneous_bootstrap_95_ci/0",
            f"/paired_probe_advantage/{factor}/simultaneous_bootstrap_95_ci/1",
            f"/paired_probe_advantage/{factor}/wins",
        ]
        values = [_require_bound(evidence, payload, pointer) for pointer in used]
        pointers.extend(used)
        rows.append(
            (
                factor.title(),
                *(float(value) for value in values[:7]),
                int(values[7]),
            )
        )

    caption = (
        "Axis-isolated entity decisions on 40 analytic confidence-age crossover "
        "cases (20 old and 20 new targets). Values report Top-1 mean and standard "
        "deviation across ten untouched confirmation seeds. Delta is full minus "
        "the matched factor-removal ablation; intervals are paired, family-wise "
        "simultaneous 95% bootstrap intervals across the three predeclared probes. "
        "This is synthetic controlled decision evidence, not natural prevalence."
    )
    markdown = [
        "<!-- Generated by scripts/build_paper_tables.py; do not edit. -->",
        f"**Table 3. {caption}**",
        "",
        "| Isolated factor | Full Top-1 ↑ | Remove factor Top-1 ↑ | Paired Δ ↑ [simultaneous 95% CI] | Wins / 10 |",
        "|---|---:|---:|---:|---:|",
    ]
    for factor, full, full_std, removed, removed_std, delta, lower, upper, wins in rows:
        markdown.append(
            f"| {factor} | **{full:.3f} ± {full_std:.3f}** | "
            f"{removed:.3f} ± {removed_std:.3f} | "
            f"**{delta:.3f} [{lower:.3f}, {upper:.3f}]** | {wins}/10 |"
        )
    markdown.extend(["", f"*Claim boundary:* {claim['scope']}", ""])

    latex = [
        "% Generated by scripts/build_paper_tables.py; do not edit.",
        r"\begin{table*}[t]",
        r"\caption{" + _escape_latex(caption) + "}",
        r"\label{tab:controlled-decision-utility}",
        r"\centering",
        r"\small",
        r"\begin{tabular}{@{}lcccc@{}}",
        r"\toprule",
        r"Isolated factor & Full Top-1 $\uparrow$ & Remove factor Top-1 $\uparrow$ & Paired $\Delta$ $\uparrow$ [simultaneous 95\% CI] & Wins / 10 \\",
        r"\midrule",
    ]
    for factor, full, full_std, removed, removed_std, delta, lower, upper, wins in rows:
        latex.append(
            f"{_escape_latex(factor)} & "
            + r"\textbf{"
            + f"{full:.3f} $\\pm$ {full_std:.3f}"
            + r"} & "
            + f"{removed:.3f} $\\pm$ {removed_std:.3f} & "
            + r"\textbf{"
            + f"{delta:.3f} [{lower:.3f}, {upper:.3f}]"
            + r"} & "
            + f"{wins}/10"
            + r" \\"
        )
    latex.extend(
        [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
    )
    return "\n".join(markdown), "\n".join(latex), pointers


def _observation_process_tables(
    claim: Mapping[str, Any],
    evidence: Mapping[str, Any],
    payload: Mapping[str, Any],
) -> tuple[str, str, list[str]]:
    pointers = [
        "/protocol/claim_scope",
        "/protocol/inspection_intervals_hours",
        "/protocol/latent_hazard_per_hour",
        "/protocol/samples_per_schedule",
        "/protocol/seeds",
        "/protocol/test_samples_per_schedule",
    ]
    protocol = {
        pointer: _require_bound(evidence, payload, pointer) for pointer in pointers
    }
    rows: list[tuple[str, list[tuple[float, float]]]] = []
    metrics = (
        "mean_absolute_hazard_error",
        "schedule_gap",
        "exact_test_nll",
    )
    for method, label in (
        ("naive", "Detected-time naive"),
        ("interval_aware", "Interval-aware"),
    ):
        values: list[tuple[float, float]] = []
        for metric in metrics:
            mean, std, used = _mean_std(evidence, payload, method, metric)
            values.append((mean, std))
            pointers.extend(used)
        rows.append((label, values))

    intervals = protocol["/protocol/inspection_intervals_hours"]
    seeds = protocol["/protocol/seeds"]
    caption = (
        "Inspection-frequency bias under an identical synthetic exponential process "
        f"(true hazard {float(protocol['/protocol/latent_hazard_per_hour']):.2f}/h; "
        f"inspection every {float(intervals[0]):.1f} or {float(intervals[1]):.1f} h). "
        f"Values are mean and standard deviation across {len(seeds)} fixed seeds, with "
        f"{int(protocol['/protocol/samples_per_schedule'])} train and "
        f"{int(protocol['/protocol/test_samples_per_schedule'])} exact-time test samples "
        "per schedule. Bold marks the better mean. This isolates interval-censoring "
        "mechanics; it is not evidence for arbitrary missingness or real observation processes."
    )
    markdown = [
        "<!-- Generated by scripts/build_paper_tables.py; do not edit. -->",
        f"**Table 4. {caption}**",
        "",
        "| Estimator | Hazard MAE ↓ | Schedule gap ↓ | Exact test NLL ↓ |",
        "|---|---:|---:|---:|",
    ]
    for label, values in rows:
        best = label == "Interval-aware"
        markdown.append(
            f"| {label} | "
            + " | ".join(
                _format_markdown(mean, std, best=best) for mean, std in values
            )
            + " |"
        )
    markdown.extend(["", f"*Claim boundary:* {claim['scope']}", ""])

    latex = [
        "% Generated by scripts/build_paper_tables.py; do not edit.",
        r"\begin{table*}[t]",
        r"\caption{" + _escape_latex(caption) + "}",
        r"\label{tab:observation-process-bias}",
        r"\centering",
        r"\small",
        r"\begin{tabular}{@{}lccc@{}}",
        r"\toprule",
        r"Estimator & Hazard MAE $\downarrow$ & Schedule gap $\downarrow$ & Exact test NLL $\downarrow$ \\",
        r"\midrule",
    ]
    for label, values in rows:
        best = label == "Interval-aware"
        latex.append(
            _escape_latex(label)
            + " & "
            + " & ".join(
                _format_latex(mean, std, best=best) for mean, std in values
            )
            + r" \\"
        )
    latex.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""])
    return "\n".join(markdown), "\n".join(latex), pointers


def _observation_grounding_tables(
    claim: Mapping[str, Any],
    evidence: Mapping[str, Any],
    payload: Mapping[str, Any],
) -> tuple[str, str, list[str]]:
    pointers = [
        "/phase",
        "/seeds",
        "/protocol/claim_scope",
        "/protocol/latent_hazard_per_hour",
        "/protocol/scene_inspection_intervals_hours/frequent-scene",
        "/protocol/scene_inspection_intervals_hours/sparse-scene",
        "/protocol/development_seeds",
        "/protocol/confirmation_seeds",
        "/protocol/samples_per_scene",
        "/protocol/cases_fixed_before_confirmation",
        "/protocol/case_design",
        "/protocol/target_scene_balance",
        "/protocol/truth_boundary",
        "/protocol/candidate_order_policy",
        "/protocol/query_policy",
        "/protocol/primary_metric",
        "/protocol/secondary_metrics",
        "/protocol/bootstrap_samples",
        "/cases/total",
        "/cases/target_frequent_scene",
        "/cases/target_sparse_scene",
    ]
    facts = {pointer: _require_bound(evidence, payload, pointer) for pointer in pointers}
    rows: list[tuple[str, list[tuple[float, float]]]] = []
    metrics = ("top1", "worst_target_scene_top1", "target_scene_gap")
    for condition, label in (
        ("naive", "Detected-time naive"),
        ("interval_aware", "Interval-aware"),
        ("oracle", "True-hazard oracle"),
    ):
        values: list[tuple[float, float]] = []
        for metric in metrics:
            mean, std, used = _mean_std(evidence, payload, condition, metric)
            values.append((mean, std))
            pointers.extend(used)
        rows.append((label, values))

    comparison_values: list[tuple[float, float, float]] = []
    for metric in metrics:
        base = f"/paired_interval_advantage/{metric}"
        used = [
            f"{base}/mean_advantage",
            f"{base}/bootstrap_95_ci/0",
            f"{base}/bootstrap_95_ci/1",
        ]
        comparison_values.append(
            tuple(float(_require_bound(evidence, payload, pointer)) for pointer in used)
        )
        pointers.extend(used)
    orientation_pointers = [
        "/paired_interval_advantage/top1/orientation",
        "/paired_interval_advantage/target_scene_gap/orientation",
    ]
    for pointer in orientation_pointers:
        _require_bound(evidence, payload, pointer)
    pointers.extend(orientation_pointers)
    seed_count_pointers = [
        "/paired_interval_advantage/top1/wins",
        "/paired_interval_advantage/top1/ties",
        "/paired_interval_advantage/top1/losses",
    ]
    wins, ties, losses = (
        int(_require_bound(evidence, payload, pointer)) for pointer in seed_count_pointers
    )
    pointers.extend(seed_count_pointers)

    seeds = facts["/seeds"]
    caption = (
        "Downstream grounding under inspection-frequency confounding. Two scene "
        f"strata share hazard {float(facts['/protocol/latent_hazard_per_hour']):.2f}/h "
        "but are inspected every "
        f"{float(facts['/protocol/scene_inspection_intervals_hours/frequent-scene']):.1f} "
        "or "
        f"{float(facts['/protocol/scene_inspection_intervals_hours/sparse-scene']):.1f} h. "
        f"The {int(facts['/cases/total'])} fixed cases balance target scene "
        f"({int(facts['/cases/target_frequent_scene'])}/{int(facts['/cases/target_sparse_scene'])}) "
        f"and use {int(facts['/protocol/samples_per_scene'])} training histories per "
        f"scene across {len(seeds)} untouched confirmation seeds. Values are mean "
        "and standard deviation; delta rows use paired seed-bootstrap 95% intervals. "
        "Scene affects persistence context but is not queried. This is analytic "
        "synthetic decision evidence, not natural prevalence or real-world grounding."
    )
    markdown = [
        "<!-- Generated by scripts/build_paper_tables.py; do not edit. -->",
        f"**Table 5. {caption}**",
        "",
        "| Estimator | Overall Top-1 ↑ | Worst-scene Top-1 ↑ | Target-scene gap ↓ | Primary W/T/L |",
        "|---|---:|---:|---:|---:|",
    ]
    for label, values in rows:
        bold = label in {"Interval-aware", "True-hazard oracle"}
        markdown.append(
            f"| {label} | "
            + " | ".join(_format_markdown(mean, std, best=bold) for mean, std in values)
            + " | — |"
        )
    delta_cells = [
        f"**+{comparison_values[0][0]:.3f} [{comparison_values[0][1]:.3f}, {comparison_values[0][2]:.3f}]**",
        f"**+{comparison_values[1][0]:.3f} [{comparison_values[1][1]:.3f}, {comparison_values[1][2]:.3f}]**",
        f"**−{comparison_values[2][0]:.3f} [−{comparison_values[2][2]:.3f}, −{comparison_values[2][1]:.3f}]**",
    ]
    markdown.append(
        "| Interval-aware advantage vs naive | "
        + " | ".join(delta_cells)
        + f" | **{wins}/{ties}/{losses}** |"
    )
    markdown.extend(["", f"*Claim boundary:* {claim['scope']}", ""])

    latex = [
        "% Generated by scripts/build_paper_tables.py; do not edit.",
        r"\begin{table*}[t]",
        r"\caption{" + _escape_latex(caption) + "}",
        r"\label{tab:observation-grounding-decisions}",
        r"\centering",
        r"\small",
        r"\begin{tabular}{@{}lcccc@{}}",
        r"\toprule",
        r"Estimator & Overall Top-1 $\uparrow$ & Worst-scene Top-1 $\uparrow$ & Target-scene gap $\downarrow$ & Primary W/T/L \\",
        r"\midrule",
    ]
    for label, values in rows:
        bold = label in {"Interval-aware", "True-hazard oracle"}
        latex.append(
            _escape_latex(label)
            + " & "
            + " & ".join(_format_latex(mean, std, best=bold) for mean, std in values)
            + r" & -- \\"
        )
    latex.append(r"\midrule")
    latex.append(
        "Interval-aware advantage vs naive & "
        + " & ".join(
            r"\textbf{" + cell.replace("**", "").replace("−", "-") + "}"
            for cell in delta_cells
        )
        + f" & \\textbf{{{wins}/{ties}/{losses}}} \\\\"
    )
    latex.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""])
    return "\n".join(markdown), "\n".join(latex), pointers


def _external_language_tables(
    claim: Mapping[str, Any],
    baseline_evidence: Mapping[str, Any],
    comparison_evidence: Mapping[str, Any],
    llm_evidence: Mapping[str, Any],
    baseline: Mapping[str, Any],
    comparison: Mapping[str, Any],
    llm: Mapping[str, Any],
) -> tuple[str, str, dict[str, list[str]]]:
    baseline_pointers = [
        "/protocol/baseline",
        "/protocol/bm25_evidence_policy/exact_train_query",
        "/protocol/bm25_evidence_policy/novel_query",
        "/protocol/bm25_evidence_policy/unsupported_state",
        "/protocol/bm25_evidence_policy/validation_labels_used",
        "/protocol/candidate_or_matcher_access",
        "/protocol/parameter_source",
        "/protocol/retriever/source_split",
        "/protocol/retriever/validation_data_used_for_fit",
        "/protocol/temporal_claim",
        "/protocol/validation_task_id_overlap_with_train",
    ]
    for pointer in baseline_pointers:
        _require_bound(baseline_evidence, baseline, pointer)
    comparison_pointers = [
        "/protocol/comparison",
        "/protocol/labels_used_for_method_selection",
        "/protocol/stratification",
    ]
    for pointer in comparison_pointers:
        _require_bound(comparison_evidence, comparison, pointer)
    llm_pointers = [
        "/protocol/bootstrap_cluster",
        "/protocol/bootstrap_stratum",
        "/protocol/method_selection_labels_used",
        "/protocol/pairing",
        "/protocol/sample",
    ]
    for pointer in llm_pointers:
        _require_bound(llm_evidence, llm, pointer)

    metrics = (
        ("property_f1", "Property F1 ↑", "Property F1 $\\uparrow$"),
        ("value_recall", "Value recall ↑", "Value recall $\\uparrow$"),
        ("exact_frame", "Exact frame ↑", "Exact frame $\\uparrow$"),
    )
    rows: list[tuple[str, int, list[tuple[str, tuple[float, ...]]]]] = []
    for split, split_label in (("valid_seen", "Valid-seen"), ("valid_unseen", "Valid-unseen")):
        cases_pointer = f"/{split}/evidence_only/cases"
        cases = int(_require_bound(baseline_evidence, baseline, cases_pointer))
        baseline_pointers.append(cases_pointer)
        split_rows: list[tuple[str, tuple[float, ...]]] = []
        for method, label in (
            ("evidence_only", "Evidence only"),
            ("bm25_top1", "BM25 top-1"),
            ("bm25_evidence", "BM25 + positive evidence"),
        ):
            values: list[float] = []
            for metric, _, _ in metrics:
                pointer = f"/{split}/{method}/{metric}"
                values.append(float(_require_bound(baseline_evidence, baseline, pointer)))
                baseline_pointers.append(pointer)
            split_rows.append((label, tuple(values)))
        delta_values: list[float] = []
        for metric, _, _ in metrics:
            base = f"/{split}/bm25_evidence_minus_bm25/{metric}"
            used = [f"{base}/delta", f"{base}/ci_95/0", f"{base}/ci_95/1"]
            delta_values.extend(
                float(_require_bound(comparison_evidence, comparison, pointer))
                for pointer in used
            )
            comparison_pointers.extend(used)
        split_rows.append(("Positive-evidence Δ vs BM25", tuple(delta_values)))
        rows.append((split_label, cases, split_rows))

    llm_values: dict[str, dict[str, tuple[float, float, float]]] = {}
    for model in ("gemma3_4b", "llama3_2"):
        llm_values[model] = {}
        for metric in ("exact_frame", "value_recall"):
            base = f"/{model}/{metric}"
            used = [f"{base}/delta", f"{base}/ci_95/0", f"{base}/ci_95/1"]
            values = tuple(
                float(_require_bound(llm_evidence, llm, pointer)) for pointer in used
            )
            llm_pointers.extend(used)
            llm_values[model][metric] = values

    caption = (
        "External ALFRED language-to-frame parsing with a fixed train-only BM25 "
        "retriever and positive span evidence. No validation labels select or fit "
        "the methods. Deltas and 95% intervals are paired by case and bootstrapped "
        "by task ID, stratified by task type. The methods have no candidate, matcher, "
        "visual, or temporal access."
    )
    markdown = [
        "<!-- Generated by scripts/build_paper_tables.py; do not edit. -->",
        f"**Table 6. {caption}**",
        "",
        "| Split | Method | Property F1 ↑ | Value recall ↑ | Exact frame ↑ |",
        "|---|---|---:|---:|---:|",
    ]
    for split_label, cases, split_rows in rows:
        for label, values in split_rows:
            if label.startswith("Positive-evidence"):
                formatted = [
                    f"**+{values[index]:.3f} [{values[index + 1]:.3f}, {values[index + 2]:.3f}]**"
                    for index in (0, 3, 6)
                ]
            else:
                formatted = [f"{value:.3f}" for value in values]
                if label == "BM25 + positive evidence":
                    formatted = [f"**{value}**" for value in formatted]
            markdown.append(
                f"| {split_label} ({cases}) | {label} | " + " | ".join(formatted) + " |"
            )
    gemma_exact = llm_values["gemma3_4b"]["exact_frame"]
    gemma_recall = llm_values["gemma3_4b"]["value_recall"]
    llama_exact = llm_values["llama3_2"]["exact_frame"]
    llama_recall = llm_values["llama3_2"]["value_recall"]
    llm_note = (
        "On the separate frozen 40-case valid-unseen confirmation sample, the hybrid "
        f"also exceeds frozen local Gemma 3 4B by {gemma_exact[0]:.3f} exact-frame "
        f"[{gemma_exact[1]:.3f}, {gemma_exact[2]:.3f}] and {gemma_recall[0]:.3f} "
        f"value recall [{gemma_recall[1]:.3f}, {gemma_recall[2]:.3f}], and Llama 3.2 "
        f"by {llama_exact[0]:.3f} [{llama_exact[1]:.3f}, {llama_exact[2]:.3f}] and "
        f"{llama_recall[0]:.3f} [{llama_recall[1]:.3f}, {llama_recall[2]:.3f}], respectively."
    )
    markdown.extend(["", llm_note, "", f"*Claim boundary:* {claim['scope']}", ""])

    latex = [
        "% Generated by scripts/build_paper_tables.py; do not edit.",
        r"\begin{table*}[t]",
        r"\caption{" + _escape_latex(caption) + "}",
        r"\label{tab:external-language-results}",
        r"\centering",
        r"\small",
        r"\begin{tabular}{@{}llccc@{}}",
        r"\toprule",
        "Split & Method & " + " & ".join(metric[2] for metric in metrics) + r" \\",
        r"\midrule",
    ]
    for split_index, (split_label, cases, split_rows) in enumerate(rows):
        for row_index, (label, values) in enumerate(split_rows):
            if label.startswith("Positive-evidence"):
                formatted = [
                    r"\textbf{" + f"+{values[index]:.3f} [{values[index + 1]:.3f}, {values[index + 2]:.3f}]" + "}"
                    for index in (0, 3, 6)
                ]
            else:
                formatted = [f"{value:.3f}" for value in values]
                if label == "BM25 + positive evidence":
                    formatted = [r"\textbf{" + value + "}" for value in formatted]
            split_cell = f"{_escape_latex(split_label)} ({cases})" if row_index == 0 else ""
            latex_label = (
                r"Positive-evidence $\Delta$ vs BM25"
                if label.startswith("Positive-evidence")
                else _escape_latex(label)
            )
            latex.append(
                f"{split_cell} & {latex_label} & " + " & ".join(formatted) + r" \\"
            )
        if split_index == 0:
            latex.append(r"\midrule")
    latex.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\vspace{2pt}",
            r"\parbox{0.98\textwidth}{\footnotesize " + _escape_latex(llm_note) + "}",
            r"\end{table*}",
            "",
        ]
    )
    return "\n".join(markdown), "\n".join(latex), {
        ALFRED_RETRIEVAL_BASELINE_ARTIFACT: baseline_pointers,
        ALFRED_RETRIEVAL_COMPARISON_ARTIFACT: comparison_pointers,
        ALFRED_RETRIEVAL_VS_LLM_ARTIFACT: llm_pointers,
    }


def _boundary_evidence(
    claims: Mapping[str, Mapping[str, Any]],
    artifacts: Mapping[str, Any],
) -> tuple[list[dict[str, str]], dict[str, list[str]]]:
    n1 = claims["N1_NEURAL_NECESSITY"]
    n1_evidence = _evidence(n1, COMPOSITIONAL_ARTIFACT)
    composition = artifacts[COMPOSITIONAL_ARTIFACT]
    n1_pointers = [
        "/aggregate/factorized/negative_log_likelihood/mean",
        "/aggregate/neural/negative_log_likelihood/mean",
        "/aggregate/factorized/concordance_index/mean",
        "/aggregate/neural/concordance_index/mean",
        "/aggregate/factorized/integrated_brier_score/mean",
        "/aggregate/neural/integrated_brier_score/mean",
        "/aggregate/factorized/grounding_top1/mean",
        "/aggregate/neural/grounding_top1/mean",
    ]
    n1_values = [
        float(_require_bound(n1_evidence, composition, pointer))
        for pointer in n1_pointers
    ]

    n2 = claims["N2_REAL_WORLD_GROUNDING"]
    if n2.get("status") != "pending_external" or n2.get("evidence"):
        raise ValueError("real-world grounding boundary must remain pending without evidence")

    n3 = claims["N3_GENERAL_ADAPTATION_SAFETY"]
    n3_evidence = _evidence(n3, REPEATED_EVIDENCE_ARTIFACT)
    repeated = artifacts[REPEATED_EVIDENCE_ARTIFACT]
    n3_pointers = [
        "/protocol/stage",
        "/activation_counts/correct_source_control/0.2/single_noisy_15",
        "/activation_counts/correct_source_control/0.2/repeat5_cases15_confident",
        "/paired_delta_vs_deployed_source/local_subject_scene_bump/0.2/repeat5_cases15_confident/affected/concordance_index/mean",
        "/paired_delta_vs_deployed_source/local_subject_scene_bump/0.2/repeat5_cases15_confident/affected/concordance_index/bootstrap_95_ci_lower",
        "/paired_delta_vs_deployed_source/local_subject_scene_bump/0.2/repeat5_cases15_confident/affected/concordance_index/minimum",
    ]
    n3_values = [
        _require_bound(n3_evidence, repeated, pointer) for pointer in n3_pointers
    ]
    rows = [
        {
            "claim": "Neural persistence is necessary",
            "status": "Contradicted",
            "evidence": (
                f"Factorized vs neural: NLL {n1_values[0]:.3f} vs {n1_values[1]:.3f}; "
                f"C-index {n1_values[2]:.3f} vs {n1_values[3]:.3f}; "
                f"IBS {n1_values[4]:.3f} vs {n1_values[5]:.3f}; "
                f"Top-1 {n1_values[6]:.3f} vs {n1_values[7]:.3f}."
            ),
            "paper_consequence": "Claim typed factorization, not neural novelty.",
        },
        {
            "claim": "Semi-real longitudinal effectiveness",
            "status": "Pending",
            "evidence": str(n2["scope"]),
            "paper_consequence": "Submission blocker; no performance wording is allowed.",
        },
        {
            "claim": "Calibration-only adaptation is generally safe",
            "status": "Contradicted",
            "evidence": (
                f"Under 20% flips, repetition changes noisy control activations from "
                f"{n3_values[1]}/10 to {n3_values[2]}/10, yet affected C-index changes "
                f"{float(n3_values[3]):.3f} (95% bootstrap lower bound "
                f"{float(n3_values[4]):.3f}; worst seed {float(n3_values[5]):.3f})."
            ),
            "paper_consequence": "Report an identifiability boundary, not a safety guarantee.",
        },
    ]
    return rows, {
        "N1_NEURAL_NECESSITY": n1_pointers,
        "N3_GENERAL_ADAPTATION_SAFETY": n3_pointers,
    }


def _boundary_tables(rows: list[dict[str, str]]) -> tuple[str, str]:
    caption = (
        "Claims excluded or pending after adversarial evaluation. Negative and "
        "missing results are retained rather than hidden behind favorable averages."
    )
    markdown_lines = [
        "<!-- Generated by scripts/build_paper_tables.py; do not edit. -->",
        f"**Table 7. {caption}**",
        "",
        "| Candidate claim | Status | Frozen evidence | Paper consequence |",
        "|---|---|---|---|",
    ]
    for row in rows:
        markdown_lines.append(
            "| "
            + " | ".join(
                row[key]
                for key in ("claim", "status", "evidence", "paper_consequence")
            )
            + " |"
        )
    markdown_lines.append("")

    latex_lines = [
        "% Generated by scripts/build_paper_tables.py; do not edit.",
        r"\begin{table*}[t]",
        r"\caption{" + _escape_latex(caption) + "}",
        r"\label{tab:claim-boundaries}",
        r"\centering",
        r"\small",
        r"\begin{tabularx}{\textwidth}{@{}p{0.20\textwidth}p{0.10\textwidth}X p{0.23\textwidth}@{}}",
        r"\toprule",
        r"Candidate claim & Status & Frozen evidence & Paper consequence \\",
        r"\midrule",
    ]
    for row in rows:
        latex_lines.append(
            " & ".join(
                _escape_latex(row[key])
                for key in ("claim", "status", "evidence", "paper_consequence")
            )
            + r" \\"
        )
    latex_lines.extend(
        [
            r"\bottomrule",
            r"\end{tabularx}",
            r"\end{table*}",
            "",
        ]
    )
    return "\n".join(markdown_lines), "\n".join(latex_lines)


def _replace_generated_block(text: str, name: str, content: str) -> str:
    start = f"<!-- BEGIN GENERATED TABLE: {name} -->"
    end = f"<!-- END GENERATED TABLE: {name} -->"
    if text.count(start) != 1 or text.count(end) != 1:
        raise ValueError(f"manuscript requires exactly one generated table block: {name}")
    before, remainder = text.split(start, 1)
    _, after = remainder.split(end, 1)
    return before + start + "\n" + content.rstrip() + "\n" + end + after


def _embedded_table_content(text: str, name: str) -> str:
    start = f"<!-- BEGIN GENERATED TABLE: {name} -->"
    end = f"<!-- END GENERATED TABLE: {name} -->"
    if start not in text or end not in text:
        raise ValueError(f"missing generated table block: {name}")
    return text.split(start, 1)[1].split(end, 1)[0].strip()


def build_paper_tables(
    manifest_path: Path,
    output_dir: Path,
    *,
    repository_root: Path | None = None,
    manuscript_path: Path | None = None,
) -> tuple[Path, ...]:
    manifest_file = manifest_path.resolve()
    root = (
        repository_root.resolve()
        if repository_root is not None
        else manifest_file.parent.parent.resolve()
    )
    verification = verify_claim_manifest(manifest_file, repository_root=root)
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    claims = _claim_map(manifest)
    required_claims = {
        MAIN_CLAIM_ID,
        "C2_TYPED_COMPONENTS",
        "C3_DECISION_UTILITY",
        "C4_INTERVAL_CENSORING",
        "C5_EXTERNAL_LANGUAGE",
        *BOUNDARY_CLAIM_IDS,
    }
    missing = required_claims.difference(claims)
    if missing:
        raise ValueError("paper table claims are missing: " + ", ".join(sorted(missing)))

    used_artifacts = (
        COMPOSITIONAL_ARTIFACT,
        COMPONENT_ABLATION_ARTIFACT,
        DECISION_UTILITY_ARTIFACT,
        OBSERVATION_PROCESS_ARTIFACT,
        OBSERVATION_GROUNDING_ARTIFACT,
        ALFRED_RETRIEVAL_BASELINE_ARTIFACT,
        ALFRED_RETRIEVAL_COMPARISON_ARTIFACT,
        ALFRED_RETRIEVAL_VS_LLM_ARTIFACT,
        REPEATED_EVIDENCE_ARTIFACT,
    )
    artifacts = {
        relative: json.loads((root / relative).read_text(encoding="utf-8"))
        for relative in used_artifacts
    }
    main_claim = claims[MAIN_CLAIM_ID]
    if main_claim.get("status") != "supported_synthetic":
        raise ValueError("main compositional table requires supported_synthetic status")
    main_evidence = _evidence(main_claim, COMPOSITIONAL_ARTIFACT)
    main_markdown, main_latex, main_pointers = _main_tables(
        main_claim,
        main_evidence,
        artifacts[COMPOSITIONAL_ARTIFACT],
    )
    component_claim = claims["C2_TYPED_COMPONENTS"]
    component_evidence = _evidence(component_claim, COMPONENT_ABLATION_ARTIFACT)
    component_markdown, component_latex, component_pointers = _component_ablation_tables(
        component_claim,
        component_evidence,
        artifacts[COMPONENT_ABLATION_ARTIFACT],
    )
    decision_claim = claims["C3_DECISION_UTILITY"]
    decision_evidence = _evidence(decision_claim, DECISION_UTILITY_ARTIFACT)
    decision_markdown, decision_latex, decision_pointers = _decision_utility_tables(
        decision_claim,
        decision_evidence,
        artifacts[DECISION_UTILITY_ARTIFACT],
    )
    observation_claim = claims["C4_INTERVAL_CENSORING"]
    observation_evidence = _evidence(observation_claim, OBSERVATION_PROCESS_ARTIFACT)
    observation_markdown, observation_latex, observation_pointers = _observation_process_tables(
        observation_claim,
        observation_evidence,
        artifacts[OBSERVATION_PROCESS_ARTIFACT],
    )
    observation_grounding_evidence = _evidence(
        observation_claim, OBSERVATION_GROUNDING_ARTIFACT
    )
    observation_grounding_markdown, observation_grounding_latex, observation_grounding_pointers = _observation_grounding_tables(
        observation_claim, observation_grounding_evidence, artifacts[OBSERVATION_GROUNDING_ARTIFACT]
    )
    language_claim = claims["C5_EXTERNAL_LANGUAGE"]
    baseline_evidence = _evidence(language_claim, ALFRED_RETRIEVAL_BASELINE_ARTIFACT)
    comparison_evidence = _evidence(language_claim, ALFRED_RETRIEVAL_COMPARISON_ARTIFACT)
    llm_evidence = _evidence(language_claim, ALFRED_RETRIEVAL_VS_LLM_ARTIFACT)
    language_markdown, language_latex, language_pointers = _external_language_tables(
        language_claim,
        baseline_evidence,
        comparison_evidence,
        llm_evidence,
        artifacts[ALFRED_RETRIEVAL_BASELINE_ARTIFACT],
        artifacts[ALFRED_RETRIEVAL_COMPARISON_ARTIFACT],
        artifacts[ALFRED_RETRIEVAL_VS_LLM_ARTIFACT],
    )
    boundary_rows, boundary_pointers = _boundary_evidence(claims, artifacts)
    boundary_markdown, boundary_latex = _boundary_tables(boundary_rows)

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "controlled_compositional_results.md": main_markdown,
        "controlled_compositional_results.tex": main_latex,
        "typed_component_ablation.md": component_markdown,
        "typed_component_ablation.tex": component_latex,
        "controlled_decision_utility.md": decision_markdown,
        "controlled_decision_utility.tex": decision_latex,
        "observation_process_bias.md": observation_markdown,
        "observation_process_bias.tex": observation_latex,
        "observation_grounding_decisions.md": observation_grounding_markdown,
        "observation_grounding_decisions.tex": observation_grounding_latex,
        "external_language_results.md": language_markdown,
        "external_language_results.tex": language_latex,
        "claim_boundaries.md": boundary_markdown,
        "claim_boundaries.tex": boundary_latex,
    }
    paths: list[Path] = []
    for name, content in outputs.items():
        path = output_dir / name
        path.write_text(content, encoding="utf-8", newline="\n")
        paths.append(path)

    if manuscript_path is not None:
        manuscript = manuscript_path.resolve()
        manuscript_text = manuscript.read_text(encoding="utf-8")
        manuscript_text = _replace_generated_block(
            manuscript_text, "controlled-compositional", main_markdown
        )
        manuscript_text = _replace_generated_block(
            manuscript_text, "typed-component-ablation", component_markdown
        )
        manuscript_text = _replace_generated_block(
            manuscript_text, "controlled-decision-utility", decision_markdown
        )
        manuscript_text = _replace_generated_block(
            manuscript_text, "observation-process-bias", observation_markdown
        )
        manuscript_text = _replace_generated_block(
            manuscript_text, "observation-grounding-decisions", observation_grounding_markdown
        )
        manuscript_text = _replace_generated_block(
            manuscript_text, "external-language-results", language_markdown
        )
        manuscript_text = _replace_generated_block(
            manuscript_text, "claim-boundaries", boundary_markdown
        )
        manuscript.write_text(manuscript_text, encoding="utf-8", newline="\n")

    source_hashes = {
        relative: _sha256(root / relative) for relative in used_artifacts
    }
    table_manifest = {
        "schema_version": 1,
        "generator": "scripts/build_paper_tables.py",
        "claim_manifest": {
            "path": "paper/claims.json",
            "sha256": _sha256(manifest_file),
        },
        "verification": {
            key: verification[key]
            for key in (
                "verified",
                "claims",
                "artifacts",
                "metric_checks",
                "status_counts",
            )
        },
        "sources": source_hashes,
        "bound_pointers": {
            MAIN_CLAIM_ID: main_pointers,
            "C2_TYPED_COMPONENTS": component_pointers,
            "C3_DECISION_UTILITY": decision_pointers,
            "C4_INTERVAL_CENSORING": {
                OBSERVATION_PROCESS_ARTIFACT: observation_pointers,
                OBSERVATION_GROUNDING_ARTIFACT: observation_grounding_pointers,
            },
            "C5_EXTERNAL_LANGUAGE": language_pointers,
            **boundary_pointers,
        },
        "outputs": {path.name: _sha256(path) for path in paths},
        "embedded_blocks": {
            "controlled-compositional": hashlib.sha256(main_markdown.encode("utf-8")).hexdigest(),
            "typed-component-ablation": hashlib.sha256(component_markdown.encode("utf-8")).hexdigest(),
            "controlled-decision-utility": hashlib.sha256(decision_markdown.encode("utf-8")).hexdigest(),
            "observation-process-bias": hashlib.sha256(observation_markdown.encode("utf-8")).hexdigest(),
            "observation-grounding-decisions": hashlib.sha256(observation_grounding_markdown.encode("utf-8")).hexdigest(),
            "external-language-results": hashlib.sha256(language_markdown.encode("utf-8")).hexdigest(),
            "claim-boundaries": hashlib.sha256(boundary_markdown.encode("utf-8")).hexdigest(),
        },
        "claim_boundary": manifest["paper_policy"]["result_boundary"],
    }
    manifest_output = output_dir / "table_manifest.json"
    manifest_output.write_text(
        json.dumps(table_manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    paths.append(manifest_output)
    return tuple(paths)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build evidence-locked Markdown and LaTeX paper tables."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("paper/claims.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("paper/tables"),
    )
    parser.add_argument(
        "--manuscript",
        type=Path,
        default=Path("paper/manuscript.md"),
    )
    parser.add_argument("--repository-root", type=Path)
    parser.add_argument(
        "--check",
        action="store_true",
        help="regenerate in a temporary directory and compare checked-in outputs",
    )
    args = parser.parse_args()
    root = (
        args.repository_root.resolve()
        if args.repository_root is not None
        else Path(__file__).resolve().parents[1]
    )
    manifest = args.manifest if args.manifest.is_absolute() else root / args.manifest
    output_dir = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    manuscript = args.manuscript if args.manuscript.is_absolute() else root / args.manuscript
    if args.check:
        with tempfile.TemporaryDirectory() as temporary:
            generated = build_paper_tables(
                manifest,
                Path(temporary),
                repository_root=root,
            )
            for candidate in generated:
                checked_in = output_dir / candidate.name
                if not checked_in.is_file() or candidate.read_bytes() != checked_in.read_bytes():
                    raise ValueError(f"checked-in paper table drifted: {checked_in}")
            manuscript_text = manuscript.read_text(encoding="utf-8")
            markdown_tables = [path for path in generated if path.suffix == ".md"]
            for name, generated_table in zip(
                (
                    "controlled-compositional",
                    "typed-component-ablation",
                    "controlled-decision-utility",
                    "observation-process-bias",
                    "observation-grounding-decisions",
                    "external-language-results",
                    "claim-boundaries",
                ),
                markdown_tables,
            ):
                if _embedded_table_content(manuscript_text, name) != generated_table.read_text(encoding="utf-8").strip():
                    raise ValueError(f"manuscript generated table block drifted: {name}")
        print(f"verified {len(generated)} checked-in paper table files and 7 manuscript blocks")
        return
    outputs = build_paper_tables(
        manifest,
        output_dir,
        repository_root=root,
        manuscript_path=manuscript,
    )
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
