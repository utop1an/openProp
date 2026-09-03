from __future__ import annotations

import argparse
import json
from pathlib import Path

from analyze_alfred_retrieval_comparison import METRICS, _paired_cluster_bootstrap


def _load_llm_rows(path: Path) -> dict[str, dict[str, object]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    report = next(
        item for item in payload["reports"] if item["strategy"] == "llm-evidence-fused"
    )
    return {row["case_id"]: row for row in report["results"]}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare frozen ALFRED LLM confirmation runs with retrieval fusion."
    )
    parser.add_argument(
        "--retrieval",
        type=Path,
        default=Path("artifacts/alfred_retrieval_baseline.json"),
    )
    parser.add_argument(
        "--gemma",
        type=Path,
        default=Path("artifacts/alfred_selection_confirmation_gemma3_4b.json"),
    )
    parser.add_argument(
        "--llama",
        type=Path,
        default=Path("artifacts/alfred_selection_confirmation_llama3_2.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/alfred_retrieval_vs_llm.json"),
    )
    parser.add_argument("--samples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260826)
    args = parser.parse_args()
    retrieval_payload = json.loads(args.retrieval.read_text(encoding="utf-8"))
    retrieval_rows = retrieval_payload["confirmation"]["results"]
    report: dict[str, object] = {
        "protocol": {
            "sample": "frozen valid_unseen confirmation sample",
            "pairing": "case_id",
            "bootstrap_cluster": "task_id",
            "bootstrap_stratum": "task_type",
            "method_selection_labels_used": False,
        }
    }
    for model_index, (model, path) in enumerate(
        (("gemma3_4b", args.gemma), ("llama3_2", args.llama))
    ):
        llm_rows = _load_llm_rows(path)
        retrieval_ids = {row["case_id"] for row in retrieval_rows}
        if retrieval_ids != set(llm_rows):
            raise ValueError(f"{model} does not contain the same confirmation cases")
        merged = []
        for row in retrieval_rows:
            llm = llm_rows[row["case_id"]]
            item = dict(row)
            item["retrieval_property_f1"] = row["bm25_evidence_property_f1"]
            item["retrieval_value_recall"] = row["bm25_evidence_value_recall"]
            item["retrieval_exact_frame"] = row["bm25_evidence_exact_frame"]
            item["llm_property_f1"] = llm["property_f1"]
            item["llm_value_recall"] = llm["value_recall"]
            item["llm_exact_frame"] = (
                float(llm["property_f1"]) == 1.0
                and float(llm["value_recall"]) == 1.0
            )
            merged.append(item)
        report[model] = {
            metric: _paired_cluster_bootstrap(
                merged,
                left="retrieval",
                right="llm",
                metric=metric,
                samples=args.samples,
                seed=args.seed + model_index * 10 + metric_index,
            )
            for metric_index, metric in enumerate(METRICS)
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    for model in ("gemma3_4b", "llama3_2"):
        print(model)
        for metric in METRICS:
            result = report[model][metric]
            print(
                f"  {metric}: delta={result['delta']:.3f} "
                f"CI=[{result['ci_95'][0]:.3f},{result['ci_95'][1]:.3f}] "
                f"W/T/L={result['wins']}/{result['ties']}/{result['losses']}"
            )
    print(f"report: {args.output}")


if __name__ == "__main__":
    main()
