from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path

from openprop.language_temporal_grounding import (
    LanguageTemporalReport,
    LanguageTemporalStrategy,
    RawLanguageResponse,
    collect_language_responses,
    evaluate_language_temporal_grounding,
)
from openprop.ollama import OllamaClient
from openprop.teach_audit import read_teach_audit_manifest
from openprop.teach_dialogue_alignment import teach_manifest_sha256
from openprop.teach_grounding import (
    TEACH_BOOLEAN_STATE_PROPERTIES,
    teach_grounding_registry,
)
from openprop.teach_layer_c import (
    prepare_teach_layer_c_cases,
    validate_teach_layer_c_gate,
)
from openprop.teach_layer_c_annotation import (
    TEACH_LAYER_C_MIN_PAIRWISE_AGREEMENT,
    apply_teach_layer_c_annotation_resolution,
    resolve_teach_layer_c_annotations,
)
from openprop.temporal_grounding import NoDecayPersistenceModel


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _report_payload(report: LanguageTemporalReport) -> dict[str, object]:
    payload = asdict(report)
    payload["strategy"] = report.strategy.value
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate TEACh Layer C type-oracle and predicted language frames."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--alignment-cases", type=Path, required=True)
    parser.add_argument("--feasibility-audit", type=Path, required=True)
    parser.add_argument("--responses-input", type=Path, default=None)
    parser.add_argument("--rich-annotation-files", nargs=3, type=Path, default=None)
    parser.add_argument(
        "--minimum-pairwise-agreement",
        type=float,
        default=TEACH_LAYER_C_MIN_PAIRWISE_AGREEMENT,
    )
    parser.add_argument("--model", default=os.environ.get("OLLAMA_MODEL", "gemma3:4b"))
    parser.add_argument(
        "--ollama-host",
        default=os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434"),
    )
    parser.add_argument(
        "--properties", nargs="+", default=list(TEACH_BOOLEAN_STATE_PROPERTIES)
    )
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/teach_layer_c_results.json")
    )
    args = parser.parse_args()

    manifest = args.manifest.resolve()
    alignment_path = args.alignment_cases.resolve()
    feasibility_path = args.feasibility_audit.resolve()
    alignment = json.loads(alignment_path.read_text(encoding="utf-8"))
    feasibility = json.loads(feasibility_path.read_text(encoding="utf-8"))
    sessions = read_teach_audit_manifest(manifest)
    properties = tuple(args.properties)
    manifest_hash = teach_manifest_sha256(manifest)
    validate_teach_layer_c_gate(
        alignment,
        feasibility,
        expected_manifest_sha256=manifest_hash,
    )
    prepared = prepare_teach_layer_c_cases(
        sessions,
        alignment,
        expected_manifest_sha256=manifest_hash,
        property_names=properties,
    )
    rich_prepared = None
    rich_annotation_paths: list[Path] = []
    if args.rich_annotation_files is not None:
        rich_annotation_paths = [path.resolve() for path in args.rich_annotation_files]
        if len(set(rich_annotation_paths)) != 3:
            parser.error("--rich-annotation-files must contain three distinct files")
        annotations = [
            json.loads(path.read_text(encoding="utf-8"))
            for path in rich_annotation_paths
        ]
        resolution = resolve_teach_layer_c_annotations(
            prepared,
            annotations,
            property_names=properties,
            min_pairwise_agreement=args.minimum_pairwise_agreement,
        )
        rich_prepared = apply_teach_layer_c_annotation_resolution(prepared, resolution)
    registry = teach_grounding_registry(properties)
    if args.responses_input is None:
        client = OllamaClient(model=args.model, base_url=args.ollama_host)
        responses = collect_language_responses(prepared.cases, registry, client)
        response_source = "live Ollama"
    else:
        stored = json.loads(args.responses_input.read_text(encoding="utf-8"))
        responses = {
            query: RawLanguageResponse(**payload)
            for query, payload in stored["raw_responses"].items()
        }
        response_source = str(args.responses_input.resolve())

    no_decay = NoDecayPersistenceModel()
    type_oracle_report = evaluate_language_temporal_grounding(
        prepared.cases,
        registry,
        LanguageTemporalStrategy.GOLD,
        persistence_model=no_decay,
    )
    rich_oracle_report = (
        evaluate_language_temporal_grounding(
            rich_prepared.cases,
            registry,
            LanguageTemporalStrategy.GOLD,
            persistence_model=no_decay,
        )
        if rich_prepared is not None
        else None
    )
    predicted_cases = (
        rich_prepared.cases if rich_prepared is not None else prepared.cases
    )
    predicted_reports = [
        evaluate_language_temporal_grounding(
            predicted_cases,
            registry,
            strategy,
            responses=responses,
            persistence_model=no_decay,
        )
        for strategy in (
            LanguageTemporalStrategy.LLM_STRICT,
            LanguageTemporalStrategy.LLM_TOLERANT,
            LanguageTemporalStrategy.LLM_SCHEMA_REPAIRED,
        )
    ]
    output = {
        "protocol": {
            "claim_scope": "Layer C language-frame grounding; not temporal-state evidence",
            "type_oracle_frame": "official target object type only",
            "rich_oracle_frame": (
                "independently annotated explicit referential attributes"
                if rich_prepared is not None
                else "unavailable: three validated annotations were not supplied"
            ),
            "predicted_report_reference": (
                "rich independent frame case population"
                if rich_prepared is not None
                else "type-only case population"
            ),
            "persistence": "no decay, held fixed to isolate language-frame effects",
            "primary_metric_denominator": "all aligned cases including input-coverage and parse failures",
            "matcher_truth_access": "none",
            "response_source": response_source,
            "model": args.model,
            "properties": list(properties),
        },
        "source": {
            "manifest": str(manifest),
            "manifest_sha256": _sha256(manifest),
            "alignment_cases": str(alignment_path),
            "alignment_cases_sha256": _sha256(alignment_path),
            "feasibility_audit": str(feasibility_path),
            "feasibility_audit_sha256": _sha256(feasibility_path),
            "rich_annotation_files": [str(path) for path in rich_annotation_paths],
            "rich_annotation_file_sha256": [
                _sha256(path) for path in rich_annotation_paths
            ],
        },
        "coverage": {
            "type_oracle": dict(prepared.audit),
            "rich_oracle": (
                dict(rich_prepared.audit) if rich_prepared is not None else None
            ),
        },
        "raw_responses": {
            query: asdict(response) for query, response in responses.items()
        },
        "type_oracle_report": _report_payload(type_oracle_report),
        "rich_oracle_report": (
            _report_payload(rich_oracle_report)
            if rich_oracle_report is not None
            else None
        ),
        "predicted_reports": [_report_payload(report) for report in predicted_reports],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "type_oracle": {
                    "top1_all_cases": type_oracle_report.top1_accuracy,
                },
                "rich_oracle": (
                    {"top1_all_cases": rich_oracle_report.top1_accuracy}
                    if rich_oracle_report is not None
                    else None
                ),
                "predicted": {
                    report.strategy.value: {
                        "parse_success": report.parse_success_rate,
                        "top1_all_cases": report.top1_accuracy,
                        "top1_parsed_only": report.conditional_top1_accuracy,
                    }
                    for report in predicted_reports
                },
            },
            indent=2,
            sort_keys=True,
        )
    )
    print(f"report: {args.output}")


if __name__ == "__main__":
    main()

