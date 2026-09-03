from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path

from openprop.alfred_adapter import load_alfred_language_dataset
from openprop.alfred_language_evaluation import (
    AlfredLanguageReport,
    AlfredLanguageStrategy,
    alfred_language_registry,
    collect_alfred_language_responses,
    evaluate_alfred_language,
    select_stratified_cases,
)
from openprop.alfred_ontology import fit_alfred_training_ontology
from openprop.language_temporal_grounding import RawLanguageResponse
from openprop.ollama import OllamaClient


def _payload(report: AlfredLanguageReport) -> dict[str, object]:
    value = asdict(report)
    value["strategy"] = report.strategy.value
    return value


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate typed goal parsing on held-out ALFRED human descriptions."
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--model", default=os.environ.get("OLLAMA_MODEL", "gemma3:4b"))
    parser.add_argument("--split", default="valid_unseen")
    parser.add_argument("--trajectories-per-task", type=int, default=10)
    parser.add_argument("--trajectory-offset", type=int, default=0)
    parser.add_argument("--annotation-index", type=int, default=0)
    parser.add_argument(
        "--ollama-host",
        default=os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434"),
    )
    parser.add_argument(
        "--responses-input",
        type=Path,
        default=None,
        help="Replay raw responses from an existing result without model calls.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/alfred_language_results.json"),
    )
    args = parser.parse_args()
    dataset = load_alfred_language_dataset(args.root, splits=(args.split,))
    cases = select_stratified_cases(
        dataset.cases,
        split=args.split,
        trajectories_per_task=args.trajectories_per_task,
        annotation_index=args.annotation_index,
        trajectory_offset=args.trajectory_offset,
    )
    registry = alfred_language_registry()
    ontology = fit_alfred_training_ontology(args.root)
    if args.responses_input is None:
        responses = collect_alfred_language_responses(
            cases,
            registry,
            OllamaClient(model=args.model, base_url=args.ollama_host),
        )
        response_source = "live Ollama"
    else:
        stored = json.loads(args.responses_input.read_text(encoding="utf-8"))
        responses = {
            query: RawLanguageResponse(**payload)
            for query, payload in stored["raw_responses"].items()
        }
        missing = {case.query for case in cases} - responses.keys()
        if missing:
            parser.error(f"responses input is missing {len(missing)} selected queries")
        response_source = str(args.responses_input)
    reports = [
        evaluate_alfred_language(cases, registry, AlfredLanguageStrategy.GOLD),
        evaluate_alfred_language(
            cases, registry, AlfredLanguageStrategy.LLM_STRICT, responses=responses
        ),
        evaluate_alfred_language(
            cases, registry, AlfredLanguageStrategy.LLM_TOLERANT, responses=responses
        ),
        evaluate_alfred_language(
            cases,
            registry,
            AlfredLanguageStrategy.LLM_SCHEMA_REPAIRED,
            responses=responses,
        ),
        evaluate_alfred_language(
            cases,
            registry,
            AlfredLanguageStrategy.LLM_ONTOLOGY_NORMALIZED,
            responses=responses,
            ontology=ontology,
        ),
        evaluate_alfred_language(
            cases,
            registry,
            AlfredLanguageStrategy.LLM_EVIDENCE_FUSED,
            responses=responses,
            ontology=ontology,
        ),
    ]
    payload = {
        "protocol": {
            "source": "official ALFRED 2.1.0 lite trajectories",
            "split": args.split,
            "model": args.model,
            "cases": len(cases),
            "unique_queries": len(responses),
            "selection": "fixed annotation from sorted trajectory window per supported task",
            "trajectories_per_task": args.trajectories_per_task,
            "trajectory_offset": args.trajectory_offset,
            "annotation_index": args.annotation_index,
            "primary_denominator": "all selected cases including request and parse failures",
            "evaluation_target": "human language to typed goal property frame",
            "visual_grounding_claim": "none",
            "temporal_observation_claim": "none",
            "repair_input_boundary": "property schema and raw response only",
            "ontology_input_boundary": "train PDDL label vocabulary and parsed frame only",
            "ontology": ontology.audit(),
            "selection_input_boundary": "query text, train PDDL vocabulary, property schema, and parsed frame only",
            "selection_evidence_requirement": "every added property logs a non-ambiguous query token span",
            "response_source": response_source,
        },
        "selected_cases": [
            {
                "case_id": case.case_id,
                "task_type": case.task_type,
                "floorplan": case.floorplan,
                "query": case.query,
                "source_path": case.source_path,
            }
            for case in cases
        ],
        "raw_responses": {
            query: asdict(response) for query, response in responses.items()
        },
        "reports": [_payload(report) for report in reports],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    for report in reports:
        print(
            f"{report.strategy.value}: parse={report.parse_success_rate:.3f} "
            f"property_f1={report.property_f1:.3f} value_recall={report.value_recall:.3f} "
            f"exact={report.exact_frame_accuracy:.3f}"
        )
    print(f"report: {args.output}")


if __name__ == "__main__":
    main()
