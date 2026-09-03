from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path

from openprop.association import AssociationPolicy
from openprop.query_decision import QueryDecisionPolicy
from openprop.visual_evaluation import write_visual_results_jsonl
from openprop.visual_replay import replay_visual_case
from openprop.visual_replay_evaluation import evaluate_visual_replay
from openprop.vlm_replay import read_captured_vlm_response


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Replay one hash-bound visual response without truth, then attach "
            "evaluation-only labels and write metric-ready JSONL."
        )
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--response", type=Path, required=True)
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--assignment", choices=("independent", "global"), required=True)
    parser.add_argument("--system", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path)
    parser.add_argument("--association-threshold", type=float, default=0.8)
    parser.add_argument("--association-margin", type=float, default=0.15)
    parser.add_argument("--association-null-weight", type=float, default=0.05)
    parser.add_argument("--query-threshold", type=float, default=0.5)
    parser.add_argument("--query-margin", type=float, default=0.1)
    parser.add_argument("--query-null-weight", type=float, default=0.05)
    parser.add_argument("--latency-seconds", type=float, default=0.0)
    parser.add_argument("--vlm-calls", type=int, default=1)
    args = parser.parse_args()
    audit_path = args.audit_output or args.output.with_suffix(args.output.suffix + ".audit.json")
    paths = tuple(
        path.resolve()
        for path in (
            args.input, args.case, args.response, args.truth, args.output, audit_path
        )
    )
    if len(set(paths)) != len(paths):
        raise ValueError("input, case, response, truth, output, and audit paths must differ")
    captured = read_captured_vlm_response(args.response, input_artifact=args.input)
    response = captured["response"]
    assert isinstance(response, Mapping)
    outcome = replay_visual_case(
        _mapping(args.input),
        _mapping(args.case),
        response,
        assignment=args.assignment,
        association_policy=AssociationPolicy(
            acceptance_threshold=args.association_threshold,
            margin_threshold=args.association_margin,
            null_weight=args.association_null_weight,
        ),
        query_policy=QueryDecisionPolicy(
            acceptance_threshold=args.query_threshold,
            margin_threshold=args.query_margin,
            null_weight=args.query_null_weight,
        ),
    )
    dataset = evaluate_visual_replay(
        outcome,
        _mapping(args.truth),
        system=args.system,
        latency_seconds=args.latency_seconds,
        vlm_calls=args.vlm_calls,
    )
    write_visual_results_jsonl(args.output, dataset)
    audit = {
        "schema_version": 1,
        "decision_frozen_before_truth_load": True,
        "test_truth_used_for_policy_selection": False,
        "system": args.system,
        "assignment": args.assignment,
        "malformed_response": outcome.malformed_response,
        "response_error": outcome.response_error,
        "captured_response": {
            "provider": captured["provider"],
            "model": captured["model"],
            "system_id": captured["system_id"],
        },
        "policies": {
            "association": {
                "acceptance_threshold": args.association_threshold,
                "margin_threshold": args.association_margin,
                "null_weight": args.association_null_weight,
            },
            "query": {
                "acceptance_threshold": args.query_threshold,
                "margin_threshold": args.query_margin,
                "null_weight": args.query_null_weight,
            },
        },
        "artifacts": {
            name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in (
                ("input", args.input),
                ("case", args.case),
                ("response", args.response),
                ("truth", args.truth),
                ("output", args.output),
            )
        },
        "denominators": {
            "property": len(dataset.properties),
            "association": len(dataset.associations),
            "query": len(dataset.queries),
        },
    }
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"system={args.system} assignment={args.assignment} "
        f"property_rows={len(dataset.properties)} "
        f"association_rows={len(dataset.associations)} "
        f"query_rows={len(dataset.queries)}"
    )
    print(f"output: {args.output}")
    print(f"audit: {audit_path}")


def _mapping(path: Path) -> Mapping[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"JSON input must be an object: {path}")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    main()
