from __future__ import annotations

import argparse
import json
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Mapping

from openprop.association import AssociationPolicy
from openprop.query_decision import QueryDecisionPolicy
from openprop.visual_evaluation import NULL_ENTITY
from openprop.visual_replay import replay_visual_case
from openprop.vlm_replay import read_captured_vlm_response


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay one truth-free visual case through update and final query."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--response", type=Path, required=True)
    parser.add_argument("--assignment", choices=("independent", "global"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--association-threshold", type=float, default=0.8)
    parser.add_argument("--association-margin", type=float, default=0.15)
    parser.add_argument("--association-null-weight", type=float, default=0.05)
    parser.add_argument("--query-threshold", type=float, default=0.5)
    parser.add_argument("--query-margin", type=float, default=0.1)
    parser.add_argument("--query-null-weight", type=float, default=0.05)
    args = parser.parse_args()
    input_payload = _mapping(args.input)
    case_payload = _mapping(args.case)
    captured = read_captured_vlm_response(args.response, input_artifact=args.input)
    response = captured["response"]
    assert isinstance(response, Mapping)
    outcome = replay_visual_case(
        input_payload,
        case_payload,
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
    payload = {
        "schema_version": 1,
        "truth_used": False,
        "case_id": outcome.case_id,
        "assignment": outcome.assignment,
        "malformed_response": outcome.malformed_response,
        "response_error": outcome.response_error,
        "captured_response": {
            "provider": captured["provider"],
            "model": captured["model"],
            "system_id": captured["system_id"],
            "input_artifact_sha256": captured["input_artifact_sha256"],
        },
        "detections": [
            {
                "detection_id": item.detection_id,
                "frame_id": item.frame.frame_id,
                "property_name": item.property_name,
                "value": _jsonable(item.value),
                "detection_confidence": item.detection_confidence,
                "value_confidence": item.value_confidence,
                "region": _jsonable(item.region),
            }
            for item in outcome.run.detections
        ],
        "associations": [
            {
                "detection_id": item.detection.detection_id,
                "decision_entity_id": item.decision_entity_id,
                "accepted_entity_id": item.accepted_entity_id,
                "probabilities": {
                    **{candidate.entity_id: candidate.posterior for candidate in item.candidates},
                    NULL_ENTITY: item.null_probability,
                },
                "update_confidence": item.update_confidence,
                "reason": item.reason,
            }
            for item in outcome.run.hypotheses
        ],
        "proposals": [
            {
                "frame_id": item.frame_id,
                "entity_id": item.entity_id,
                "property_name": item.property_name,
                "observation": _jsonable(item.observation),
            }
            for item in outcome.run.proposals
        ],
        "query_decision": _jsonable(outcome.query_decision),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"case={outcome.case_id} assignment={outcome.assignment} "
        f"detections={len(outcome.run.detections)} "
        f"updates={len(outcome.run.proposals)} "
        f"query={outcome.query_decision.accepted_entity_id or NULL_ENTITY}"
    )
    print(f"output: {args.output}")


def _mapping(path: Path) -> Mapping[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"JSON input must be an object: {path}")
    return value


def _jsonable(value: object) -> object:
    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable(asdict(value))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


if __name__ == "__main__":
    main()
