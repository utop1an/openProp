from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from openprop.teach_audit import read_teach_audit_manifest
from openprop.teach_dialogue_alignment import teach_manifest_sha256
from openprop.teach_grounding import TEACH_BOOLEAN_STATE_PROPERTIES
from openprop.teach_layer_c import prepare_teach_layer_c_cases, validate_teach_layer_c_gate
from openprop.teach_layer_c_annotation import (
    TEACH_LAYER_C_MIN_PAIRWISE_AGREEMENT,
    resolve_teach_layer_c_annotations,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Resolve three target-blind TEACh Layer C semantic annotations."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--alignment-cases", type=Path, required=True)
    parser.add_argument("--feasibility-audit", type=Path, required=True)
    parser.add_argument("--annotation-files", nargs=3, type=Path, required=True)
    parser.add_argument(
        "--properties", nargs="+", default=list(TEACH_BOOLEAN_STATE_PROPERTIES)
    )
    parser.add_argument(
        "--minimum-pairwise-agreement",
        type=float,
        default=TEACH_LAYER_C_MIN_PAIRWISE_AGREEMENT,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/teach_layer_c_rich_frames.json"),
    )
    args = parser.parse_args()
    annotation_paths = [path.resolve() for path in args.annotation_files]
    if len(set(annotation_paths)) != 3:
        parser.error("--annotation-files must contain three distinct files")

    manifest = args.manifest.resolve()
    alignment_path = args.alignment_cases.resolve()
    feasibility_path = args.feasibility_audit.resolve()
    alignment = json.loads(alignment_path.read_text(encoding="utf-8"))
    feasibility = json.loads(feasibility_path.read_text(encoding="utf-8"))
    manifest_hash = teach_manifest_sha256(manifest)
    validate_teach_layer_c_gate(
        alignment,
        feasibility,
        expected_manifest_sha256=manifest_hash,
    )
    prepared = prepare_teach_layer_c_cases(
        read_teach_audit_manifest(manifest),
        alignment,
        expected_manifest_sha256=manifest_hash,
        property_names=tuple(args.properties),
    )
    annotations = [
        json.loads(path.read_text(encoding="utf-8")) for path in annotation_paths
    ]
    resolution = resolve_teach_layer_c_annotations(
        prepared,
        annotations,
        property_names=tuple(args.properties),
        min_pairwise_agreement=args.minimum_pairwise_agreement,
    )
    payload = {
        "protocol": {
            "claim_scope": "independent text-frame annotation; not model performance",
            "properties": list(args.properties),
            "target_candidate_model_blind": True,
        },
        "source": {
            "manifest": str(manifest),
            "manifest_sha256": _sha256(manifest),
            "alignment_cases": str(alignment_path),
            "alignment_cases_sha256": _sha256(alignment_path),
            "feasibility_audit": str(feasibility_path),
            "feasibility_audit_sha256": _sha256(feasibility_path),
            "annotation_files": [str(path) for path in annotation_paths],
            "annotation_file_sha256": [_sha256(path) for path in annotation_paths],
        },
        "audit": dict(resolution.audit),
        "frames": [
            {
                "case_id": case_id,
                "query": frame.text,
                "constraints": [
                    {
                        "property_name": item.property_name,
                        "desired_value": item.desired_value,
                        "relevance": item.relevance,
                    }
                    for item in frame.constraints
                ],
            }
            for case_id, frame in sorted(resolution.frames.items())
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"cases={resolution.audit['cases']} "
        f"pairwise_agreement={resolution.audit['pairwise_semantic_agreement']:.3f}"
    )
    print(f"report: {args.output}")


if __name__ == "__main__":
    main()

