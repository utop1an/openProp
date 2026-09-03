from __future__ import annotations

import argparse
import json
from pathlib import Path

from openprop.teach_audit import read_teach_audit_manifest
from openprop.teach_dialogue_alignment import teach_manifest_sha256
from openprop.teach_grounding import TEACH_BOOLEAN_STATE_PROPERTIES
from openprop.teach_layer_c import prepare_teach_layer_c_cases, validate_teach_layer_c_gate
from openprop.teach_layer_c_annotation import build_teach_layer_c_annotation_template


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create three independent target-blind TEACh Layer C annotation templates."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--alignment-cases", type=Path, required=True)
    parser.add_argument("--feasibility-audit", type=Path, required=True)
    parser.add_argument("--annotator-ids", nargs=3, required=True)
    parser.add_argument("--outputs", nargs=3, type=Path, required=True)
    parser.add_argument(
        "--properties", nargs="+", default=list(TEACH_BOOLEAN_STATE_PROPERTIES)
    )
    args = parser.parse_args()
    if len(set(args.annotator_ids)) != 3 or any(not value.strip() for value in args.annotator_ids):
        parser.error("--annotator-ids must contain three distinct nonempty IDs")
    resolved_outputs = [path.resolve() for path in args.outputs]
    if len(set(resolved_outputs)) != 3:
        parser.error("--outputs must contain three distinct files")

    manifest = args.manifest.resolve()
    alignment = json.loads(args.alignment_cases.read_text(encoding="utf-8"))
    feasibility = json.loads(args.feasibility_audit.read_text(encoding="utf-8"))
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
    for annotator_id, output_path in zip(
        args.annotator_ids, resolved_outputs, strict=True
    ):
        template = build_teach_layer_c_annotation_template(
            prepared,
            annotator_id=annotator_id,
            property_names=tuple(args.properties),
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(template, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"{annotator_id}: {output_path}")


if __name__ == "__main__":
    main()

