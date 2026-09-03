from __future__ import annotations

import argparse
from pathlib import Path

from openprop.teach_audit import read_teach_audit_manifest
from openprop.teach_dialogue_alignment import (
    TEACH_DIALOGUE_ALIGNMENT_SAMPLE_SEED,
    audit_teach_dialogue_alignments,
    build_teach_dialogue_alignment_label_template,
    teach_manifest_sha256,
    write_teach_dialogue_alignment_json,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Construct frozen high-precision TEACh dialogue/object alignments and "
            "an incomplete manual-label template."
        )
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--output-cases",
        type=Path,
        default=Path("artifacts/teach_dialogue_alignment_cases.json"),
    )
    parser.add_argument(
        "--output-label-template",
        type=Path,
        default=Path("artifacts/teach_dialogue_alignment_labels.json"),
    )
    parser.add_argument("--sample-size", type=int, default=50)
    parser.add_argument(
        "--sample-seed", type=int, default=TEACH_DIALOGUE_ALIGNMENT_SAMPLE_SEED
    )
    args = parser.parse_args()

    sessions = read_teach_audit_manifest(args.manifest)
    report = audit_teach_dialogue_alignments(
        sessions,
        frozen_manifest_sha256=teach_manifest_sha256(args.manifest),
    )
    if report["sessions_with_game_file"] != report["sessions"]:
        raise ValueError(
            "every manifest row requires game_file before Layer C cases can be frozen"
        )
    template = build_teach_dialogue_alignment_label_template(
        report,
        sample_size=args.sample_size,
        seed=args.sample_seed,
    )
    write_teach_dialogue_alignment_json(args.output_cases, report)
    write_teach_dialogue_alignment_json(args.output_label_template, template)
    print(
        f"sessions={report['sessions']} "
        f"successful_object_interactions={report['successful_object_interactions']} "
        f"aligned_cases={report['aligned_cases']} "
        f"manual_sample={template['sample_size']}"
    )
    print(f"cases: {args.output_cases}")
    print(f"label template: {args.output_label_template}")


if __name__ == "__main__":
    main()
