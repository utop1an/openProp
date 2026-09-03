from __future__ import annotations

import argparse
from pathlib import Path

from openprop.teach_audit import (
    audit_teach_sessions,
    read_teach_audit_manifest,
    write_teach_audit_report,
)
from openprop.teach_dialogue_alignment import (
    TEACH_DIALOGUE_ALIGNMENT_POLICY_ID,
    audit_teach_dialogue_alignments,
    teach_manifest_sha256,
)
from openprop.teach_feasibility import read_teach_dialogue_alignment_audit


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit prepared TEACh replay data for longitudinal grounding."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/teach_feasibility_audit.json"),
    )
    parser.add_argument(
        "--require-ready",
        choices=("layer-a", "layer-b", "main"),
        help="exit with status 2 unless the selected feasibility layer passes",
    )
    parser.add_argument(
        "--dialogue-alignment-audit",
        type=Path,
        help="frozen manual alignment audit JSON used only for Layer C readiness",
    )
    args = parser.parse_args()
    sessions = read_teach_audit_manifest(args.manifest)
    manifest_hash = teach_manifest_sha256(args.manifest)
    automatic_alignment = audit_teach_dialogue_alignments(
        sessions,
        frozen_manifest_sha256=manifest_hash,
    )
    dialogue_alignment = (
        read_teach_dialogue_alignment_audit(
            args.dialogue_alignment_audit,
            expected_manifest_sha256=manifest_hash,
            expected_policy_id=TEACH_DIALOGUE_ALIGNMENT_POLICY_ID,
            expected_aligned_case_ids=automatic_alignment["case_ids"],
            expected_aligned_cases=automatic_alignment["aligned_cases"],
        )
        if args.dialogue_alignment_audit
        else None
    )
    report = audit_teach_sessions(
        sessions,
        dialogue_alignment=dialogue_alignment,
        dialogue_alignment_auto=automatic_alignment,
    )
    write_teach_audit_report(args.output, report)
    totals = report["totals"]
    print(
        f"sessions={totals['sessions']} floorplans={totals['floorplans']} "
        f"snapshots={totals['snapshots']} "
        f"visible_entities={totals['unique_visible_entities']}"
    )
    gate = report["feasibility_gate"]
    print(
        f"ready: layer-a={gate['layer_a_ready']} "
        f"layer-b={gate['layer_b_ready']} main={gate['main_claim_ready']}"
    )
    print(
        f"dialogue: games={automatic_alignment['sessions_with_game_file']}/"
        f"{automatic_alignment['sessions']} aligned={automatic_alignment['aligned_cases']}"
    )
    print(f"report: {args.output}")
    readiness = {
        "layer-a": gate["layer_a_ready"],
        "layer-b": gate["layer_b_ready"],
        "main": gate["main_claim_ready"],
    }
    if args.require_ready and not readiness[args.require_ready]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
