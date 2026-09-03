from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from openprop.teach_audit import (
    audit_teach_sessions,
    read_teach_audit_manifest,
    write_teach_audit_report,
)
from openprop.teach_experiment import (
    prepare_teach_layer_b_experiment,
    run_teach_layer_b_experiment,
    write_teach_layer_b_report,
)
from openprop.teach_grounding import TEACH_BOOLEAN_STATE_PROPERTIES

def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run leakage-safe TEACh Layer B persistence and grounding evaluation."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--audit-output",
        type=Path,
        default=Path("artifacts/teach_feasibility_audit.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/teach_layer_b_results.json"),
    )
    parser.add_argument(
        "--properties",
        nargs="+",
        default=list(TEACH_BOOLEAN_STATE_PROPERTIES),
    )
    parser.add_argument(
        "--half-life-grid-hours",
        type=float,
        nargs="+",
        default=[0.25, 1.0, 4.0, 16.0, 64.0, 256.0],
    )
    parser.add_argument("--factorized-epochs", type=int, default=1200)
    args = parser.parse_args()
    property_names = tuple(args.properties)
    sessions = read_teach_audit_manifest(args.manifest)
    audit = audit_teach_sessions(sessions, property_names=property_names)
    write_teach_audit_report(args.audit_output, audit)
    gate = audit["feasibility_gate"]
    if not gate["layer_b_ready"]:
        print("Layer B gate failed: " + ", ".join(gate["failed_checks"]))
        print(f"audit: {args.audit_output}")
        raise SystemExit(2)
    prepared = prepare_teach_layer_b_experiment(
        sessions,
        property_names=property_names,
    )
    report = run_teach_layer_b_experiment(
        prepared,
        gate,
        half_life_grid_hours=tuple(args.half_life_grid_hours),
        factorized_epochs=args.factorized_epochs,
    )
    manifest_path = args.manifest.resolve()
    audit_path = args.audit_output.resolve()
    report["source"] = {
        "manifest": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "feasibility_audit": str(audit_path),
        "feasibility_audit_sha256": _sha256(audit_path),
        "feasibility_profile": gate["criteria"]["profile"],
        "layer_b_ready": True,
        "properties": list(property_names),
    }
    write_teach_layer_b_report(args.output, report)
    print(
        json.dumps(
            {
                name: values["grounding"]["top1_accuracy"]
                for name, values in report["test"].items()
            },
            indent=2,
            sort_keys=True,
        )
    )
    print(f"report: {args.output}")


if __name__ == "__main__":
    main()
