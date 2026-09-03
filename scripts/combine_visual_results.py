from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from openprop.visual_evaluation import (
    read_visual_results_jsonl,
    write_visual_results_jsonl,
)
from openprop.visual_matrix import combine_visual_evaluation_datasets


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Combine visual result JSONLs with paired-population gates."
    )
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--require-systems", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path)
    args = parser.parse_args()
    audit_path = args.audit_output or args.output.with_suffix(args.output.suffix + ".audit.json")
    resolved = [path.resolve() for path in (*args.input, args.output, audit_path)]
    if len(resolved) != len(set(resolved)):
        raise ValueError("matrix inputs, output, and audit paths must differ")
    dataset, report = combine_visual_evaluation_datasets(
        tuple(read_visual_results_jsonl(path) for path in args.input),
        required_systems=args.require_systems,
    )
    write_visual_results_jsonl(args.output, dataset)
    audit = {
        **report,
        "inputs": [
            {"path": str(path), "sha256": _sha256(path)} for path in args.input
        ],
        "output": {"path": str(args.output), "sha256": _sha256(args.output)},
    }
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"systems={','.join(report['systems'])} "
        f"property_rows={len(dataset.properties)} "
        f"association_rows={len(dataset.associations)} "
        f"query_rows={len(dataset.queries)}"
    )
    print(f"output: {args.output}")
    print(f"audit: {audit_path}")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    main()
