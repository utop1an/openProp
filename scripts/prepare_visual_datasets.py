from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _marker_status(marker: Path, dataset_id: str) -> tuple[bool, dict[str, Any] | None]:
    if not marker.is_file():
        return False, None
    payload = _load_json(marker)
    if payload.get("dataset_id") != dataset_id:
        raise ValueError(f"dataset marker ID mismatch: {marker}")
    if payload.get("license_accepted_by_user") is not True:
        raise ValueError(f"dataset marker lacks explicit license acceptance: {marker}")
    files = payload.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError(f"dataset marker must enumerate at least one file: {marker}")
    marker_root = marker.parent.resolve()
    for row in files:
        if not isinstance(row, dict) or not isinstance(row.get("path"), str):
            raise ValueError(f"malformed file row in marker: {marker}")
        candidate = (marker.parent / row["path"]).resolve()
        try:
            candidate.relative_to(marker_root)
        except ValueError as error:
            raise ValueError(f"dataset marker path escapes its root: {candidate}") from error
        if not candidate.is_file():
            raise FileNotFoundError(f"dataset file is missing: {candidate}")
        if row.get("bytes") != candidate.stat().st_size:
            raise ValueError(f"dataset byte count drifted: {candidate}")
        if row.get("sha256") != _sha256(candidate):
            raise ValueError(f"dataset hash drifted: {candidate}")
    return True, payload


def build_status(registry_path: Path, data_root: Path, repository_root: Path) -> dict[str, Any]:
    registry = _load_json(registry_path)
    if registry.get("protocol_id") != "openprop-visual-datasets-v1":
        raise ValueError("unsupported visual dataset registry")
    datasets = registry.get("datasets")
    if not isinstance(datasets, list) or not datasets:
        raise ValueError("visual dataset registry has no datasets")

    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for dataset in datasets:
        if not isinstance(dataset, dict) or not isinstance(dataset.get("id"), str):
            raise ValueError("malformed dataset registry row")
        dataset_id = dataset["id"]
        if dataset_id in seen:
            raise ValueError(f"duplicate dataset ID: {dataset_id}")
        seen.add(dataset_id)

        repository_artifacts: list[dict[str, Any]] = []
        artifacts_ready = True
        for relative in dataset.get("required_repository_artifacts", []):
            path = (repository_root / relative).resolve()
            exists = path.is_file()
            artifacts_ready = artifacts_ready and exists
            repository_artifacts.append(
                {
                    "path": relative,
                    "exists": exists,
                    "bytes": path.stat().st_size if exists else None,
                    "sha256": _sha256(path) if exists else None,
                }
            )

        credential_env = dataset.get("credential_env")
        credential_configured = True
        credential_kind = "not_required"
        if isinstance(credential_env, str):
            value = os.environ.get(credential_env)
            credential_configured = bool(value)
            credential_kind = "configured" if value else "missing"
            if value and credential_env.endswith("_FILE"):
                credential_configured = Path(value).expanduser().is_file()
                credential_kind = "configured_file" if credential_configured else "missing_file"

        marker_relative = dataset.get("completion_marker")
        marker_ready = True
        marker_summary = None
        if isinstance(marker_relative, str):
            marker_ready, marker_payload = _marker_status(data_root / marker_relative, dataset_id)
            if marker_payload is not None:
                marker_summary = {
                    "file_count": len(marker_payload["files"]),
                    "source_release": marker_payload.get("source_release"),
                }

        ready = artifacts_ready and marker_ready and credential_configured
        if dataset_id == "ai2thor_ithor":
            # The Unity binary is packaged separately because it is platform-specific;
            # repository protocol artifacts are enough to call this dataset layer ready.
            ready = artifacts_ready
        rows.append(
            {
                "id": dataset_id,
                "priority": dataset.get("priority"),
                "claim_role": dataset.get("claim_role"),
                "end_to_end_query_claim_eligible": bool(
                    dataset.get("end_to_end_query_claim_eligible", False)
                ),
                "repository_artifacts_ready": artifacts_ready,
                "repository_artifacts": repository_artifacts,
                "credential_status": credential_kind,
                "completion_marker_ready": marker_ready,
                "completion_marker_summary": marker_summary,
                "ready": ready,
            }
        )

    return {
        "schema_version": 1,
        "protocol_id": registry["protocol_id"],
        "registry_path": registry_path.as_posix(),
        "data_root": data_root.as_posix(),
        "datasets": rows,
        "required_now_ready": all(
            row["ready"] for row in rows if row["priority"] == "required_now"
        ),
        "real_world_claim_ready": all(
            row["ready"]
            for row in rows
            if row["priority"] == "required_before_real_world_claim"
        ),
        "performance_evidence": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Initialize and audit OpenProp visual dataset material without downloading licensed media."
    )
    parser.add_argument(
        "--registry", type=Path, default=Path("data/visual/registry.json")
    )
    parser.add_argument(
        "--data-root", type=Path, default=Path("artifacts/external/visual")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/visual_dataset_status.json")
    )
    parser.add_argument("--initialize", action="store_true")
    parser.add_argument("--require-ready", action="append", default=[])
    args = parser.parse_args()

    repository_root = Path(__file__).resolve().parents[1]
    registry_path = (
        args.registry if args.registry.is_absolute() else repository_root / args.registry
    )
    data_root = args.data_root if args.data_root.is_absolute() else repository_root / args.data_root
    output = args.output if args.output.is_absolute() else repository_root / args.output

    registry = _load_json(registry_path)
    if args.initialize:
        data_root.mkdir(parents=True, exist_ok=True)
        for dataset in registry.get("datasets", []):
            (data_root / dataset["id"]).mkdir(parents=True, exist_ok=True)

    status = build_status(registry_path, data_root, repository_root)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(status, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(status, ensure_ascii=False, indent=2, sort_keys=True))

    by_id = {row["id"]: row for row in status["datasets"]}
    unknown = sorted(set(args.require_ready) - set(by_id))
    if unknown:
        raise ValueError(f"unknown required dataset IDs: {unknown}")
    missing = [dataset_id for dataset_id in args.require_ready if not by_id[dataset_id]["ready"]]
    if missing:
        raise SystemExit(f"required datasets are not ready: {missing}")


if __name__ == "__main__":
    main()
