"""Audit the frozen ADT pilot subset and emit a credential-free completion marker."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


REQUIRED_GROUNDTRUTH_FILES = {
    "instances.json",
    "scene_objects.csv",
    "2d_bounding_box.csv",
    "3d_bounding_box.csv",
    "aria_trajectory.csv",
    "metadata.json",
}


def digest(path: Path, algorithm: str) -> str:
    value = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def load_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def safe_child(root: Path, name: str) -> Path:
    if Path(name).name != name:
        raise ValueError(f"unsafe sequence name: {name!r}")
    root = root.resolve()
    child = (root / name).resolve()
    if root not in child.parents:
        raise ValueError(f"sequence path escapes dataset root: {name!r}")
    return child


def audit_sequence(
    root: Path,
    row: dict[str, Any],
    manifest_sequences: dict[str, Any],
) -> dict[str, Any]:
    name = row["sequence_name"]
    sequence_root = safe_child(root, name)
    if not sequence_root.is_dir():
        raise FileNotFoundError(f"missing sequence directory: {name}")

    missing_groundtruth = sorted(
        filename
        for filename in REQUIRED_GROUNDTRUTH_FILES
        if not (sequence_root / filename).is_file()
    )
    status_path = sequence_root / ".download_status.json"
    status = load_object(status_path) if status_path.is_file() else {}
    if missing_groundtruth or status.get("main_groundtruth") is not True:
        raise RuntimeError(
            f"ground-truth audit failed for {name}; missing={missing_groundtruth}"
        )

    manifest_row = manifest_sequences.get(name)
    if not isinstance(manifest_row, dict):
        raise KeyError(f"sequence absent from CDN manifest: {name}")
    preview_meta = manifest_row.get("video_main_rgb")
    if not isinstance(preview_meta, dict):
        raise KeyError(f"preview metadata absent from CDN manifest: {name}")
    filename = preview_meta.get("filename")
    expected_bytes = preview_meta.get("file_size_bytes")
    expected_sha1 = preview_meta.get("sha1sum")
    if not isinstance(filename, str) or Path(filename).name != filename:
        raise ValueError(f"invalid preview filename in CDN manifest: {name}")
    preview = sequence_root / filename
    if (
        not preview.is_file()
        or preview.stat().st_size != expected_bytes
        or digest(preview, "sha1") != expected_sha1
    ):
        raise RuntimeError(f"preview integrity audit failed for {name}")

    segmentation_expected = row.get("download", {}).get("segmentation") is True
    segmentation = sequence_root / "segmentations.vrs"
    if segmentation_expected and (
        not segmentation.is_file() or status.get("segmentation") is not True
    ):
        raise RuntimeError(f"segmentation audit failed for {name}")

    return {
        "sequence_name": name,
        "activity": row.get("activity"),
        "condition": row.get("condition"),
        "split": row.get("split"),
        "groundtruth_ready": True,
        "preview_ready": True,
        "segmentation_expected": segmentation_expected,
        "segmentation_ready": segmentation.is_file() if segmentation_expected else None,
    }


def marker_files(root: Path) -> list[dict[str, Any]]:
    files = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name == ".openprop-dataset-complete.json":
            continue
        files.append(
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": digest(path, "sha256"),
            }
        )
    return files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--cdn-manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--write-completion-marker", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selection = load_object(args.selection)
    manifest = load_object(args.cdn_manifest)
    manifest_sequences = manifest.get("sequences")
    if not isinstance(manifest_sequences, dict):
        raise ValueError("CDN manifest has no sequence map")
    rows = selection.get("sequences")
    if not isinstance(rows, list) or not rows:
        raise ValueError("selection has no sequences")

    audited = [audit_sequence(args.data_root, row, manifest_sequences) for row in rows]
    file_rows = marker_files(args.data_root)
    total_bytes = sum(row["bytes"] for row in file_rows)
    selection_sha256 = digest(args.selection, "sha256")
    manifest_sha256 = digest(args.cdn_manifest, "sha256")
    segmentation_count = sum(row["segmentation_ready"] is True for row in audited)
    report = {
        "schema_version": 1,
        "protocol_id": "openprop-adt-pilot-download-audit-v1",
        "dataset_id": "aria_digital_twin",
        "selection_sha256": selection_sha256,
        "source_manifest_sha256": manifest_sha256,
        "contains_download_urls": False,
        "sequence_count": len(audited),
        "groundtruth_ready_count": len(audited),
        "preview_ready_count": len(audited),
        "segmentation_ready_count": segmentation_count,
        "file_count": len(file_rows),
        "total_bytes": total_bytes,
        "split_scope": selection.get("split_scope"),
        "selection_uses_screening_outcomes": selection.get(
            "selection_uses_screening_outcomes"
        ),
        "performance_evidence": False,
        "sequences": audited,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    if args.write_completion_marker:
        marker = {
            "schema_version": 1,
            "dataset_id": "aria_digital_twin",
            "source_release": f"ADT manifest sha256:{manifest_sha256}",
            "license_accepted_by_user": True,
            "selection_sha256": selection_sha256,
            "scope": "OpenProp pilot subset: ground truth and preview for 63 sequences; segmentation for 18",
            "files": file_rows,
        }
        marker_path = args.data_root / ".openprop-dataset-complete.json"
        marker_path.write_text(
            json.dumps(marker, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    print(
        f"audited sequences={len(audited)} previews={len(audited)} "
        f"segmentations={segmentation_count} files={len(file_rows)} "
        f"bytes={total_bytes}"
    )


if __name__ == "__main__":
    main()
