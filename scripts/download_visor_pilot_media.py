"""Plan, download, and audit selected VISOR sparse RGB frame archives."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path, PurePosixPath
from typing import Any


DOI_ID = "2v6cgv1x04ol22qp9rm9x2j6a7"
RESOURCE_BASE = f"https://data.bris.ac.uk/datasets/{DOI_ID}"
CONTENT_LENGTH = re.compile(rb"^content-length:\s*(\d+)\s*$", re.IGNORECASE | re.MULTILINE)
ETAG = re.compile(rb"^etag:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def media_path(root: Path, official_split: str, participant: str, video_id: str) -> Path:
    for value in (official_split, participant, video_id):
        if Path(value).name != value:
            raise ValueError(f"unsafe VISOR path component: {value!r}")
    root = root.resolve()
    target = (
        root
        / "GroundTruth-SparseAnnotations"
        / "rgb_frames"
        / official_split
        / participant
        / f"{video_id}.zip"
    ).resolve()
    if root not in target.parents:
        raise ValueError("VISOR media path escapes dataset root")
    return target


def media_url(official_split: str, participant: str, video_id: str) -> str:
    return (
        f"{RESOURCE_BASE}/GroundTruth-SparseAnnotations/rgb_frames/"
        f"{official_split}/{participant}/{video_id}.zip"
    )


def probe(url: str) -> tuple[int, str | None]:
    result = subprocess.run(
        [
            "curl",
            "--head",
            "--location",
            "--fail",
            "--silent",
            "--show-error",
            "--retry",
            "3",
            "--max-time",
            "120",
            url,
        ],
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"VISOR media probe failed: curl exit {result.returncode}")
    lengths = [int(value) for value in CONTENT_LENGTH.findall(result.stdout)]
    if not lengths or lengths[-1] <= 0:
        raise RuntimeError("VISOR media probe returned no positive content length")
    etags = ETAG.findall(result.stdout)
    return lengths[-1], etags[-1].decode("utf-8", errors="replace") if etags else None


def verify_zip(path: Path) -> int:
    with zipfile.ZipFile(path) as archive:
        members = archive.infolist()
        if not members:
            raise ValueError(f"empty VISOR archive: {path}")
        for member in members:
            name = PurePosixPath(member.filename.replace("\\", "/"))
            if name.is_absolute() or ".." in name.parts:
                raise ValueError(f"unsafe member in VISOR archive: {path}")
        bad = archive.testzip()
        if bad is not None:
            raise ValueError(f"corrupt VISOR archive member: {bad}")
        image_count = sum(
            member.filename.lower().endswith((".jpg", ".jpeg", ".png"))
            for member in members
        )
    if image_count == 0:
        raise ValueError(f"VISOR archive contains no images: {path}")
    return image_count


def download(url: str, target: Path, expected_bytes: int) -> int:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file() and target.stat().st_size == expected_bytes:
        return verify_zip(target)
    partial = target.with_suffix(target.suffix + ".part")
    result = subprocess.run(
        [
            "curl",
            "--location",
            "--fail",
            "--silent",
            "--show-error",
            "--retry",
            "5",
            "--retry-all-errors",
            "--continue-at",
            "-",
            "--connect-timeout",
            "30",
            "--max-time",
            "3600",
            "--output",
            str(partial),
            url,
        ],
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"VISOR media download failed: curl exit {result.returncode}")
    if partial.stat().st_size != expected_bytes:
        raise RuntimeError(f"VISOR media byte count mismatch: {target.name}")
    image_count = verify_zip(partial)
    partial.replace(target)
    return image_count


def completion_files(root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name == ".openprop-dataset-complete.json" or path.suffix == ".part":
            continue
        rows.append(
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--plan-output", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    parser.add_argument("--max-bytes", type=int, default=5 * 1024**3)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--plan-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    videos = selection.get("videos")
    if not isinstance(videos, list) or not videos:
        raise ValueError("VISOR selection has no videos")

    plan_rows = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        jobs = {}
        for row in videos:
            video_id = row["video_id"]
            participant = row["participant_id"]
            official_split = row["official_split"]
            url = media_url(official_split, participant, video_id)
            jobs[pool.submit(probe, url)] = (row, url)
        for future in as_completed(jobs):
            row, url = jobs[future]
            size, etag = future.result()
            target = media_path(
                args.data_root, row["official_split"], row["participant_id"], row["video_id"]
            )
            plan_rows.append(
                {
                    "video_id": row["video_id"],
                    "participant_id": row["participant_id"],
                    "official_split": row["official_split"],
                    "split": row["split"],
                    "path": target.relative_to(args.data_root).as_posix(),
                    "declared_bytes": size,
                    "source_etag": etag,
                    "_url": url,
                }
            )
    total = sum(row["declared_bytes"] for row in plan_rows)
    if total > args.max_bytes:
        raise RuntimeError(f"selected VISOR media requires {total} bytes; cap is {args.max_bytes}")
    public_plan = {
        "schema_version": 1,
        "protocol_id": "openprop-visor-pilot-media-plan-v1",
        "source_doi": "10.5523/bris.2v6cgv1x04ol22qp9rm9x2j6a7",
        "selection_sha256": sha256(args.selection),
        "file_count": len(plan_rows),
        "declared_download_bytes": total,
        "contains_private_credentials": False,
        "performance_evidence": False,
        "files": [
            {key: value for key, value in row.items() if key != "_url"}
            for row in sorted(plan_rows, key=lambda value: value["video_id"])
        ],
    }
    args.plan_output.parent.mkdir(parents=True, exist_ok=True)
    args.plan_output.write_text(
        json.dumps(public_plan, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"planned files={len(plan_rows)} bytes={total}", flush=True)
    if args.plan_only:
        return

    media_audit = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        jobs = {
            pool.submit(
                download,
                row["_url"],
                args.data_root / row["path"],
                row["declared_bytes"],
            ): row
            for row in plan_rows
        }
        for index, future in enumerate(as_completed(jobs), start=1):
            row = jobs[future]
            image_count = future.result()
            target = args.data_root / row["path"]
            media_audit.append(
                {
                    "video_id": row["video_id"],
                    "split": row["split"],
                    "path": row["path"],
                    "bytes": target.stat().st_size,
                    "sha256": sha256(target),
                    "image_count": image_count,
                }
            )
            print(f"media={index}/{len(plan_rows)} video={row['video_id']}", flush=True)

    files = completion_files(args.data_root)
    report = {
        "schema_version": 1,
        "protocol_id": "openprop-visor-pilot-download-audit-v1",
        "dataset_id": "epic_kitchens_visor",
        "source_doi": "10.5523/bris.2v6cgv1x04ol22qp9rm9x2j6a7",
        "selection_sha256": sha256(args.selection),
        "media_file_count": len(media_audit),
        "media_bytes": sum(row["bytes"] for row in media_audit),
        "image_count": sum(row["image_count"] for row in media_audit),
        "participant_disjoint": selection.get("split_is_participant_disjoint"),
        "performance_evidence": False,
        "media": sorted(media_audit, key=lambda row: row["video_id"]),
    }
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    args.audit_output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    marker = {
        "schema_version": 1,
        "dataset_id": "epic_kitchens_visor",
        "source_release": "VISOR DOI 10.5523/bris.2v6cgv1x04ol22qp9rm9x2j6a7",
        "license_accepted_by_user": True,
        "scope": "all train/val sparse annotations and metadata; sparse RGB for 44 selected videos",
        "selection_sha256": sha256(args.selection),
        "files": files,
    }
    (args.data_root / ".openprop-dataset-complete.json").write_text(
        json.dumps(marker, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"audited media={len(media_audit)} images={report['image_count']} files={len(files)}")


if __name__ == "__main__":
    main()
