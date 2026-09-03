"""Download the lightweight VISOR metadata and sparse annotation inventory."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


DOI_ID = "2v6cgv1x04ol22qp9rm9x2j6a7"
RESOURCE_BASE = f"https://data.bris.ac.uk/datasets/{DOI_ID}"
ANNOTATION_PAGES = {
    "train": "https://data.bris.ac.uk/data/dataset/b0cad676b4fa0bb37f0909be6c4f1db1",
    "val": "https://data.bris.ac.uk/data/dataset/9fc235ca3842ebe9854fabc6f4b74106",
}
METADATA_FILES = (
    "EPIC_100_noun_classes_v2.csv",
    "frame_mapping.json",
    "README.txt",
)
VIDEO_PATTERN = re.compile(r"P\d{2}_\d{2,3}\.json")


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def fetch(url: str, *, timeout: int = 180, attempts: int = 4) -> bytes:
    command = [
        "curl",
        "--location",
        "--fail",
        "--silent",
        "--show-error",
        "--retry",
        str(attempts - 1),
        "--retry-all-errors",
        "--max-time",
        str(timeout),
        url,
    ]
    result = subprocess.run(command, capture_output=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"official VISOR fetch failed: curl exit {result.returncode}")
    return result.stdout


def annotation_names(page: bytes) -> list[str]:
    names = sorted(set(VIDEO_PATTERN.findall(page.decode("utf-8", errors="replace"))))
    if not names:
        raise ValueError("VISOR annotation page contained no video JSON files")
    return names


def write_verified(path: Path, content: bytes, max_total: int, used: int) -> int:
    if used + len(content) > max_total:
        raise RuntimeError("VISOR screening download exceeds configured byte cap")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".json":
        json.loads(content.decode("utf-8"))
    path.write_bytes(content)
    return used + len(content)


def download_json(url: str, target: Path) -> int:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file():
        json.loads(target.read_text(encoding="utf-8"))
        return target.stat().st_size
    partial = target.with_suffix(target.suffix + ".part")
    command = [
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
        "1800",
        "--output",
        str(partial),
        url,
    ]
    result = subprocess.run(command, capture_output=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"official VISOR file fetch failed: curl exit {result.returncode}")
    json.loads(partial.read_text(encoding="utf-8"))
    partial.replace(target)
    return target.stat().st_size


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    parser.add_argument("--max-bytes", type=int, default=1024**3)
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    used = 0
    downloaded: list[dict[str, object]] = []
    split_counts: dict[str, int] = {}

    for filename in METADATA_FILES:
        target = args.output_root / "metadata" / filename
        content = target.read_bytes() if target.is_file() else fetch(f"{RESOURCE_BASE}/{filename}")
        used = write_verified(target, content, args.max_bytes, used)
        downloaded.append(
            {"path": target.relative_to(args.output_root).as_posix(), "bytes": len(content)}
        )

    for split, page_url in ANNOTATION_PAGES.items():
        names = annotation_names(fetch(page_url))
        split_counts[split] = len(names)
        jobs = {}
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for filename in names:
                target = args.output_root / "GroundTruth-SparseAnnotations" / "annotations" / split / filename
                url = (
                    f"{RESOURCE_BASE}/GroundTruth-SparseAnnotations/annotations/"
                    f"{split}/{filename}"
                )
                jobs[pool.submit(download_json, url, target)] = (filename, target)
            for index, future in enumerate(as_completed(jobs), start=1):
                filename, target = jobs[future]
                size = future.result()
                if used + size > args.max_bytes:
                    raise RuntimeError("VISOR screening download exceeds configured byte cap")
                used += size
                downloaded.append(
                    {"path": target.relative_to(args.output_root).as_posix(), "bytes": size}
                )
                print(
                    f"split={split} annotation={index}/{len(names)} file={filename}",
                    flush=True,
                )

    for row in downloaded:
        row["sha256"] = sha256(args.output_root / str(row["path"]))
    report = {
        "schema_version": 1,
        "protocol_id": "openprop-visor-screening-download-v1",
        "dataset_id": "epic_kitchens_visor",
        "source_doi": "10.5523/bris.2v6cgv1x04ol22qp9rm9x2j6a7",
        "license": "CC BY-NC 4.0",
        "license_accepted_by_user": True,
        "split_counts": split_counts,
        "file_count": len(downloaded),
        "total_bytes": used,
        "performance_evidence": False,
        "files": downloaded,
    }
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    args.audit_output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"downloaded files={len(downloaded)} bytes={used}")


if __name__ == "__main__":
    main()
