"""Download frozen ADT previews and segmentation without logging CDN URLs."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests


def file_sha1(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def preview_target(output_root: Path, sequence_name: str, filename: str) -> Path:
    if Path(filename).name != filename:
        raise ValueError(f"unsafe preview filename for {sequence_name}")
    target = (output_root / sequence_name / filename).resolve()
    root = output_root.resolve()
    if root not in target.parents:
        raise ValueError(f"preview path escapes output root for {sequence_name}")
    return target


def download_preview(
    output_root: Path,
    sequence_name: str,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    url = metadata.get("download_url")
    filename = metadata.get("filename")
    expected_bytes = metadata.get("file_size_bytes")
    expected_sha1 = metadata.get("sha1sum")
    if (
        not isinstance(url, str)
        or urlparse(url).scheme != "https"
        or not isinstance(filename, str)
        or not isinstance(expected_bytes, int)
        or expected_bytes <= 0
        or not isinstance(expected_sha1, str)
    ):
        raise ValueError(f"invalid preview metadata for {sequence_name}")
    target = preview_target(output_root, sequence_name, filename)
    target.parent.mkdir(parents=True, exist_ok=True)
    if (
        target.is_file()
        and target.stat().st_size == expected_bytes
        and file_sha1(target) == expected_sha1
    ):
        return {"sequence_name": sequence_name, "status": "verified_existing"}

    partial = target.with_suffix(target.suffix + ".part")
    offset = partial.stat().st_size if partial.exists() else 0
    headers = {"Range": f"bytes={offset}-"} if offset else {}
    try:
        with requests.get(url, headers=headers, stream=True, timeout=(30, 120)) as response:
            response.raise_for_status()
            append = offset > 0 and response.status_code == 206
            mode = "ab" if append else "wb"
            with partial.open(mode) as handle:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        handle.write(chunk)
    except Exception as error:
        raise RuntimeError(
            f"preview download failed for {sequence_name}: {type(error).__name__}"
        ) from None
    if partial.stat().st_size != expected_bytes or file_sha1(partial) != expected_sha1:
        raise RuntimeError(f"preview integrity check failed for {sequence_name}")
    partial.replace(target)
    return {"sequence_name": sequence_name, "status": "downloaded_verified"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--cdn-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--downloader", default="aria_dataset_downloader")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--preview-only", action="store_true")
    parser.add_argument("--segmentation-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.preview_only and args.segmentation_only:
        raise SystemExit("preview-only and segmentation-only are mutually exclusive")
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    if selection.get("protocol_id") != "openprop-adt-pilot-selection-v1":
        raise SystemExit("unexpected ADT selection protocol")
    cdn = json.loads(args.cdn_manifest.read_text(encoding="utf-8"))
    args.output_root.mkdir(parents=True, exist_ok=True)

    if not args.segmentation_only:
        futures = []
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            for row in selection["sequences"]:
                name = row["sequence_name"]
                metadata = cdn["sequences"][name]["video_main_rgb"]
                futures.append(
                    executor.submit(download_preview, args.output_root, name, metadata)
                )
            completed = 0
            for future in as_completed(futures):
                result = future.result()
                completed += 1
                print(
                    f"preview={completed}/{len(futures)} "
                    f"sequence={result['sequence_name']} status={result['status']}"
                )

    if not args.preview_only:
        segmentation_names = [
            row["sequence_name"]
            for row in selection["sequences"]
            if row["download"]["segmentation"]
        ]
        subprocess.run(
            [
                args.downloader,
                "-c",
                str(args.cdn_manifest.resolve(strict=True)),
                "-o",
                str(args.output_root),
                "-l",
                *segmentation_names,
                "-d",
                "7",
            ],
            check=True,
        )


if __name__ == "__main__":
    main()
