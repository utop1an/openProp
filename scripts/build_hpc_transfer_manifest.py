from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


DEFAULT_FILES = (
    "openprop-ai2thor.sif",
    "ai2thor-cloudrendering-f0825767-cache.tar.gz",
    "openprop-ai2thor.packages.txt",
    "openprop-hpc-source.tar.gz",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(directory: Path) -> dict[str, Any]:
    rows = []
    for name in DEFAULT_FILES:
        path = directory / name
        if not path.is_file():
            raise FileNotFoundError(f"missing HPC transfer artifact: {path}")
        rows.append({"path": name, "bytes": path.stat().st_size, "sha256": _sha256(path)})
    return {
        "schema_version": 1,
        "bundle_id": "openprop-hpc-transfer-v1",
        "ai2thor_version": "5.0.0",
        "ai2thor_commit": "f0825767cd50d69f666c7f282e54abfe58f1e917",
        "container_runtime_built_with": "SingularityCE 4.2.0",
        "source_platform": "WSL2 Ubuntu x86_64",
        "hpc_runtime_requirement": "Apptainer or Singularity with NVIDIA --nv and Vulkan ICD",
        "files": rows,
        "performance_evidence": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build or verify the OpenProp HPC transfer receipt.")
    parser.add_argument("--directory", type=Path, default=Path("dist"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    repository_root = Path(__file__).resolve().parents[1]
    directory = args.directory if args.directory.is_absolute() else repository_root / args.directory
    output = args.output or directory / "HPC_TRANSFER_MANIFEST.json"
    if not output.is_absolute():
        output = repository_root / output
    payload = build_manifest(directory)
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.check:
        if not output.is_file() or output.read_text(encoding="utf-8") != rendered:
            raise ValueError("HPC transfer manifest is missing or drifted")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
