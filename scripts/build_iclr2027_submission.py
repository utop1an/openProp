from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper" / "iclr2027"
BUILD = PAPER / "build"
OUTPUT = ROOT / "output" / "pdf" / "OpenProp_ICLR2027_submission.pdf"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def validate_source() -> None:
    integrity = json.loads((PAPER / "template-integrity.json").read_text(encoding="utf-8"))
    for relative, expected in integrity["files"].items():
        actual = sha256(PAPER / relative)
        if actual != expected:
            raise SystemExit(f"Official template drift: {relative}: {actual} != {expected}")

    source = (PAPER / "main.tex").read_text(encoding="utf-8")
    required = [
        r"\usepackage{iclr2027_conference,times}",
        r"\author{Anonymous Authors}",
        r"\label{sec:main-end}",
        r"\section*{AI Use Statement}",
        r"\section*{Ethics Statement}",
        r"\section*{Reproducibility Statement}",
    ]
    for token in required:
        if token not in source:
            raise SystemExit(f"Required submission token missing: {token}")
    if re.search(r"(?im)^\s*\\iclrfinalcopy", source):
        raise SystemExit("Submission source enables camera-ready mode and is not anonymous")
    forbidden = [r"\bTODO\b", r"\bTBD\b", r"SUBMISSION BLOCKER", r"Anonymous University"]
    for pattern in forbidden:
        if re.search(pattern, source, flags=re.IGNORECASE):
            raise SystemExit(f"Forbidden placeholder or identity pattern: {pattern}")


def label_page(aux: str, label: str) -> int:
    match = re.search(r"\\newlabel\{" + re.escape(label) + r"\}\{\{[^}]*\}\{(\d+)\}", aux)
    if not match:
        raise SystemExit(f"Could not find page for label {label}")
    return int(match.group(1))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and validate the anonymous ICLR 2027 submission")
    parser.add_argument("--tectonic", type=Path, required=True)
    args = parser.parse_args()
    tectonic = args.tectonic.resolve()
    if not tectonic.is_file():
        raise SystemExit(f"Tectonic executable not found: {tectonic}")

    validate_source()
    BUILD.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["SOURCE_DATE_EPOCH"] = "1787702400"
    command = [
        str(tectonic),
        "main.tex",
        "--outdir", str(BUILD),
        "--keep-intermediates",
        "--keep-logs",
        "-Z", "search-path=official",
        "-Z", "deterministic-mode",
    ]
    completed = subprocess.run(command, cwd=PAPER, env=env, text=True, capture_output=True)
    combined = completed.stdout + "\n" + completed.stderr
    print(combined)
    if completed.returncode:
        raise SystemExit(completed.returncode)

    log = (BUILD / "main.log").read_text(encoding="utf-8", errors="replace")
    bad_patterns = [
        r"LaTeX Warning: There were undefined references",
        r"Citation `[^']+' on page .* undefined",
        r"Reference `[^']+' on page .* undefined",
        r"Empty bibliography",
        r"Overfull \\hbox",
    ]
    for pattern in bad_patterns:
        if re.search(pattern, log):
            raise SystemExit(f"LaTeX validation failed: {pattern}")

    aux = (BUILD / "main.aux").read_text(encoding="utf-8", errors="replace")
    main_end = label_page(aux, "sec:main-end")
    if main_end > 9:
        raise SystemExit(f"Main text ends on page {main_end}; ICLR 2027 limit is 9")

    pdf = BUILD / "main.pdf"
    if not pdf.is_file() or pdf.stat().st_size < 10_000:
        raise SystemExit("Compiled PDF is missing or unexpectedly small")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(pdf, OUTPUT)
    print(json.dumps({"pdf": str(OUTPUT), "main_text_end_page": main_end, "sha256": sha256(OUTPUT)}, indent=2))


if __name__ == "__main__":
    main()
