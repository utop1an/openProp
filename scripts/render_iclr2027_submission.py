from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Render every PDF page to PNG for visual QA")
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--output", type=Path, default=Path("tmp/pdfs/iclr2027-rendered"))
    args = parser.parse_args()
    try:
        import fitz
    except ImportError as exc:
        raise SystemExit("PyMuPDF is required: install it into the active Python environment") from exc

    pdf = args.pdf.resolve()
    output = args.output.resolve()
    safe_root = (Path(__file__).resolve().parents[1] / "tmp" / "pdfs").resolve()
    if output != safe_root and safe_root not in output.parents:
        raise SystemExit(f"Refusing to replace render directory outside {safe_root}: {output}")

    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    document = fitz.open(pdf)
    matrix = fitz.Matrix(1.7, 1.7)
    for index, page in enumerate(document):
        target = output / f"page-{index + 1:02d}.png"
        page.get_pixmap(matrix=matrix, alpha=False).save(target)
        print(target)
    print(f"Rendered {document.page_count} pages")


if __name__ == "__main__":
    main()
