from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser(description="Build two-page JPEG contact sheets for PDF visual QA")
    parser.add_argument("pages", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--page-width", type=int, default=700)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    images = []
    for path in sorted(args.pages.glob("page-*.png")):
        image = Image.open(path).convert("RGB")
        height = round(image.height * args.page_width / image.width)
        images.append(image.resize((args.page_width, height)))
    if not images:
        raise SystemExit("No rendered page PNGs found")
    page_height = max(image.height for image in images)
    for offset in range(0, len(images), 2):
        sheet = Image.new("RGB", (2 * args.page_width + 30, page_height + 20), (210, 210, 210))
        for column, image in enumerate(images[offset : offset + 2]):
            sheet.paste(image, (10 + column * (args.page_width + 10), 10))
        target = args.output / f"sheet-{offset // 2 + 1:02d}.jpg"
        sheet.save(target, quality=72, optimize=True)
        print(target)


if __name__ == "__main__":
    main()
