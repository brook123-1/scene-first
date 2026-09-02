from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / ".local" / "app" / "benchmark"
APP_LOCAL = ROOT / ".local" / "app"


def media_path(url: str) -> Path:
    return APP_LOCAL / url.removeprefix("/media/")


def main() -> int:
    review = json.loads((BENCHMARK / "review.json").read_text("utf8"))
    cells = []
    for item in review:
        with Image.open(media_path(item["result_url"])) as source:
            preview = ImageOps.contain(source.convert("RGB"), (400, 280), Image.Resampling.LANCZOS)
        cell = Image.new("RGB", (420, 320), "#f2f0e9")
        cell.paste(preview, ((420 - preview.width) // 2, 4))
        draw = ImageDraw.Draw(cell)
        draw.text((12, 292), f"sample {item['sample_id']} · candidate {item['blind_label']}", fill="#10221a")
        cells.append(cell)
    columns = 4
    rows = (len(cells) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * 420, rows * 320), "#dedfd9")
    for index, cell in enumerate(cells):
        sheet.paste(cell, ((index % columns) * 420, (index // columns) * 320))
    destination = BENCHMARK / "result-contact-sheet.jpg"
    sheet.save(destination, "JPEG", quality=90, optimize=True)
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
