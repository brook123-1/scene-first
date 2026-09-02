from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.detector import detector  # noqa: E402
from app.config import UPLOAD_DIR  # noqa: E402
from app.image_ops import load_image_path, save_clean  # noqa: E402


INBOX = ROOT / "samples" / "inbox"
MANIFEST = ROOT / "samples" / "manifest.csv"
OUT = ROOT / ".local" / "app" / "benchmark" / "detection-audit-v2"
MAX_DETECTION_EDGE = 1600


def find_sample(filename: str) -> Path:
    matches = list(INBOX.rglob(Path(filename).name))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one match for {filename}, found {len(matches)}")
    return matches[0]


def annotated_cell(image: Image.Image, detections: list[dict], label: str) -> Image.Image:
    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    line = max(4, round(max(canvas.size) / 700))
    for index, detection in enumerate(detections, 1):
        x, y, w, h = detection["head_box"]
        color = "#baff35" if detection["source"] != "mediapipe-person" else "#ffb52e"
        draw.rounded_rectangle((x, y, x + w, y + h), radius=max(4, line * 2), outline=color, width=line)
        draw.rectangle((x, y, x + line * 7, y + line * 5), fill="#132119")
        draw.text((x + line, y), str(index), fill=color, stroke_width=1, stroke_fill="#132119")
    canvas.thumbnail((520, 340), Image.Resampling.LANCZOS)
    cell = Image.new("RGB", (540, 390), "#f2f0e9")
    cell.paste(canvas, ((540 - canvas.width) // 2, 4))
    caption = ImageDraw.Draw(cell)
    caption.text((12, 352), f"{label} · detected {len(detections)}", fill="#10221a")
    caption.text((12, 371), "green=face  orange=body-derived head", fill="#52645b")
    return cell


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit Scene First head detection on the 20-sample set.")
    parser.add_argument("--max-edge", type=int, default=MAX_DETECTION_EDGE, help="0 keeps full source resolution")
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args(argv)
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(MANIFEST.open("r", encoding="utf-8-sig")))
    records = []
    cells = []
    for raw in rows:
        row = {key: (value or "").strip() for key, value in raw.items()}
        path = find_sample(row["filename"])
        image = load_image_path(path)
        # The preflight page only needs a visually faithful review copy. JPEG
        # is far faster and smaller than a 12–24MP PNG; metadata is still clean.
        save_clean(image, UPLOAD_DIR / f"sample-{row['id']}.jpg", "JPEG", 92)
        # Match the production browser path: metadata-free JPEG, longest edge
        # 1600 px, then map the detector coordinates back to the source image.
        ratio = min(1, args.max_edge / max(image.size)) if args.max_edge > 0 else 1
        detection_size = (max(1, round(image.width * ratio)), max(1, round(image.height * ratio)))
        detection_image = image.resize(detection_size, Image.Resampling.LANCZOS) if ratio < 1 else image.copy()
        started = time.perf_counter()
        detections = detector.detect(detection_image)
        elapsed_ms = round((time.perf_counter() - started) * 1000)
        scale_x, scale_y = image.width / detection_image.width, image.height / detection_image.height
        for detection in detections:
            for key in ("box", "head_box"):
                x, y, width, height = detection[key]
                detection[key] = [round(x * scale_x), round(y * scale_y), round(width * scale_x), round(height * scale_y)]
        records.append({
            "sample_id": row["id"], "filename": row["filename"], "width": image.width, "height": image.height,
            "detection_count": len(detections), "elapsed_ms": elapsed_ms, "detection_size": detection_size,
            "keep_people": row["keep_people"],
            "detections": detections,
        })
        cells.append(annotated_cell(image, detections, row["id"]))
        print(f"{row['id']}: {len(detections)} regions in {elapsed_ms} ms")
    columns = 3
    sheet_rows = (len(cells) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * 540, sheet_rows * 390), "#dedfd9")
    for index, cell in enumerate(cells):
        sheet.paste(cell, ((index % columns) * 540, (index // columns) * 390))
    sheet.save(output / "contact-sheet.jpg", "JPEG", quality=90, optimize=True)
    summary = {
        "samples": len(records),
        "total_regions": sum(record["detection_count"] for record in records),
        "zero_detection_samples": [record["sample_id"] for record in records if not record["detection_count"]],
        "average_elapsed_ms": round(sum(record["elapsed_ms"] for record in records) / len(records)),
        "records": records,
    }
    (output / "detections.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), "utf8")
    print(json.dumps({key: summary[key] for key in ("samples", "total_regions", "zero_detection_samples", "average_elapsed_ms")}, ensure_ascii=False))
    print(output / "contact-sheet.jpg")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
