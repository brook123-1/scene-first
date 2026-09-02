from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "samples" / "manifest.csv"
INBOX = ROOT / "samples" / "inbox"
OUT = ROOT / ".local" / "app" / "benchmark"
ALLOWED_RIGHTS = {"owned", "consented", "licensed"}


def truth(value: str) -> bool:
    return value.strip().lower() in {"yes", "true", "1"}


def load_rows() -> list[dict]:
    return list(csv.DictReader(MANIFEST.open("r", encoding="utf-8-sig")))


def image_info(path: Path) -> dict:
    raw = path.read_bytes()
    with Image.open(io.BytesIO(raw)) as source:
        exif = source.getexif()
        gps = exif.get(0x8825) if exif else None
        oriented = ImageOps.exif_transpose(source)
        return {
            "format": source.format,
            "width": oriented.width,
            "height": oriented.height,
            "megapixels": round(oriented.width * oriented.height / 1_000_000, 2),
            "exif_fields": len(exif),
            "has_gps": bool(gps),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "bytes": len(raw),
        }


def contact_sheet(records: list[dict], destination: Path) -> None:
    cells = []
    for record in records:
        if record["status"] != "ready":
            continue
        with Image.open(ROOT / record["relative_path"]) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
            image.thumbnail((420, 260), Image.Resampling.LANCZOS)
            cell = Image.new("RGB", (440, 315), "#f2f0e9")
            cell.paste(image, ((440 - image.width) // 2, 8))
            draw = ImageDraw.Draw(cell)
            draw.text((12, 276), f"{record['id']}  {record['width']}x{record['height']}  {record['megapixels']}MP", fill="#10221a")
            draw.text((12, 294), f"{record['scene_type'][:36]}  EXIF:{record['exif_fields']} GPS:{record['has_gps']}", fill="#52645b")
            cells.append(cell)
    columns = 4
    rows = (len(cells) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * 440, max(1, rows) * 315), "#dedfd9")
    for index, cell in enumerate(cells):
        sheet.paste(cell, ((index % columns) * 440, (index // columns) * 315))
    sheet.save(destination, "JPEG", quality=88, optimize=True)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    records = []
    errors = []
    seen_ids = set()
    seen_files = set()
    for raw in load_rows():
        row = {key: (value or "").strip() for key, value in raw.items()}
        sample_id = row.get("id", "")
        filename = Path(row.get("filename", "")).name
        issues = []
        if not sample_id or sample_id in seen_ids:
            issues.append("missing_or_duplicate_id")
        if not filename or filename in seen_files:
            issues.append("missing_or_duplicate_filename")
        seen_ids.add(sample_id)
        seen_files.add(filename)
        if row.get("rights", "").lower() not in ALLOWED_RIGHTS:
            issues.append("rights_not_allowed")
        if truth(row.get("contains_minors", "")):
            issues.append("contains_minors_excluded_in_round_one")
        matches = list(INBOX.rglob(filename)) if filename else []
        if len(matches) != 1:
            issues.append("file_missing" if not matches else "duplicate_file_match")
        record = {
            "id": sample_id,
            "filename": filename,
            "rights": row.get("rights", ""),
            "scene_type": row.get("scene_type", ""),
            "intended_subject": row.get("intended_subject", ""),
            "keep_people": row.get("keep_people", ""),
            "contains_minors": truth(row.get("contains_minors", "")),
            "issues": issues,
            "status": "invalid" if issues else "ready",
        }
        if len(matches) == 1:
            path = matches[0]
            record["relative_path"] = str(path.relative_to(ROOT))
            try:
                record.update(image_info(path))
            except Exception as exc:
                record["status"] = "invalid"
                record["issues"].append(f"image_decode_failed:{exc}")
        if record["issues"]:
            errors.append({"id": sample_id, "issues": record["issues"]})
        records.append(record)
    payload = {
        "total_manifest_rows": len(records),
        "ready": sum(record["status"] == "ready" for record in records),
        "invalid": sum(record["status"] != "ready" for record in records),
        "with_exif": sum(bool(record.get("exif_fields")) for record in records),
        "with_gps": sum(bool(record.get("has_gps")) for record in records),
        "errors": errors,
        "samples": records,
    }
    (OUT / "sample-audit.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), "utf8")
    contact_sheet(records, OUT / "sample-contact-sheet.jpg")
    print(json.dumps({key: payload[key] for key in ("total_manifest_rows", "ready", "invalid", "with_exif", "with_gps", "errors")}, ensure_ascii=False, indent=2))
    print(OUT / "sample-contact-sheet.jpg")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
