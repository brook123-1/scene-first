"""Bounded, synthetic 12MP master-preparation benchmark.

Outputs only anonymous timings to .local; no user photo or generated image is
retained.  Network upload time must still be measured on a real phone after an
explicit staging deployment.
"""
from __future__ import annotations

import argparse
import io
import json
import shutil
import sys
import time
import uuid
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import LOCAL_DIR, PREVIEW_DIR, UPLOAD_DIR
from app.image_ops import load_image, save_clean, save_preview
from app.master_upload import master_upload_store


def synthetic_photo() -> Image.Image:
    width, height = 4032, 3024
    x = np.linspace(0, 1, width, dtype=np.float32)[None, :]
    y = np.linspace(0, 1, height, dtype=np.float32)[:, None]
    base = np.empty((height, width, 3), dtype=np.uint8)
    base[..., 0] = (55 + 155 * x).astype(np.uint8)
    base[..., 1] = (40 + 125 * y).astype(np.uint8)
    base[..., 2] = (65 + 90 * (x * y)).astype(np.uint8)
    rng = np.random.default_rng(20260813)
    noise = rng.integers(0, 18, size=(height, width, 1), dtype=np.uint8)
    return Image.fromarray(np.clip(base + noise, 0, 255).astype(np.uint8), "RGB")


def encode(image: Image.Image, fmt: str) -> bytes:
    output = io.BytesIO()
    if fmt == "JPEG":
        image.save(output, "JPEG", quality=92)
    elif fmt == "PNG":
        # A normal photo-like PNG can exceed the 30 MB product upload ceiling;
        # use a smooth synthetic image to exercise the decoder within policy.
        image.resize((2016, 1512)).resize(image.size).save(output, "PNG", compress_level=6)
    elif fmt == "HEIF":
        image.save(output, "HEIF", quality=88)
    return output.getvalue()


def legacy(payload: bytes) -> dict:
    image_id = uuid.uuid4().hex
    started = time.perf_counter()
    step = time.perf_counter()
    image = load_image(payload)
    decode_ms = round((time.perf_counter() - step) * 1000)
    step = time.perf_counter()
    save_clean(image, UPLOAD_DIR / f"{image_id}.png", "PNG")
    master_ms = round((time.perf_counter() - step) * 1000)
    step = time.perf_counter()
    save_preview(image, PREVIEW_DIR / f"{image_id}-before.jpg")
    preview_ms = round((time.perf_counter() - step) * 1000)
    (UPLOAD_DIR / f"{image_id}.png").unlink(missing_ok=True)
    (PREVIEW_DIR / f"{image_id}-before.jpg").unlink(missing_ok=True)
    return {"decode_ms": decode_ms, "master_encode_ms": master_ms, "preview_ms": preview_ms, "total_ms": round((time.perf_counter() - started) * 1000)}


def profiled(payload: bytes, fmt: str) -> dict:
    image_id = uuid.uuid4().hex
    temporary = master_upload_store.temp_path(image_id)
    started = time.perf_counter()
    temporary.write_bytes(payload)
    receive_ms = round((time.perf_counter() - started) * 1000)
    result = master_upload_store._process(image_id, temporary, {"bytes": len(payload), "receive_ms": receive_ms, "format_hint": fmt.lower()})
    (UPLOAD_DIR / f"{image_id}.png").unlink(missing_ok=True)
    (PREVIEW_DIR / f"{image_id}-before.jpg").unlink(missing_ok=True)
    master_upload_store._state_path(image_id).unlink(missing_ok=True)
    return {"status": result["status"], **result.get("timings", {}), "peak_rss_kb": result.get("peak_rss_kb")}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=LOCAL_DIR / "master-upload-benchmark.json")
    args = parser.parse_args()
    image = synthetic_photo()
    payloads = {fmt: encode(image, fmt) for fmt in ("JPEG", "PNG", "HEIF")}
    report = {
        "kind": "synthetic-12mp-local-server-benchmark",
        "width": image.width,
        "height": image.height,
        "network_upload_ms": None,
        "network_note": "Requires an explicitly approved staging deployment and real phone.",
        "legacy_jpeg": legacy(payloads["JPEG"]),
        "profiled": {fmt.lower(): {"bytes": len(payload), **profiled(payload, fmt)} for fmt, payload in payloads.items()},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), "utf8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
