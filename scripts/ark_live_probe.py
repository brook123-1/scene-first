from __future__ import annotations

import base64
import io
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import httpx
from PIL import Image, ImageDraw, ImageFilter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import ARK_IMAGE_BASE_URL, ENV, LOCAL_DIR, PROVIDER_SETTINGS
from app.providers import prompt_for_profile


OUTPUT_DIR = LOCAL_DIR / "ark-validation"


def _synthetic_test_crop() -> Image.Image:
    """Build a deliberately non-photographic test head crop.

    The image contains no real person and no real identity; it exists only to
    exercise the remote endpoint without uploading private material.
    """
    width = height = 1024
    image = Image.new("RGB", (width, height), "#cfd8dd")
    draw = ImageDraw.Draw(image)

    for y in range(height):
        t = y / max(1, height - 1)
        color = (
            int(180 + (215 - 180) * t),
            int(205 + (235 - 205) * t),
            int(220 + (244 - 220) * t),
        )
        draw.line([(0, y), (width, y)], fill=color)

    # Neck and shoulders as simple synthetic blocks.
    draw.rounded_rectangle((360, 640, 664, 1024), radius=70, fill="#c89a78")
    draw.rounded_rectangle((250, 820, 774, 1060), radius=110, fill="#30434f")

    # Head silhouette.
    draw.ellipse((292, 120, 732, 650), fill="#d6a57e")
    draw.ellipse((330, 122, 694, 590), fill="#d6a57e")

    # Hair: a synthetic block rather than biometric hairstyle.
    draw.pieslice((270, 80, 754, 590), 180, 360, fill="#3b3f46")
    draw.ellipse((330, 130, 694, 585), fill="#d6a57e")

    # Eyes.
    draw.ellipse((392, 350, 472, 420), fill="#f4f7f9")
    draw.ellipse((552, 350, 632, 420), fill="#f4f7f9")
    draw.ellipse((420, 372, 446, 398), fill="#26313b")
    draw.ellipse((578, 372, 604, 398), fill="#26313b")

    # Nose and mouth are simple non-biometric placeholders.
    draw.line((512, 410, 498, 490, 520, 500), fill="#9b6d57", width=8, joint="curve")
    draw.arc((450, 505, 574, 565), 20, 160, fill="#8a5848", width=8)

    return image.filter(ImageFilter.GaussianBlur(0.6))


def _data_uri(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode('ascii')}"


def _safe_response_summary(response: httpx.Response) -> dict:
    summary: dict = {
        "status_code": response.status_code,
        "content_type": response.headers.get("content-type", ""),
    }
    try:
        payload = response.json()
    except ValueError:
        summary["body_preview"] = response.text[:500]
        return summary

    if isinstance(payload, dict):
        summary["top_level_keys"] = sorted(payload.keys())
        if "data" in payload and isinstance(payload["data"], list) and payload["data"]:
            first = payload["data"][0]
            if isinstance(first, dict):
                summary["first_data_keys"] = sorted(first.keys())
                if isinstance(first.get("url"), str):
                    summary["url_returned"] = True
                if isinstance(first.get("b64_json"), str):
                    summary["b64_json_length"] = len(first["b64_json"])
        for key in ("error", "code", "message", "detail"):
            if key in payload:
                summary[key] = payload[key]
        if payload.get("error") and isinstance(payload["error"], dict):
            summary["error_keys"] = sorted(payload["error"].keys())
    else:
        summary["body_preview"] = str(payload)[:500]
    return summary


def main() -> None:
    settings = PROVIDER_SETTINGS["ark"]
    key = ENV.get(settings.key_name or "")
    if not key:
        raise SystemExit("ARK_API_KEY 未配置；无法发起受控调用。")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    crop = _synthetic_test_crop()
    input_path = OUTPUT_DIR / "ark-probe-input.png"
    crop.save(input_path, format="PNG")

    payload = {
        "model": settings.model,
        "prompt": prompt_for_profile("balanced_portrait"),
        "image": _data_uri(crop),
        "size": "2K",
        "response_format": "url",
    }

    started = datetime.now(timezone.utc)
    response = httpx.post(
        ARK_IMAGE_BASE_URL,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json=payload,
        timeout=240,
    )
    elapsed_ms = int((datetime.now(timezone.utc) - started).total_seconds() * 1000)
    summary = _safe_response_summary(response)
    summary["elapsed_ms"] = elapsed_ms
    summary["input_path"] = str(input_path)
    summary["endpoint"] = ARK_IMAGE_BASE_URL

    if response.is_success:
        data = response.json().get("data", [])
        if not data:
            summary["image_returned"] = False
            summary["error"] = "成功响应中缺少 data。"
        else:
            item = data[0]
            if item.get("b64_json"):
                encoded = item["b64_json"]
                image = Image.open(io.BytesIO(base64.b64decode(encoded))).convert("RGB")
            elif item.get("url"):
                image_response = httpx.get(item["url"], timeout=60, follow_redirects=True)
                image_response.raise_for_status()
                image = Image.open(io.BytesIO(image_response.content)).convert("RGB")
            else:
                image = None
                summary["image_returned"] = False
                summary["error"] = "data[0] 中既无 url 也无 b64_json。"
            if image:
                output_path = OUTPUT_DIR / "ark-probe-output.png"
                image.save(output_path, format="PNG")
                summary["image_returned"] = True
                summary["output_path"] = str(output_path)
                summary["output_size"] = list(image.size)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
