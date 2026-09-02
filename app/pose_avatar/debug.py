from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw

from .models import DetectedHead, RenderBatchResult


def annotated_input(image: Image.Image, heads: list[DetectedHead]) -> Image.Image:
    output = image.convert("RGB").copy()
    draw = ImageDraw.Draw(output)
    for head in heads:
        x, y, width, height = head.bbox
        draw.rectangle((x, y, x + width, y + height), outline=(255, 62, 90), width=3)
        for point in head.face_landmarks.values():
            draw.ellipse((point[0] - 3, point[1] - 3, point[0] + 3, point[1] + 3), fill=(40, 210, 255))
        for point in head.body_landmarks.values():
            draw.ellipse((point[0] - 4, point[1] - 4, point[0] + 4, point[1] + 4), fill=(77, 230, 130))
        label = f"{head.head_id} yaw={head.pose.yaw_deg:.1f} pitch={head.pose.pitch_deg:.1f} roll={head.pose.roll_deg:.1f}"
        draw.text((x, max(0, y - 14)), label, fill=(255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0))
    return output


def write_debug_bundle(
    directory: Path,
    original: Image.Image,
    heads: list[DetectedHead],
    result: RenderBatchResult,
) -> dict:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    files = {
        "original": directory / "01-original.png",
        "detections": directory / "02-head-bbox-landmarks-pose.png",
        "transformed_overlay": directory / "03-transformed-overlay.png",
        "composite": directory / "04-final-composite.png",
        "trace": directory / "05-decision-trace.json",
    }
    original.convert("RGB").save(files["original"], "PNG")
    annotated_input(original, heads).save(files["detections"], "PNG")
    result.transformed_overlay.save(files["transformed_overlay"], "PNG")
    result.image.save(files["composite"], "PNG")
    files["trace"].write_text(
        json.dumps({
            "schema_version": "1.0",
            "heads": [head.model_dump(mode="json") for head in heads],
            "decisions": [value.model_dump(mode="json") for value in result.decisions],
            "transforms": [value.model_dump(mode="json") if value else None for value in result.transforms],
            "traces": [value.model_dump(mode="json") for value in result.traces],
        }, ensure_ascii=False, indent=2),
        "utf8",
    )
    return {key: str(value) for key, value in files.items()}
