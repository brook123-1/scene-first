from __future__ import annotations

import hashlib
import io
import math
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageOps
from pillow_heif import register_heif_opener

from .config import SAFE_COVER_DIR, SAFE_COVERS

register_heif_opener()


def load_image(data: bytes, *, max_pixels: int | None = None) -> Image.Image:
    image = Image.open(io.BytesIO(data))
    width, height = image.size
    if max_pixels and width * height > max_pixels:
        raise ValueError(f"image_pixels_exceeded:{width}x{height}:{max_pixels}")
    if getattr(image, "is_animated", False):
        image.seek(0)
    image = ImageOps.exif_transpose(image)
    image.load()
    return image.convert("RGB")


def load_image_path(path: Path, *, max_pixels: int | None = None) -> Image.Image:
    """Decode directly from a path without first duplicating the file in RAM."""
    with Image.open(path) as opened:
        width, height = opened.size
        if max_pixels and width * height > max_pixels:
            raise ValueError(f"image_pixels_exceeded:{width}x{height}:{max_pixels}")
        if getattr(opened, "is_animated", False):
            opened.seek(0)
        normalized = ImageOps.exif_transpose(opened)
        normalized.load()
        return normalized.convert("RGB")


def image_bytes(image: Image.Image, fmt: str = "PNG", quality: int = 94) -> bytes:
    output = io.BytesIO()
    image = image.convert("RGB")
    if fmt.upper() in {"JPG", "JPEG"}:
        image.save(output, format="JPEG", quality=quality, optimize=True, exif=b"")
    else:
        image.save(output, format="PNG", optimize=True)
    return output.getvalue()


def save_clean(image: Image.Image, path: Path, fmt: str = "PNG", quality: int = 94) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(image_bytes(image, fmt, quality))


def save_clean_atomic(image: Image.Image, path: Path, fmt: str = "PNG", quality: int = 94) -> None:
    """Write a metadata-free image and expose it only after encoding succeeds."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    converted = image.convert("RGB")
    try:
        if fmt.upper() in {"JPG", "JPEG"}:
            converted.save(temporary, format="JPEG", quality=quality, optimize=True, exif=b"")
        else:
            # The server-side master is a lossless working copy.  Level 3 keeps
            # exact pixels while avoiding the large CPU cost of PNG optimize.
            converted.save(temporary, format="PNG", compress_level=3)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def save_preview(image: Image.Image, path: Path, max_dimension: int = 2048) -> None:
    """Save a small, metadata-free workbench preview; never replace the master."""
    preview = image.convert("RGB").copy()
    preview.thumbnail((max_dimension, max_dimension), Image.Resampling.LANCZOS)
    save_clean(preview, path, "JPEG", quality=86)


def clamp_box(box: list[int] | tuple[int, int, int, int], size: tuple[int, int]) -> list[int]:
    x, y, w, h = [int(round(value)) for value in box]
    width, height = size
    x = max(0, min(x, width - 1))
    y = max(0, min(y, height - 1))
    w = max(1, min(w, width - x))
    h = max(1, min(h, height - y))
    return [x, y, w, h]


def face_to_head_box(face_box: list[int], size: tuple[int, int]) -> list[int]:
    x, y, w, h = face_box
    return clamp_box([x - 0.33 * w, y - 0.52 * h, 1.66 * w, 1.92 * h], size)


def padded_crop_box(head_box: list[int], size: tuple[int, int], padding: float = 0.42) -> list[int]:
    x, y, w, h = head_box
    return clamp_box([x - w * padding, y - h * padding, w * (1 + 2 * padding), h * (1 + 2 * padding)], size)


def head_mask(
    size: tuple[int, int],
    head_box: list[int],
    feather: int | None = None,
    neck_width: float = 0.32,
    neck_start: float = 0.58,
) -> Image.Image:
    width, height = size
    x, y, w, h = clamp_box(head_box, size)
    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)
    # Hair-to-jaw ellipse plus a narrow neck transition; shoulders remain untouched.
    draw.ellipse([x, y, x + w, y + int(h * 0.83)], fill=255)
    neck_left = x + int(w * ((1 - neck_width) / 2))
    neck_right = x + int(w * ((1 + neck_width) / 2))
    draw.rounded_rectangle(
        [neck_left, y + int(h * neck_start), neck_right, y + h],
        radius=max(2, int(w * 0.08)), fill=255,
    )
    radius = feather if feather is not None else max(2, min(18, int(min(w, h) * 0.035)))
    if radius:
        mask = mask.filter(ImageFilter.GaussianBlur(radius))
        # Preserve a finite support so mask-external pixels can be verified exactly.
        mask = mask.point(lambda value: 0 if value < 3 else value)
    return mask


def crop_mask(mask: Image.Image, crop_box: list[int]) -> Image.Image:
    x, y, w, h = crop_box
    return mask.crop((x, y, x + w, y + h))


def composite_patch(base: Image.Image, patch: Image.Image, crop_box: list[int], mask: Image.Image) -> Image.Image:
    x, y, w, h = crop_box
    patch = patch.convert("RGB").resize((w, h), Image.Resampling.LANCZOS)
    region = base.crop((x, y, x + w, y + h))
    local_mask = crop_mask(mask, crop_box)
    merged = Image.composite(patch, region, local_mask)
    output = base.copy()
    output.paste(merged, (x, y))
    return output


def blur_region(base: Image.Image, crop_box: list[int], mask: Image.Image) -> Image.Image:
    x, y, w, h = crop_box
    region = base.crop((x, y, x + w, y + h))
    radius = max(6, min(38, int(min(w, h) * 0.12)))
    blurred = region.filter(ImageFilter.GaussianBlur(radius))
    blurred = ImageEnhance.Contrast(blurred).enhance(0.82)
    return composite_patch(base, blurred, crop_box, mask)


def safe_cover_patch(
    original: Image.Image,
    crop_box: list[int],
    head_box: list[int],
    cover_id: str,
) -> Image.Image:
    """Place a bundled transparent avatar over the detected head.

    The caller still applies the head mask when compositing.  That restriction
    is intentional: the person's body, clothing and the surrounding scene stay
    exactly as in the source photograph.
    """
    if cover_id not in SAFE_COVERS:
        raise ValueError("Unknown safe-cover asset")
    path = SAFE_COVER_DIR / f"{cover_id}.png"
    if not path.is_file():
        raise FileNotFoundError(f"Safe-cover asset is unavailable: {cover_id}")

    asset = Image.open(path).convert("RGBA")
    alpha_bounds = asset.getchannel("A").getbbox()
    if not alpha_bounds:
        raise ValueError(f"Safe-cover asset is empty: {cover_id}")
    asset = asset.crop(alpha_bounds)

    x, y, width, height = crop_box
    head_x, head_y, head_w, head_h = head_box
    local_x, local_y = head_x - x, head_y - y
    # The artwork includes a little torso.  Scale its alpha bounds so the
    # illustrated head fills the original head mask, while leaving a soft neck
    # transition rather than covering clothing.
    target_w = max(2, int(head_w * 1.42))
    target_h = max(2, int(asset.height * target_w / asset.width))
    asset = asset.resize((target_w, target_h), Image.Resampling.LANCZOS)
    left = int(local_x + head_w / 2 - target_w / 2)
    top = int(local_y - head_h * 0.12)

    patch = original.convert("RGBA")
    patch.alpha_composite(asset, (left, top))
    return patch.convert("RGB")


def _sample_palette(region: Image.Image) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    thumb = region.resize((32, 32)).convert("RGB")
    pixels = np.asarray(thumb).reshape(-1, 3)
    brightness = pixels.mean(axis=1)
    skin = tuple(np.percentile(pixels[brightness >= np.percentile(brightness, 45)], 62, axis=0).astype(int))
    hair = tuple(np.percentile(pixels[brightness <= np.percentile(brightness, 35)], 28, axis=0).astype(int))
    return skin, hair


def illustrated_patch(
    original: Image.Image,
    crop_box: list[int],
    head_box: list[int],
    subject_id: str,
    retry_nonce: int = 0,
) -> Image.Image:
    x, y, w, h = crop_box
    hx, hy, hw, hh = head_box
    local_head = [hx - x, hy - y, hw, hh]
    region = original.crop((x, y, x + w, y + h))
    skin, hair = _sample_palette(region)
    seed = int(hashlib.sha256(f"{subject_id}:{retry_nonce}".encode()).hexdigest()[:12], 16)
    rng = random.Random(seed)

    canvas = region.filter(ImageFilter.GaussianBlur(max(1, int(min(w, h) * 0.006))))
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    lx, ly, lw, lh = local_head
    outline = tuple(max(0, channel - 45) for channel in hair) + (255,)
    face_color = tuple(min(255, int(channel * 1.04 + 5)) for channel in skin) + (255,)
    hair_color = hair + (255,)

    # A deliberately synthetic, non-likeness-preserving avatar.
    head_rect = [lx + int(lw * 0.12), ly + int(lh * 0.10), lx + int(lw * 0.88), ly + int(lh * 0.76)]
    draw.ellipse(head_rect, fill=face_color, outline=outline, width=max(2, int(lw * 0.035)))
    hair_rect = [lx + int(lw * 0.08), ly + int(lh * 0.02), lx + int(lw * 0.92), ly + int(lh * 0.46)]
    draw.pieslice(hair_rect, 180, 360, fill=hair_color, outline=outline, width=max(2, int(lw * 0.03)))

    eye_y = ly + int(lh * 0.43)
    eye_gap = int(lw * 0.16)
    eye_radius = max(2, int(lw * 0.045))
    center_x = lx + lw // 2
    for eye_x in (center_x - eye_gap, center_x + eye_gap):
        draw.ellipse(
            [eye_x - eye_radius, eye_y - eye_radius, eye_x + eye_radius, eye_y + eye_radius],
            fill=(250, 250, 245, 255), outline=outline, width=max(1, eye_radius // 3),
        )
        pupil = max(1, eye_radius // 2)
        draw.ellipse([eye_x - pupil, eye_y - pupil, eye_x + pupil, eye_y + pupil], fill=outline)

    mouth_y = ly + int(lh * 0.62)
    smile = retry_nonce % 2
    draw.arc(
        [center_x - int(lw * 0.12), mouth_y - int(lh * 0.025), center_x + int(lw * 0.12), mouth_y + int(lh * (0.07 if smile else 0.035))],
        start=5 if smile else 185, end=175 if smile else 355, fill=outline, width=max(2, int(lw * 0.025)),
    )
    # Neck bridge uses a neutral flat tone and stops above the clothing line.
    draw.rounded_rectangle(
        [center_x - int(lw * 0.13), ly + int(lh * 0.68), center_x + int(lw * 0.13), ly + int(lh * 0.94)],
        radius=max(2, int(lw * 0.05)), fill=face_color, outline=outline, width=max(1, int(lw * 0.018)),
    )
    return Image.alpha_composite(canvas.convert("RGBA"), overlay).convert("RGB")


def outside_mask_is_exact(before: Image.Image, after: Image.Image, mask: Image.Image) -> bool:
    a = np.asarray(before.convert("RGB"))
    b = np.asarray(after.convert("RGB"))
    m = np.asarray(mask) > 0
    return bool(np.array_equal(a[~m], b[~m]))
