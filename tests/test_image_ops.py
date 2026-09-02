from __future__ import annotations

import io

import numpy as np
from PIL import Image

from app.image_ops import (
    blur_region,
    composite_patch,
    head_mask,
    image_bytes,
    illustrated_patch,
    load_image,
    outside_mask_is_exact,
)


def sample_image() -> Image.Image:
    array = np.zeros((180, 260, 3), dtype=np.uint8)
    array[:, :, 0] = np.arange(260, dtype=np.uint8)[None, :]
    array[:, :, 1] = 120
    array[:, :, 2] = np.arange(180, dtype=np.uint8)[:, None]
    return Image.fromarray(array, "RGB")


def test_clean_encoding_removes_metadata():
    image = sample_image()
    exif = Image.Exif()
    exif[0x010E] = "sensitive description"
    source = io.BytesIO()
    image.save(source, "JPEG", exif=exif)
    loaded = load_image(source.getvalue())
    clean = Image.open(io.BytesIO(image_bytes(loaded, "PNG")))
    assert not clean.getexif()
    assert clean.size == image.size


def test_blur_never_changes_pixels_outside_mask():
    image = sample_image()
    head_box = [80, 35, 75, 110]
    mask = head_mask(image.size, head_box)
    output = blur_region(image, [50, 10, 135, 160], mask)
    assert outside_mask_is_exact(image, output, mask)


def test_illustration_composite_never_changes_pixels_outside_mask():
    image = sample_image()
    head_box = [82, 35, 72, 110]
    crop_box = [52, 9, 132, 160]
    mask = head_mask(image.size, head_box)
    patch = illustrated_patch(image, crop_box, head_box, "person-test", 0)
    output = composite_patch(image, patch, crop_box, mask)
    assert outside_mask_is_exact(image, output, mask)
    assert not np.array_equal(np.asarray(image), np.asarray(output))


def test_retry_nonce_changes_local_avatar():
    image = sample_image()
    crop = [52, 9, 132, 160]
    head = [82, 35, 72, 110]
    first = illustrated_patch(image, crop, head, "same-person", 0)
    second = illustrated_patch(image, crop, head, "same-person", 1)
    assert not np.array_equal(np.asarray(first), np.asarray(second))
