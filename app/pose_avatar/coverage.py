from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from .models import DetectedHead
from .safety import conservative_head_envelope


@dataclass(frozen=True)
class CoverageMasks:
    original_head_mask: Image.Image
    avatar_coverage_mask: Image.Image
    exposed_head_mask: Image.Image
    exposed_head_ratio: float


class HeadCoverageMaskProvider(Protocol):
    provider_id: str

    def evaluate(self, size: tuple[int, int], head: DetectedHead, overlay_alpha: Image.Image) -> CoverageMasks: ...


def _evaluate(original: Image.Image, overlay_alpha: Image.Image) -> CoverageMasks:
    original_array = np.asarray(original.convert("L"), dtype=np.uint8) > 0
    avatar_array = np.asarray(overlay_alpha.convert("L"), dtype=np.uint8) > 0
    exposed_array = original_array & ~avatar_array
    denominator = int(original_array.sum())
    exposed = Image.fromarray((exposed_array * 255).astype(np.uint8), "L")
    return CoverageMasks(
        original_head_mask=original.convert("L"),
        avatar_coverage_mask=overlay_alpha.convert("L"),
        exposed_head_mask=exposed,
        exposed_head_ratio=round(float(exposed_array.sum()) / denominator, 6) if denominator else 0.0,
    )


class EllipseCoverageMaskProvider:
    provider_id = "bbox_ellipse_v1"

    def evaluate(self, size: tuple[int, int], head: DetectedHead, overlay_alpha: Image.Image) -> CoverageMasks:
        return _evaluate(conservative_head_envelope(size, head), overlay_alpha)


class LandmarkSilhouetteMaskProvider:
    """Lightweight local silhouette proxy; geometry only, not identity protection proof."""

    provider_id = "landmark_silhouette_v1"

    def evaluate(self, size: tuple[int, int], head: DetectedHead, overlay_alpha: Image.Image) -> CoverageMasks:
        x, y, width, height = head.bbox
        points = list(head.face_landmarks.values())
        mask = Image.new("L", size, 0)
        draw = ImageDraw.Draw(mask)
        if len(points) >= 3:
            min_x = max(x, min(point[0] for point in points) - width * 0.22)
            max_x = min(x + width, max(point[0] for point in points) + width * 0.22)
            min_y = max(y, min(point[1] for point in points) - height * 0.30)
            max_y = min(y + height, max(point[1] for point in points) + height * 0.30)
            draw.ellipse((min_x, min_y, max_x, max_y), fill=255)
            draw.polygon([(x + width * .18, y + height * .42), (x + width * .82, y + height * .42),
                          (x + width * .70, y + height), (x + width * .30, y + height)], fill=255)
        else:
            draw.ellipse((x + width * .08, y + height * .02, x + width * .92, y + height), fill=255)
        mask = mask.filter(ImageFilter.GaussianBlur(radius=max(1.0, width * .012)))
        return _evaluate(mask, overlay_alpha)
