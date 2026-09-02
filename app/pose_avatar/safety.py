from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw

from .models import DetectedHead, RenderTransform, Scene


@dataclass(frozen=True)
class SafetyExpansionPolicy:
    """Optional P1 coverage expansion, disabled on every production path.

    Expansion is applied around the fitted neck anchor.  It therefore adds
    head coverage without moving the neck/shoulder relationship.  The policy
    is roll- and scene-dependent instead of permanently enlarging all assets.
    """

    enabled: bool = False
    roll_start_deg: float = 12.0
    roll_full_deg: float = 35.0
    max_roll_scale: float = 1.08
    three_quarter_bonus: float = 0.015
    back_bonus: float = 0.0

    def multiplier(self, scene: Scene, roll_deg: float) -> float:
        if not self.enabled:
            return 1.0
        span = max(1.0, self.roll_full_deg - self.roll_start_deg)
        progress = min(1.0, max(0.0, abs(roll_deg) - self.roll_start_deg) / span)
        value = 1.0 + progress * (self.max_roll_scale - 1.0)
        if scene in {Scene.S04_L34_NEUTRAL, Scene.S07_R34_NEUTRAL}:
            value += self.three_quarter_bonus
        elif scene == Scene.S12_BACK:
            value += self.back_bonus
        return value


DISABLED_SAFETY_POLICY = SafetyExpansionPolicy()
P1_CANDIDATE_SAFETY_POLICY = SafetyExpansionPolicy(enabled=True)


def expand_transform_about_neck(
    transform: RenderTransform,
    head: DetectedHead,
    scene: Scene,
    policy: SafetyExpansionPolicy,
) -> RenderTransform:
    multiplier = policy.multiplier(scene, head.pose.roll_deg)
    if abs(multiplier - 1.0) < 1e-9:
        return transform
    matrix = np.asarray(transform.matrix, dtype=np.float64)
    neck = np.asarray(head.neck_center, dtype=np.float64)
    linear = matrix[:, :2] * multiplier
    translation = neck - linear @ np.linalg.solve(matrix[:, :2], neck - matrix[:, 2])
    expanded = np.c_[linear, translation]
    return transform.model_copy(update={
        "matrix": expanded.tolist(),
        "target_scale": transform.target_scale * multiplier,
        "translation": (float(translation[0]), float(translation[1])),
    })


def conservative_head_envelope(size: tuple[int, int], head: DetectedHead) -> Image.Image:
    """Return a geometry-only proxy mask; it is not a segmentation ground truth."""
    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)
    x, y, width, height = head.bbox
    # The confirmed head box includes hair and is the only real-photo ground
    # truth P1 currently owns.  A slightly inset ellipse avoids pretending the
    # rectangular corners are actual head pixels.
    inset_x, inset_y = width * 0.035, height * 0.015
    draw.ellipse((x + inset_x, y + inset_y, x + width - inset_x, y + height - inset_y), fill=255)
    return mask


def coverage_proxies(
    overlay_alpha: Image.Image,
    head: DetectedHead,
    *,
    envelope: Image.Image | None = None,
) -> dict[str, float]:
    """Measure reproducible geometry proxies, never human placement quality."""
    target = np.asarray(envelope or conservative_head_envelope(overlay_alpha.size, head)) > 0
    overlay = np.asarray(overlay_alpha.convert("L")) >= 16
    target_pixels = int(target.sum())
    exposed = int(np.logical_and(target, ~overlay).sum())
    overlay_pixels = int(overlay.sum())
    outside = int(np.logical_and(overlay, ~target).sum())
    neck_y = int(round(head.neck_center[1] + head.bbox[3] * 0.08))
    below = int(overlay[max(0, neck_y):, :].sum()) if neck_y < overlay.shape[0] else 0
    return {
        "exposed_head_proxy_rate": exposed / max(1, target_pixels),
        "overlay_outside_envelope_rate": outside / max(1, overlay_pixels),
        "below_neck_proxy_rate": below / max(1, overlay_pixels),
    }
