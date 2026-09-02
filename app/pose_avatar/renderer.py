from __future__ import annotations

import time

import cv2
import numpy as np
from PIL import Image, ImageDraw

from app.image_ops import blur_region, head_mask, outside_mask_is_exact, padded_crop_box

from .models import (
    DecisionTrace,
    DetectedHead,
    Expression,
    OverlayDecision,
    RenderBatchResult,
    RenderRoute,
)
from .registry import AssetRecord, AssetRegistry
from .routing import DEFAULT_THRESHOLDS, RoutingThresholds, classify_scene, fallback_route, quality_gate
from .safety import DISABLED_SAFETY_POLICY, SafetyExpansionPolicy, expand_transform_about_neck
from .transform import select_neck_adapter, solve_transform


def _maximum_mask(first: Image.Image, second: Image.Image) -> Image.Image:
    return Image.fromarray(np.maximum(np.asarray(first, dtype=np.uint8), np.asarray(second, dtype=np.uint8)))


def _composite_rgba(base: Image.Image, overlay: Image.Image) -> Image.Image:
    return Image.alpha_composite(base.convert("RGBA"), overlay.convert("RGBA")).convert("RGB")


def _warp_asset(asset: Image.Image, matrix: list[list[float]], size: tuple[int, int]) -> Image.Image:
    rgba = np.asarray(asset.convert("RGBA"))
    warped = cv2.warpAffine(
        rgba,
        np.asarray(matrix, dtype=np.float32),
        size,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )
    return Image.fromarray(warped, "RGBA")


def _fallback_overlay(size: tuple[int, int], head: DetectedHead, route: RenderRoute) -> tuple[Image.Image, Image.Image]:
    mask = head_mask(size, [round(value) for value in head.bbox], feather=2)
    overlay = Image.new("RGBA", size, (0, 0, 0, 0))
    color = {
        RenderRoute.SIMPLIFIED_AVATAR: (103, 173, 214, 255),
        RenderRoute.CROP_SAFE_AVATAR: (115, 132, 162, 255),
        RenderRoute.SILHOUETTE: (48, 55, 70, 255),
    }.get(route, (70, 74, 82, 255))
    solid = Image.new("RGBA", size, color)
    overlay = Image.composite(solid, overlay, mask)
    if route == RenderRoute.SIMPLIFIED_AVATAR:
        draw = ImageDraw.Draw(overlay)
        x, y, width, height = head.bbox
        radius = max(1, round(width * 0.035))
        eye_y = y + height * 0.40
        for eye_x in (x + width * 0.36, x + width * 0.64):
            draw.ellipse((eye_x - radius, eye_y - radius, eye_x + radius, eye_y + radius), fill=(245, 247, 250, 255))
    return overlay, mask


class PoseAvatarRenderer:
    def __init__(
        self,
        registry: AssetRegistry,
        thresholds: RoutingThresholds = DEFAULT_THRESHOLDS,
        safety_policy: SafetyExpansionPolicy = DISABLED_SAFETY_POLICY,
    ) -> None:
        self.registry = registry
        self.thresholds = thresholds
        self.safety_policy = safety_policy

    def render(
        self,
        image: Image.Image,
        heads: list[DetectedHead],
        *,
        image_id: str,
        family_id: str | None = None,
    ) -> RenderBatchResult:
        original = image.convert("RGB")
        working = original.copy()
        combined_mask = Image.new("L", original.size, 0)
        combined_overlay = Image.new("RGBA", original.size, (0, 0, 0, 0))
        decisions: list[OverlayDecision] = []
        transforms = []
        traces: list[DecisionTrace] = []

        # Existing screen-space heuristic is retained as the deterministic
        # fallback ordering: low/large heads paint later than far/small heads.
        ordered = sorted(heads, key=lambda head: (head.depth_order, head.head_id))
        for head in ordered:
            started = time.perf_counter()
            gate = quality_gate(head, self.thresholds)
            scene = None
            scene_reason = None
            record: AssetRecord | None = None
            transform = None
            selected_asset = None

            if gate.passed:
                scene, scene_reason = classify_scene(head, self.thresholds)
            failure_reason = gate.reason or scene_reason
            if scene is not None and failure_reason is None:
                record = self.registry.select(scene, family_id)
                if record is None:
                    failure_reason = "unsupported_scene_asset"

            if record is not None and failure_reason is None:
                transform = solve_transform(head, scene, record.anchors, self.thresholds)
                if transform.residual_normalized > self.thresholds.max_transform_residual_ratio:
                    failure_reason = "transform_residual_too_high"

            if record is not None and transform is not None and failure_reason is None:
                transform = expand_transform_about_neck(transform, head, scene, self.safety_policy)
                decision = OverlayDecision(
                    head_id=head.head_id,
                    route_type=RenderRoute.STANDARD_AVATAR,
                    scene_id=scene,
                    expression=Expression.NEUTRAL,
                    family_id=record.family.family_id,
                    family_version=record.family.version,
                    neck_adapter=select_neck_adapter(head),
                    decision_confidence=min(head.confidence, head.pose.confidence, head.visibility),
                )
                overlay = _warp_asset(self.registry.image(record), transform.matrix, original.size)
                local_mask = overlay.getchannel("A")
                working = _composite_rgba(working, overlay)
                selected_asset = str(record.path.relative_to(self.registry.root.parent.parent))
            else:
                reason = failure_reason or "unsupported_scene"
                route = fallback_route(reason)
                decision = OverlayDecision(
                    head_id=head.head_id,
                    route_type=route,
                    scene_id=scene,
                    fallback_reason=reason,
                    decision_confidence=0.0,
                )
                if route == RenderRoute.BLUR_FALLBACK:
                    local_mask = head_mask(original.size, [round(value) for value in head.bbox])
                    working = blur_region(working, padded_crop_box([round(value) for value in head.bbox], original.size), local_mask)
                    overlay = Image.new("RGBA", original.size, (0, 0, 0, 0))
                    blurred_only = Image.composite(working, Image.new("RGB", original.size), local_mask).convert("RGBA")
                    blurred_only.putalpha(local_mask)
                    overlay = blurred_only
                else:
                    overlay, local_mask = _fallback_overlay(original.size, head, route)
                    working = _composite_rgba(working, overlay)

            combined_mask = _maximum_mask(combined_mask, local_mask)
            combined_overlay = Image.alpha_composite(combined_overlay, overlay)
            exact_so_far = outside_mask_is_exact(original, working, combined_mask)
            trace = DecisionTrace(
                image_id=image_id,
                head_id=head.head_id,
                scene_id=decision.scene_id,
                route_type=decision.route_type,
                family_id=decision.family_id,
                family_version=decision.family_version,
                selected_asset=selected_asset,
                yaw_deg=head.pose.yaw_deg,
                pitch_deg=head.pose.pitch_deg,
                roll_deg=head.pose.roll_deg,
                pose_confidence=head.pose.confidence,
                visibility=head.visibility,
                occlusion_score=head.occlusion_score,
                head_bbox=head.bbox,
                face_landmarks=head.face_landmarks,
                body_landmarks=head.body_landmarks,
                transform_mode=transform.warp_mode if transform else None,
                transform_scale=transform.target_scale if transform else None,
                transform_rotation_deg=transform.rotation_deg if transform else None,
                transform_translation=transform.translation if transform else None,
                transform_residual=transform.residual_normalized if transform else None,
                transform_residual_components=transform.residual_components if transform else {},
                fallback_reason=decision.fallback_reason,
                quality_gates=gate.checks + ([{"gate": "scene_support", "passed": scene_reason is None, "reason": scene_reason}] if gate.passed else []),
                render_ms=round((time.perf_counter() - started) * 1000),
                outside_mask_verified=exact_so_far,
            )
            decisions.append(decision)
            transforms.append(transform)
            traces.append(trace)

        final_exact = outside_mask_is_exact(original, working, combined_mask)
        if not final_exact:
            raise RuntimeError("pose avatar renderer changed pixels outside the combined alpha mask")
        return RenderBatchResult(
            image=working,
            alpha_mask=combined_mask,
            transformed_overlay=combined_overlay,
            decisions=decisions,
            transforms=transforms,
            traces=traces,
        )
