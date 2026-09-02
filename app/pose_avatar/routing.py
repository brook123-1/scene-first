from __future__ import annotations

from dataclasses import dataclass

from .models import DetectedHead, RenderRoute, Scene, View


@dataclass(frozen=True)
class RoutingThresholds:
    # Source: Pose-aware Avatar Overlay specification v0.1, section 8.2.
    tiny_head_px: float = 48.0
    min_visibility: float = 0.45
    max_occlusion: float = 0.55
    max_crop_ratio: float = 0.20
    min_detection_confidence: float = 0.65
    min_pose_confidence: float = 0.60
    front_yaw_deg: float = 15.0
    three_quarter_yaw_deg: float = 50.0
    neutral_pitch_deg: float = 15.0
    max_roll_deg: float = 35.0
    target_head_width_multiplier: float = 1.12
    max_transform_residual_ratio: float = 0.08


DEFAULT_THRESHOLDS = RoutingThresholds()


@dataclass(frozen=True)
class GateResult:
    passed: bool
    reason: str | None
    checks: list[dict]


def quality_gate(head: DetectedHead, thresholds: RoutingThresholds = DEFAULT_THRESHOLDS) -> GateResult:
    checks: list[dict] = []

    def check(name: str, value: float, threshold: float, passed: bool) -> bool:
        checks.append({"gate": name, "value": round(value, 4), "threshold": threshold, "passed": passed})
        return passed

    edge = min(head.bbox[2], head.bbox[3])
    if not check("tiny_head", edge, thresholds.tiny_head_px, edge >= thresholds.tiny_head_px):
        return GateResult(False, "tiny_head", checks)
    if not check("low_visibility", head.visibility, thresholds.min_visibility, head.visibility >= thresholds.min_visibility):
        return GateResult(False, "low_visibility", checks)
    if not check("high_occlusion", head.occlusion_score, thresholds.max_occlusion, head.occlusion_score <= thresholds.max_occlusion):
        return GateResult(False, "high_occlusion", checks)
    if not check("edge_cropped", head.out_of_frame_ratio, thresholds.max_crop_ratio, head.out_of_frame_ratio <= thresholds.max_crop_ratio):
        return GateResult(False, "edge_cropped", checks)
    if not check("low_confidence", head.confidence, thresholds.min_detection_confidence, head.confidence >= thresholds.min_detection_confidence):
        return GateResult(False, "low_confidence", checks)
    if not check("pose_unavailable", head.pose.confidence, thresholds.min_pose_confidence, head.pose.confidence >= thresholds.min_pose_confidence):
        return GateResult(False, "pose_unavailable", checks)
    if not check("roll_out_of_range", abs(head.pose.roll_deg), thresholds.max_roll_deg, abs(head.pose.roll_deg) <= thresholds.max_roll_deg):
        return GateResult(False, "roll_out_of_range", checks)
    return GateResult(True, None, checks)


def classify_scene(head: DetectedHead, thresholds: RoutingThresholds = DEFAULT_THRESHOLDS) -> tuple[Scene | None, str | None]:
    if head.view_hint == View.BACK:
        return Scene.S12_BACK, None
    yaw, pitch = head.pose.yaw_deg, head.pose.pitch_deg
    if abs(pitch) > thresholds.neutral_pitch_deg:
        return None, "unsupported_pitch"
    if abs(yaw) <= thresholds.front_yaw_deg:
        return Scene.S01_FRONT_NEUTRAL, None
    if yaw < -thresholds.three_quarter_yaw_deg:
        return None, "unsupported_left_profile"
    if yaw > thresholds.three_quarter_yaw_deg:
        return None, "unsupported_right_profile"
    return (Scene.S04_L34_NEUTRAL if yaw < 0 else Scene.S07_R34_NEUTRAL), None


def fallback_route(reason: str) -> RenderRoute:
    if reason == "tiny_head":
        return RenderRoute.SIMPLIFIED_AVATAR
    if reason in {"low_visibility", "high_occlusion", "roll_out_of_range"}:
        return RenderRoute.SILHOUETTE
    if reason == "edge_cropped":
        return RenderRoute.CROP_SAFE_AVATAR
    return RenderRoute.BLUR_FALLBACK
