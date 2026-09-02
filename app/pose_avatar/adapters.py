from __future__ import annotations

import math
from typing import Any

from .models import CropStatus, DetectedHead, PoseEstimate, View


def _point(value: Any) -> tuple[float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    try:
        return float(value[0]), float(value[1])
    except (TypeError, ValueError):
        return None


def _landmarks(value: Any) -> dict[str, tuple[float, float]]:
    if not isinstance(value, dict):
        return {}
    result = {}
    for key, raw in value.items():
        parsed = _point(raw)
        if parsed is not None:
            result[str(key)] = parsed
    return result


def estimate_pose(
    bbox: tuple[float, float, float, float],
    face_landmarks: dict[str, tuple[float, float]],
    detection_confidence: float,
) -> PoseEstimate:
    """Estimate a conservative 2D pose from image-left/image-right landmarks.

    This is an adapter, not a learned pose model. Positive yaw means the nose
    moves toward image-right and maps to RIGHT_34; negative maps to LEFT_34.
    """
    left_eye = face_landmarks.get("left_eye")
    right_eye = face_landmarks.get("right_eye")
    nose = face_landmarks.get("nose")
    mouth = face_landmarks.get("mouth")
    _, _, _, head_height = bbox
    if not left_eye or not right_eye or not nose:
        return PoseEstimate(yaw_deg=0.0, pitch_deg=0.0, roll_deg=0.0, confidence=0.0)

    if left_eye[0] > right_eye[0]:
        left_eye, right_eye = right_eye, left_eye
    eye_dx = right_eye[0] - left_eye[0]
    eye_dy = right_eye[1] - left_eye[1]
    eye_distance = max(1.0, math.hypot(eye_dx, eye_dy))
    eye_mid = ((left_eye[0] + right_eye[0]) / 2, (left_eye[1] + right_eye[1]) / 2)
    yaw = max(-65.0, min(65.0, (nose[0] - eye_mid[0]) / eye_distance * 75.0))
    roll = math.degrees(math.atan2(eye_dy, eye_dx))

    # YuNet gives eye, nose and mouth corners but no 3D pose.  The normalized
    # eye->nose / eye->mouth ratio is a weak pitch signal and is deliberately
    # kept close to neutral unless it is well outside the nominal 0.52 ratio.
    pitch = 0.0
    if mouth:
        eye_to_mouth = max(1.0, mouth[1] - eye_mid[1])
        ratio = (nose[1] - eye_mid[1]) / eye_to_mouth
        pitch = max(-30.0, min(30.0, (0.52 - ratio) * 75.0))
    confidence = min(0.90, max(0.0, detection_confidence) * 0.92)
    if eye_distance < max(6.0, head_height * 0.12):
        confidence *= 0.55
    return PoseEstimate(yaw_deg=yaw, pitch_deg=pitch, roll_deg=roll, confidence=confidence)


def detected_head_from_region(
    region: dict[str, Any],
    image_size: tuple[int, int],
) -> DetectedHead:
    raw_bbox = region.get("head_box") or region.get("bbox") or region.get("box")
    if not isinstance(raw_bbox, (list, tuple)) or len(raw_bbox) != 4:
        raise ValueError("confirmed region has no valid head_box")
    x, y, width, height = (float(value) for value in raw_bbox)
    image_width, image_height = image_size
    if x < 0 or y < 0 or width <= 0 or height <= 0 or x + width > image_width + 1 or y + height > image_height + 1:
        raise ValueError("confirmed region head_box is outside image bounds")
    bbox = (x, y, width, height)
    confidence = float(region.get("confidence", 1.0))
    face = _landmarks(region.get("face_landmarks"))
    body = _landmarks(region.get("body_landmarks"))
    provenance = {str(k): str(v) for k, v in (region.get("anchor_provenance") or {}).items()}

    # The current detectors do not expose chin/neck landmarks.  These bbox
    # estimates are explicit in provenance and only support coarse P0 fitting.
    if "chin" not in face:
        face["chin"] = (x + width * 0.5, y + height * 0.76)
        provenance["chin"] = "head_bbox_estimate"
    if "neck_center" not in body:
        body["neck_center"] = (x + width * 0.5, y + height * 0.88)
        provenance["neck_center"] = "head_bbox_estimate"

    pose_value = region.get("pose")
    if isinstance(pose_value, dict) and all(key in pose_value for key in ("yaw_deg", "pitch_deg", "roll_deg")):
        pose = PoseEstimate.model_validate(pose_value)
    else:
        pose = estimate_pose(bbox, face, confidence)

    view_hint = region.get("view_hint")
    return DetectedHead(
        head_id=str(region.get("head_id") or region.get("id") or "head"),
        bbox=bbox,
        face_landmarks=face,
        body_landmarks=body,
        anchor_provenance=provenance,
        pose=pose,
        view_hint=View(view_hint) if view_hint else None,
        visibility=float(region.get("visibility", 1.0)),
        crop_status=CropStatus(region.get("crop_status", CropStatus.FULL_IN_FRAME)),
        out_of_frame_ratio=float(region.get("out_of_frame_ratio", 0.0)),
        occlusion_score=float(region.get("occlusion_score", 0.0)),
        accessory_flags=list(region.get("accessory_flags") or []),
        depth_order=float(region.get("depth_order") if region.get("depth_order") is not None else y + height + math.sqrt(width * height) * 0.25),
        confidence=confidence,
    )


def enrich_detection(detection: dict[str, Any], image_size: tuple[int, int]) -> dict[str, Any]:
    """Add the versioned pose contract to a detector result when P0 is enabled."""
    head = detected_head_from_region(detection, image_size)
    return {
        **detection,
        "pose_schema_version": head.schema_version,
        "face_landmarks": {key: list(value) for key, value in head.face_landmarks.items()},
        "body_landmarks": {key: list(value) for key, value in head.body_landmarks.items()},
        "anchor_provenance": head.anchor_provenance,
        "pose": head.pose.model_dump(),
        "size_tier": head.size_tier.value,
        "visibility": head.visibility,
        "crop_status": head.crop_status.value,
        "out_of_frame_ratio": head.out_of_frame_ratio,
        "occlusion_score": head.occlusion_score,
        "depth_order": head.depth_order,
    }
