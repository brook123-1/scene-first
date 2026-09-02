from __future__ import annotations

import math

import numpy as np
import cv2

from .models import DetectedHead, RenderTransform, Scene
from .registry import SceneAnchors
from .routing import DEFAULT_THRESHOLDS, RoutingThresholds


def _transform_from_matrix(head: DetectedHead, matrix: np.ndarray, mode: str, fit_points: dict[str, tuple[float, float]]) -> RenderTransform:
    scale = math.sqrt(matrix[0, 0] ** 2 + matrix[0, 1] ** 2)
    rotation = math.degrees(math.atan2(matrix[1, 0], matrix[0, 0]))
    errors = [math.dist(apply_matrix(matrix, fit_points[key]), fit_points[key.replace("src_", "dst_")])
              for key in fit_points if key.startswith("src_") and key.replace("src_", "dst_") in fit_points]
    residual = float(np.sqrt(np.mean(np.square(errors)))) if errors else 0.0
    return RenderTransform(head_id=head.head_id, matrix=matrix.tolist(), target_scale=scale, rotation_deg=rotation,
                           translation=(float(matrix[0, 2]), float(matrix[1, 2])), warp_mode=mode,
                           fit_points=fit_points, residual_error_px=residual,
                           residual_normalized=residual / max(head.bbox[2], 1.0),
                           residual_components={"final_normalized_residual": residual / max(head.bbox[2], 1.0)})


def solve_face_primary_transform(head: DetectedHead, asset_anchors: SceneAnchors, *, allow_neck_correction: bool = True) -> RenderTransform:
    mapping = (("left_eye_center", "left_eye"), ("right_eye_center", "right_eye"), ("nose_tip", "nose"), ("chin", "chin"))
    pairs = [(asset_anchors.anchors.get(source), head.face_landmarks.get(target), target) for source, target in mapping]
    pairs = [(source, target, name) for source, target, name in pairs if source is not None and target is not None]
    if len(pairs) < 2:
        raise ValueError("face-primary fitting requires at least two face anchors")
    source = np.asarray([pair[0] for pair in pairs], dtype=np.float32)
    target = np.asarray([pair[1] for pair in pairs], dtype=np.float32)
    matrix, _ = cv2.estimateAffinePartial2D(source, target, method=cv2.LMEDS)
    if matrix is None:
        raise ValueError("face-primary similarity solve failed")
    source_neck = tuple(np.mean(np.asarray([asset_anchors.anchors["neck_left"], asset_anchors.anchors["neck_right"]]), axis=0))
    if allow_neck_correction:
        predicted_neck = apply_matrix(matrix, source_neck)
        correction = max(-head.bbox[2] * .05, min(head.bbox[2] * .05, head.neck_center[1] - predicted_neck[1]))
        matrix[1, 2] += correction
    fit_points = {f"src_{name}": tuple(src) for src, _, name in pairs} | {f"dst_{name}": tuple(dst) for _, dst, name in pairs}
    result = _transform_from_matrix(head, matrix, "SIMILARITY", fit_points)
    result.residual_components["neck_center_error_px"] = math.dist(apply_matrix(matrix, source_neck), head.neck_center)
    result.residual_components["neck_center_error_head_width"] = result.residual_components["neck_center_error_px"] / max(head.bbox[2], 1.0)
    return result


def solve_two_stage_head_neck(head: DetectedHead, asset_anchors: SceneAnchors) -> tuple[RenderTransform, RenderTransform | None]:
    head_transform = solve_face_primary_transform(head, asset_anchors, allow_neck_correction=False)
    source_left, source_right = asset_anchors.anchors.get("neck_left"), asset_anchors.anchors.get("neck_right")
    target_left, target_right = head.body_landmarks.get("neck_left"), head.body_landmarks.get("neck_right")
    if not (source_left and source_right and target_left and target_right):
        return head_transform, None
    source = np.asarray([source_left, source_right], dtype=np.float32)
    target = np.asarray([target_left, target_right], dtype=np.float32)
    matrix, _ = cv2.estimateAffinePartial2D(source, target, method=cv2.LMEDS)
    if matrix is None:
        return head_transform, None
    fit = {"src_neck_left": source_left, "dst_neck_left": target_left, "src_neck_right": source_right, "dst_neck_right": target_right}
    return head_transform, _transform_from_matrix(head, matrix, "SIMILARITY", fit)


def apply_matrix(matrix: list[list[float]] | np.ndarray, point: tuple[float, float]) -> tuple[float, float]:
    value = np.asarray(matrix, dtype=np.float64)
    mapped = value @ np.asarray([point[0], point[1], 1.0], dtype=np.float64)
    return float(mapped[0]), float(mapped[1])


def select_neck_adapter(head: DetectedHead) -> str:
    left = head.body_landmarks.get("neck_left")
    right = head.body_landmarks.get("neck_right")
    if not left or not right:
        # P0 ships only adapter B. Shoulder width is not treated as a neck
        # width measurement; doing so selected thin necks too aggressively.
        return "B"
    ratio = math.dist(left, right) / max(head.bbox[2], 1.0)
    if ratio < 0.30:
        return "A"
    if ratio <= 0.42:
        return "B"
    return "C"


def solve_transform(
    head: DetectedHead,
    scene: Scene,
    asset_anchors: SceneAnchors,
    thresholds: RoutingThresholds = DEFAULT_THRESHOLDS,
) -> RenderTransform:
    overlay_x, _, overlay_width, _ = asset_anchors.overlay_bbox
    target_width = head.bbox[2] * thresholds.target_head_width_multiplier
    scale = target_width / max(overlay_width, 1.0)
    angle = math.radians(head.pose.roll_deg)
    cosine, sine = math.cos(angle), math.sin(angle)

    neck_left = asset_anchors.anchors["neck_left"]
    neck_right = asset_anchors.anchors["neck_right"]
    assert neck_left is not None and neck_right is not None
    source_neck = ((neck_left[0] + neck_right[0]) / 2, (neck_left[1] + neck_right[1]) / 2)
    target_neck = head.neck_center
    tx = target_neck[0] - scale * (cosine * source_neck[0] - sine * source_neck[1])
    ty = target_neck[1] - scale * (sine * source_neck[0] + cosine * source_neck[1])
    matrix = np.asarray([
        [scale * cosine, -scale * sine, tx],
        [scale * sine, scale * cosine, ty],
    ], dtype=np.float64)

    anchor_mapping = {
        "left_eye_center": "left_eye",
        "right_eye_center": "right_eye",
        "nose_tip": "nose",
        "chin": "chin",
    }
    fit_points: dict[str, tuple[float, float]] = {
        "src_neck_center": source_neck,
        "dst_neck_center": target_neck,
    }
    residuals = []
    component_px: dict[str, float] = {}
    for source_name, target_name in anchor_mapping.items():
        source = asset_anchors.anchors.get(source_name)
        target = head.face_landmarks.get(target_name)
        if source is None or target is None:
            continue
        predicted = apply_matrix(matrix, source)
        error = math.dist(predicted, target)
        residuals.append(error)
        component_px[f"{target_name}_error_px"] = error
        fit_points[f"src_{target_name}"] = source
        fit_points[f"dst_{target_name}"] = target
    residual = float(np.sqrt(np.mean(np.square(residuals)))) if residuals else 0.0
    normalized = residual / max(head.bbox[2], 1.0)
    predicted_neck_left = apply_matrix(matrix, neck_left)
    predicted_neck_right = apply_matrix(matrix, neck_right)
    target_neck_left = head.body_landmarks.get("neck_left")
    target_neck_right = head.body_landmarks.get("neck_right")
    if target_neck_left:
        component_px["neck_left_error_px"] = math.dist(predicted_neck_left, target_neck_left)
    if target_neck_right:
        component_px["neck_right_error_px"] = math.dist(predicted_neck_right, target_neck_right)
    component_px["neck_center_error_px"] = math.dist(apply_matrix(matrix, source_neck), target_neck)
    component_px["vertical_offset_error_px"] = abs(apply_matrix(matrix, source_neck)[1] - target_neck[1])
    left_eye = head.face_landmarks.get("left_eye")
    right_eye = head.face_landmarks.get("right_eye")
    source_left_eye = asset_anchors.anchors.get("left_eye_center")
    source_right_eye = asset_anchors.anchors.get("right_eye_center")
    if left_eye and right_eye and source_left_eye and source_right_eye:
        predicted_left = apply_matrix(matrix, source_left_eye)
        predicted_right = apply_matrix(matrix, source_right_eye)
        predicted_angle = math.degrees(math.atan2(predicted_right[1] - predicted_left[1], predicted_right[0] - predicted_left[0]))
        target_angle = math.degrees(math.atan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0]))
        component_px["eye_pair_angle_error_deg"] = abs((predicted_angle - target_angle + 180) % 360 - 180)
        target_distance = math.dist(left_eye, right_eye)
        component_px["eye_distance_scale_error"] = abs(math.dist(predicted_left, predicted_right) / max(target_distance, 1.0) - 1.0)
    components = dict(component_px)
    for name, value in component_px.items():
        if name.endswith("_px"):
            components[name.removesuffix("_px") + "_head_width"] = value / max(head.bbox[2], 1.0)
    components["final_normalized_residual"] = normalized
    return RenderTransform(
        head_id=head.head_id,
        matrix=matrix.tolist(),
        target_scale=scale,
        rotation_deg=head.pose.roll_deg,
        translation=(tx, ty),
        warp_mode="AFFINE" if scene in {Scene.S04_L34_NEUTRAL, Scene.S07_R34_NEUTRAL} else "SIMILARITY",
        fit_points=fit_points,
        residual_error_px=residual,
        residual_normalized=normalized,
        residual_components=components,
    )
