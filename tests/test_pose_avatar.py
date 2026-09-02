from __future__ import annotations

import json
import math
import shutil

import numpy as np
import pytest
from PIL import Image

from app.config import AVATAR_ASSET_DIR
from app.image_ops import outside_mask_is_exact
from app.pose_avatar.adapters import detected_head_from_region
from app.pose_avatar.models import DetectedHead, PoseEstimate, RenderRoute, Scene, View
from app.pose_avatar.registry import AssetRegistry, AssetValidationError
from app.pose_avatar.renderer import PoseAvatarRenderer
from app.pose_avatar.routing import classify_scene, quality_gate
from app.pose_avatar.transform import apply_matrix, select_neck_adapter, solve_transform


@pytest.fixture(scope="module")
def registry() -> AssetRegistry:
    return AssetRegistry(AVATAR_ASSET_DIR)


def head(
    *,
    head_id: str = "head-1",
    bbox=(100.0, 70.0, 100.0, 120.0),
    yaw=0.0,
    pitch=0.0,
    roll=0.0,
    confidence=0.96,
    pose_confidence=0.95,
    visibility=0.95,
    occlusion=0.05,
    out_of_frame=0.0,
    view_hint=None,
    face_landmarks=None,
    body_landmarks=None,
    depth=0.0,
) -> DetectedHead:
    x, y, width, height = bbox
    return DetectedHead(
        head_id=head_id,
        bbox=bbox,
        face_landmarks=face_landmarks or {},
        body_landmarks=body_landmarks or {"neck_center": (x + width / 2, y + height * 0.88)},
        pose=PoseEstimate(yaw_deg=yaw, pitch_deg=pitch, roll_deg=roll, confidence=pose_confidence),
        view_hint=view_hint,
        confidence=confidence,
        visibility=visibility,
        occlusion_score=occlusion,
        out_of_frame_ratio=out_of_frame,
        depth_order=depth,
    )


def aligned_head(registry: AssetRegistry, scene: Scene, *, head_id="aligned", bbox=(100.0, 70.0, 100.0, 120.0), roll=0.0, depth=0.0) -> DetectedHead:
    yaw = {Scene.S01_FRONT_NEUTRAL: 0.0, Scene.S04_L34_NEUTRAL: -30.0, Scene.S07_R34_NEUTRAL: 30.0, Scene.S12_BACK: 0.0}[scene]
    view_hint = View.BACK if scene == Scene.S12_BACK else None
    base = head(head_id=head_id, bbox=bbox, yaw=yaw, roll=roll, view_hint=view_hint, depth=depth)
    record = registry.select(scene)
    assert record is not None
    transform = solve_transform(base, scene, record.anchors)
    mapping = {"left_eye_center": "left_eye", "right_eye_center": "right_eye", "nose_tip": "nose", "chin": "chin"}
    face = {
        target_name: apply_matrix(transform.matrix, source)
        for source_name, target_name in mapping.items()
        if (source := record.anchors.anchors.get(source_name)) is not None
    }
    return base.model_copy(update={"face_landmarks": face})


@pytest.mark.parametrize(
    ("yaw", "expected"),
    [(-50.0, Scene.S04_L34_NEUTRAL), (-15.01, Scene.S04_L34_NEUTRAL), (-15.0, Scene.S01_FRONT_NEUTRAL), (15.0, Scene.S01_FRONT_NEUTRAL), (15.01, Scene.S07_R34_NEUTRAL), (50.0, Scene.S07_R34_NEUTRAL)],
)
def test_scene_classifier_boundaries(yaw, expected):
    assert classify_scene(head(yaw=yaw)) == (expected, None)


def test_scene_classifier_rejects_profiles_and_non_neutral_pitch():
    assert classify_scene(head(yaw=-50.01))[1] == "unsupported_left_profile"
    assert classify_scene(head(yaw=50.01))[1] == "unsupported_right_profile"
    assert classify_scene(head(pitch=15.01))[1] == "unsupported_pitch"
    assert classify_scene(head(view_hint=View.BACK)) == (Scene.S12_BACK, None)


def test_yaw_adapter_uses_image_left_negative_and_image_right_positive():
    common = {"left_eye": [120, 110], "right_eye": [180, 110], "mouth": [150, 160]}
    left = detected_head_from_region({"id": "left", "head_box": [100, 60, 100, 140], "confidence": 0.95, "face_landmarks": {**common, "nose": [132, 138]}}, (400, 300))
    right = detected_head_from_region({"id": "right", "head_box": [100, 60, 100, 140], "confidence": 0.95, "face_landmarks": {**common, "nose": [168, 138]}}, (400, 300))
    assert left.pose.yaw_deg < 0
    assert right.pose.yaw_deg > 0
    assert classify_scene(left)[0] == Scene.S04_L34_NEUTRAL
    assert classify_scene(right)[0] == Scene.S07_R34_NEUTRAL


def test_roll_is_a_continuous_transform(registry):
    value = aligned_head(registry, Scene.S01_FRONT_NEUTRAL, roll=19.25)
    record = registry.select(Scene.S01_FRONT_NEUTRAL)
    transform = solve_transform(value, Scene.S01_FRONT_NEUTRAL, record.anchors)
    assert transform.rotation_deg == pytest.approx(19.25)
    assert math.degrees(math.atan2(transform.matrix[1][0], transform.matrix[0][0])) == pytest.approx(19.25)


def test_bbox_width_controls_overlay_scale(registry):
    record = registry.select(Scene.S01_FRONT_NEUTRAL)
    small = solve_transform(head(bbox=(10, 10, 80, 100)), Scene.S01_FRONT_NEUTRAL, record.anchors)
    large = solve_transform(head(bbox=(10, 10, 160, 200)), Scene.S01_FRONT_NEUTRAL, record.anchors)
    assert large.target_scale == pytest.approx(small.target_scale * 2)


def test_neck_anchor_is_aligned_exactly(registry):
    value = head(body_landmarks={"neck_center": (177.0, 211.0)})
    record = registry.select(Scene.S01_FRONT_NEUTRAL)
    transform = solve_transform(value, Scene.S01_FRONT_NEUTRAL, record.anchors)
    left, right = record.anchors.anchors["neck_left"], record.anchors.anchors["neck_right"]
    source_neck = ((left[0] + right[0]) / 2, (left[1] + right[1]) / 2)
    assert apply_matrix(transform.matrix, source_neck) == pytest.approx((177.0, 211.0))


def test_neck_adapter_interface_boundaries():
    assert select_neck_adapter(head(body_landmarks={})) == "B"
    assert select_neck_adapter(head(body_landmarks={"neck_left": (110, 170), "neck_right": (139, 170), "neck_center": (124.5, 170)})) == "A"
    assert select_neck_adapter(head(body_landmarks={"neck_left": (100, 170), "neck_right": (145, 170), "neck_center": (122.5, 170)})) == "C"


def test_manifest_validation_accepts_p0_assets_and_rejects_missing_asset(registry, tmp_path):
    assert {record.scene for record in registry.records if record.family.family_id == "generic"} == {
        Scene.S01_FRONT_NEUTRAL, Scene.S04_L34_NEUTRAL, Scene.S07_R34_NEUTRAL, Scene.S12_BACK,
    }
    copy = tmp_path / "avatar_families"
    shutil.copytree(AVATAR_ASSET_DIR, copy)
    manifest_path = copy / "generic" / "manifest.json"
    manifest = json.loads(manifest_path.read_text("utf8"))
    manifest["scenes"]["S01_FRONT_NEUTRAL"]["asset"] = "runtime_png/missing.png"
    manifest_path.write_text(json.dumps(manifest), "utf8")
    with pytest.raises(AssetValidationError, match="asset file missing"):
        AssetRegistry(copy)


@pytest.mark.parametrize(
    ("value", "route", "reason"),
    [
        (head(bbox=(10, 10, 47, 80)), RenderRoute.SIMPLIFIED_AVATAR, "tiny_head"),
        (head(confidence=0.64), RenderRoute.BLUR_FALLBACK, "low_confidence"),
        (head(visibility=0.44), RenderRoute.SILHOUETTE, "low_visibility"),
        (head(occlusion=0.56), RenderRoute.SILHOUETTE, "high_occlusion"),
        (head(out_of_frame=0.21), RenderRoute.CROP_SAFE_AVATAR, "edge_cropped"),
    ],
)
def test_quality_fallbacks_are_deterministic(registry, value, route, reason):
    result = PoseAvatarRenderer(registry).render(Image.new("RGB", (320, 260), "white"), [value], image_id="fixture")
    assert result.decisions[0].route_type == route
    assert result.decisions[0].fallback_reason == reason


def test_unsupported_scene_falls_back_instead_of_using_front_asset(registry):
    result = PoseAvatarRenderer(registry).render(Image.new("RGB", (320, 260), "white"), [head(pitch=20)], image_id="fixture")
    assert result.decisions[0].route_type == RenderRoute.BLUR_FALLBACK
    assert result.decisions[0].fallback_reason == "unsupported_pitch"


def test_multi_person_decisions_and_composites_are_isolated(registry):
    original = Image.fromarray(np.full((280, 520, 3), 235, dtype=np.uint8), "RGB")
    standard = aligned_head(registry, Scene.S04_L34_NEUTRAL, head_id="left-person", bbox=(80, 70, 120, 145), depth=1)
    tiny = head(head_id="right-person", bbox=(360, 100, 40, 47), depth=2)
    result = PoseAvatarRenderer(registry).render(original, [tiny, standard], image_id="multi")
    by_id = {decision.head_id: decision for decision in result.decisions}
    assert by_id["left-person"].scene_id == Scene.S04_L34_NEUTRAL
    assert by_id["left-person"].route_type == RenderRoute.STANDARD_AVATAR
    assert by_id["right-person"].route_type == RenderRoute.SIMPLIFIED_AVATAR
    assert {trace.head_id for trace in result.traces} == {"left-person", "right-person"}
    assert outside_mask_is_exact(original, result.image, result.alpha_mask)
    assert np.any(np.asarray(result.image)[70:220, 80:210] != np.asarray(original)[70:220, 80:210])
    assert np.any(np.asarray(result.image)[100:150, 360:405] != np.asarray(original)[100:150, 360:405])
