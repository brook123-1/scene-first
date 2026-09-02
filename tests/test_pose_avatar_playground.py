from __future__ import annotations

import json

from PIL import Image, ImageDraw

from app.pose_avatar.models import Scene
from app.pose_avatar.playground import PlaygroundRenderRequest, PoseAvatarPlayground
from app.pose_avatar.registry import AssetRegistry


def _write_json(path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf8")


def _playground(tmp_path) -> PoseAvatarPlayground:
    root = tmp_path
    assets = root / "assets" / "avatar_families"
    family = assets / "playground-demo-01"
    family.mkdir(parents=True)
    avatar = Image.new("RGBA", (100, 120), (0, 0, 0, 0))
    ImageDraw.Draw(avatar).rounded_rectangle((10, 5, 90, 115), radius=20, fill=(230, 180, 120, 255))
    avatar.save(family / "front.png")
    _write_json(family / "manifest.json", {
        "schema_version": "1.1", "family_id": "playground-demo-01", "version": "1.0.0",
        "style_group": "round-neutral", "privacy_mode": "non_identity_preserving",
        "supported_scenes": ["S01_FRONT_NEUTRAL"], "supported_expressions": ["NEUTRAL"],
        "playground_only": True, "anchors_file": "anchors.json", "constraints_file": "constraints.json",
        "scenes": {"S01_FRONT_NEUTRAL": {"asset": "front.png", "anchor_key": "front", "constraint_key": "front"}},
    })
    _write_json(family / "anchors.json", {"schema_version": "1.1", "scenes": {"front": {
        "canvas_size": [100, 120],
        "anchors": {"head_top": [50, 5], "chin": [50, 85], "left_eye_center": [35, 45],
                    "right_eye_center": [65, 45], "nose_tip": [50, 60], "neck_left": [40, 95], "neck_right": [60, 95]},
        "overlay_bbox": [10, 5, 80, 110], "safe_mask_polygon": [[50, 5], [10, 40], [30, 115], [70, 115], [90, 40]],
        "silhouette_bbox": [10, 5, 80, 110], "silhouette_polygon": [[50, 5], [10, 40], [30, 115], [70, 115], [90, 40]],
        "neck_overlap_zone": [[35, 82], [65, 82], [70, 102], [30, 102]], "transparent_canvas_bounds": [10, 5, 80, 110],
    }}})
    _write_json(family / "constraints.json", {"schema_version": "1.1", "scenes": {"front": {
        "scene_id": "S01_FRONT_NEUTRAL", "yaw_range_deg": [-15, 15], "pitch_range_deg": [-20, 20],
        "roll_range_deg": [-35, 35], "min_head_size_px": 20, "max_head_size_px": 500,
        "visibility_min": .4, "crop_tolerance": .2, "max_occlusion_score": .6,
    }}})

    photo = root / "samples" / "inbox" / "photo.png"
    photo.parent.mkdir(parents=True)
    Image.new("RGB", (200, 180), (35, 45, 55)).save(photo)
    local = root / ".local" / "app" / "pose-avatar-p1"
    case = {
        "case_id": "p1-001", "local_image_ref": "samples/inbox/photo.png", "source_sample_id": "01",
        "view": "FRONT", "expected_scene": "S01_FRONT_NEUTRAL", "route_judgment": "STANDARD_ELIGIBLE",
        "observations": {
            "yunet_5pt_heuristic": {"actual_route": "BLUR_FALLBACK", "actual_scene": "S01_FRONT_NEUTRAL"},
            "opencv_mediapipe_pose_hybrid": {"body_landmarks": {"neck_left": [83, 120], "neck_right": [117, 120], "neck_center": [100, 120]}},
        },
    }
    _write_json(local / "annotations.json", {"cases": [case]})
    _write_json(local / "cases" / "p1-001" / "07-trace.json", {"head": {
        "head_id": "p1-001", "bbox": [60, 20, 80, 105],
        "face_landmarks": {"left_eye": [85, 60], "right_eye": [115, 60], "nose": [100, 78], "chin": [100, 105]},
        "body_landmarks": {"neck_center": [100, 120]}, "pose": {"yaw_deg": 0, "pitch_deg": 0, "roll_deg": 3},
    }, "coverage": {"landmark_silhouette_v1": {"exposed_head_ratio": .1}}})
    return PoseAvatarPlayground(root=root, local_root=local, asset_root=assets)


def test_playground_forces_only_human_standard_case_and_renders_both_fittings(tmp_path):
    playground = _playground(tmp_path)
    payload = playground.list_payload()
    assert payload["items"][0]["case_id"] == "p1-001"
    assert payload["families"][0]["id"] == "playground-demo-01"

    face = playground.render(PlaygroundRenderRequest(case_id="p1-001", family_id="playground-demo-01"))
    assert face["image_data_url"].startswith("data:image/jpeg;base64,")
    assert face["meta"]["scene"] == Scene.S01_FRONT_NEUTRAL
    assert face["meta"]["forced_preview"] is True
    assert face["meta"]["neck_stage"] == "not_used"

    two_stage = playground.render(PlaygroundRenderRequest(
        case_id="p1-001", family_id="playground-demo-01", fitting="two-stage", scale=1.1, x_offset=4,
    ))
    assert two_stage["meta"]["neck_stage"] == "independent"
    assert two_stage["meta"]["transform"]["target_scale"] > face["meta"]["transform"]["target_scale"]


def test_playground_only_family_is_excluded_from_production_selection(tmp_path):
    playground = _playground(tmp_path)
    registry = AssetRegistry(playground.asset_root)
    assert registry.select(Scene.S01_FRONT_NEUTRAL, "playground-demo-01") is None
    assert registry.select(Scene.S01_FRONT_NEUTRAL, "playground-demo-01", include_playground=True) is not None


def test_registry_rasterizes_transparent_svg_assets(tmp_path):
    path = tmp_path / "avatar.svg"
    path.write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" width="32" height="40"><path fill="#eeaa77" d="M4 2h24v36H4z"/></svg>',
        encoding="utf8",
    )
    image = AssetRegistry._open_asset(path)
    assert image.mode == "RGBA"
    assert image.size == (32, 40)
    assert image.getchannel("A").getbbox() == (4, 2, 28, 38)
