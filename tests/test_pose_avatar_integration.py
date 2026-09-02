from __future__ import annotations

from PIL import Image

import app.service as service_module
from app.config import AVATAR_ASSET_DIR
from app.pose_avatar.models import DetectedHead, PoseEstimate, Scene
from app.pose_avatar.registry import AssetRegistry
from app.pose_avatar.transform import apply_matrix, solve_transform
from app.schemas import EditRequest


def test_feature_flag_off_keeps_legacy_local_provider_path(monkeypatch, tmp_path):
    paths = {name: tmp_path / name.lower() for name in ("UPLOAD_DIR", "OUTPUT_DIR", "PREVIEW_DIR", "JOB_DIR")}
    for path in paths.values():
        path.mkdir()
    for name, path in paths.items():
        monkeypatch.setattr(service_module, name, path)
    monkeypatch.setattr(service_module, "POSE_AVATAR_OVERLAY_ENABLED", False)
    monkeypatch.setattr(service_module, "pose_avatar_renderer", lambda: (_ for _ in ()).throw(AssertionError("pose renderer must stay off")))

    image_id = "a" * 32
    Image.new("RGB", (240, 180), (120, 130, 140)).save(paths["UPLOAD_DIR"] / f"{image_id}.png")
    request = EditRequest(
        image_id=image_id,
        regions=[{
            "id": "person-1", "box": [70, 35, 80, 110], "head_box": [70, 35, 80, 110],
            "confidence": 0.95, "selected": True,
            "pose": {"yaw_deg": 0, "pitch_deg": 0, "roll_deg": 0, "confidence": 1},
        }],
        provider="local",
        selection_confirmed=True,
        pose_aware_overlay=True,
    )
    store = service_module.JobStore()
    job_id = "feature-off"
    store._save({"id": job_id, "status": "queued", "progress": 0, "request": request.model_dump(), "people": []})
    store._run(job_id, request)
    job = store.get(job_id)
    store.executor.shutdown(wait=True)
    assert job["status"] == "completed"
    assert job.get("pose_avatar_overlay") is not True
    assert job["outside_mask_exact"] is True


def test_feature_flag_and_request_opt_in_use_pose_renderer(monkeypatch, tmp_path):
    paths = {name: tmp_path / name.lower() for name in ("UPLOAD_DIR", "OUTPUT_DIR", "PREVIEW_DIR", "JOB_DIR")}
    for path in paths.values():
        path.mkdir()
    for name, path in paths.items():
        monkeypatch.setattr(service_module, name, path)
    monkeypatch.setattr(service_module, "POSE_AVATAR_OVERLAY_ENABLED", True)
    monkeypatch.setattr(service_module, "_pose_registry", None)

    image_id = "b" * 32
    Image.new("RGB", (320, 260), (180, 190, 200)).save(paths["UPLOAD_DIR"] / f"{image_id}.png")
    registry = AssetRegistry(AVATAR_ASSET_DIR)
    record = registry.select(Scene.S01_FRONT_NEUTRAL)
    base = DetectedHead(
        head_id="person-pose",
        bbox=(110, 55, 100, 120),
        body_landmarks={"neck_center": (160, 160.6)},
        pose=PoseEstimate(yaw_deg=0, pitch_deg=0, roll_deg=12, confidence=0.98),
        confidence=0.98,
    )
    transform = solve_transform(base, Scene.S01_FRONT_NEUTRAL, record.anchors)
    mapping = {"left_eye_center": "left_eye", "right_eye_center": "right_eye", "nose_tip": "nose", "chin": "chin"}
    face = {
        target: apply_matrix(transform.matrix, source)
        for source_name, target in mapping.items()
        if (source := record.anchors.anchors.get(source_name)) is not None
    }
    request = EditRequest(
        image_id=image_id,
        regions=[{
            "id": "person-pose", "box": [110, 55, 100, 120], "head_box": [110, 55, 100, 120],
            "confidence": 0.98, "selected": True, "face_landmarks": face,
            "body_landmarks": {"neck_center": [160, 160.6]},
            "pose": {"yaw_deg": 0, "pitch_deg": 0, "roll_deg": 12, "confidence": 0.98},
        }],
        provider="local",
        selection_confirmed=True,
        pose_aware_overlay=True,
    )
    store = service_module.JobStore()
    job_id = "feature-on"
    store._save({"id": job_id, "status": "queued", "progress": 0, "request": request.model_dump(), "people": []})
    store._run(job_id, request)
    job = store.get(job_id)
    store.executor.shutdown(wait=True)
    assert job["status"] == "completed"
    assert job["pose_avatar_overlay"] is True
    assert job["outside_mask_exact"] is True
    assert job["decisions"][0]["route_type"] == "STANDARD_AVATAR"
    assert job["decisions"][0]["scene_id"] == "S01_FRONT_NEUTRAL"
