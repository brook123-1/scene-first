from __future__ import annotations

from base64 import b64encode
from io import BytesIO
import json
import math
from pathlib import Path
import re
from typing import Literal

import numpy as np
from PIL import Image
from pydantic import BaseModel, Field

from .benchmark import RouteJudgment
from .models import DetectedHead, RenderRoute, RenderTransform, Scene, View
from .registry import AssetRecord, AssetRegistry, AssetValidationError
from .renderer import _composite_rgba, _warp_asset
from .transform import solve_face_primary_transform, solve_two_stage_head_neck


BASELINE_ADAPTER = "yunet_5pt_heuristic"
NECK_ADAPTER = "opencv_mediapipe_pose_hybrid"
CASE_ID = re.compile(r"^p1-[0-9]{3}$")
PLAYGROUND_SCENES = (
    Scene.S01_FRONT_NEUTRAL,
    Scene.S04_L34_NEUTRAL,
    Scene.S07_R34_NEUTRAL,
    Scene.S10_L_PROFILE,
    Scene.S11_R_PROFILE,
)
VIEW_SCENES = {
    View.FRONT: Scene.S01_FRONT_NEUTRAL,
    View.LEFT_34: Scene.S04_L34_NEUTRAL,
    View.RIGHT_34: Scene.S07_R34_NEUTRAL,
    View.LEFT_PROFILE: Scene.S10_L_PROFILE,
    View.RIGHT_PROFILE: Scene.S11_R_PROFILE,
}


class PlaygroundRenderRequest(BaseModel):
    case_id: str = Field(pattern=r"^p1-[0-9]{3}$")
    family_id: str
    fitting: Literal["face-primary", "two-stage"] = "face-primary"
    scene_override: Scene | None = None
    scale: float = Field(default=1.0, ge=0.5, le=1.75)
    x_offset: float = Field(default=0.0, ge=-500, le=500)
    y_offset: float = Field(default=0.0, ge=-500, le=500)
    rotation: float = Field(default=0.0, ge=-60, le=60)


def _adjust_transform(
    transform: RenderTransform,
    center: tuple[float, float],
    *,
    scale: float,
    x_offset: float,
    y_offset: float,
    rotation: float,
) -> RenderTransform:
    angle = math.radians(rotation)
    cosine = math.cos(angle) * scale
    sine = math.sin(angle) * scale
    cx, cy = center
    adjustment = np.asarray(
        [
            [cosine, -sine, cx + x_offset - cosine * cx + sine * cy],
            [sine, cosine, cy + y_offset - sine * cx - cosine * cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    source = np.vstack((np.asarray(transform.matrix, dtype=np.float64), [0.0, 0.0, 1.0]))
    matrix = (adjustment @ source)[:2]
    return transform.model_copy(
        update={
            "matrix": matrix.tolist(),
            "target_scale": transform.target_scale * scale,
            "rotation_deg": transform.rotation_deg + rotation,
            "translation": (float(matrix[0, 2]), float(matrix[1, 2])),
        }
    )


def _split_head_and_neck(asset: Image.Image, record: AssetRecord) -> tuple[Image.Image, Image.Image]:
    rgba = np.asarray(asset.convert("RGBA")).copy()
    zone = record.anchors.neck_overlap_zone
    if zone:
        top = max(0, round(min(point[1] for point in zone)))
        bottom = min(rgba.shape[0], round(max(point[1] for point in zone)))
    else:
        neck_y = round(np.mean([record.anchors.anchors["neck_left"][1], record.anchors.anchors["neck_right"][1]]))
        top, bottom = max(0, neck_y - 12), min(rgba.shape[0], neck_y + 12)
    head = rgba.copy()
    neck = rgba.copy()
    head[bottom:, 3] = 0
    neck[:top, 3] = 0
    return Image.fromarray(head, "RGBA"), Image.fromarray(neck, "RGBA")


def _data_url(image: Image.Image) -> str:
    output = BytesIO()
    image.convert("RGB").save(output, "JPEG", quality=90, optimize=True)
    return "data:image/jpeg;base64," + b64encode(output.getvalue()).decode("ascii")


class PoseAvatarPlayground:
    def __init__(self, *, root: Path, local_root: Path, asset_root: Path) -> None:
        self.root = Path(root).resolve()
        self.local_root = Path(local_root).resolve()
        self.asset_root = Path(asset_root).resolve()

    @property
    def annotations_path(self) -> Path:
        return self.local_root / "annotations.json"

    def _document(self) -> dict:
        if not self.annotations_path.is_file():
            return {"cases": []}
        return json.loads(self.annotations_path.read_text(encoding="utf8"))

    def _case(self, case_id: str) -> dict:
        if not CASE_ID.fullmatch(case_id):
            raise KeyError(case_id)
        for case in self._document().get("cases", []):
            if case.get("case_id") == case_id:
                return case
        raise KeyError(case_id)

    def _registry(self) -> AssetRegistry:
        return AssetRegistry(self.asset_root)

    def _source_path(self, case: dict) -> Path:
        allowed = (self.root / "samples" / "inbox").resolve()
        candidate = (self.root / str(case["local_image_ref"])).resolve()
        if allowed not in candidate.parents or not candidate.is_file():
            raise FileNotFoundError(candidate)
        return candidate

    def _trace(self, case_id: str) -> dict:
        path = self.local_root / "cases" / case_id / "07-trace.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        return json.loads(path.read_text(encoding="utf8"))

    def list_payload(self) -> dict:
        cases = self._document().get("cases", [])
        try:
            records = self._registry().records
            registry_error = None
        except AssetValidationError as exc:
            records = []
            registry_error = str(exc)
        grouped: dict[str, dict] = {}
        for record in records:
            item = grouped.setdefault(
                record.family.family_id,
                {
                    "id": record.family.family_id,
                    "version": record.family.version,
                    "style_group": record.family.style_group,
                    "fixture": "fixture" in record.family.style_group,
                    "scenes": [],
                },
            )
            item["scenes"].append(record.scene.value)
        items = []
        for case in cases:
            observation = case.get("observations", {}).get(BASELINE_ADAPTER, {})
            items.append(
                {
                    "case_id": case["case_id"],
                    "source_sample_id": case.get("source_sample_id"),
                    "view": case.get("view"),
                    "route_judgment": case.get("route_judgment"),
                    "actual_route": observation.get("actual_route"),
                    "actual_scene": observation.get("actual_scene"),
                    "original_url": f"/api/pose-avatar-playground/cases/{case['case_id']}/original",
                }
            )
        return {
            "items": items,
            "families": sorted(grouped.values(), key=lambda value: (value["fixture"], value["id"])),
            "scenes": [scene.value for scene in PLAYGROUND_SCENES],
            "registry_error": registry_error,
        }

    def original_path(self, case_id: str) -> Path:
        return self._source_path(self._case(case_id))

    @staticmethod
    def _automatic_scene(head: DetectedHead, case: dict, records: list[AssetRecord]) -> Scene | None:
        supported = {record.scene for record in records}
        candidates = [
            record for record in records
            if record.scene in PLAYGROUND_SCENES
            and record.constraints.yaw_range_deg[0] <= head.pose.yaw_deg <= record.constraints.yaw_range_deg[1]
        ]
        if candidates:
            candidates.sort(key=lambda record: abs(head.pose.yaw_deg - sum(record.constraints.yaw_range_deg) / 2))
            return candidates[0].scene
        observation = case.get("observations", {}).get(BASELINE_ADAPTER, {})
        if observation.get("actual_scene"):
            scene = Scene(observation["actual_scene"])
            if scene in supported:
                return scene
        if case.get("expected_scene"):
            scene = Scene(case["expected_scene"])
            if scene in supported:
                return scene
        if case.get("view"):
            return VIEW_SCENES.get(View(case["view"]))
        return None

    def render(self, request: PlaygroundRenderRequest) -> dict:
        case = self._case(request.case_id)
        trace = self._trace(request.case_id)
        head = DetectedHead.model_validate(trace["head"])
        neck_observation = case.get("observations", {}).get(NECK_ADAPTER, {})
        body = {**head.body_landmarks, **neck_observation.get("body_landmarks", {})}
        head = head.model_copy(update={"body_landmarks": body})

        registry = self._registry()
        family_records = [record for record in registry.records if record.family.family_id == request.family_id]
        if not family_records:
            raise ValueError(f"unknown avatar family: {request.family_id}")
        automatic_scene = self._automatic_scene(head, case, family_records)
        scene = request.scene_override or automatic_scene
        if scene is None:
            raise ValueError("no supported scene could be selected")
        record = registry.select(scene, request.family_id, include_playground=True)
        if record is None:
            raise ValueError(f"{request.family_id} does not provide {scene.value}")

        observation = case.get("observations", {}).get(BASELINE_ADAPTER, {})
        production_route = observation.get("actual_route") or RenderRoute.BLUR_FALLBACK.value
        human_standard = case.get("route_judgment") == RouteJudgment.STANDARD_ELIGIBLE.value
        forced_preview = production_route != RenderRoute.STANDARD_AVATAR.value and human_standard
        preview_allowed = production_route == RenderRoute.STANDARD_AVATAR.value or forced_preview

        with Image.open(self._source_path(case)) as source_image:
            original = source_image.convert("RGB")
        warnings: list[str] = []
        head_transform: RenderTransform | None = None
        neck_transform: RenderTransform | None = None
        composite = original.copy()
        if preview_allowed:
            if request.fitting == "two-stage":
                head_transform, neck_transform = solve_two_stage_head_neck(head, record.anchors)
                if neck_transform is None:
                    warnings.append("neck_landmarks_unavailable; rendered face-primary head stage")
            else:
                head_transform = solve_face_primary_transform(head, record.anchors, allow_neck_correction=False)
            center = (head.bbox[0] + head.bbox[2] / 2, head.bbox[1] + head.bbox[3] / 2)
            head_transform = _adjust_transform(
                head_transform,
                center,
                scale=request.scale,
                x_offset=request.x_offset,
                y_offset=request.y_offset,
                rotation=request.rotation,
            )
            asset = registry.image(record)
            if request.fitting == "two-stage" and neck_transform is not None:
                neck_transform = _adjust_transform(
                    neck_transform,
                    center,
                    scale=request.scale,
                    x_offset=request.x_offset,
                    y_offset=request.y_offset,
                    rotation=request.rotation,
                )
                head_asset, neck_asset = _split_head_and_neck(asset, record)
                neck_overlay = _warp_asset(neck_asset, neck_transform.matrix, original.size)
                head_overlay = _warp_asset(head_asset, head_transform.matrix, original.size)
                composite = _composite_rgba(_composite_rgba(original, neck_overlay), head_overlay)
            else:
                overlay = _warp_asset(asset, head_transform.matrix, original.size)
                composite = _composite_rgba(original, overlay)
        else:
            warnings.append("human review does not authorize forced avatar preview for this fallback case")

        coverage = trace.get("coverage", {})
        return {
            "image_data_url": _data_url(composite),
            "meta": {
                "case_id": request.case_id,
                "yaw": head.pose.yaw_deg,
                "roll": head.pose.roll_deg,
                "scene": scene.value,
                "automatic_scene": automatic_scene.value if automatic_scene else None,
                "head_bbox": list(head.bbox),
                "route": production_route,
                "preview_route": "FORCED_STANDARD_PREVIEW" if forced_preview else production_route,
                "forced_preview": forced_preview,
                "fitting": request.fitting,
                "neck_stage": "independent" if neck_transform is not None else "not_used",
                "asset": str(record.path.relative_to(self.root)),
                "coverage": coverage,
                "warnings": warnings,
                "transform": head_transform.model_dump() if head_transform else None,
            },
        }
