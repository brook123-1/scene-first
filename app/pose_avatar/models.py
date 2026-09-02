from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Literal

from PIL import Image
from pydantic import BaseModel, Field, model_validator


SCHEMA_VERSION = "1.0"
Point = tuple[float, float]
BBox = tuple[float, float, float, float]


class View(StrEnum):
    FRONT = "FRONT"
    LEFT_34 = "LEFT_34"
    RIGHT_34 = "RIGHT_34"
    LEFT_PROFILE = "LEFT_PROFILE"
    RIGHT_PROFILE = "RIGHT_PROFILE"
    BACK = "BACK"


class Scene(StrEnum):
    S01_FRONT_NEUTRAL = "S01_FRONT_NEUTRAL"
    S02_FRONT_UP = "S02_FRONT_UP"
    S03_FRONT_DOWN = "S03_FRONT_DOWN"
    S04_L34_NEUTRAL = "S04_L34_NEUTRAL"
    S05_L34_UP = "S05_L34_UP"
    S06_L34_DOWN = "S06_L34_DOWN"
    S07_R34_NEUTRAL = "S07_R34_NEUTRAL"
    S08_R34_UP = "S08_R34_UP"
    S09_R34_DOWN = "S09_R34_DOWN"
    S10_L_PROFILE = "S10_L_PROFILE"
    S11_R_PROFILE = "S11_R_PROFILE"
    S12_BACK = "S12_BACK"


P0_SCENES = frozenset({
    Scene.S01_FRONT_NEUTRAL,
    Scene.S04_L34_NEUTRAL,
    Scene.S07_R34_NEUTRAL,
    Scene.S12_BACK,
})


class Expression(StrEnum):
    NEUTRAL = "NEUTRAL"
    SMILE = "SMILE"
    OPEN_MOUTH = "OPEN_MOUTH"
    CLOSED_EYE_SMILE = "CLOSED_EYE_SMILE"


class RenderRoute(StrEnum):
    STANDARD_AVATAR = "STANDARD_AVATAR"
    SIMPLIFIED_AVATAR = "SIMPLIFIED_AVATAR"
    CROP_SAFE_AVATAR = "CROP_SAFE_AVATAR"
    SILHOUETTE = "SILHOUETTE"
    BLUR_FALLBACK = "BLUR_FALLBACK"


class SizeTier(StrEnum):
    SMALL = "SMALL"
    MEDIUM = "MEDIUM"
    LARGE = "LARGE"


class CropStatus(StrEnum):
    FULL_IN_FRAME = "FULL_IN_FRAME"
    EDGE_CROPPED = "EDGE_CROPPED"


class PoseEstimate(BaseModel):
    yaw_deg: float
    pitch_deg: float
    roll_deg: float
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class DetectedHead(BaseModel):
    schema_version: Literal["1.0"] = SCHEMA_VERSION
    head_id: str = Field(min_length=1, max_length=96)
    bbox: BBox
    face_landmarks: dict[str, Point] = Field(default_factory=dict)
    body_landmarks: dict[str, Point] = Field(default_factory=dict)
    anchor_provenance: dict[str, str] = Field(default_factory=dict)
    pose: PoseEstimate
    view_hint: View | None = None
    size_tier: SizeTier | None = None
    visibility: float = Field(default=1.0, ge=0.0, le=1.0)
    crop_status: CropStatus = CropStatus.FULL_IN_FRAME
    out_of_frame_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    occlusion_score: float = Field(default=0.0, ge=0.0, le=1.0)
    accessory_flags: list[str] = Field(default_factory=list)
    depth_order: float = 0.0
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_geometry(self) -> "DetectedHead":
        x, y, width, height = self.bbox
        if x < 0 or y < 0 or width <= 0 or height <= 0:
            raise ValueError("bbox must be positive image-space [x,y,w,h]")
        if self.size_tier is None:
            edge = min(width, height)
            self.size_tier = SizeTier.SMALL if edge < 48 else SizeTier.LARGE if edge > 128 else SizeTier.MEDIUM
        return self

    @property
    def neck_center(self) -> Point:
        if "neck_center" in self.body_landmarks:
            return self.body_landmarks["neck_center"]
        x, y, width, height = self.bbox
        return (x + width / 2, y + height)


class OverlayDecision(BaseModel):
    schema_version: Literal["1.0"] = SCHEMA_VERSION
    head_id: str
    route_type: RenderRoute
    scene_id: Scene | None = None
    expression: Expression = Expression.NEUTRAL
    family_id: str | None = None
    family_version: str | None = None
    accessories: list[str] = Field(default_factory=list)
    neck_adapter: str | None = None
    fallback_reason: str | None = None
    decision_confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class RenderTransform(BaseModel):
    schema_version: Literal["1.0"] = SCHEMA_VERSION
    head_id: str
    matrix: list[list[float]]
    target_scale: float = Field(gt=0.0)
    rotation_deg: float
    translation: Point
    warp_mode: Literal["SCALE_TRANSLATE", "SIMILARITY", "AFFINE"]
    fit_points: dict[str, Point] = Field(default_factory=dict)
    residual_error_px: float = Field(default=0.0, ge=0.0)
    residual_normalized: float = Field(default=0.0, ge=0.0)
    residual_components: dict[str, float] = Field(default_factory=dict)


class DecisionTrace(BaseModel):
    schema_version: Literal["1.0"] = SCHEMA_VERSION
    image_id: str
    head_id: str
    scene_id: Scene | None
    route_type: RenderRoute
    family_id: str | None
    family_version: str | None
    selected_asset: str | None
    yaw_deg: float
    pitch_deg: float
    roll_deg: float
    pose_confidence: float
    visibility: float
    occlusion_score: float
    head_bbox: BBox
    face_landmarks: dict[str, Point]
    body_landmarks: dict[str, Point]
    transform_mode: str | None
    transform_scale: float | None
    transform_rotation_deg: float | None
    transform_translation: Point | None
    transform_residual: float | None
    transform_residual_components: dict[str, float] = Field(default_factory=dict)
    fallback_reason: str | None
    quality_gates: list[dict[str, Any]] = Field(default_factory=list)
    render_ms: int
    outside_mask_verified: bool


@dataclass
class RenderBatchResult:
    image: Image.Image
    alpha_mask: Image.Image
    transformed_overlay: Image.Image
    decisions: list[OverlayDecision]
    transforms: list[RenderTransform | None]
    traces: list[DecisionTrace]
