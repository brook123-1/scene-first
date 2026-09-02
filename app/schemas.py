from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class Region(BaseModel):
    id: str
    box: list[int] = Field(min_length=4, max_length=4)
    head_box: list[int] | None = Field(default=None, min_length=4, max_length=4)
    confidence: float = 1.0
    source: str = "manual"
    selected: bool = True
    mode: Literal["anime", "safe"] | None = None
    face_landmarks: dict[str, list[float]] = Field(default_factory=dict)
    body_landmarks: dict[str, list[float]] = Field(default_factory=dict)
    anchor_provenance: dict[str, str] = Field(default_factory=dict)
    pose: dict[str, float] | None = None
    view_hint: Literal["FRONT", "LEFT_34", "RIGHT_34", "LEFT_PROFILE", "RIGHT_PROFILE", "BACK"] | None = None
    visibility: float = Field(default=1.0, ge=0.0, le=1.0)
    crop_status: Literal["FULL_IN_FRAME", "EDGE_CROPPED"] = "FULL_IN_FRAME"
    out_of_frame_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    occlusion_score: float = Field(default=0.0, ge=0.0, le=1.0)
    accessory_flags: list[str] = Field(default_factory=list)
    depth_order: float | None = None


class EditRequest(BaseModel):
    image_id: str
    regions: list[Region]
    mode: Literal["anime", "safe"] = "anime"
    provider: Literal["local", "openai", "gemini", "fal", "ark", "bfl", "qwen"] = "local"
    cloud_scope: Literal["crop", "full"] = "crop"
    retry_nonce: int = 0
    prompt_profile: Literal["default", "balanced_portrait", "balanced_painterly"] = "balanced_portrait"
    mask_profile: Literal["standard", "neck_blend"] = "standard"
    safe_cover_id: Literal[
        "bald-bearded", "graduate", "architect", "office", "programmer", "blue-collar"
    ] = "bald-bearded"
    # Set only after the user has reviewed the automatic candidate set.  A
    # reviewed target must not be silently downgraded merely because the
    # detector that originally proposed it was low-confidence.
    selection_confirmed: bool = False
    # Both this request opt-in and the server feature flag are required.  The
    # existing production path remains the default when either is false.
    pose_aware_overlay: bool = False


class ExportRequest(BaseModel):
    job_id: str
    format: Literal["png", "jpeg"] = "png"
    quality: int = Field(default=94, ge=80, le=100)


class ReviewRating(BaseModel):
    item_id: str
    publishable: bool
    naturalness: int = Field(ge=1, le=5)
    privacy: int = Field(ge=1, le=5)
    notes: str = ""


class PreflightSave(BaseModel):
    sample_id: str
    regions: list[Region]


class ProviderConfigUpdate(BaseModel):
    provider: Literal["openai", "gemini", "fal", "ark", "bfl", "qwen"]
    api_key: str | None = Field(default=None, min_length=1)
    model: str | None = Field(default=None, min_length=1, max_length=120)


class PublishabilityUpdate(BaseModel):
    publishable: bool
