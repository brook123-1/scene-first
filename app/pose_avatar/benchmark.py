from __future__ import annotations

from collections import Counter
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from .models import CropStatus, RenderRoute, Scene, View


BENCHMARK_SCHEMA_VERSION = "1.0"


class ReviewStatus(StrEnum):
    PENDING = "PENDING"
    REVIEWED = "REVIEWED"


class PlacementResult(StrEnum):
    PASS = "PASS"
    BORDERLINE = "BORDERLINE"
    FAIL = "FAIL"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class RouteJudgment(StrEnum):
    STANDARD_ELIGIBLE = "STANDARD_ELIGIBLE"
    SIMPLIFIED_ELIGIBLE = "SIMPLIFIED_ELIGIBLE"
    SHOULD_FALLBACK = "SHOULD_FALLBACK"
    UNSURE = "UNSURE"


class PitchClass(StrEnum):
    UP = "UP"
    NEUTRAL = "NEUTRAL"
    DOWN = "DOWN"
    UNKNOWN = "UNKNOWN"


class RollClass(StrEnum):
    LEVEL = "LEVEL"
    MODERATE = "MODERATE"
    LARGE = "LARGE"
    UNKNOWN = "UNKNOWN"


class VisibilityClass(StrEnum):
    CLEAR = "CLEAR"
    PARTIAL = "PARTIAL"
    POOR = "POOR"
    UNKNOWN = "UNKNOWN"


class OcclusionType(StrEnum):
    NONE = "NONE"
    HAIR = "HAIR"
    HAND = "HAND"
    OBJECT = "OBJECT"
    OTHER_PERSON = "OTHER_PERSON"
    FRAME_EDGE = "FRAME_EDGE"
    UNKNOWN = "UNKNOWN"


class OcclusionLevel(StrEnum):
    LOW = "LOW"
    HIGH = "HIGH"
    UNKNOWN = "UNKNOWN"


class FailureReason(StrEnum):
    EXPOSED_ORIGINAL_HEAD = "EXPOSED_ORIGINAL_HEAD"
    WRONG_SCENE = "WRONG_SCENE"
    WRONG_ROUTE = "WRONG_ROUTE"
    SCALE_TOO_SMALL = "SCALE_TOO_SMALL"
    SCALE_TOO_LARGE = "SCALE_TOO_LARGE"
    ROLL_MISMATCH = "ROLL_MISMATCH"
    NECK_FLOATING = "NECK_FLOATING"
    NECK_OVERLAP = "NECK_OVERLAP"
    OCCLUSION_ORDER = "OCCLUSION_ORDER"
    POSE_UNSTABLE = "POSE_UNSTABLE"
    UNSUPPORTED_ASSET = "UNSUPPORTED_ASSET"
    OTHER = "OTHER"
    WRONG_SCALE = "WRONG_SCALE"
    WRONG_ROLL = "WRONG_ROLL"
    WRONG_POSITION = "WRONG_POSITION"
    FLOATING_NECK = "FLOATING_NECK"
    WRONG_OCCLUSION = "WRONG_OCCLUSION"
    BAD_ASSET_GEOMETRY = "BAD_ASSET_GEOMETRY"
    SHOULD_HAVE_FALLBACK = "SHOULD_HAVE_FALLBACK"
    FALSE_FALLBACK = "FALSE_FALLBACK"
    UNSUPPORTED_SCENE = "UNSUPPORTED_SCENE"


class AdapterObservation(BaseModel):
    adapter_id: str
    available: bool
    elapsed_ms: int = Field(ge=0)
    adapter_elapsed_ms: int = Field(default=0, ge=0)
    render_elapsed_ms: int = Field(default=0, ge=0)
    yaw_deg: float | None = None
    pitch_deg: float | None = None
    roll_deg: float | None = None
    pose_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    body_landmarks: dict[str, tuple[float, float]] = Field(default_factory=dict)
    actual_scene: Scene | None = None
    actual_route: RenderRoute | None = None
    fallback_reason: str | None = None
    selected_asset: str | None = None
    safety_multiplier: float = Field(default=1.0, ge=1.0)
    proxies: dict[str, float] = Field(default_factory=dict)
    residual_components: dict[str, float] = Field(default_factory=dict)
    error: str | None = None


class BenchmarkAnnotation(BaseModel):
    schema_version: Literal["1.0"] = BENCHMARK_SCHEMA_VERSION
    case_id: str = Field(pattern=r"^p1-[0-9]{3}$")
    # This is deliberately present only in the gitignored local annotation
    # file.  sanitize_for_commit removes it from aggregate exports.
    local_image_ref: str = Field(min_length=1)
    source_sample_id: str = Field(min_length=1)
    source_region_id: str = Field(min_length=1)
    head_bbox: tuple[float, float, float, float]
    head_size_px: float = Field(gt=0)
    detector_source: str
    view: View | None = None
    pitch: PitchClass = PitchClass.UNKNOWN
    roll: RollClass = RollClass.UNKNOWN
    visibility: VisibilityClass = VisibilityClass.UNKNOWN
    crop_status: CropStatus = CropStatus.FULL_IN_FRAME
    occlusion_type: list[OcclusionType] = Field(default_factory=lambda: [OcclusionType.UNKNOWN])
    occlusion_level: OcclusionLevel = OcclusionLevel.UNKNOWN
    neck_visible: bool | None = None
    shoulder_visible: bool | None = None
    expected_route: RenderRoute | None = None
    expected_scene: Scene | None = None
    route_judgment: RouteJudgment | None = None
    review_status: ReviewStatus = ReviewStatus.PENDING
    placement_result: PlacementResult | None = None
    failure_reasons: list[FailureReason] = Field(default_factory=list)
    primary_failure_reason: FailureReason | None = None
    reviewer_note: str | None = Field(default=None, max_length=500)
    observations: dict[str, AdapterObservation] = Field(default_factory=dict)

    @model_validator(mode="after")
    def reviewed_fields_are_complete(self) -> "BenchmarkAnnotation":
        x, y, width, height = self.head_bbox
        if x < 0 or y < 0 or width <= 0 or height <= 0:
            raise ValueError("head_bbox must be positive [x,y,w,h]")
        if abs(self.head_size_px - min(width, height)) > 0.51:
            raise ValueError("head_size_px must equal the bbox short edge")
        if self.review_status == ReviewStatus.REVIEWED:
            if self.route_judgment is None and self.expected_route is not None:
                self.route_judgment = (
                    RouteJudgment.STANDARD_ELIGIBLE if self.expected_route == RenderRoute.STANDARD_AVATAR
                    else RouteJudgment.SIMPLIFIED_ELIGIBLE if self.expected_route == RenderRoute.SIMPLIFIED_AVATAR
                    else RouteJudgment.SHOULD_FALLBACK
                )
            if self.placement_result is None or self.route_judgment is None:
                raise ValueError("reviewed cases require placement_result and expected_route or route_judgment")
            if self.placement_result == PlacementResult.FAIL and not self.failure_reasons:
                raise ValueError("failed reviewed cases require at least one failure reason")
            if self.failure_reasons and self.primary_failure_reason is None:
                self.primary_failure_reason = self.failure_reasons[0]
            if self.failure_reasons and self.primary_failure_reason not in self.failure_reasons:
                raise ValueError("primary_failure_reason must be one of failure_reasons")
        elif self.placement_result is not None:
            raise ValueError("pending cases cannot carry placement_result")
        return self


class BenchmarkDocument(BaseModel):
    schema_version: Literal["1.0"] = BENCHMARK_SCHEMA_VERSION
    benchmark_id: str
    created_at: str
    sampling_seed: str
    cases: list[BenchmarkAnnotation] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_cases(self) -> "BenchmarkDocument":
        ids = [case.case_id for case in self.cases]
        if len(ids) != len(set(ids)):
            raise ValueError("case_id values must be unique")
        return self


def benchmark_metrics(document: BenchmarkDocument, adapter_id: str) -> dict:
    reviewed = [case for case in document.cases if case.review_status == ReviewStatus.REVIEWED]
    observed = [case for case in reviewed if adapter_id in case.observations]
    applicable = [case for case in observed if case.placement_result != PlacementResult.NOT_APPLICABLE]
    counts = Counter(case.placement_result.value for case in observed if case.placement_result)
    standard = [case for case in applicable if case.route_judgment == RouteJudgment.STANDARD_ELIGIBLE]
    scene_cases = [case for case in observed if case.expected_scene is not None]
    def route_class(route: RenderRoute | None) -> RouteJudgment:
        if route == RenderRoute.STANDARD_AVATAR:
            return RouteJudgment.STANDARD_ELIGIBLE
        if route == RenderRoute.SIMPLIFIED_AVATAR:
            return RouteJudgment.SIMPLIFIED_ELIGIBLE
        return RouteJudgment.SHOULD_FALLBACK

    route_eligible = [case for case in observed if case.route_judgment != RouteJudgment.UNSURE]
    route_correct = [case for case in route_eligible if route_class(case.observations[adapter_id].actual_route) == case.route_judgment]
    false_standard = [case for case in route_eligible if case.observations[adapter_id].actual_route == RenderRoute.STANDARD_AVATAR and case.route_judgment != RouteJudgment.STANDARD_ELIGIBLE]
    false_fallback = [case for case in route_eligible if route_class(case.observations[adapter_id].actual_route) == RouteJudgment.SHOULD_FALLBACK and case.route_judgment in {RouteJudgment.STANDARD_ELIGIBLE, RouteJudgment.SIMPLIFIED_ELIGIBLE}]

    def rate(numerator: int, denominator: int) -> float | None:
        return round(numerator / denominator, 4) if denominator else None

    def failures(reason: FailureReason) -> int:
        return sum(reason in case.failure_reasons for case in observed)

    return {
        "adapter_id": adapter_id,
        "total_cases": len(document.cases),
        "reviewed_cases": len(reviewed),
        "observed_reviewed_cases": len(observed),
        "pending_cases": len(document.cases) - len(reviewed),
        "placement_counts": dict(counts),
        "placement_pass_rate": rate(counts[PlacementResult.PASS], len(applicable)),
        "standard_eligible_rate": rate(sum(case.placement_result == PlacementResult.PASS for case in standard), len(standard)),
        "scene_accuracy": rate(sum(case.observations[adapter_id].actual_scene == case.expected_scene for case in scene_cases), len(scene_cases)),
        "route_accuracy": rate(len(route_correct), len(route_eligible)),
        "false_standard_rate": rate(len(false_standard), len(route_eligible)),
        "false_fallback_rate": rate(len(false_fallback), len(route_eligible)),
        "failure_taxonomy": dict(Counter(reason.value for case in observed for reason in case.failure_reasons)),
        "exposed_original_head_rate": rate(failures(FailureReason.EXPOSED_ORIGINAL_HEAD), len(applicable)),
        "neck_failure_rate": rate(failures(FailureReason.NECK_FLOATING) + failures(FailureReason.FLOATING_NECK) + failures(FailureReason.NECK_OVERLAP), len(applicable)),
        "scale_failure_rate": rate(failures(FailureReason.SCALE_TOO_SMALL) + failures(FailureReason.SCALE_TOO_LARGE) + failures(FailureReason.WRONG_SCALE), len(applicable)),
        "roll_failure_rate": rate(failures(FailureReason.ROLL_MISMATCH) + failures(FailureReason.WRONG_ROLL), len(applicable)),
    }


def anonymized_distribution(document: BenchmarkDocument) -> dict:
    return {
        "cases": len(document.cases),
        "samples": len({case.source_sample_id for case in document.cases}),
        "size_buckets": dict(Counter(
            "tiny" if case.head_size_px < 48 else "small" if case.head_size_px < 96 else "medium" if case.head_size_px < 192 else "large"
            for case in document.cases
        )),
        "detector_sources": dict(Counter(case.detector_source for case in document.cases)),
        "review_status": dict(Counter(case.review_status.value for case in document.cases)),
        "views": dict(Counter((case.view.value if case.view else "UNLABELED") for case in document.cases)),
    }
