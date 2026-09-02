from __future__ import annotations

from datetime import datetime, timezone

import pytest
from PIL import Image

from app.config import AVATAR_ASSET_DIR
from app.pose_avatar.benchmark import (
    BenchmarkAnnotation,
    BenchmarkDocument,
    FailureReason,
    PlacementResult,
    ReviewStatus,
    RouteJudgment,
    benchmark_metrics,
)
from app.pose_avatar.calibration import calibration_proposal
from app.pose_avatar.models import DetectedHead, PoseEstimate, RenderRoute, Scene
from app.pose_avatar.registry import AssetRegistry
from app.pose_avatar.safety import (
    DISABLED_SAFETY_POLICY,
    P1_CANDIDATE_SAFETY_POLICY,
    expand_transform_about_neck,
)
from app.pose_avatar.transform import apply_matrix, solve_transform
from app.pose_avatar.transform import solve_face_primary_transform, solve_two_stage_head_neck
from app.pose_avatar.coverage import EllipseCoverageMaskProvider, LandmarkSilhouetteMaskProvider


def _head(*, roll=0.0, yaw=0.0):
    return DetectedHead(
        head_id="p1-head",
        bbox=(100, 80, 120, 145),
        body_landmarks={"neck_center": (160, 207.6)},
        pose=PoseEstimate(yaw_deg=yaw, pitch_deg=0, roll_deg=roll, confidence=.9),
        confidence=.9,
    )


def test_safety_expansion_is_identity_when_disabled_and_neck_fixed_when_enabled():
    registry = AssetRegistry(AVATAR_ASSET_DIR)
    record = registry.select(Scene.S01_FRONT_NEUTRAL)
    head = _head(roll=30)
    original = solve_transform(head, Scene.S01_FRONT_NEUTRAL, record.anchors)
    assert expand_transform_about_neck(original, head, Scene.S01_FRONT_NEUTRAL, DISABLED_SAFETY_POLICY) == original
    expanded = expand_transform_about_neck(original, head, Scene.S01_FRONT_NEUTRAL, P1_CANDIDATE_SAFETY_POLICY)
    source_neck = original.fit_points["src_neck_center"]
    assert expanded.target_scale > original.target_scale
    assert apply_matrix(expanded.matrix, source_neck) == pytest.approx(head.neck_center)


def test_safety_expansion_only_activates_for_roll_or_supported_scene_bonus():
    policy = P1_CANDIDATE_SAFETY_POLICY
    assert policy.multiplier(Scene.S01_FRONT_NEUTRAL, 12) == 1.0
    assert policy.multiplier(Scene.S01_FRONT_NEUTRAL, 30) > 1.0
    assert policy.multiplier(Scene.S04_L34_NEUTRAL, 0) == pytest.approx(1.015)
    assert policy.multiplier(Scene.S12_BACK, 0) == 1.0


def _case(**updates):
    values = dict(
        case_id="p1-001", local_image_ref="samples/inbox/local.jpg", source_sample_id="01",
        source_region_id="r1", head_bbox=(1, 2, 80, 100), head_size_px=80, detector_source="yunet",
    )
    values.update(updates)
    return BenchmarkAnnotation(**values)


def test_pending_case_cannot_smuggle_a_placement_result():
    with pytest.raises(ValueError, match="pending cases"):
        _case(placement_result=PlacementResult.PASS)


def test_reviewed_fail_requires_expected_route_and_failure_reason():
    with pytest.raises(ValueError, match="placement_result and expected_route"):
        _case(review_status=ReviewStatus.REVIEWED, placement_result=PlacementResult.FAIL)
    with pytest.raises(ValueError, match="failure reason"):
        _case(
            review_status=ReviewStatus.REVIEWED,
            placement_result=PlacementResult.FAIL,
            expected_route=RenderRoute.STANDARD_AVATAR,
        )


def test_metrics_exclude_pending_and_keep_denominators_explicit():
    observation = {
        "adapter_id": "candidate", "available": True, "elapsed_ms": 12,
        "actual_route": "STANDARD_AVATAR", "actual_scene": "S01_FRONT_NEUTRAL",
    }
    reviewed = _case(
        review_status=ReviewStatus.REVIEWED,
        placement_result=PlacementResult.PASS,
        expected_route=RenderRoute.STANDARD_AVATAR,
        expected_scene=Scene.S01_FRONT_NEUTRAL,
        observations={"candidate": observation},
    )
    pending = _case(case_id="p1-002")
    document = BenchmarkDocument(
        benchmark_id="test", created_at=datetime.now(timezone.utc).isoformat(), sampling_seed="fixed",
        cases=[reviewed, pending],
    )
    metrics = benchmark_metrics(document, "candidate")
    assert metrics["reviewed_cases"] == 1
    assert metrics["pending_cases"] == 1
    assert metrics["placement_pass_rate"] == 1.0
    assert metrics["scene_accuracy"] == 1.0


def test_reviewed_failure_taxonomy_is_accepted():
    case = _case(
        review_status=ReviewStatus.REVIEWED,
        placement_result=PlacementResult.FAIL,
        expected_route=RenderRoute.STANDARD_AVATAR,
        failure_reasons=[FailureReason.EXPOSED_ORIGINAL_HEAD],
    )
    assert case.failure_reasons == [FailureReason.EXPOSED_ORIGINAL_HEAD]


def test_calibration_refuses_to_tune_on_an_undersized_review_set():
    document = BenchmarkDocument(
        benchmark_id="test", created_at=datetime.now(timezone.utc).isoformat(), sampling_seed="fixed",
        cases=[_case()],
    )
    proposal = calibration_proposal(document)
    assert proposal["status"] == "insufficient_reviewed_cases"
    assert proposal["baseline_preserved"] is True
    assert proposal["proposal"] is None


def test_calibration_keeps_a_deterministic_held_out_split():
    cases = []
    for index in range(1, 33):
        cases.append(_case(
            case_id=f"p1-{index:03d}", review_status=ReviewStatus.REVIEWED,
            source_sample_id=f"{index:02d}",
            placement_result=PlacementResult.PASS, route_judgment=RouteJudgment.STANDARD_ELIGIBLE,
            expected_scene=Scene.S01_FRONT_NEUTRAL,
            observations={"yunet_5pt_heuristic": {"adapter_id": "yunet_5pt_heuristic", "available": True,
                "elapsed_ms": 1, "pose_confidence": .9, "yaw_deg": 0, "pitch_deg": 0, "roll_deg": 0,
                "actual_route": RenderRoute.STANDARD_AVATAR, "actual_scene": Scene.S01_FRONT_NEUTRAL}},
        ))
    document = BenchmarkDocument(benchmark_id="test", created_at=datetime.now(timezone.utc).isoformat(),
                                 sampling_seed="fixed", cases=cases)
    proposal = calibration_proposal(document)
    assert proposal["status"] == "proposal_ready_not_applied"
    assert proposal["calibration"]["cases"] == 22
    assert proposal["held_out_validation"]["proposal"]["cases"] == 10
    assert set(proposal["split"]["calibration_source_samples"]).isdisjoint(proposal["split"]["held_out_source_samples"])


def test_p15_review_requires_primary_failure_and_accepts_not_applicable():
    with pytest.raises(ValueError, match="primary_failure_reason"):
        _case(review_status=ReviewStatus.REVIEWED, placement_result=PlacementResult.FAIL,
              route_judgment=RouteJudgment.SHOULD_FALLBACK,
              failure_reasons=[FailureReason.WRONG_POSITION], primary_failure_reason=FailureReason.OTHER)
    case = _case(review_status=ReviewStatus.REVIEWED, placement_result=PlacementResult.NOT_APPLICABLE,
                 route_judgment=RouteJudgment.SHOULD_FALLBACK)
    assert case.placement_result == PlacementResult.NOT_APPLICABLE


def test_residual_components_and_experimental_fits_are_explainable():
    registry = AssetRegistry(AVATAR_ASSET_DIR)
    record = registry.select(Scene.S01_FRONT_NEUTRAL)
    head = _head()
    head.face_landmarks = {"left_eye": (135, 125), "right_eye": (185, 125), "nose": (160, 150), "chin": (160, 186)}
    transform = solve_transform(head, Scene.S01_FRONT_NEUTRAL, record.anchors)
    assert "left_eye_error_px" in transform.residual_components
    assert "final_normalized_residual" in transform.residual_components
    assert solve_face_primary_transform(head, record.anchors).warp_mode == "SIMILARITY"
    head_fit, neck_fit = solve_two_stage_head_neck(head, record.anchors)
    assert head_fit.warp_mode == "SIMILARITY"
    assert neck_fit is None


def test_local_coverage_providers_return_explicit_masks():
    head = _head()
    overlay = Image.new("L", (320, 320), 0)
    for provider in (EllipseCoverageMaskProvider(), LandmarkSilhouetteMaskProvider()):
        result = provider.evaluate((320, 320), head, overlay)
        assert result.original_head_mask.size == (320, 320)
        assert result.avatar_coverage_mask.size == (320, 320)
        assert result.exposed_head_ratio == 1.0
