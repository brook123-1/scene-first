from __future__ import annotations

from dataclasses import asdict, replace
from itertools import product
import hashlib

from .benchmark import BenchmarkDocument, ReviewStatus, RouteJudgment
from .models import RenderRoute, Scene
from .routing import DEFAULT_THRESHOLDS, RoutingThresholds


def _predict(case, observation, thresholds: RoutingThresholds) -> tuple[RenderRoute, Scene | None]:
    if case.head_size_px < thresholds.tiny_head_px:
        return RenderRoute.SIMPLIFIED_AVATAR, None
    if not observation.available or (observation.pose_confidence or 0) < thresholds.min_pose_confidence:
        return RenderRoute.BLUR_FALLBACK, None
    yaw, pitch, roll = observation.yaw_deg or 0, observation.pitch_deg or 0, observation.roll_deg or 0
    if abs(roll) > thresholds.max_roll_deg:
        return RenderRoute.SILHOUETTE, None
    if abs(pitch) > thresholds.neutral_pitch_deg or abs(yaw) > thresholds.three_quarter_yaw_deg:
        return RenderRoute.BLUR_FALLBACK, None
    if abs(yaw) <= thresholds.front_yaw_deg:
        return RenderRoute.STANDARD_AVATAR, Scene.S01_FRONT_NEUTRAL
    return RenderRoute.STANDARD_AVATAR, Scene.S04_L34_NEUTRAL if yaw < 0 else Scene.S07_R34_NEUTRAL


def calibration_proposal(
    document: BenchmarkDocument,
    adapter_id: str = "yunet_5pt_heuristic",
    *,
    minimum_reviewed: int = 30,
) -> dict:
    eligible = [
        case for case in document.cases
        if case.review_status == ReviewStatus.REVIEWED
        and case.route_judgment not in {None, RouteJudgment.UNSURE}
        and adapter_id in case.observations
    ]
    baseline = asdict(DEFAULT_THRESHOLDS)
    if len(eligible) < minimum_reviewed:
        return {
            "status": "insufficient_reviewed_cases",
            "adapter_id": adapter_id,
            "reviewed_cases": len(eligible),
            "minimum_reviewed": minimum_reviewed,
            "baseline_preserved": True,
            "baseline_thresholds": baseline,
            "proposal": None,
        }
    split_seed = "pose-avatar-calibration-source-group-v1"
    samples = sorted({case.source_sample_id for case in eligible},
                     key=lambda value: hashlib.sha256(f"{split_seed}|{value}".encode()).hexdigest())
    calibration_count = max(1, min(len(samples) - 1, round(len(samples) * .70))) if len(samples) > 1 else len(samples)
    calibration_samples = set(samples[:calibration_count])
    calibration = [case for case in eligible if case.source_sample_id in calibration_samples]
    held_out = [case for case in eligible if case.source_sample_id not in calibration_samples]
    candidates = product(
        (12.0, 15.0, 18.0), (45.0, 50.0, 55.0), (12.0, 15.0, 18.0),
        (30.0, 35.0, 40.0), (0.55, 0.60, 0.65),
    )
    best = None
    for front, three_quarter, pitch, roll, pose_confidence in candidates:
        thresholds = replace(
            DEFAULT_THRESHOLDS,
            front_yaw_deg=front,
            three_quarter_yaw_deg=three_quarter,
            neutral_pitch_deg=pitch,
            max_roll_deg=roll,
            min_pose_confidence=pose_confidence,
        )
        route_correct = scene_correct = scene_total = unsafe_standard = 0
        for case in calibration:
            route, scene = _predict(case, case.observations[adapter_id], thresholds)
            predicted = RouteJudgment.STANDARD_ELIGIBLE if route == RenderRoute.STANDARD_AVATAR else RouteJudgment.SIMPLIFIED_ELIGIBLE if route == RenderRoute.SIMPLIFIED_AVATAR else RouteJudgment.SHOULD_FALLBACK
            route_correct += predicted == case.route_judgment
            if case.expected_scene is not None:
                scene_total += 1
                scene_correct += scene == case.expected_scene
            unsafe_standard += route == RenderRoute.STANDARD_AVATAR and case.route_judgment != RouteJudgment.STANDARD_ELIGIBLE
        route_accuracy = route_correct / len(calibration)
        scene_accuracy = scene_correct / scene_total if scene_total else 0.0
        unsafe_rate = unsafe_standard / len(calibration)
        drift = sum(abs(getattr(thresholds, key) - getattr(DEFAULT_THRESHOLDS, key)) for key in (
            "front_yaw_deg", "three_quarter_yaw_deg", "neutral_pitch_deg", "max_roll_deg",
        )) / 100 + abs(thresholds.min_pose_confidence - DEFAULT_THRESHOLDS.min_pose_confidence)
        score = route_accuracy * 2 + scene_accuracy - unsafe_rate * 3 - drift * 0.02
        candidate = (score, -drift, thresholds, route_accuracy, scene_accuracy, unsafe_rate)
        if best is None or candidate[:2] > best[:2]:
            best = candidate
    assert best is not None
    _, _, thresholds, route_accuracy, scene_accuracy, unsafe_rate = best
    def evaluate(cases, candidate_thresholds):
        route_correct = scene_correct = scene_total = unsafe_standard = 0
        for case in cases:
            route, scene = _predict(case, case.observations[adapter_id], candidate_thresholds)
            predicted = RouteJudgment.STANDARD_ELIGIBLE if route == RenderRoute.STANDARD_AVATAR else RouteJudgment.SIMPLIFIED_ELIGIBLE if route == RenderRoute.SIMPLIFIED_AVATAR else RouteJudgment.SHOULD_FALLBACK
            route_correct += predicted == case.route_judgment
            if case.expected_scene is not None:
                scene_total += 1
                scene_correct += scene == case.expected_scene
            unsafe_standard += route == RenderRoute.STANDARD_AVATAR and case.route_judgment != RouteJudgment.STANDARD_ELIGIBLE
        return {"cases": len(cases), "route_accuracy": round(route_correct / len(cases), 4) if cases else None,
                "scene_accuracy": round(scene_correct / scene_total, 4) if scene_total else None,
                "unsafe_standard_rate": round(unsafe_standard / len(cases), 4) if cases else None}
    return {
        "status": "proposal_ready_not_applied",
        "adapter_id": adapter_id,
        "reviewed_cases": len(eligible),
        "baseline_preserved": True,
        "baseline_thresholds": baseline,
        "proposal": asdict(thresholds),
        "split": {"seed": split_seed, "group_key": "source_sample_id",
                  "calibration_source_samples": sorted(calibration_samples),
                  "held_out_source_samples": sorted(set(samples) - calibration_samples)},
        "calibration": evaluate(calibration, thresholds),
        "held_out_validation": {"baseline": evaluate(held_out, DEFAULT_THRESHOLDS), "proposal": evaluate(held_out, thresholds)},
        "warning": "Proposal requires a held-out review before any production adoption.",
    }
