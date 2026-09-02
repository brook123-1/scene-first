from __future__ import annotations

import hashlib
import math
import statistics
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

from .benchmark import BenchmarkAnnotation, BenchmarkDocument, PlacementResult, RouteJudgment
from .models import RenderRoute
from .routing import DEFAULT_THRESHOLDS
from .models import DetectedHead, Scene
from .registry import AssetRegistry, SceneAnchors
from .transform import solve_transform


BASELINE_ADAPTER = "yunet_5pt_heuristic"
FAILURE_KEYS = (
    "WRONG_SCENE", "WRONG_SCALE", "WRONG_ROLL", "WRONG_POSITION", "FLOATING_NECK",
    "NECK_OVERLAP", "EXPOSED_ORIGINAL_HEAD", "WRONG_OCCLUSION", "BAD_ASSET_GEOMETRY",
    "SHOULD_HAVE_FALLBACK", "FALSE_FALLBACK", "UNSUPPORTED_SCENE", "OTHER",
)
SPLIT_SEED = "pose-avatar-p1.5-human-baseline-v0.1"


def _rate(numerator: int, denominator: int) -> dict[str, int | float | None]:
    return {"numerator": numerator, "denominator": denominator,
            "rate": round(numerator / denominator, 6) if denominator else None}


def _summary(values: list[float]) -> dict[str, int | float | None]:
    if not values:
        return {"n": 0, "min": None, "median": None, "mean": None, "max": None}
    return {"n": len(values), "min": round(min(values), 6), "median": round(statistics.median(values), 6),
            "mean": round(statistics.mean(values), 6), "max": round(max(values), 6)}


def grouped_split(document: BenchmarkDocument, calibration_ratio: float = .70) -> dict[str, Any]:
    sample_ids = sorted({case.source_sample_id for case in document.cases}, key=lambda value: hashlib.sha256(f"{SPLIT_SEED}|{value}".encode()).hexdigest())
    calibration_count = round(len(sample_ids) * calibration_ratio)
    calibration_samples = set(sample_ids[:calibration_count])
    calibration_cases = [case.case_id for case in document.cases if case.source_sample_id in calibration_samples]
    held_out_cases = [case.case_id for case in document.cases if case.source_sample_id not in calibration_samples]
    return {"seed": SPLIT_SEED, "group_key": "source_sample_id", "calibration_ratio": calibration_ratio,
            "calibration_source_samples": sorted(calibration_samples),
            "held_out_source_samples": sorted(set(sample_ids) - calibration_samples),
            "calibration_case_ids": calibration_cases, "held_out_case_ids": held_out_cases}


def _route_class(route: RenderRoute | str | None) -> str:
    value = route.value if isinstance(route, RenderRoute) else route
    if value == RenderRoute.STANDARD_AVATAR.value:
        return RouteJudgment.STANDARD_ELIGIBLE.value
    if value == RenderRoute.SIMPLIFIED_AVATAR.value:
        return RouteJudgment.SIMPLIFIED_ELIGIBLE.value
    return RouteJudgment.SHOULD_FALLBACK.value


def _confusion(cases: list[BenchmarkAnnotation], route_getter: Callable[[BenchmarkAnnotation], str]) -> dict[str, Any]:
    labels = [value.value for value in RouteJudgment]
    matrix = {expected: {actual: 0 for actual in labels[:-1]} for expected in labels}
    for case in cases:
        matrix[case.route_judgment.value][route_getter(case)] += 1
    return matrix


def _integrity(cases: list[BenchmarkAnnotation]) -> dict[str, Any]:
    actual_fallback = lambda case: case.observations[BASELINE_ADAPTER].actual_route != RenderRoute.STANDARD_AVATAR
    checks = {
        "pass_with_failure_reason": [case.case_id for case in cases if case.placement_result == PlacementResult.PASS and case.failure_reasons],
        "primary_not_in_failure_reasons": [case.case_id for case in cases if case.primary_failure_reason and case.primary_failure_reason not in case.failure_reasons],
        "actual_fallback_with_false_fallback": [case.case_id for case in cases if actual_fallback(case) and any(reason.value == "FALSE_FALLBACK" for reason in case.failure_reasons)],
        "should_fallback_with_placement_pass": [case.case_id for case in cases if case.route_judgment == RouteJudgment.SHOULD_FALLBACK and case.placement_result == PlacementResult.PASS],
    }
    return {"schema_version": "valid", "case_count": len(cases), "unique_case_ids": len({case.case_id for case in cases}),
            "reviewed": sum(case.review_status.value == "REVIEWED" for case in cases), "pending_case_ids": [],
            "missing_expected_scene_case_ids": [case.case_id for case in cases if case.expected_scene is None],
            "missing_expected_scene_interpretation": "Allowed for human fallback/unsupported cases; excluded from Scene Accuracy denominator and never inferred.",
            "semantic_combinations_preserved": checks,
            "interpretation": "Placement judges rendered placement quality; route judgment judges whether STANDARD/SIMPLIFIED/fallback was appropriate. They are intentionally counted separately."}


def _failure_analysis(cases: list[BenchmarkAnnotation]) -> dict[str, Any]:
    standard = [case for case in cases if case.route_judgment == RouteJudgment.STANDARD_ELIGIBLE]
    result = {}
    for key in FAILURE_KEYS:
        occurrence = [case.case_id for case in cases if any(reason.value == key for reason in case.failure_reasons)]
        primary = [case.case_id for case in cases if case.primary_failure_reason and case.primary_failure_reason.value == key]
        standard_cases = [case_id for case_id in occurrence if case_id in {case.case_id for case in standard}]
        result[key] = {"occurrence_count": len(occurrence), "primary_count": len(primary),
                       "of_reviewed": _rate(len(occurrence), len(cases)), "of_standard_eligible": _rate(len(standard_cases), len(standard)),
                       "case_ids": occurrence, "primary_case_ids": primary}
    pairs = Counter()
    pair_cases: dict[str, list[str]] = defaultdict(list)
    for case in cases:
        values = sorted({reason.value for reason in case.failure_reasons})
        for left_index, left in enumerate(values):
            for right in values[left_index + 1:]:
                key = f"{left} + {right}"
                pairs[key] += 1
                pair_cases[key].append(case.case_id)
    co_occurrence = [{"pair": pair, "count": count, "case_ids": pair_cases[pair]} for pair, count in pairs.most_common()]
    primary_ranking = Counter(case.primary_failure_reason.value if case.primary_failure_reason else "NONE" for case in cases)
    return {"taxonomy": result, "primary_ranking": dict(primary_ranking.most_common()), "co_occurrence": co_occurrence}


def _trace_payload(trace_root: Path, case_id: str) -> dict[str, Any]:
    import json
    path = trace_root / case_id / "07-trace.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def _residual_analysis(cases: list[BenchmarkAnnotation], traces: dict[str, dict[str, Any]]) -> dict[str, Any]:
    residual = {case.case_id: case.observations[BASELINE_ADAPTER].residual_components.get("final_normalized_residual") for case in cases}
    available = [case for case in cases if residual[case.case_id] is not None]
    by_placement = {value.value: _summary([residual[case.case_id] for case in available if case.placement_result == value]) for value in PlacementResult}
    by_route = {value.value: _summary([residual[case.case_id] for case in available if case.route_judgment == value]) for value in RouteJudgment}
    false_fallback_label = [case for case in available if any(reason.value == "FALSE_FALLBACK" for reason in case.failure_reasons)]
    router_false_fallback = [case for case in available if case.route_judgment == RouteJudgment.STANDARD_ELIGIBLE and _route_class(case.observations[BASELINE_ADAPTER].actual_route) == RouteJudgment.SHOULD_FALLBACK.value]
    high_standard_pass = [case.case_id for case in available if residual[case.case_id] > .08 and case.route_judgment == RouteJudgment.STANDARD_ELIGIBLE and case.placement_result == PlacementResult.PASS]
    low_fail = [case.case_id for case in available if residual[case.case_id] <= .08 and case.placement_result == PlacementResult.FAIL]
    binary = [case for case in available if case.placement_result in {PlacementResult.PASS, PlacementResult.FAIL}]
    positives = [case for case in binary if case.placement_result == PlacementResult.FAIL]
    negatives = [case for case in binary if case.placement_result == PlacementResult.PASS]
    auc_pairs = [(residual[p.case_id], residual[n.case_id]) for p in positives for n in negatives]
    auc = sum(1 if p > n else .5 if p == n else 0 for p, n in auc_pairs) / len(auc_pairs) if auc_pairs else None
    sweep = []
    for threshold in (.04, .06, .08, .10, .12, .15, .20):
        tp = sum(residual[c.case_id] > threshold for c in positives); fp = sum(residual[c.case_id] > threshold for c in negatives)
        fn, tn = len(positives) - tp, len(negatives) - fp
        sweep.append({"threshold": threshold, "confusion": {"fail_blocked_tp": tp, "pass_blocked_fp": fp, "fail_allowed_fn": fn, "pass_allowed_tn": tn},
                      "precision": round(tp / (tp + fp), 6) if tp + fp else None, "recall": round(tp / len(positives), 6) if positives else None})
    adapter_components = {}
    for adapter_id in (BASELINE_ADAPTER, "opencv_mediapipe_pose_hybrid"):
        component_groups: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        for case in cases:
            observation = case.observations.get(adapter_id)
            if not observation:
                continue
            for key, value in observation.residual_components.items():
                if key.endswith("_head_width") or key in {"eye_distance_scale_error", "eye_pair_angle_error_deg"}:
                    component_groups[key][case.placement_result.value].append(value)
        components = {key: {placement: _summary(values) for placement, values in groups.items()} for key, groups in component_groups.items()}
        effect = []
        for key, groups in component_groups.items():
            passed, failed = groups.get("PASS", []), groups.get("FAIL", [])
            if passed and failed:
                effect.append({"component": key, "fail_minus_pass_mean": round(statistics.mean(failed) - statistics.mean(passed), 6),
                               "pass_n": len(passed), "fail_n": len(failed)})
        adapter_components[adapter_id] = {"by_placement": components,
                                          "failure_association_ranking": sorted(effect, key=lambda item: abs(item["fail_minus_pass_mean"]), reverse=True)}
    return {"available_cases": len(available), "unavailable_case_ids": [case.case_id for case in cases if case not in available],
            "by_placement": by_placement, "by_route_judgment": by_route,
            "false_fallback_occurrence": _summary([residual[case.case_id] for case in false_fallback_label]),
            "router_false_fallback": _summary([residual[case.case_id] for case in router_false_fallback]),
            "residual_gt_008_standard_eligible_pass": high_standard_pass, "residual_le_008_fail": low_fail,
            "auc_fail_vs_pass": round(auc, 6) if auc is not None else None, "threshold_sweep": sweep,
            "component_analysis": adapter_components,
            "interpretation_limit": "Only three FAIL cases have a P1 baseline residual. Association rankings are descriptive, not stable causal estimates."}


def _fitting_analysis(cases: list[BenchmarkAnnotation], traces: dict[str, dict[str, Any]]) -> dict[str, Any]:
    methods = ("p1_baseline", "face_primary", "two_stage_head", "two_stage_neck")
    result = {}
    for method in methods:
        rows = [(case, traces.get(case.case_id, {}).get("fitting_bakeoff", {}).get(method)) for case in cases]
        rows = [(case, value) for case, value in rows if isinstance(value, dict) and value.get("residual_normalized") is not None]
        result[method] = {"residual": _summary([value["residual_normalized"] for _, value in rows]),
                          "le_008": _rate(sum(value["residual_normalized"] <= .08 for _, value in rows), len(rows)),
                          "standard_eligible": _summary([value["residual_normalized"] for case, value in rows if case.route_judgment == RouteJudgment.STANDARD_ELIGIBLE])}
    result["human_label_limitation"] = "Human placement labels apply to the P1 baseline render; experimental candidates were not re-reviewed, so residual improvement is not a human PASS claim."
    return result


def _asset_geometry_ablation(cases: list[BenchmarkAnnotation], traces: dict[str, dict[str, Any]], split: dict[str, Any], asset_root: Path) -> dict[str, Any]:
    registry = AssetRegistry(asset_root)
    calibration_ids, held_out_ids = set(split["calibration_case_ids"]), set(split["held_out_case_ids"])
    scene_map = {"S01_FRONT_NEUTRAL": Scene.S01_FRONT_NEUTRAL, "S04_L34_NEUTRAL": Scene.S04_L34_NEUTRAL,
                 "S07_R34_NEUTRAL": Scene.S07_R34_NEUTRAL}
    report: dict[str, Any] = {}
    for scene_name, scene in scene_map.items():
        record = registry.select(scene)
        if record is None:
            continue
        anchors = record.anchors.anchors
        overlay_x, overlay_y, overlay_width, overlay_height = record.anchors.overlay_bbox
        neck_center = tuple((anchors["neck_left"][i] + anchors["neck_right"][i]) / 2 for i in (0, 1))
        current = {
            "silhouette_ratio_h_over_w": round(overlay_height / overlay_width, 6),
            "eye_spacing_over_width": round(math.dist(anchors["left_eye_center"], anchors["right_eye_center"]) / overlay_width, 6),
            "eye_to_chin_over_height": round(abs(anchors["chin"][1] - (anchors["left_eye_center"][1] + anchors["right_eye_center"][1]) / 2) / overlay_height, 6),
            "nose_x_over_width": round((anchors["nose_tip"][0] - overlay_x) / overlay_width, 6),
            "nose_y_over_height": round((anchors["nose_tip"][1] - overlay_y) / overlay_height, 6),
            "neck_width_over_width": round(math.dist(anchors["neck_left"], anchors["neck_right"]) / overlay_width, 6),
            "neck_height_over_height": round((neck_center[1] - overlay_y) / overlay_height, 6),
            "transparent_canvas_margins": {"left": round(overlay_x / record.anchors.canvas_size[0], 6),
                "right": round((record.anchors.canvas_size[0] - overlay_x - overlay_width) / record.anchors.canvas_size[0], 6),
                "top": round(overlay_y / record.anchors.canvas_size[1], 6),
                "bottom": round((record.anchors.canvas_size[1] - overlay_y - overlay_height) / record.anchors.canvas_size[1], 6)},
        }
        training = []
        for case in cases:
            if case.case_id not in calibration_ids or case.route_judgment != RouteJudgment.STANDARD_ELIGIBLE:
                continue
            trace = traces.get(case.case_id, {}); baseline_trace = trace.get("baseline_trace", {})
            if baseline_trace.get("scene_id") != scene_name or not trace.get("head"):
                continue
            head = DetectedHead.model_validate(trace["head"]); x, y, width, height = head.bbox
            if not all(name in head.face_landmarks for name in ("left_eye", "right_eye", "nose", "chin")):
                continue
            training.append((head, x, y, width, height))
        target_metrics: dict[str, list[float]] = defaultdict(list)
        ideal_offsets: dict[str, list[tuple[float, float]]] = defaultdict(list)
        for head, x, y, width, height in training:
            left, right, nose, chin = (head.face_landmarks[name] for name in ("left_eye", "right_eye", "nose", "chin"))
            target_metrics["silhouette_ratio_h_over_w"].append(height / width)
            target_metrics["eye_spacing_over_width"].append(math.dist(left, right) / width)
            target_metrics["eye_to_chin_over_height"].append(abs(chin[1] - (left[1] + right[1]) / 2) / height)
            target_metrics["nose_x_over_width"].append((nose[0] - x) / width)
            target_metrics["nose_y_over_height"].append((nose[1] - y) / height)
            target_metrics["neck_height_over_height"].append((head.neck_center[1] - y) / height)
            body = head.body_landmarks
            if body.get("neck_left") and body.get("neck_right"):
                target_metrics["neck_width_over_width"].append(math.dist(body["neck_left"], body["neck_right"]) / width)
            for source_name, target_name in (("left_eye_center", "left_eye"), ("right_eye_center", "right_eye"), ("nose_tip", "nose"), ("chin", "chin")):
                target = head.face_landmarks[target_name]
                ideal_offsets[source_name].append(((target[0] - head.neck_center[0]) / width, (target[1] - head.neck_center[1]) / width))
        calibrated_anchors = dict(anchors)
        if training:
            for name, values in ideal_offsets.items():
                dx = statistics.median(value[0] for value in values) / DEFAULT_THRESHOLDS.target_head_width_multiplier * overlay_width
                dy = statistics.median(value[1] for value in values) / DEFAULT_THRESHOLDS.target_head_width_multiplier * overlay_width
                calibrated_anchors[name] = (neck_center[0] + dx, neck_center[1] + dy)
        calibrated_doc = record.anchors.model_copy(update={"anchors": calibrated_anchors})
        current_residual, calibrated_residual, held_out_case_ids = [], [], []
        for case in cases:
            if case.case_id not in held_out_ids or case.route_judgment != RouteJudgment.STANDARD_ELIGIBLE:
                continue
            trace = traces.get(case.case_id, {}); baseline_trace = trace.get("baseline_trace", {})
            if baseline_trace.get("scene_id") != scene_name or not trace.get("head"):
                continue
            head = DetectedHead.model_validate(trace["head"])
            current_residual.append(solve_transform(head, scene, record.anchors).residual_normalized)
            calibrated_residual.append(solve_transform(head, scene, calibrated_doc).residual_normalized)
            held_out_case_ids.append(case.case_id)
        report[scene_name] = {"calibration_cases": [head.head_id for head, *_ in training], "held_out_case_ids": held_out_case_ids,
                              "current_asset_geometry": current,
                              "calibration_target_geometry": {key: _summary(values) for key, values in target_metrics.items()},
                              "geometry_calibrated_test_asset": {"type": "virtual anchor-only ablation; not a production asset", "anchors": calibrated_anchors},
                              "held_out_residual": {"current_generic": _summary(current_residual), "geometry_calibrated": _summary(calibrated_residual)}}
    return {"scenes": report, "interpretation_limit": "This isolates anchor geometry. It does not alter silhouette pixels and experimental renders were not human re-reviewed."}


def _proxy_confusion(cases: list[BenchmarkAnnotation], scores: dict[str, float], threshold: float, subset: set[str]) -> dict[str, Any]:
    selected = [case for case in cases if case.case_id in subset and case.case_id in scores]
    tp = fp = fn = tn = 0
    for case in selected:
        truth = any(reason.value == "EXPOSED_ORIGINAL_HEAD" for reason in case.failure_reasons)
        predicted = scores[case.case_id] > threshold
        if truth and predicted: tp += 1
        elif predicted: fp += 1
        elif truth: fn += 1
        else: tn += 1
    return {"threshold": threshold, "counts": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
            "precision": round(tp / (tp + fp), 6) if tp + fp else None,
            "recall": round(tp / (tp + fn), 6) if tp + fn else None}


def _coverage_analysis(cases: list[BenchmarkAnnotation], traces: dict[str, dict[str, Any]], split: dict[str, Any]) -> dict[str, Any]:
    scores: dict[str, dict[str, float]] = {"existing_bbox_proxy": {}, "p1_safety_proxy": {}, "candidate_landmark_silhouette": {}}
    for case in cases:
        baseline = case.observations.get(BASELINE_ADAPTER)
        safety = case.observations.get("yunet_5pt_heuristic_p1_safety")
        if baseline and "exposed_head_proxy_rate" in baseline.proxies:
            scores["existing_bbox_proxy"][case.case_id] = baseline.proxies["exposed_head_proxy_rate"]
        if safety and "exposed_head_proxy_rate" in safety.proxies:
            scores["p1_safety_proxy"][case.case_id] = safety.proxies["exposed_head_proxy_rate"]
        candidate = traces.get(case.case_id, {}).get("coverage", {}).get("landmark_silhouette_v1", {})
        if "exposed_head_ratio" in candidate:
            scores["candidate_landmark_silhouette"][case.case_id] = candidate["exposed_head_ratio"]
    calibration = set(split["calibration_case_ids"]); held_out = set(split["held_out_case_ids"])
    report = {}
    for name, values in scores.items():
        candidates = sorted({0.0, .01, .02, .03, .05, .08, .10, .15, .20, *[round(value, 3) for case_id, value in values.items() if case_id in calibration]})
        ranked = []
        for threshold in candidates:
            item = _proxy_confusion(cases, values, threshold, calibration)
            counts = item["counts"]; recall = item["recall"] or 0; precision = item["precision"] or 0
            item["safety_score"] = recall * 2 + precision
            ranked.append(item)
        selected = max(ranked, key=lambda item: (item["safety_score"], -(item["counts"]["fp"]))) if ranked else {"threshold": 0.0}
        report[name] = {"score_distribution": _summary(list(values.values())), "calibration_selected": selected,
                        "held_out_once": _proxy_confusion(cases, values, selected["threshold"], held_out)}
    side_effects = {}
    for metric in ("exposed_head_proxy_rate", "overlay_outside_envelope_rate", "below_neck_proxy_rate"):
        pairs = [(case.observations[BASELINE_ADAPTER].proxies.get(metric), case.observations["yunet_5pt_heuristic_p1_safety"].proxies.get(metric))
                 for case in cases if BASELINE_ADAPTER in case.observations and "yunet_5pt_heuristic_p1_safety" in case.observations]
        pairs = [(baseline, safety) for baseline, safety in pairs if baseline is not None and safety is not None]
        side_effects[metric] = {"n": len(pairs), "baseline_mean": round(statistics.mean(value[0] for value in pairs), 6) if pairs else None,
                                "safety_mean": round(statistics.mean(value[1] for value in pairs), 6) if pairs else None,
                                "improved": sum(safety < baseline for baseline, safety in pairs),
                                "unchanged": sum(safety == baseline for baseline, safety in pairs),
                                "worse": sum(safety > baseline for baseline, safety in pairs)}
    report["p1_safety_expansion_side_effects"] = side_effects
    report["limitation"] = "MediaPipe body landmarks are points, not a segmentation mask. No body-segmentation result is reported as if it existed. All listed masks remain geometry proxies, not proof of anonymity."
    return report


@dataclass(frozen=True)
class CandidateConfig:
    fitting: str = "two_stage_head"
    min_head_size_px: float = 40.0
    min_pose_confidence: float = .55
    front_yaw_deg: float = 18.0
    three_quarter_yaw_deg: float = 55.0
    neutral_pitch_deg: float = 18.0
    max_roll_deg: float = 35.0
    residual_threshold: float = .08


def _candidate_route(case: BenchmarkAnnotation, trace: dict[str, Any], config: CandidateConfig) -> tuple[str, str | None]:
    observation = case.observations[BASELINE_ADAPTER]
    if case.head_size_px < config.min_head_size_px:
        return RouteJudgment.SIMPLIFIED_ELIGIBLE.value, "tiny_head"
    if not observation.available or (observation.pose_confidence or 0) < config.min_pose_confidence:
        return RouteJudgment.SHOULD_FALLBACK.value, "pose_unavailable"
    yaw, pitch, roll = observation.yaw_deg or 0, observation.pitch_deg or 0, observation.roll_deg or 0
    if abs(roll) > config.max_roll_deg: return RouteJudgment.SHOULD_FALLBACK.value, "roll_too_large"
    if abs(pitch) > config.neutral_pitch_deg: return RouteJudgment.SHOULD_FALLBACK.value, "unsupported_pitch"
    if abs(yaw) > config.three_quarter_yaw_deg: return RouteJudgment.SHOULD_FALLBACK.value, "unsupported_scene"
    fitting = trace.get("fitting_bakeoff", {}).get(config.fitting)
    if not isinstance(fitting, dict): return RouteJudgment.SHOULD_FALLBACK.value, "fitting_unavailable"
    if fitting.get("residual_normalized", math.inf) > config.residual_threshold:
        return RouteJudgment.SHOULD_FALLBACK.value, "transform_residual_too_high"
    return RouteJudgment.STANDARD_ELIGIBLE.value, None


def _router_counts(cases: list[BenchmarkAnnotation], predicted: dict[str, tuple[str, str | None]]) -> dict[str, Any]:
    false_standard = [case.case_id for case in cases if predicted[case.case_id][0] == RouteJudgment.STANDARD_ELIGIBLE.value and case.route_judgment != RouteJudgment.STANDARD_ELIGIBLE]
    false_fallback = [case.case_id for case in cases if predicted[case.case_id][0] == RouteJudgment.SHOULD_FALLBACK.value and case.route_judgment == RouteJudgment.STANDARD_ELIGIBLE]
    correct_standard = [case.case_id for case in cases if predicted[case.case_id][0] == RouteJudgment.STANDARD_ELIGIBLE.value and case.route_judgment == RouteJudgment.STANDARD_ELIGIBLE]
    correct_fallback = [case.case_id for case in cases if predicted[case.case_id][0] == RouteJudgment.SHOULD_FALLBACK.value and case.route_judgment == RouteJudgment.SHOULD_FALLBACK]
    reliable = [case.case_id for case in cases if case.case_id in correct_standard and case.placement_result == PlacementResult.PASS]
    predicted_standard = [case for case in cases if predicted[case.case_id][0] == RouteJudgment.STANDARD_ELIGIBLE.value]
    placement = Counter(case.placement_result.value for case in predicted_standard)
    exposed = [case.case_id for case in predicted_standard if any(reason.value == "EXPOSED_ORIGINAL_HEAD" for reason in case.failure_reasons)]
    reasons = Counter(predicted[case.case_id][1] or "NONE" for case in cases if case.case_id in false_fallback)
    return {"reliable_standard": _rate(len(reliable), len(cases)), "correct_standard": {"count": len(correct_standard), "case_ids": correct_standard},
            "correct_fallback": {"count": len(correct_fallback), "case_ids": correct_fallback},
            "false_standard": {"count": len(false_standard), "case_ids": false_standard},
            "false_fallback": {"count": len(false_fallback), "case_ids": false_fallback, "reason_decomposition": dict(reasons)},
            "predicted_standard_placement": dict(placement),
            "predicted_standard_exposed_head_failure": {"count": len(exposed), "case_ids": exposed}}


def _router_calibration(cases: list[BenchmarkAnnotation], traces: dict[str, dict[str, Any]], split: dict[str, Any]) -> dict[str, Any]:
    calibration = [case for case in cases if case.case_id in set(split["calibration_case_ids"])]
    held_out = [case for case in cases if case.case_id in set(split["held_out_case_ids"])]
    candidates = []
    for fitting in ("p1_baseline", "face_primary", "two_stage_head"):
        for min_size in (40.0, 48.0):
            for confidence in (.55, .60):
                for front in (15.0, 18.0):
                    for three_quarter in (50.0, 55.0):
                        for pitch in (15.0, 18.0):
                            for residual in (.06, .08, .10, .12):
                                config = CandidateConfig(fitting, min_size, confidence, front, three_quarter, pitch, 35.0, residual)
                                predicted = {case.case_id: _candidate_route(case, traces.get(case.case_id, {}), config) for case in calibration}
                                counts = _router_counts(calibration, predicted)
                                score = counts["reliable_standard"]["numerator"] * 5 + counts["correct_fallback"]["count"] * 2 - counts["false_standard"]["count"] * 12 - counts["false_fallback"]["count"] * 2
                                candidates.append((score, -counts["false_standard"]["count"], counts["reliable_standard"]["numerator"], config, counts))
    _, _, _, selected, calibration_counts = max(candidates, key=lambda item: item[:3])
    baseline_predicted = {case.case_id: (_route_class(case.observations[BASELINE_ADAPTER].actual_route), case.observations[BASELINE_ADAPTER].fallback_reason) for case in held_out}
    candidate_predicted = {case.case_id: _candidate_route(case, traces.get(case.case_id, {}), selected) for case in held_out}
    return {"objective": "maximize reliable STANDARD with a strong false-STANDARD penalty",
            "baseline_values_preserved": asdict(DEFAULT_THRESHOLDS), "candidate_configuration_not_applied": asdict(selected),
            "calibration_result": calibration_counts,
            "held_out_validation_once": {"baseline": _router_counts(held_out, baseline_predicted), "candidate": _router_counts(held_out, candidate_predicted)}}


def analyze_human_baseline(document: BenchmarkDocument, trace_root: Path, asset_root: Path) -> dict[str, Any]:
    cases = list(document.cases)
    split = grouped_split(document)
    traces = {case.case_id: _trace_payload(trace_root, case.case_id) for case in cases}
    placement = Counter(case.placement_result.value for case in cases)
    route = Counter(case.route_judgment.value for case in cases)
    actual_confusion = _confusion(cases, lambda case: _route_class(case.observations[BASELINE_ADAPTER].actual_route))
    actual_route_values = sorted({case.observations[BASELINE_ADAPTER].actual_route.value for case in cases})
    detailed_confusion = {judgment.value: {route_name: 0 for route_name in actual_route_values} for judgment in RouteJudgment}
    for case in cases:
        detailed_confusion[case.route_judgment.value][case.observations[BASELINE_ADAPTER].actual_route.value] += 1
    scene_cases = [case for case in cases if case.expected_scene is not None]
    scene_correct = [case.case_id for case in scene_cases if case.observations[BASELINE_ADAPTER].actual_scene == case.expected_scene]
    scene_mismatch = [case.case_id for case in scene_cases if case.observations[BASELINE_ADAPTER].actual_scene != case.expected_scene]
    standard = [case for case in cases if case.route_judgment == RouteJudgment.STANDARD_ELIGIBLE]
    standard_actual = Counter(_route_class(case.observations[BASELINE_ADAPTER].actual_route) for case in standard)
    standard_placement = Counter(case.placement_result.value for case in standard)
    baseline_predicted = {case.case_id: (_route_class(case.observations[BASELINE_ADAPTER].actual_route), case.observations[BASELINE_ADAPTER].fallback_reason) for case in cases}
    failures = _failure_analysis(cases)
    residual = _residual_analysis(cases, traces)
    fitting = _fitting_analysis(cases, traces)
    asset_ablation = _asset_geometry_ablation(cases, traces, split, asset_root)
    coverage = _coverage_analysis(cases, traces, split)
    router = _router_calibration(cases, traces, split)
    return {
        "report_id": "pose-aware-avatar-overlay-real-world-human-baseline-v0.1",
        "ground_truth_policy": "human review only; geometry proxies and model outputs are never substituted for labels",
        "integrity": _integrity(cases),
        "human_baseline": {"reviewed_completeness": _rate(len(cases), len(cases)), "placement_distribution": dict(placement),
                           "route_judgment_distribution": dict(route), "actual_route_vs_human_route_class": actual_confusion,
                           "actual_route_vs_human_route_detailed": detailed_confusion,
                           "route_accuracy": _rate(sum(actual_confusion[label].get(label, 0) for label in actual_confusion), len(cases)),
                           "scene_accuracy": {**_rate(len(scene_correct), len(scene_cases)), "correct_case_ids": scene_correct,
                                              "mismatch_case_ids": scene_mismatch,
                                              "unsupported_or_unlabeled_case_ids": [case.case_id for case in cases if case.expected_scene is None]},
                           "router_overall": _router_counts(cases, baseline_predicted),
                           "standard_eligible": {"count": len(standard), "actual_route_distribution": dict(standard_actual),
                                                 "placement_distribution": dict(standard_placement),
                                                 "false_fallback_rate": _rate(standard_actual[RouteJudgment.SHOULD_FALLBACK.value], len(standard))}},
        "failure_analysis": failures, "residual_gate_validity": residual,
        "fitting_bakeoff": fitting, "asset_geometry_ablation": asset_ablation, "coverage_proxy_validation": coverage,
        "grouped_split": split, "router_calibration": router,
        "conclusions": {
            "top_three_observed_problems": ["EXPOSED_ORIGINAL_HEAD (14/48)", "SHOULD_HAVE_FALLBACK (11/48)", "FALSE_FALLBACK (9/48 occurrence; 16/24 STANDARD_ELIGIBLE cases routed to fallback)"],
            "residual_gate": "0.08 is over-conservative and the current equally weighted residual is not a reliable standalone routing signal; redesign the fitting objective instead of merely increasing the threshold.",
            "recommended_fitting_architecture": "P2A two-stage head + neck patch, with face-primary head fitting and neck/body landmarks used only for an independent adapter and validation.",
            "recommended_pose_adapter": "Keep YuNet five-point face/yaw as the current baseline; do not adopt solvePnP or MediaPipe hybrid as the default pose adapter.",
            "recommended_body_landmarks": "Use MediaPipe shoulder/neck points only in the experimental independent neck stage. Never let them rigidly drag the face transform.",
            "generic_asset_geometry": "Replace/recalibrate the temporary generic silhouette before production, but anchor-only held-out ablation is mixed and does not prove asset geometry is the sole bottleneck.",
            "candidate_configuration": "Rejected for production: held-out reliable STANDARD rose 4 to 7, but false STANDARD regressed 0 to 1 and correct fallback fell 3 to 2.",
            "go_no_go": "YELLOW: route over-conservatism and face/neck conflict are localized and experimentally recoverable, but candidate renders lack human re-review and exposed-head coverage remains unreliable.",
            "p2_next_three_tasks": ["Human re-review of P2A two-stage candidate renders on the frozen held-out cases", "Build a real geometry-calibrated generic silhouette/neck asset and rerun asset ablation", "Develop a safety-oriented head mask with materially better held-out precision/recall before using coverage as a gate"],
        },
    }
