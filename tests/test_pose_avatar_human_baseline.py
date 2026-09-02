from __future__ import annotations

from datetime import datetime, timezone

from app.pose_avatar.benchmark import (
    AdapterObservation, BenchmarkAnnotation, BenchmarkDocument, PlacementResult, ReviewStatus, RouteJudgment,
)
from app.pose_avatar.human_baseline import grouped_split
from app.pose_avatar.models import RenderRoute, Scene


def _case(number: int, sample: str) -> BenchmarkAnnotation:
    return BenchmarkAnnotation(
        case_id=f"p1-{number:03d}", local_image_ref=f"samples/inbox/{sample}.jpg",
        source_sample_id=sample, source_region_id=str(number), head_bbox=(1, 2, 80, 100),
        head_size_px=80, detector_source="yunet", review_status=ReviewStatus.REVIEWED,
        placement_result=PlacementResult.PASS, route_judgment=RouteJudgment.STANDARD_ELIGIBLE,
        expected_scene=Scene.S01_FRONT_NEUTRAL,
        observations={"yunet_5pt_heuristic": AdapterObservation(
            adapter_id="yunet_5pt_heuristic", available=True, elapsed_ms=1,
            actual_route=RenderRoute.STANDARD_AVATAR, actual_scene=Scene.S01_FRONT_NEUTRAL,
        )},
    )


def test_grouped_split_never_leaks_people_from_the_same_source_photo():
    cases = [_case(index, f"sample-{(index - 1) // 2:02d}") for index in range(1, 21)]
    document = BenchmarkDocument(benchmark_id="test", created_at=datetime.now(timezone.utc).isoformat(),
                                 sampling_seed="fixed", cases=cases)
    split = grouped_split(document)
    calibration = set(split["calibration_source_samples"])
    held_out = set(split["held_out_source_samples"])
    assert calibration.isdisjoint(held_out)
    assert calibration | held_out == {case.source_sample_id for case in cases}
    for sample in calibration:
        assert all(case.case_id in split["calibration_case_ids"] for case in cases if case.source_sample_id == sample)


def test_grouped_split_is_deterministic():
    cases = [_case(index, f"sample-{index:02d}") for index in range(1, 11)]
    document = BenchmarkDocument(benchmark_id="test", created_at=datetime.now(timezone.utc).isoformat(),
                                 sampling_seed="fixed", cases=cases)
    assert grouped_split(document) == grouped_split(document)
