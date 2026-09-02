from app.detector import FaceDetector


def test_mediapipe_anchor_generator_has_reference_shape():
    anchors = FaceDetector._build_person_anchors()
    assert anchors.shape == (2254, 2)
    assert anchors[0].tolist() == anchors[1].tolist()


def test_dedupe_removes_nested_body_head_hypothesis():
    detector = FaceDetector()
    candidates = [
        ([100, 100, 80, 100], 0.92, "yunet"),
        ([80, 70, 140, 180], 0.54, "mediapipe-person"),
        ([300, 100, 80, 100], 0.80, "yunet"),
    ]
    kept = detector._dedupe(candidates)
    assert len(kept) == 2
    assert [item[2] for item in kept] == ["yunet", "yunet"]


def test_fusion_compares_candidates_after_face_to_head_normalization():
    detector = FaceDetector()
    candidates = [
        ([110, 110, 60, 70], 0.92, "yunet"),
        ([86, 70, 116, 134], 0.54, "mediapipe-person"),
    ]
    kept = detector._fuse_candidates(candidates, (800, 600))
    assert len(kept) == 1
    raw_box, head_box, confidence, source, sources = kept[0]
    assert raw_box == [110, 110, 60, 70]
    assert head_box != raw_box
    assert confidence == 0.92
    assert source == "yunet"
    assert sources == ["yunet", "mediapipe-person"]


def test_person_head_hypothesis_is_not_expanded_twice():
    detector = FaceDetector()
    candidate = ([80, 70, 140, 180], 0.54, "mediapipe-person")
    kept = detector._fuse_candidates([candidate], (800, 600))
    assert kept[0][1] == candidate[0]


def test_fusion_preserves_adjacent_overlapping_heads():
    detector = FaceDetector()
    candidates = [
        ([100, 100, 100, 140], 0.90, "mediapipe-person"),
        ([175, 100, 100, 140], 0.88, "mediapipe-person"),
    ]
    assert len(detector._fuse_candidates(candidates, (800, 600))) == 2


def test_fusion_does_not_merge_a_smaller_occluded_head_at_edge_of_foreground_head():
    detector = FaceDetector()
    candidates = [
        ([100, 100, 160, 200], 0.90, "mediapipe-person"),
        ([190, 120, 70, 90], 0.88, "mediapipe-person"),
    ]
    assert len(detector._fuse_candidates(candidates, (800, 600))) == 2


def test_cross_detector_containment_is_merged_despite_box_convention_offset():
    detector = FaceDetector()
    candidates = [
        ([100, 100, 160, 200], 0.90, "mediapipe-person"),
        ([190, 120, 70, 90], 0.52, "haar"),
    ]
    assert len(detector._fuse_candidates(candidates, (800, 600))) == 1


def test_degenerate_standalone_body_tile_candidate_is_removed():
    detector = FaceDetector()
    candidate = ([798, 598, 2, 2], 0.50, "mediapipe-person")
    assert detector._fuse_candidates([candidate], (800, 600)) == []


def test_tiny_standalone_haar_false_positive_is_removed_but_supported_one_stays():
    detector = FaceDetector()
    tiny = ([300, 80, 24, 24], 0.52, "haar")
    assert detector._fuse_candidates([tiny], (1600, 1200)) == []
    supported = detector._fuse_candidates([
        tiny,
        ([294, 70, 40, 52], 0.50, "mediapipe-person"),
    ], (1600, 1200))
    assert len(supported) == 1
    assert set(supported[0][4]) == {"haar", "mediapipe-person"}
