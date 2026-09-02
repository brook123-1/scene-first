from __future__ import annotations

import math
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import httpx
import numpy as np
from PIL import Image

from .adapters import estimate_pose
from .models import DetectedHead, PoseEstimate, View


MP_POSE_MODEL_URL = (
    "https://huggingface.co/opencv/pose_estimation_mediapipe/resolve/main/"
    "pose_estimation_mediapipe_2023mar.onnx?download=true"
)


@dataclass
class PoseAdapterResult:
    adapter_id: str
    available: bool
    elapsed_ms: int
    pose: PoseEstimate | None = None
    face_landmarks: dict[str, tuple[float, float]] = field(default_factory=dict)
    body_landmarks: dict[str, tuple[float, float]] = field(default_factory=dict)
    anchor_provenance: dict[str, str] = field(default_factory=dict)
    view_hint: View | None = None
    error: str | None = None

    def apply(self, head: DetectedHead) -> DetectedHead:
        if not self.available or self.pose is None:
            return head.model_copy(update={"pose": PoseEstimate(yaw_deg=0, pitch_deg=0, roll_deg=0, confidence=0)})
        face = {**head.face_landmarks, **self.face_landmarks}
        body = {**head.body_landmarks, **self.body_landmarks}
        provenance = {**head.anchor_provenance, **self.anchor_provenance}
        return head.model_copy(update={
            "pose": self.pose,
            "face_landmarks": face,
            "body_landmarks": body,
            "anchor_provenance": provenance,
            "view_hint": self.view_hint or head.view_hint,
        })


class YuNetHeuristicAdapter:
    adapter_id = "yunet_5pt_heuristic"

    def estimate(self, image: Image.Image, head: DetectedHead) -> PoseAdapterResult:
        started = time.perf_counter()
        pose = estimate_pose(head.bbox, head.face_landmarks, head.confidence)
        return PoseAdapterResult(
            adapter_id=self.adapter_id,
            available=pose.confidence > 0,
            elapsed_ms=round((time.perf_counter() - started) * 1000),
            pose=pose,
            face_landmarks=head.face_landmarks,
            error=None if pose.confidence > 0 else "yunet_landmarks_unavailable",
        )


class YuNetSolvePnPAdapter:
    """Experimental 5-point + bbox-chin PnP candidate.

    The chin remains a bbox estimate, so this adapter is intentionally kept in
    the bake-off and never selected by production routing without calibration.
    """

    adapter_id = "yunet_5pt_solvepnp"

    _MODEL_POINTS = np.asarray([
        (0.0, 0.0, 0.0),          # nose
        (-43.0, 32.0, -26.0),     # image-left eye
        (43.0, 32.0, -26.0),      # image-right eye
        (-28.0, -28.0, -24.0),    # image-left mouth
        (28.0, -28.0, -24.0),     # image-right mouth
        (0.0, -63.0, -12.0),      # chin, bbox-derived in P1
    ], dtype=np.float64)

    def estimate(self, image: Image.Image, head: DetectedHead) -> PoseAdapterResult:
        started = time.perf_counter()
        face = head.face_landmarks
        mouth_left, mouth_right = face.get("mouth_left"), face.get("mouth_right")
        required = [face.get("nose"), face.get("left_eye"), face.get("right_eye"), mouth_left, mouth_right, face.get("chin")]
        if any(value is None for value in required):
            return PoseAdapterResult(self.adapter_id, False, round((time.perf_counter() - started) * 1000), error="six_correspondences_unavailable")
        image_points = np.asarray(required, dtype=np.float64)
        focal = float(max(image.size))
        camera = np.asarray([[focal, 0, image.width / 2], [0, focal, image.height / 2], [0, 0, 1]], dtype=np.float64)
        success, rotation, _ = cv2.solvePnP(
            self._MODEL_POINTS, image_points, camera, np.zeros((4, 1)), flags=cv2.SOLVEPNP_EPNP,
        )
        if not success:
            return PoseAdapterResult(self.adapter_id, False, round((time.perf_counter() - started) * 1000), error="solvepnp_failed")
        rotation_matrix, _ = cv2.Rodrigues(rotation)
        angles = cv2.RQDecomp3x3(rotation_matrix)[0]
        # OpenCV's canonical axes differ from the overlay contract.  Roll from
        # the observed eye line is more stable and preserves P0 sign semantics.
        left_eye, right_eye = face["left_eye"], face["right_eye"]
        if left_eye[0] > right_eye[0]:
            left_eye, right_eye = right_eye, left_eye
        roll = math.degrees(math.atan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0]))
        pose = PoseEstimate(
            yaw_deg=float(max(-80, min(80, angles[1]))),
            pitch_deg=float(max(-60, min(60, -angles[0]))),
            roll_deg=float(roll),
            confidence=min(0.82, head.confidence * 0.84),
        )
        return PoseAdapterResult(self.adapter_id, True, round((time.perf_counter() - started) * 1000), pose=pose, face_landmarks=face)


class OpenCvMediaPipePoseAdapter:
    """CPU-only OpenCV Zoo MediaPipe Pose adapter for local benchmark use."""

    adapter_id = "opencv_mediapipe_pose_hybrid"
    _NAMES = {
        0: "nose", 2: "left_eye", 5: "right_eye", 7: "left_ear", 8: "right_ear",
        11: "left_shoulder", 12: "right_shoulder",
    }

    def __init__(self, model_path: Path, *, allow_download: bool = False) -> None:
        self.model_path = Path(model_path)
        self.allow_download = allow_download
        self._net = None

    def _ensure_model(self) -> None:
        if self._net is not None:
            return
        if self.model_path.exists() and self.model_path.stat().st_size < 1_000_000:
            self.model_path.unlink(missing_ok=True)
        if not self.model_path.exists():
            if not self.allow_download:
                raise FileNotFoundError("MediaPipe Pose ONNX is not installed; rerun with --download-models")
            response = httpx.get(MP_POSE_MODEL_URL, follow_redirects=True, timeout=90)
            response.raise_for_status()
            if len(response.content) < 1_000_000:
                raise ValueError("MediaPipe Pose download was not an ONNX model")
            self.model_path.parent.mkdir(parents=True, exist_ok=True)
            self.model_path.write_bytes(response.content)
        # OpenCV on Windows can fail to open ONNX files below a non-ASCII
        # workspace path.  Mirror the proven detector workaround and keep the
        # model itself local in an ASCII-only temporary cache.
        cache = Path(tempfile.gettempdir()) / "SceneFirstPrivacy" / "models"
        cache.mkdir(parents=True, exist_ok=True)
        safe_path = cache / self.model_path.name
        if not safe_path.exists() or safe_path.stat().st_size != self.model_path.stat().st_size:
            shutil.copy2(self.model_path, safe_path)
        self._net = cv2.dnn.readNet(str(safe_path))

    @staticmethod
    def _output_by_size(outputs: list[np.ndarray], size: int) -> np.ndarray:
        match = next((value for value in outputs if value.size == size), None)
        if match is None:
            raise ValueError(f"MediaPipe Pose output with {size} values is missing")
        return match

    def estimate(self, image: Image.Image, head: DetectedHead) -> PoseAdapterResult:
        started = time.perf_counter()
        try:
            self._ensure_model()
            assert self._net is not None
            bgr = cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2BGR)
            x, y, width, height = head.bbox
            cx, cy = x + width / 2, y + height * 2.0
            radius = max(width * 2.2, height * 2.25)
            scale = 256.0 / max(radius * 2, 1.0)
            affine = np.asarray([[scale, 0, 128 - cx * scale], [0, scale, 128 - cy * scale]], dtype=np.float32)
            crop = cv2.warpAffine(bgr, affine, (256, 256), flags=cv2.INTER_AREA, borderMode=cv2.BORDER_CONSTANT)
            blob = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32)[None, ...] / 255.0
            self._net.setInput(blob)
            outputs = self._net.forward(self._net.getUnconnectedOutLayersNames())
            raw = self._output_by_size(outputs, 195).reshape(39, 5).astype(np.float64)
            confidence = float(self._output_by_size(outputs, 1).reshape(-1)[0])
            raw[:, 3:] = 1.0 / (1.0 + np.exp(-np.clip(raw[:, 3:], -100, 100)))
            raw[:, 0] = (raw[:, 0] - 128) / scale + cx
            raw[:, 1] = (raw[:, 1] - 128) / scale + cy
            points: dict[str, tuple[float, float]] = {}
            strengths: dict[str, float] = {}
            for index, name in self._NAMES.items():
                strengths[name] = float(min(raw[index, 3], raw[index, 4]))
                if strengths[name] >= 0.35:
                    points[name] = (float(raw[index, 0]), float(raw[index, 1]))
            shoulders = [points.get("left_shoulder"), points.get("right_shoulder")]
            body = {key: value for key, value in points.items() if "shoulder" in key}
            if all(shoulders):
                left, right = shoulders
                shoulder_width = math.dist(left, right)
                body["neck_center"] = ((left[0] + right[0]) / 2, (left[1] + right[1]) / 2 - shoulder_width * 0.10)
                body["neck_left"] = (body["neck_center"][0] - shoulder_width * 0.10, body["neck_center"][1])
                body["neck_right"] = (body["neck_center"][0] + shoulder_width * 0.10, body["neck_center"][1])
            face = {key: value for key, value in points.items() if key not in body and "shoulder" not in key}
            # This adapter is deliberately hybrid: a real YuNet five-point set
            # remains authoritative for yaw/pitch and transform residuals;
            # MediaPipe contributes roll and body anchors.  Pose-model face
            # points only fill gaps instead of silently replacing YuNet.
            merged_face = {**face, **head.face_landmarks}
            supplemental_face = {key: value for key, value in face.items() if key not in head.face_landmarks}
            baseline = estimate_pose(head.bbox, merged_face, head.confidence)
            eye_left, eye_right = merged_face.get("left_eye"), merged_face.get("right_eye")
            roll = baseline.roll_deg
            if eye_left and eye_right:
                if eye_left[0] > eye_right[0]:
                    eye_left, eye_right = eye_right, eye_left
                roll = math.degrees(math.atan2(eye_right[1] - eye_left[1], eye_right[0] - eye_left[0]))
            relevant = [strengths.get(name, 0.0) for name in ("nose", "left_eye", "right_eye", "left_shoulder", "right_shoulder")]
            mp_confidence = min(0.90, max(0.0, confidence) * (sum(relevant) / len(relevant)))
            pose_confidence = min(baseline.confidence, mp_confidence) if baseline.confidence > 0 else mp_confidence * 0.72
            pose = baseline.model_copy(update={"roll_deg": roll, "confidence": pose_confidence})
            face_strength = max(strengths.get("nose", 0), strengths.get("left_eye", 0), strengths.get("right_eye", 0))
            ear_strength = min(strengths.get("left_ear", 0), strengths.get("right_ear", 0))
            view_hint = View.BACK if face_strength < 0.25 and ear_strength > 0.55 and all(shoulders) else None
            provenance = {key: "opencv_mediapipe_pose" for key in (*supplemental_face.keys(), *body.keys())}
            return PoseAdapterResult(
                self.adapter_id, True, round((time.perf_counter() - started) * 1000), pose,
                supplemental_face, body, provenance, view_hint,
            )
        except Exception as exc:
            return PoseAdapterResult(
                self.adapter_id, False, round((time.perf_counter() - started) * 1000), error=f"{type(exc).__name__}: {exc}",
            )
