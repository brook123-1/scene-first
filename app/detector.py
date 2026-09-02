from __future__ import annotations

import threading
import uuid
import os
import shutil
import tempfile
import time
from pathlib import Path

import cv2
import httpx
import numpy as np
from PIL import Image

from .config import (
    DETECTION_ENABLE_TILES,
    DETECTION_TILE_MIN_EDGE,
    PERSON_MODEL_PATH,
    PERSON_MODEL_URL,
    YUNET_MODEL_PATH,
    YUNET_MODEL_URL,
)
from .image_ops import clamp_box, face_to_head_box


class FaceDetector:
    """Replaceable face/head detector with YuNet and offline Haar fallback."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._yunet = None
        self._yunet_attempted = False
        self._person_net = None
        self._person_attempted = False
        self._person_anchors = self._build_person_anchors()
        # Some OpenCV Windows builds cannot open cascade files through a path
        # containing non-ASCII characters. Keep an ASCII-path copy outside the
        # project so the Chinese workspace name never breaks detection.
        source = Path(cv2.data.haarcascades) / "haarcascade_frontalface_alt2.xml"
        # Use the OS temporary directory rather than LOCALAPPDATA. It is an
        # ASCII-only path on Windows and remains writable in restricted local
        # runners as well as inside the production container.
        self._opencv_model_cache = Path(tempfile.gettempdir()) / "SceneFirstPrivacy" / "models"
        target = self._opencv_model_cache / source.name
        try:
            self._opencv_model_cache.mkdir(parents=True, exist_ok=True)
            if source.exists() and not target.exists():
                shutil.copy2(source, target)
        except OSError:
            target = source
        self._haar = cv2.CascadeClassifier(str(target))

    def _opencv_safe_path(self, source: Path) -> Path:
        """Copy a model to an ASCII-only path for OpenCV builds on Windows."""
        target = self._opencv_model_cache / source.name
        self._opencv_model_cache.mkdir(parents=True, exist_ok=True)
        if not target.exists() or target.stat().st_size != source.stat().st_size:
            shutil.copy2(source, target)
        return target

    @staticmethod
    def _build_person_anchors() -> np.ndarray:
        # MediaPipe Pose detector's 2,254 fixed-size SSD anchors. This compact
        # generator is equivalent to the Apache-2.0 OpenCV Zoo reference list.
        anchors = []
        for grid, copies in ((28, 2), (14, 2), (7, 6)):
            for y in range(grid):
                for x in range(grid):
                    anchors.extend([[(x + 0.5) / grid, (y + 0.5) / grid]] * copies)
        return np.asarray(anchors, dtype=np.float32)

    def _ensure_person_model(self) -> None:
        if self._person_net is not None or self._person_attempted:
            return
        with self._lock:
            if self._person_net is not None or self._person_attempted:
                return
            self._person_attempted = True
            if PERSON_MODEL_PATH.exists() and PERSON_MODEL_PATH.stat().st_size < 1_000_000:
                PERSON_MODEL_PATH.unlink(missing_ok=True)
            if not PERSON_MODEL_PATH.exists():
                try:
                    response = httpx.get(PERSON_MODEL_URL, timeout=60, follow_redirects=True)
                    response.raise_for_status()
                    if len(response.content) < 1_000_000:
                        raise ValueError("Person detector download did not contain an ONNX model.")
                    PERSON_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
                    PERSON_MODEL_PATH.write_bytes(response.content)
                except Exception:
                    return
            try:
                self._person_net = cv2.dnn.readNet(str(self._opencv_safe_path(PERSON_MODEL_PATH)))
            except Exception:
                self._person_net = None

    def _person_infer(self, bgr: np.ndarray, offset: tuple[int, int] = (0, 0)) -> list[tuple[list[int], float, str]]:
        if self._person_net is None:
            return []
        height, width = bgr.shape[:2]
        ratio = min(224 / height, 224 / width)
        resized_width, resized_height = max(1, round(width * ratio)), max(1, round(height * ratio))
        # Resize while pixels are still uint8. Converting a 12 MP source to
        # float32 first creates several full-size arrays and was the largest
        # avoidable memory spike in the container detection path.
        resized_bgr = cv2.resize(bgr, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
        resized = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
        resized = (resized / 127.5) - 1.0
        pad_x, pad_y = (224 - resized_width) // 2, (224 - resized_height) // 2
        padded = cv2.copyMakeBorder(
            resized, pad_y, 224 - resized_height - pad_y, pad_x, 224 - resized_width - pad_x,
            cv2.BORDER_CONSTANT, value=(0, 0, 0),
        )
        self._person_net.setInput(np.transpose(padded, (2, 0, 1))[None, ...])
        outputs = self._person_net.forward(self._person_net.getUnconnectedOutLayersNames())
        box_data, score_data = outputs[0][0, :, :4], outputs[1][0, :, 0]
        scores = 1.0 / (1.0 + np.exp(-np.clip(score_data.astype(np.float64), -100, 100)))
        scale = max(width, height)
        centers = box_data[:, :2] / 224 + self._person_anchors
        sizes = box_data[:, 2:] / 224
        xy1 = (centers - sizes / 2) * scale
        xy2 = (centers + sizes / 2) * scale
        boxes = np.concatenate((xy1, xy2), axis=1)
        bias = np.asarray([pad_x / ratio, pad_y / ratio, pad_x / ratio, pad_y / ratio])
        boxes -= bias
        candidate_indices = np.flatnonzero(scores >= 0.48)
        if not len(candidate_indices):
            return []
        nms_boxes = []
        nms_scores = []
        for index in candidate_indices:
            x1, y1, x2, y2 = boxes[index]
            nms_boxes.append([float(x1), float(y1), float(x2 - x1), float(y2 - y1)])
            nms_scores.append(float(scores[index]))
        kept = cv2.dnn.NMSBoxes(nms_boxes, nms_scores, 0.48, 0.3, top_k=500)
        results = []
        ox, oy = offset
        for local_index in np.asarray(kept).reshape(-1) if len(kept) else []:
            x, y, w, h = nms_boxes[int(local_index)]
            # This model returns a person/upper-body box. Convert its top-center
            # into a deliberately generous head hypothesis; face detectors will
            # supersede it when they overlap. SAM-style masks come later.
            head = [
                round(ox + x + 0.27 * w), round(oy + y - 0.03 * h),
                round(0.46 * w), round(0.37 * h),
            ]
            # Body-derived head regions are useful for recall but intentionally
            # remain lower-confidence than a real face detection.
            results.append((head, min(0.54, nms_scores[int(local_index)] * 0.62), "mediapipe-person"))
        return results

    def _ensure_yunet(self) -> None:
        if self._yunet is not None or self._yunet_attempted:
            return
        with self._lock:
            if self._yunet is not None or self._yunet_attempted:
                return
            self._yunet_attempted = True
            # Git/LFS pointer files are only a few hundred bytes. They are not
            # valid ONNX models and must not be handed to OpenCV.
            if YUNET_MODEL_PATH.exists() and YUNET_MODEL_PATH.stat().st_size < 100_000:
                YUNET_MODEL_PATH.unlink(missing_ok=True)
            if not YUNET_MODEL_PATH.exists():
                try:
                    response = httpx.get(YUNET_MODEL_URL, timeout=30, follow_redirects=True)
                    response.raise_for_status()
                    if len(response.content) < 100_000:
                        raise ValueError("YuNet download did not contain an ONNX model.")
                    YUNET_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
                    YUNET_MODEL_PATH.write_bytes(response.content)
                except Exception:
                    return
            try:
                self._yunet = cv2.FaceDetectorYN.create(
                    str(self._opencv_safe_path(YUNET_MODEL_PATH)), "", (320, 320), 0.55, 0.3, 5000,
                )
            except Exception:
                self._yunet = None

    @staticmethod
    def _overlap(a: list[int], b: list[int]) -> tuple[float, float]:
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        left, top = max(ax, bx), max(ay, by)
        right, bottom = min(ax + aw, bx + bw), min(ay + ah, by + bh)
        inter = max(0, right - left) * max(0, bottom - top)
        union = aw * ah + bw * bh - inter
        smaller = min(aw * ah, bw * bh)
        return (inter / union if union else 0.0, inter / smaller if smaller else 0.0)

    @staticmethod
    def _same_head(
        a: list[int], b: list[int], source_a: str | None = None, source_b: str | None = None,
    ) -> bool:
        """Return True when two *head-space* boxes describe one person.

        Face detectors and the upper-body branch produce boxes with very
        different proportions. IoU alone therefore leaves duplicate boxes
        around the same head. Centre distance is the second signal, but it is
        deliberately strict so two genuinely overlapping people stay apart.
        """
        iou, containment = FaceDetector._overlap(a, b)
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        centre_distance = ((ax + aw / 2 - bx - bw / 2) ** 2 + (ay + ah / 2 - by - bh / 2) ** 2) ** 0.5
        reference = max(1.0, min((aw * ah) ** 0.5, (bw * bh) ** 0.5))
        area_ratio = max(aw * ah, bw * bh) / max(1, min(aw * ah, bw * bh))
        if iou >= 0.30:
            return True
        # Cross-model candidates routinely use different box conventions. A
        # contained YuNet/Haar/body pair is therefore much more likely to be a
        # duplicate than two contained boxes emitted by the same model. Keep
        # the stricter centre test for same-source boxes so genuinely
        # overlapping people are not collapsed into one.
        cross_source = bool(source_a and source_b and source_a != source_b)
        body_cross_source = cross_source and "mediapipe-person" in {source_a, source_b}
        if containment >= 0.64:
            if body_cross_source and centre_distance / reference <= 0.90:
                return True
            if area_ratio <= 4.0 and (cross_source or centre_distance / reference <= 0.60):
                return True
        return centre_distance / reference <= 0.34 and area_ratio <= 3.4

    @staticmethod
    def _head_box(box: list[int], source: str, size: tuple[int, int]) -> list[int]:
        # _person_infer already returns a head hypothesis. Expanding it again
        # as if it were a face was the main cause of oversized duplicate boxes.
        return clamp_box(box if source == "mediapipe-person" else face_to_head_box(box, size), size)

    def _fuse_candidates(
        self,
        detections: list[tuple[list[int], float, str]],
        size: tuple[int, int],
    ) -> list[tuple[list[int], list[int], float, str, list[str]]]:
        """Normalize candidates to head space, merge duplicates, keep evidence."""
        source_rank = {"yunet": 3, "haar": 2, "mediapipe-person": 1}
        ordered = sorted(detections, key=lambda value: (value[1], source_rank.get(value[2], 0)), reverse=True)
        groups: list[dict] = []
        for raw_box, confidence, source in ordered:
            head_box = self._head_box(raw_box, source, size)
            match = next((
                group for group in groups
                if self._same_head(head_box, group["head_box"], source, group["source"])
            ), None)
            if match is None:
                groups.append({
                    "box": raw_box,
                    "head_box": head_box,
                    "confidence": confidence,
                    "source": source,
                    "sources": [source],
                })
            elif source not in match["sources"]:
                match["sources"].append(source)

        # Haar has no calibrated confidence score. Very small, standalone Haar
        # hits are its most common false-positive mode (signage, foliage, table
        # objects). Keep them when another detector corroborates the same head;
        # larger standalone hits remain available for recall.
        min_edge = min(size)
        filtered = []
        for group in groups:
            raw_w, raw_h = group["box"][2:]
            tiny_standalone_haar = (
                group["source"] == "haar"
                and len(group["sources"]) == 1
                and max(raw_w, raw_h) < max(22, min_edge * 0.022)
            )
            degenerate_standalone_body = (
                group["source"] == "mediapipe-person"
                and len(group["sources"]) == 1
                and min(raw_w, raw_h) < max(8, min_edge * 0.006)
            )
            if not tiny_standalone_haar and not degenerate_standalone_body:
                filtered.append((
                    group["box"], group["head_box"], group["confidence"],
                    group["source"], group["sources"],
                ))
        return filtered

    def _dedupe(self, detections: list[tuple[list[int], float, str]]) -> list[tuple[list[int], float, str]]:
        kept: list[tuple[list[int], float, str]] = []
        for item in sorted(detections, key=lambda value: value[1], reverse=True):
            distinct = True
            for existing in kept:
                iou, containment = self._overlap(item[0], existing[0])
                if iou >= 0.32 or containment >= 0.58:
                    distinct = False
                    break
            if distinct:
                kept.append(item)
        return kept

    def detect(
        self,
        image: Image.Image,
        *,
        include_tiles: bool | None = None,
        timings: dict | None = None,
        include_pose_data: bool = False,
    ) -> list[dict]:
        started = time.perf_counter()
        rgb = np.asarray(image.convert("RGB"))
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        del rgb
        height, width = bgr.shape[:2]
        detections: list[tuple[list[int], float, str]] = []
        landmarks_by_box: dict[tuple[int, int, int, int], dict[str, list[float]]] = {}
        self._ensure_yunet()
        self._ensure_person_model()
        models_ready = time.perf_counter()

        # The global pass covers large people; overlapping tiles recover small
        # people in group photos without sending the image anywhere.
        person_candidates = self._person_infer(bgr)
        use_tiles = DETECTION_ENABLE_TILES if include_tiles is None else include_tiles
        if use_tiles and max(width, height) >= DETECTION_TILE_MIN_EDGE:
            overlap = 0.14
            tile_width, tile_height = round(width * (0.5 + overlap / 2)), round(height * (0.5 + overlap / 2))
            for top in (0, height - tile_height):
                for left in (0, width - tile_width):
                    person_candidates.extend(self._person_infer(bgr[top:top + tile_height, left:left + tile_width], (left, top)))
        person_done = time.perf_counter()
        for box, confidence, source in person_candidates:
            detections.append((clamp_box(box, image.size), confidence, source))

        if self._yunet is not None:
            max_dim = max(width, height)
            scale = min(2.0, max(1.0, 1400 / max_dim))
            scaled = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC) if scale > 1.01 else bgr
            sh, sw = scaled.shape[:2]
            self._yunet.setInputSize((sw, sh))
            _, faces = self._yunet.detect(scaled)
            if faces is not None:
                for face in faces:
                    x, y, w, h = [int(round(value / scale)) for value in face[:4]]
                    box = clamp_box([x, y, w, h], image.size)
                    detections.append((box, float(face[-1]), "yunet"))
                    if include_pose_data:
                        raw = [(float(face[index] / scale), float(face[index + 1] / scale)) for index in range(4, 14, 2)]
                        eyes = sorted(raw[:2], key=lambda point: point[0])
                        mouths = sorted(raw[3:5], key=lambda point: point[0])
                        landmarks_by_box[tuple(box)] = {
                            "left_eye": list(eyes[0]),
                            "right_eye": list(eyes[1]),
                            "nose": list(raw[2]),
                            "mouth_left": list(mouths[0]),
                            "mouth_right": list(mouths[1]),
                            "mouth": [(mouths[0][0] + mouths[1][0]) / 2, (mouths[0][1] + mouths[1][1]) / 2],
                        }
        yunet_done = time.perf_counter()

        # Haar contributes a second view and keeps the prototype usable offline.
        haar_scale = min(1.0, 1100 / max(width, height))
        haar_source = cv2.resize(bgr, None, fx=haar_scale, fy=haar_scale, interpolation=cv2.INTER_AREA) if haar_scale < 0.999 else bgr
        gray = cv2.cvtColor(haar_source, cv2.COLOR_BGR2GRAY)
        min_side = max(18, int(min(gray.shape[:2]) * 0.018))
        if not self._haar.empty():
            for x, y, w, h in self._haar.detectMultiScale(gray, 1.08, 4, minSize=(min_side, min_side)):
                detections.append((clamp_box([
                    round(x / haar_scale), round(y / haar_scale), round(w / haar_scale), round(h / haar_scale),
                ], image.size), 0.52, "haar"))
        haar_done = time.perf_counter()

        results = []
        for face_box, head_box, confidence, source, sources in self._fuse_candidates(detections, image.size):
            result = {
                "id": f"person-{uuid.uuid4().hex[:8]}",
                "box": face_box,
                "head_box": head_box,
                "confidence": round(confidence, 4),
                "source": source,
                "sources": sources,
                "support_count": len(sources),
                "selected": True,
            }
            if include_pose_data and tuple(face_box) in landmarks_by_box:
                result["face_landmarks"] = landmarks_by_box[tuple(face_box)]
            results.append(result)
        if timings is not None:
            timings.update({
                "models_ms": round((models_ready - started) * 1000),
                "person_ms": round((person_done - models_ready) * 1000),
                "yunet_ms": round((yunet_done - person_done) * 1000),
                "haar_ms": round((haar_done - yunet_done) * 1000),
                "total_ms": round((time.perf_counter() - started) * 1000),
                "tiles": bool(use_tiles and max(width, height) >= DETECTION_TILE_MIN_EDGE),
            })
        return results


detector = FaceDetector()
