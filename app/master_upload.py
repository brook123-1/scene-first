from __future__ import annotations

import json
import logging
import os
import threading
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps

from .config import MASTER_STATE_DIR, MASTER_TEMP_DIR, MAX_IMAGE_PIXELS, PREVIEW_DIR, UPLOAD_DIR
from .image_ops import save_clean_atomic


logger = logging.getLogger("scene_first.master")
PROCESS_STARTED_AT = time.time()
INSTANCE_ID = uuid.uuid4().hex[:12]


def peak_rss_kb() -> int | None:
    try:
        import resource
        return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except (ImportError, AttributeError):
        try:
            import psutil
            return int(psutil.Process().memory_info().rss / 1024)
        except (ImportError, OSError):
            if os.name != "nt":
                return None
            try:
                import ctypes
                from ctypes import wintypes

                class ProcessMemoryCounters(ctypes.Structure):
                    _fields_ = [
                        ("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
                    ]

                counters = ProcessMemoryCounters()
                counters.cb = ctypes.sizeof(counters)
                ctypes.windll.kernel32.GetCurrentProcess.restype = wintypes.HANDLE
                ctypes.windll.psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD]
                ctypes.windll.psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
                handle = ctypes.windll.kernel32.GetCurrentProcess()
                ok = ctypes.windll.psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb)
                return int(counters.PeakWorkingSetSize / 1024) if ok else None
            except (AttributeError, OSError, ctypes.ArgumentError):
                return None


def _now_ms() -> int:
    return int(time.time() * 1000)


class MasterUploadStore:
    """Durable master preparation states with one decoder per image id."""

    def __init__(self) -> None:
        self._guard = threading.Lock()
        self._futures: dict[str, Future[dict[str, Any]]] = {}
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="master")

    def _state_path(self, image_id: str) -> Path:
        return MASTER_STATE_DIR / f"{image_id}.json"

    def temp_path(self, image_id: str) -> Path:
        return MASTER_TEMP_DIR / f"{image_id}-{uuid.uuid4().hex}.part"

    def _write(self, image_id: str, value: dict[str, Any]) -> dict[str, Any]:
        value = {**value, "image_id": image_id, "updated_at_ms": _now_ms()}
        destination = self._state_path(image_id)
        temporary = destination.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(value, ensure_ascii=False), "utf8")
        os.replace(temporary, destination)
        return value

    def status(self, image_id: str) -> dict[str, Any]:
        path = self._state_path(image_id)
        if path.is_file():
            try:
                return json.loads(path.read_text("utf8"))
            except (json.JSONDecodeError, OSError):
                pass
        master = UPLOAD_DIR / f"{image_id}.png"
        if master.is_file():
            # Backwards compatibility for masters created before state files.
            with Image.open(master) as image:
                return {
                    "image_id": image_id,
                    "status": "ready",
                    "width": image.width,
                    "height": image.height,
                    "image_url": f"/media/uploads/{image_id}.png",
                    "preview_url": f"/media/previews/{image_id}-before.jpg",
                    "reused": True,
                }
        return {"image_id": image_id, "status": "missing"}

    def begin_upload(self, image_id: str) -> tuple[dict[str, Any], bool]:
        with self._guard:
            current = self.status(image_id)
            if current["status"] in {"ready", "uploading", "processing"}:
                return current, False
            return self._write(image_id, {
                "status": "uploading",
                "instance_id": INSTANCE_ID,
                "process_uptime_ms": round((time.time() - PROCESS_STARTED_AT) * 1000),
            }), True

    def fail(self, image_id: str, code: str, phase: str) -> dict[str, Any]:
        with self._guard:
            state = self._write(image_id, {
                "status": "failed",
                "error_code": code,
                "failure_phase": phase,
                "instance_id": INSTANCE_ID,
                "peak_rss_kb": peak_rss_kb(),
            })
        logger.warning("master_failed %s", json.dumps(state, ensure_ascii=False))
        return state

    def submit(self, image_id: str, temporary: Path, upload: dict[str, Any]) -> Future[dict[str, Any]]:
        with self._guard:
            existing = self._futures.get(image_id)
            if existing and not existing.done():
                temporary.unlink(missing_ok=True)
                return existing
            self._write(image_id, {
                "status": "processing",
                "bytes": upload["bytes"],
                "receive_ms": upload["receive_ms"],
                "format_hint": upload.get("format_hint", "unknown"),
                "instance_id": INSTANCE_ID,
                "process_uptime_ms": round((time.time() - PROCESS_STARTED_AT) * 1000),
            })
            future = self._executor.submit(self._process, image_id, temporary, upload)
            self._futures[image_id] = future
            return future

    def _process(self, image_id: str, temporary: Path, upload: dict[str, Any]) -> dict[str, Any]:
        started = time.perf_counter()
        timings: dict[str, int] = {"receive_ms": int(upload["receive_ms"])}
        phase = "decode"
        try:
            step = time.perf_counter()
            with Image.open(temporary) as opened:
                image_format = (opened.format or "unknown").lower()
                width, height = opened.size
                if width * height > MAX_IMAGE_PIXELS:
                    raise ValueError(f"image_pixels_exceeded:{width}x{height}:{MAX_IMAGE_PIXELS}")
                opened.load()
                timings["decode_ms"] = round((time.perf_counter() - step) * 1000)

                phase = "orientation"
                step = time.perf_counter()
                image = ImageOps.exif_transpose(opened).convert("RGB")
                image.load()
                timings["orientation_ms"] = round((time.perf_counter() - step) * 1000)

            phase = "master_encode"
            step = time.perf_counter()
            master_path = UPLOAD_DIR / f"{image_id}.png"
            save_clean_atomic(image, master_path, "PNG")
            timings["master_encode_ms"] = round((time.perf_counter() - step) * 1000)

            phase = "preview"
            step = time.perf_counter()
            preview = image.copy()
            preview.thumbnail((2048, 2048), Image.Resampling.LANCZOS)
            save_clean_atomic(preview, PREVIEW_DIR / f"{image_id}-before.jpg", "JPEG", 86)
            timings["preview_ms"] = round((time.perf_counter() - step) * 1000)
            timings["processing_total_ms"] = round((time.perf_counter() - started) * 1000)
            timings["server_total_ms"] = timings["receive_ms"] + timings["processing_total_ms"]

            with self._guard:
                state = self._write(image_id, {
                    "status": "ready",
                    "width": width,
                    "height": height,
                    "pixels": width * height,
                    "bytes": upload["bytes"],
                    "format": image_format,
                    "image_url": f"/media/uploads/{image_id}.png",
                    "preview_url": f"/media/previews/{image_id}-before.jpg",
                    "timings": timings,
                    "peak_rss_kb": peak_rss_kb(),
                    "instance_id": INSTANCE_ID,
                    "reused": False,
                })
            logger.info("master_metrics %s", json.dumps(state, ensure_ascii=False))
            return state
        except ValueError as exc:
            code = "pixel_limit" if str(exc).startswith("image_pixels_exceeded:") else "unsupported_format"
            return self.fail(image_id, code, phase)
        except Exception as exc:
            logger.exception("master processing failed image_id=%s phase=%s type=%s", image_id, phase, type(exc).__name__)
            return self.fail(image_id, "server_processing", phase)
        finally:
            temporary.unlink(missing_ok=True)


master_upload_store = MasterUploadStore()
