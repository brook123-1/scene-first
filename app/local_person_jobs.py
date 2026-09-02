from __future__ import annotations

import hashlib
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

from .config import LOCAL_PERSON_JOB_DIR, LOCAL_PERSON_OUTPUT_DIR, PROVIDER_SETTINGS
from .cost_ledger import cost_ledger
from .image_ops import head_mask, load_image_path, save_clean_atomic
from .providers import PROVIDERS, prompt_for_profile


class LocalPersonJobStore:
    """Crop-only generation store; it never receives an authoritative scene."""

    def __init__(self) -> None:
        self.jobs: dict[str, dict] = {}
        self.idempotency: dict[str, str] = {}
        self.lock = threading.Lock()
        self.executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="local-person")

    def _save(self, job: dict) -> None:
        with self.lock:
            self.jobs[job["id"]] = job.copy()
            destination = LOCAL_PERSON_JOB_DIR / f"{job['id']}.json"
            temporary = destination.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(job, ensure_ascii=False, indent=2), "utf8")
            temporary.replace(destination)

    def get(self, job_id: str) -> dict | None:
        with self.lock:
            if job_id in self.jobs:
                return self.jobs[job_id].copy()
        path = LOCAL_PERSON_JOB_DIR / f"{job_id}.json"
        return json.loads(path.read_text("utf8")) if path.is_file() else None

    def create(self, crop_path: Path, metadata: dict, provider: str) -> tuple[dict, bool]:
        key_material = json.dumps({
            "image_id": metadata["image_id"], "person_id": metadata["person_id"],
            "head_box": metadata["head_box"], "crop_box": metadata["crop_box"],
            "retry_nonce": metadata.get("retry_nonce", 0), "provider": provider,
            "prompt_profile": metadata.get("prompt_profile", "balanced_portrait"),
        }, sort_keys=True, separators=(",", ":"))
        key = hashlib.sha256(key_material.encode()).hexdigest()
        with self.lock:
            existing_id = self.idempotency.get(key)
            if existing_id:
                crop_path.unlink(missing_ok=True)
                return self.jobs[existing_id].copy(), True
            job_id = hashlib.sha256((key + ":scene-first").encode()).hexdigest()[:32]
            persisted = LOCAL_PERSON_JOB_DIR / f"{job_id}.json"
            if persisted.is_file():
                job = json.loads(persisted.read_text("utf8"))
                self.jobs[job_id] = job
                self.idempotency[key] = job_id
                crop_path.unlink(missing_ok=True)
                return job.copy(), True
            job = {
                "id": job_id, "status": "queued", "person_id": metadata["person_id"],
                "provider": provider, "created_at": time.time(), "retry_nonce": metadata.get("retry_nonce", 0),
                "queue_started_at": time.time(),
            }
            self.jobs[job_id] = job.copy()
            self.idempotency[key] = job_id
        self._save(job)
        self.executor.submit(self._run, job_id, crop_path, metadata, provider)
        return job, False

    def _run(self, job_id: str, crop_path: Path, metadata: dict, provider: str) -> None:
        job = self.get(job_id) or {"id": job_id, "person_id": metadata["person_id"]}
        started = time.perf_counter()
        try:
            job.update({
                "status": "processing",
                "queue_ms": round((time.time() - job.get("queue_started_at", time.time())) * 1000),
            })
            self._save(job)
            decode_started = time.perf_counter()
            crop = load_image_path(crop_path, max_pixels=4_000_000)
            job["local_decode_ms"] = round((time.perf_counter() - decode_started) * 1000)
            relative_head = metadata["head_box_in_crop"]
            mask = head_mask(crop.size, relative_head)
            provider_started = time.perf_counter()
            generated = PROVIDERS[provider].edit(
                crop, mask, subject_id=metadata["person_id"],
                retry_nonce=metadata.get("retry_nonce", 0),
                prompt=prompt_for_profile(metadata.get("prompt_profile", "balanced_portrait")),
            )
            job["provider_ms"] = round((time.perf_counter() - provider_started) * 1000)
            output = LOCAL_PERSON_OUTPUT_DIR / f"{job_id}.png"
            save_clean_atomic(generated, output, "PNG")
            job.update({
                "status": "completed", "output_url": f"/media/local-person/{job_id}.png",
                "elapsed_ms": round((time.perf_counter() - started) * 1000),
                "fallback": False,
            })
            audit = {
                "id": metadata["person_id"], "actual_mode": "anime", "provider": provider,
                "provider_model": PROVIDER_SETTINGS[provider].model, "attempted_provider": provider,
                "attempted_model": PROVIDER_SETTINGS[provider].model, "generation_attempted": True,
                "requested_mode": "anime", "fallback_reason": None,
                "elapsed_ms": job["elapsed_ms"], "retry_nonce": metadata.get("retry_nonce", 0),
                "cloud_scope": "crop-only-local-master",
            }
            cost_ledger.record_attempt(
                job_id=job_id, image_id=metadata["image_id"], audit=audit, generation_attempted=True,
            )
        except Exception as exc:
            job.update({
                "status": "failed", "error_code": "generation_failed",
                "error": str(exc)[:240], "elapsed_ms": round((time.perf_counter() - started) * 1000),
                "fallback": True,
            })
        finally:
            crop_path.unlink(missing_ok=True)
            self._save(job)


local_person_job_store = LocalPersonJobStore()
