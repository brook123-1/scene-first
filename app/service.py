from __future__ import annotations

import json
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from PIL import Image

from .config import (
    AVATAR_ASSET_DIR,
    JOB_DIR,
    LOW_CONFIDENCE_THRESHOLD,
    MAX_PARALLEL_PERSON_EDITS,
    OUTPUT_DIR,
    PREVIEW_DIR,
    POSE_AVATAR_OVERLAY_ENABLED,
    PROVIDER_SETTINGS,
    UPLOAD_DIR,
)
from .cost_ledger import cost_ledger
from .image_ops import (
    blur_region,
    clamp_box,
    composite_patch,
    head_mask,
    load_image_path,
    outside_mask_is_exact,
    padded_crop_box,
    save_clean,
    save_preview,
    safe_cover_patch,
)
from .providers import PROVIDERS, prompt_for_profile
from .schemas import EditRequest
from .pose_avatar.adapters import detected_head_from_region
from .pose_avatar.registry import AssetRegistry
from .pose_avatar.renderer import PoseAvatarRenderer


_pose_registry: AssetRegistry | None = None
_pose_registry_lock = threading.Lock()


def pose_avatar_renderer() -> PoseAvatarRenderer:
    global _pose_registry
    if _pose_registry is None:
        with _pose_registry_lock:
            if _pose_registry is None:
                _pose_registry = AssetRegistry(AVATAR_ASSET_DIR)
    return PoseAvatarRenderer(_pose_registry)


class JobStore:
    def __init__(self) -> None:
        self.jobs: dict[str, dict] = {}
        self.lock = threading.Lock()
        self.executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="privacy-edit")

    def create(self, request: EditRequest) -> dict:
        job_id = uuid.uuid4().hex
        job = {
            "id": job_id,
            "status": "queued",
            "progress": 0,
            "created_at": time.time(),
            "request": request.model_dump(),
            "people": [],
        }
        self._save(job)
        self.executor.submit(self._run, job_id, request)
        return job

    def get(self, job_id: str) -> dict | None:
        with self.lock:
            if job_id in self.jobs:
                return self.jobs[job_id].copy()
        path = JOB_DIR / f"{job_id}.json"
        return json.loads(path.read_text("utf8")) if path.exists() else None

    def _save(self, job: dict) -> None:
        with self.lock:
            self.jobs[job["id"]] = job.copy()
            (JOB_DIR / f"{job['id']}.json").write_text(json.dumps(job, ensure_ascii=False, indent=2), "utf8")

    def _run(self, job_id: str, request: EditRequest) -> None:
        job = self.get(job_id)
        if not job:
            return
        started = time.perf_counter()
        try:
            job["status"] = "processing"
            self._save(job)
            source_path = UPLOAD_DIR / f"{request.image_id}.png"
            if not source_path.exists():
                raise FileNotFoundError("Original image does not exist or was cleaned up.")
            original = load_image_path(source_path)
            working = original.copy()
            selected = [region for region in request.regions if region.selected]
            total = max(1, len(selected))

            # Experimental P0 path: it is reachable only through two explicit
            # opt-ins and never changes the provider or Local Master defaults.
            # The already-decoded local master is passed directly to the local
            # renderer, so this branch does not decode or upload the image again.
            if POSE_AVATAR_OVERLAY_ENABLED and request.pose_aware_overlay and request.provider == "local":
                heads = [detected_head_from_region(region.model_dump(), original.size) for region in selected]
                rendered = pose_avatar_renderer().render(original, heads, image_id=request.image_id)
                output_path = OUTPUT_DIR / f"{job_id}.png"
                save_clean(rendered.image, output_path, "PNG")
                save_preview(rendered.image, PREVIEW_DIR / f"{job_id}-after.jpg")
                job.update({
                    "status": "completed",
                    "progress": 100,
                    "output_url": f"/media/outputs/{job_id}.png",
                    "output_preview_url": f"/media/previews/{job_id}-after.jpg",
                    "outside_mask_exact": outside_mask_is_exact(original, rendered.image, rendered.alpha_mask),
                    "estimated_cost_cny": 0.0,
                    "elapsed_ms": round((time.perf_counter() - started) * 1000),
                    "pose_avatar_overlay": True,
                    "people": [trace.model_dump(mode="json") for trace in rendered.traces],
                    "decisions": [decision.model_dump(mode="json") for decision in rendered.decisions],
                    "transforms": [value.model_dump(mode="json") if value else None for value in rendered.transforms],
                })
                self._save(job)
                return
            combined_mask = Image.new("L", original.size, 0)
            prepared: list[dict] = []

            # Each crop is independently produced from the immutable source.
            # This lets two remote image calls run at once; generated patches
            # are still composited later in deterministic screen order.
            for index, region in enumerate(selected):
                head_box = clamp_box(region.head_box or region.box, original.size)
                if request.mask_profile == "neck_blend":
                    mask = head_mask(
                        original.size,
                        head_box,
                        feather=max(5, min(26, int(min(head_box[2], head_box[3]) * 0.065))),
                        neck_width=0.38,
                        neck_start=0.52,
                    )
                else:
                    mask = head_mask(original.size, head_box)
                combined_mask = Image.fromarray(
                    np.maximum(np.asarray(combined_mask), np.asarray(mask)).astype("uint8")
                )
                crop_box = padded_crop_box(head_box, original.size)
                x, y, width, height = crop_box
                prepared.append({
                    "index": index,
                    "region": region,
                    "head_box": head_box,
                    "mask": mask,
                    "crop_box": crop_box,
                    "crop": original.crop((x, y, x + width, y + height)),
                    "local_mask": mask.crop((x, y, x + width, y + height)),
                })

            # Near / large heads are the most useful first visible result.
            # Keep each original index so final compositing remains stable.
            prepared.sort(key=lambda item: item["head_box"][2] * item["head_box"][3], reverse=True)

            parallelism = min(max(1, MAX_PARALLEL_PERSON_EDITS), len(prepared)) if prepared else 1
            job.update({"total_people": len(prepared), "completed_people": 0, "parallelism": parallelism})
            self._save(job)

            def render_person(item: dict) -> dict:
                person_started = time.perf_counter()
                region = item["region"]
                requested_mode = region.mode or request.mode
                actual_mode = requested_mode
                fallback_reason = None
                generation_attempted = False
                provider_name = request.provider
                provider_model = PROVIDER_SETTINGS[provider_name].model
                attempted_provider = provider_name
                attempted_model = provider_model
                patch = None

                if requested_mode == "safe" or (
                    not request.selection_confirmed and region.confidence < LOW_CONFIDENCE_THRESHOLD
                ):
                    if not request.selection_confirmed and region.confidence < LOW_CONFIDENCE_THRESHOLD and requested_mode != "safe":
                        fallback_reason = "Low detection confidence: used the safe fallback."
                    actual_mode = "safe"
                    provider_name = "local"
                    provider_model = PROVIDER_SETTINGS[provider_name].model
                    try:
                        patch = safe_cover_patch(
                            item["crop"], item["crop_box"], item["head_box"], request.safe_cover_id
                        )
                    except Exception as exc:
                        fallback_reason = (
                            f"Safe-cover asset unavailable; blur fallback used: {str(exc)[:180]}"
                        )
                else:
                    try:
                        generation_attempted = True
                        edit_kwargs = {"subject_id": region.id, "retry_nonce": request.retry_nonce}
                        if request.prompt_profile == "balanced_painterly":
                            edit_kwargs["prompt"] = prompt_for_profile(request.prompt_profile)
                        if request.cloud_scope == "full" and provider_name != "local":
                            generated = PROVIDERS[provider_name].edit(original, item["mask"], **edit_kwargs)
                            if generated.size != original.size:
                                generated = generated.resize(original.size, Image.Resampling.LANCZOS)
                            x, y, width, height = item["crop_box"]
                            patch = generated.crop((x, y, x + width, y + height))
                        else:
                            patch = PROVIDERS[provider_name].edit(item["crop"], item["local_mask"], **edit_kwargs)
                    except Exception as exc:
                        fallback_reason = f"Generation failed; safe fallback used: {str(exc)[:240]}"
                        actual_mode = "safe"
                        provider_name = "local"
                        provider_model = PROVIDER_SETTINGS[provider_name].model

                provider_elapsed_ms = round((time.perf_counter() - person_started) * 1000)
                audit = {
                    "id": region.id,
                    "requested_mode": requested_mode,
                    "actual_mode": actual_mode,
                    "provider": provider_name,
                    "provider_model": provider_model,
                    "attempted_provider": attempted_provider,
                    "attempted_model": attempted_model,
                    "cloud_scope": request.cloud_scope if provider_name != "local" else "local",
                    "prompt_profile": request.prompt_profile,
                    "mask_profile": request.mask_profile,
                    "safe_cover_id": request.safe_cover_id if actual_mode == "safe" else None,
                    "fallback_reason": fallback_reason,
                    "generation_attempted": generation_attempted,
                    "retry_nonce": request.retry_nonce,
                    "elapsed_ms": provider_elapsed_ms,
                    "provider_ms": provider_elapsed_ms if generation_attempted else 0,
                }
                return {**item, "patch": patch, "audit": audit}

            rendered: dict[int, dict] = {}
            if prepared:
                with ThreadPoolExecutor(max_workers=parallelism, thread_name_prefix="person-edit") as renderer:
                    futures = [renderer.submit(render_person, item) for item in prepared]
                    for future in as_completed(futures):
                        result = future.result()
                        rendered[result["index"]] = result
                        job["completed_people"] = len(rendered)
                        job["people"] = [rendered[key]["audit"] for key in sorted(rendered)]
                        cost_ledger.record_attempt(
                            job_id=job_id,
                            image_id=request.image_id,
                            audit=result["audit"],
                            generation_attempted=result["audit"]["generation_attempted"],
                        )
                        job["progress"] = round(len(rendered) / total * 100)
                        self._save(job)

            total_cost = 0.0
            for index in sorted(rendered):
                result = rendered[index]
                audit = result["audit"]
                if result["patch"] is None:
                    working = blur_region(working, result["crop_box"], result["mask"])
                else:
                    working = composite_patch(working, result["patch"], result["crop_box"], result["mask"])
                    total_cost += PROVIDER_SETTINGS[audit["provider"]].estimated_cny
            job["people"] = [rendered[key]["audit"] for key in sorted(rendered)]

            output_path = OUTPUT_DIR / f"{job_id}.png"
            save_clean(working, output_path, "PNG")
            save_preview(working, PREVIEW_DIR / f"{job_id}-after.jpg")
            exact = outside_mask_is_exact(original, working, combined_mask)
            job.update({
                "status": "completed",
                "progress": 100,
                "output_url": f"/media/outputs/{job_id}.png",
                "output_preview_url": f"/media/previews/{job_id}-after.jpg",
                "outside_mask_exact": exact,
                "estimated_cost_cny": round(total_cost, 4),
                "elapsed_ms": round((time.perf_counter() - started) * 1000),
            })
        except Exception as exc:
            job.update({"status": "failed", "error": str(exc), "progress": 100})
        self._save(job)


job_store = JobStore()
