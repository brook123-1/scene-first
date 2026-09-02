from __future__ import annotations

import json
import logging
import re
import time
import uuid
import asyncio
from base64 import b64encode
from dataclasses import replace
from pathlib import Path
from secrets import compare_digest

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from .config import (
    BENCHMARK_DIR,
    AVATAR_ASSET_DIR,
    COMPARISON_DIR,
    PROMPT_COMPARISON_DIR,
    ENV,
    FAL_EDIT_MODELS,
    JOB_DIR,
    LOCAL_MASTER_ENABLED,
    LOCAL_PERSON_OUTPUT_DIR,
    MASTER_TEMP_DIR,
    MAX_UPLOAD_BYTES,
    MAX_IMAGE_PIXELS,
    MAX_DETECTION_PIXELS,
    OUTPUT_DIR,
    PUBLIC_MODE,
    PREVIEW_DIR,
    POSE_AVATAR_OVERLAY_ENABLED,
    POSE_AVATAR_PLAYGROUND_DIR,
    PROVIDER_SETTINGS,
    ROOT,
    STATIC_DIR,
    UPLOAD_DIR,
    fal_model_label,
)
from .detector import detector
from .image_ops import load_image, load_image_path, save_clean, save_preview
from .master_upload import master_upload_store
from .local_person_jobs import local_person_job_store
from .schemas import EditRequest, ExportRequest, PreflightSave, ProviderConfigUpdate, PublishabilityUpdate, ReviewRating
from .cost_ledger import cost_ledger
from .service import job_store
from .pose_avatar.adapters import enrich_detection
from .pose_avatar.playground import PlaygroundRenderRequest, PoseAvatarPlayground
from .pose_avatar.registry import AssetValidationError


app = FastAPI(
    title="Scene First Privacy Lab",
    version="0.1.0",
    docs_url=None if PUBLIC_MODE else "/docs",
    redoc_url=None if PUBLIC_MODE else "/redoc",
    openapi_url=None if PUBLIC_MODE else "/openapi.json",
)
logger = logging.getLogger("scene_first.detect")
pose_avatar_playground = PoseAvatarPlayground(
    root=ROOT,
    local_root=POSE_AVATAR_PLAYGROUND_DIR,
    asset_root=AVATAR_ASSET_DIR,
)


def _peak_rss_kb() -> int | None:
    try:
        import resource
        return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except (ImportError, AttributeError):
        return None
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
TEST_TUNNEL_PASSWORD = ENV.get("SCENE_FIRST_TEST_PASSWORD", "").strip()
TEST_ACCESS_COOKIE = "scene_first_test_access"


def _require_lab_access() -> None:
    """Keep prototype administration and evaluation data off public builds."""
    if PUBLIC_MODE:
        raise HTTPException(404)


@app.middleware("http")
async def local_development_no_cache(request: Request, call_next):
    """Avoid Chrome retaining an old local UI after a prototype update."""
    # A temporary Cloudflare tunnel adds this header.  Keep local development
    # frictionless, but prevent a random public tunnel URL from exposing the
    # settings screen or private test material without a shared test password.
    if TEST_TUNNEL_PASSWORD and request.headers.get("cf-connecting-ip"):
        expected = "Basic " + b64encode(f"scene:{TEST_TUNNEL_PASSWORD}".encode()).decode()
        basic_authorized = compare_digest(request.headers.get("authorization", ""), expected)
        cookie_authorized = request.cookies.get(TEST_ACCESS_COOKIE) == "granted"
        if not basic_authorized and not cookie_authorized:
            return PlainTextResponse(
                "This private prototype requires the test access code.",
                status_code=401,
                headers={"WWW-Authenticate": 'Basic realm="Scene First test"'},
            )
    response = await call_next(request)
    # Browser <img> requests do not consistently resend the Basic header on
    # mobile.  After the first successful challenge, a same-site HTTPS cookie
    # keeps preview and generated-image URLs accessible as well.
    if TEST_TUNNEL_PASSWORD and request.headers.get("cf-connecting-ip") and basic_authorized:
        response.set_cookie(
            TEST_ACCESS_COOKIE,
            "granted",
            max_age=8 * 60 * 60,
            httponly=True,
            secure=True,
            samesite="strict",
        )
    if request.url.path in {"/", "/settings", "/review", "/comparison", "/prompt-comparison", "/preflight", "/batch-review", "/costs", "/pose-avatar-playground"} or request.url.path.startswith("/static/"):
        response.headers["Cache-Control"] = "no-store, max-age=0"
    return response


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/sw.js", include_in_schema=False)
def service_worker():
    return FileResponse(
        STATIC_DIR / "sw.js",
        media_type="application/javascript",
        headers={"Cache-Control": "no-cache", "Service-Worker-Allowed": "/"},
    )


@app.get("/review", include_in_schema=False)
def review_page():
    _require_lab_access()
    return FileResponse(STATIC_DIR / "review.html")


@app.get("/comparison", include_in_schema=False)
def comparison_page():
    _require_lab_access()
    return FileResponse(STATIC_DIR / "comparison.html")


@app.get("/prompt-comparison", include_in_schema=False)
def prompt_comparison_page():
    _require_lab_access()
    return FileResponse(STATIC_DIR / "prompt-comparison.html")


@app.get("/batch-review", include_in_schema=False)
def batch_review_page():
    _require_lab_access()
    return FileResponse(STATIC_DIR / "batch-review.html")


@app.get("/preflight", include_in_schema=False)
def preflight_page():
    _require_lab_access()
    return FileResponse(STATIC_DIR / "preflight.html")


@app.get("/settings", include_in_schema=False)
def settings_page():
    _require_lab_access()
    return FileResponse(STATIC_DIR / "settings.html")


@app.get("/costs", include_in_schema=False)
def costs_page():
    _require_lab_access()
    return FileResponse(STATIC_DIR / "costs.html")


@app.get("/pose-avatar-playground", include_in_schema=False)
def pose_avatar_playground_page():
    _require_lab_access()
    return FileResponse(STATIC_DIR / "pose-avatar-playground.html")


@app.get("/api/pose-avatar-playground")
def pose_avatar_playground_items():
    _require_lab_access()
    return pose_avatar_playground.list_payload()


@app.get("/api/pose-avatar-playground/cases/{case_id}/original", include_in_schema=False)
def pose_avatar_playground_original(case_id: str):
    _require_lab_access()
    try:
        path = pose_avatar_playground.original_path(case_id)
    except (KeyError, FileNotFoundError):
        raise HTTPException(404, "Playground case or local source image not found")
    return FileResponse(path, headers={"Cache-Control": "no-store"})


@app.post("/api/pose-avatar-playground/render")
def pose_avatar_playground_render(request: PlaygroundRenderRequest):
    _require_lab_access()
    try:
        return pose_avatar_playground.render(request)
    except (KeyError, FileNotFoundError) as exc:
        raise HTTPException(404, "Playground case data not found") from exc
    except (AssetValidationError, ValueError) as exc:
        raise HTTPException(422, str(exc)) from exc


@app.get("/api/health")
def health():
    return {"ok": True, "version": app.version, "privacy_notice": "原型不构成绝对匿名保证。"}


@app.get("/api/providers")
def providers():
    return {
        "providers": [
            {
                "id": settings.name,
                "label": settings.label,
                "model": settings.model,
                "configured": settings.configured,
                "supports_mask": settings.supports_mask,
                "estimated_cny": settings.estimated_cny,
                "model_options": [
                    {"id": endpoint, "label": profile["label"], "description": profile["description"]}
                    for endpoint, profile in FAL_EDIT_MODELS.items()
                ] if settings.name == "fal" else [],
            }
            for settings in PROVIDER_SETTINGS.values()
        ]
    }


@app.get("/api/features")
def features():
    return {
        "local_master": LOCAL_MASTER_ENABLED,
        "local_master_mode": "crop-only" if LOCAL_MASTER_ENABLED else "traditional-master",
        "traditional_master_fallback": True,
        "max_parallel_people": 2,
        "pose_avatar_overlay": POSE_AVATAR_OVERLAY_ENABLED,
        "pose_avatar_overlay_default": False,
    }


def _write_local_env(key: str, value: str) -> None:
    """Update one local config entry without reading it back to a client."""
    path = ROOT / ".env.local"
    lines = path.read_text("utf8").splitlines() if path.exists() else []
    pattern = re.compile(rf"^{re.escape(key)}=")
    replacement = f"{key}={value}"
    for index, line in enumerate(lines):
        if pattern.match(line):
            lines[index] = replacement
            break
    else:
        lines.append(replacement)
    path.write_text("\n".join(lines) + "\n", "utf8")


@app.post("/api/settings/provider")
def update_provider_settings(value: ProviderConfigUpdate):
    _require_lab_access()
    settings = PROVIDER_SETTINGS[value.provider]
    if value.provider == "fal" and value.model and value.model.strip() not in FAL_EDIT_MODELS:
        raise HTTPException(400, "该 fal.ai 端点尚未经过本地隐私合成兼容性验证。")
    if value.provider == "ark" and value.model and value.model.strip() != "doubao-seedream-5.0-lite":
        raise HTTPException(400, "当前 Agent Plan 配置卡仅支持套餐内的 Seedream 5 Lite。")
    if value.api_key:
        if not settings.key_name:
            raise HTTPException(400, "该路线不需要 API Key。")
        _write_local_env(settings.key_name, value.api_key.strip())
        ENV[settings.key_name] = value.api_key.strip()
    if value.model:
        model_key = f"{value.provider.upper()}_IMAGE_MODEL"
        _write_local_env(model_key, value.model.strip())
        ENV[model_key] = value.model.strip()
        changes = {"model": value.model.strip()}
        if value.provider == "fal":
            changes["label"] = fal_model_label(value.model.strip())
        PROVIDER_SETTINGS[value.provider] = replace(settings, **changes)
    current = PROVIDER_SETTINGS[value.provider]
    return {
        "saved": True, "provider": current.name, "configured": current.configured,
        "model": current.model, "label": current.label,
    }


@app.post("/api/detect")
async def detect(file: UploadFile = File(...)):
    started = time.perf_counter()
    data = await file.read()
    if not data:
        raise HTTPException(400, "文件为空。")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "文件超过 30MB。")
    try:
        image = load_image(data, max_pixels=MAX_DETECTION_PIXELS)
    except ValueError as exc:
        if str(exc).startswith("image_pixels_exceeded:"):
            raise HTTPException(413, "检查副本像素过大；请将最长边缩至1600像素后重试。") from exc
        raise HTTPException(415, "无法读取图片；请使用 JPEG、PNG、WebP 或 HEIC。") from exc
    except Exception as exc:
        raise HTTPException(415, "无法读取图片；请使用 JPEG、PNG、WebP 或 HEIC。") from exc
    image_id = uuid.uuid4().hex
    timings = {"decode_ms": round((time.perf_counter() - started) * 1000)}
    detections = detector.detect(image, timings=timings, include_pose_data=POSE_AVATAR_OVERLAY_ENABLED)
    if POSE_AVATAR_OVERLAY_ENABLED:
        detections = [enrich_detection(value, image.size) for value in detections]
    timings["request_total_ms"] = round((time.perf_counter() - started) * 1000)
    timings["peak_rss_kb"] = _peak_rss_kb()
    logger.info(
        "detection_metrics %s",
        json.dumps({
            "image_id": image_id,
            "width": image.width,
            "height": image.height,
            "detections": len(detections),
            **timings,
        }, ensure_ascii=False),
    )
    return {
        "image_id": image_id,
        "filename": file.filename or "image",
        "width": image.width,
        "height": image.height,
        "image_url": None,
        "preview_url": None,
        "detections": detections,
        "detection_timings": timings,
        "warnings": [
            "自动检测可能漏掉远处、侧脸或严重遮挡的人物，请放大检查并手动补框。"
        ] if detections else ["没有自动检测到人脸；若画面有人，请使用手动补框。"],
    }


@app.post("/api/images/{image_id}/master")
async def upload_master(image_id: str, file: UploadFile = File(...)):
    if not re.fullmatch(r"[a-f0-9]{32}", image_id):
        raise HTTPException(400, "图片标识无效。")
    current, claimed = master_upload_store.begin_upload(image_id)
    if not claimed:
        return {**current, "reused": current["status"] == "ready"}

    temporary = master_upload_store.temp_path(image_id)
    bytes_received = 0
    receive_started = time.perf_counter()
    try:
        with temporary.open("xb") as handle:
            while chunk := await file.read(1024 * 1024):
                bytes_received += len(chunk)
                if bytes_received > MAX_UPLOAD_BYTES:
                    raise HTTPException(413, "文件超过 30MB。")
                handle.write(chunk)
        if not bytes_received:
            raise HTTPException(400, "文件为空。")
    except HTTPException as exc:
        temporary.unlink(missing_ok=True)
        code = "file_too_large" if exc.status_code == 413 else "empty_file"
        master_upload_store.fail(image_id, code, "receive")
        raise
    except Exception as exc:
        temporary.unlink(missing_ok=True)
        master_upload_store.fail(image_id, "connection_interrupted", "receive")
        raise HTTPException(499, "上传连接中断，请检查网络后重试。") from exc

    future = master_upload_store.submit(image_id, temporary, {
        "bytes": bytes_received,
        "receive_ms": round((time.perf_counter() - receive_started) * 1000),
        "format_hint": (file.content_type or "unknown").split(";", 1)[0][:64],
    })
    try:
        result = await asyncio.shield(asyncio.wrap_future(future))
    except asyncio.CancelledError:
        # Preparation continues after a mobile connection disappears.  The
        # client can recover by querying this image_id instead of uploading twice.
        raise
    if result["status"] == "failed":
        if result.get("error_code") == "pixel_limit":
            raise HTTPException(413, "原图像素过大；当前最多支持约1600万像素。")
        if result.get("error_code") == "unsupported_format":
            raise HTTPException(415, "无法读取图片；请使用 JPEG、PNG、WebP 或 HEIC。")
        raise HTTPException(500, "服务端准备高清工作副本失败，请安全重试。")
    return result


@app.get("/api/images/{image_id}/master")
def master_status(image_id: str):
    if not re.fullmatch(r"[a-f0-9]{32}", image_id):
        raise HTTPException(400, "图片标识无效。")
    return master_upload_store.status(image_id)


@app.post("/api/local-person-jobs", status_code=202)
async def create_local_person_job(
    crop: UploadFile = File(...), metadata: str = Form(...), provider: str = Form(...),
):
    if not LOCAL_MASTER_ENABLED:
        raise HTTPException(404, "Local Master 实验路线未启用。")
    if provider not in PROVIDER_SETTINGS or provider == "local":
        raise HTTPException(400, "Local Master AI 路线无效。")
    if not PROVIDER_SETTINGS[provider].configured:
        raise HTTPException(400, "当前生成服务尚未配置。")
    try:
        value = json.loads(metadata)
        image_id = str(value["image_id"])
        person_id = str(value["person_id"])
        original = [int(number) for number in value["original_size"]]
        crop_box = [int(number) for number in value["crop_box"]]
        head_box = [int(number) for number in value["head_box"]]
        head_in_crop = [int(number) for number in value["head_box_in_crop"]]
        upload_scale = float(value["upload_scale"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise HTTPException(400, "局部任务元数据无效。") from exc
    if not re.fullmatch(r"[a-f0-9]{32}", image_id) or not re.fullmatch(r"[A-Za-z0-9_.:-]{1,96}", person_id):
        raise HTTPException(400, "局部任务标识无效。")
    if value.get("selected") is not True:
        raise HTTPException(400, "只有人工最终确认 selected=true 的人物才可创建局部任务。")
    if len(original) != 2 or len(crop_box) != 4 or len(head_box) != 4 or len(head_in_crop) != 4:
        raise HTTPException(400, "局部任务坐标无效。")
    original_width, original_height = original
    if original_width < 1 or original_height < 1 or original_width * original_height > 16_000_000:
        raise HTTPException(413, "原图尺寸超过 Local Master 当前安全边界。")
    crop_x, crop_y, crop_width, crop_height = crop_box
    if min(crop_x, crop_y) < 0 or min(crop_width, crop_height) < 1 or crop_x + crop_width > original_width or crop_y + crop_height > original_height:
        raise HTTPException(400, "局部裁剪超出原图边界。")
    if crop_width * crop_height > original_width * original_height * 0.55:
        raise HTTPException(413, "局部裁剪意外包含过多场景；服务端已拒绝整图上传。")
    if not 0 < upload_scale <= 1:
        raise HTTPException(400, "局部缩放参数无效。")
    head_x, head_y, head_width, head_height = head_box
    if min(head_x, head_y) < 0 or min(head_width, head_height) < 1 or head_x + head_width > original_width or head_y + head_height > original_height:
        raise HTTPException(400, "头部坐标超出原图边界。")
    expected_head = [
        round((head_x - crop_x) * upload_scale), round((head_y - crop_y) * upload_scale),
        round(head_width * upload_scale), round(head_height * upload_scale),
    ]
    if any(abs(actual - expected) > 2 for actual, expected in zip(head_in_crop, expected_head)):
        raise HTTPException(400, "头部、裁剪与局部坐标不一致。")

    temporary = MASTER_TEMP_DIR / f"local-{uuid.uuid4().hex}.part"
    bytes_received = 0
    try:
        with temporary.open("xb") as handle:
            while chunk := await crop.read(512 * 1024):
                bytes_received += len(chunk)
                if bytes_received > 8 * 1024 * 1024:
                    raise HTTPException(413, "单个人物局部文件超过8MB。")
                handle.write(chunk)
        if not bytes_received:
            raise HTTPException(400, "人物局部文件为空。")
        with Image.open(temporary) as local_crop:
            local_width, local_height = local_crop.size
            if local_width * local_height > 2048 * 2048:
                raise HTTPException(413, "单个人物局部像素超过安全上限（最多 2048×2048）。")
            if local_width == original_width and local_height == original_height:
                raise HTTPException(413, "服务端拒绝接收完整原图。")
            expected_width = max(1, round(crop_width * upload_scale))
            expected_height = max(1, round(crop_height * upload_scale))
            if abs(local_width - expected_width) > 2 or abs(local_height - expected_height) > 2:
                raise HTTPException(400, "上传局部尺寸与裁剪坐标不一致。")
            hx, hy, hw, hh = head_in_crop
            if min(hx, hy) < 0 or min(hw, hh) < 1 or hx + hw > local_width + 2 or hy + hh > local_height + 2:
                raise HTTPException(400, "头部坐标超出局部图片。")
    except HTTPException:
        temporary.unlink(missing_ok=True)
        raise
    except Exception as exc:
        temporary.unlink(missing_ok=True)
        raise HTTPException(415, "无法读取人物局部图片。") from exc

    value.update({
        "image_id": image_id, "person_id": person_id, "original_size": original,
        "crop_box": crop_box, "head_box": head_box, "head_box_in_crop": head_in_crop,
        "bytes": bytes_received, "upload_size": [local_width, local_height],
    })
    job, reused = local_person_job_store.create(temporary, value, provider)
    return {"job_id": job["id"], "person_id": person_id, "status": job["status"], "reused": reused}


@app.get("/api/local-person-jobs/{job_id}")
def local_person_job(job_id: str):
    if not LOCAL_MASTER_ENABLED or not re.fullmatch(r"[a-f0-9]{32}", job_id):
        raise HTTPException(404)
    job = local_person_job_store.get(job_id)
    if not job:
        raise HTTPException(404, "局部人物任务不存在。")
    return job


@app.post("/api/edit", status_code=202)
def edit(request: EditRequest):
    if not (UPLOAD_DIR / f"{request.image_id}.png").exists():
        raise HTTPException(404, "图片不存在或已清理。")
    if request.provider != "local" and not PROVIDER_SETTINGS[request.provider].configured:
        raise HTTPException(400, f"{PROVIDER_SETTINGS[request.provider].label} 尚未配置 API Key。")
    if request.cloud_scope == "full" and request.provider == "local":
        request.cloud_scope = "crop"
    job = job_store.create(request)
    return {"job_id": job["id"], "status": job["status"]}


@app.get("/api/jobs/{job_id}")
def job(job_id: str):
    value = job_store.get(job_id)
    if not value:
        raise HTTPException(404, "任务不存在。")
    return value


@app.get("/api/costs/summary")
def costs_summary():
    _require_lab_access()
    return cost_ledger.summary()


@app.post("/api/costs/jobs/{job_id}/publishable")
def mark_job_publishable(job_id: str, request: PublishabilityUpdate):
    _require_lab_access()
    job = job_store.get(job_id)
    if not job or job.get("status") != "completed":
        raise HTTPException(404, "没有可标记的已完成任务。")
    return {"job_id": job_id, **cost_ledger.mark_publishable(job_id, request.publishable)}


@app.post("/api/export")
def export(request: ExportRequest):
    job = job_store.get(request.job_id)
    if not job or job.get("status") != "completed":
        raise HTTPException(404, "没有可导出的已完成任务。")
    source = OUTPUT_DIR / f"{request.job_id}.png"
    if request.format == "png":
        return {"url": f"/media/outputs/{request.job_id}.png", "format": "png", "pixel_exact_outside_mask": True}
    destination = OUTPUT_DIR / f"{request.job_id}.jpg"
    save_clean(load_image_path(source), destination, "JPEG", request.quality)
    return {
        "url": f"/media/outputs/{request.job_id}.jpg",
        "format": "jpeg",
        "pixel_exact_outside_mask": False,
        "warning": "JPEG 会重新编码整张图；像素级保真请下载 PNG。",
    }


@app.get("/media/{kind}/{filename}", include_in_schema=False)
def media(kind: str, filename: str):
    roots = {"uploads": UPLOAD_DIR, "outputs": OUTPUT_DIR, "previews": PREVIEW_DIR, "local-person": LOCAL_PERSON_OUTPUT_DIR}
    root = roots.get(kind)
    if not root:
        raise HTTPException(404)
    safe_name = Path(filename).name
    path = root / safe_name
    if not path.exists():
        raise HTTPException(404)
    return FileResponse(path, headers={"Cache-Control": "no-store"})


@app.get("/media/benchmark/{relative_path:path}", include_in_schema=False)
def benchmark_media(relative_path: str):
    _require_lab_access()
    root = BENCHMARK_DIR.resolve()
    candidate = (root / relative_path).resolve()
    if root not in candidate.parents and candidate != root:
        raise HTTPException(404)
    if not candidate.is_file():
        raise HTTPException(404)
    return FileResponse(candidate, headers={"Cache-Control": "no-store"})


@app.get("/api/review/items")
def review_items():
    _require_lab_access()
    path = BENCHMARK_DIR / "review.json"
    if not path.exists():
        return {"items": [], "message": "尚未运行模型竞赛。"}
    data = json.loads(path.read_text("utf8"))
    # Provider identities stay server-side; the browser gets blinded labels only.
    return {"items": [{key: value for key, value in item.items() if key != "provider"} for item in data]}


@app.get("/api/comparison/items")
def comparison_items():
    _require_lab_access()
    path = COMPARISON_DIR / "review.json"
    if not path.exists():
        return {"items": [], "message": "三模型对照尚未准备完成。"}
    data = json.loads(path.read_text("utf8"))
    return {"items": [{key: value for key, value in item.items() if key != "route"} for item in data]}


@app.get("/api/prompt-comparison/items")
def prompt_comparison_items():
    _require_lab_access()
    path = PROMPT_COMPARISON_DIR / "review.json"
    if not path.exists():
        return {"items": [], "message": "四项轻动漫档位对照尚未准备完成。"}
    data = json.loads(path.read_text("utf8"))
    # Keep model and prompt identities server-side until the rating is complete.
    return {"items": [{key: value for key, value in item.items() if key not in {"route", "profile"}} for item in data]}


def _batch_run_dir(run_id: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9_-]+", run_id):
        raise HTTPException(404)
    directory = BENCHMARK_DIR / "runs" / run_id
    if not directory.is_dir():
        raise HTTPException(404)
    return directory


@app.get("/api/batch-runs")
def batch_runs():
    _require_lab_access()
    root = BENCHMARK_DIR / "runs"
    items = []
    if root.exists():
        for directory in root.iterdir():
            manifest = directory / "manifest.json"
            if manifest.is_file():
                data = json.loads(manifest.read_text("utf8"))
                items.append({key: data.get(key) for key in ("run_id", "status", "samples", "confirmed_people", "completed_images", "failed_images", "provider", "started_at", "completed_at")})
    return {"runs": sorted(items, key=lambda value: value.get("started_at") or 0, reverse=True)}


@app.get("/api/batch-runs/{run_id}/items")
def batch_run_items(run_id: str):
    _require_lab_access()
    directory = _batch_run_dir(run_id)
    review_path = directory / "review.json"
    results_path = directory / "results.json"
    manifest_path = directory / "manifest.json"
    review = json.loads(review_path.read_text("utf8")) if review_path.exists() else []
    results = json.loads(results_path.read_text("utf8")) if results_path.exists() else []
    details = {item.get("sample_id"): item for item in results}
    items = []
    for item in review:
        result = details.get(item["sample_id"], {})
        people = result.get("people", [])
        items.append({
            **item,
            "confirmed_people": result.get("confirmed_people", 0),
            "anime_people": sum(person.get("actual_mode") == "anime" for person in people),
            "safe_people": sum(person.get("actual_mode") == "safe" for person in people),
            "elapsed_ms": result.get("elapsed_ms"),
        })
    manifest = json.loads(manifest_path.read_text("utf8")) if manifest_path.exists() else {}
    return {"run": manifest, "items": items}


@app.post("/api/review/rate")
def review_rate(rating: ReviewRating):
    _require_lab_access()
    path = BENCHMARK_DIR / "ratings.jsonl"
    record = {**rating.model_dump(), "created_at": time.time()}
    with path.open("a", encoding="utf8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return {"saved": True}


@app.post("/api/comparison/rate")
def comparison_rate(rating: ReviewRating):
    _require_lab_access()
    path = COMPARISON_DIR / "ratings.jsonl"
    record = {**rating.model_dump(), "created_at": time.time()}
    with path.open("a", encoding="utf8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return {"saved": True}


@app.post("/api/prompt-comparison/rate")
def prompt_comparison_rate(rating: ReviewRating):
    _require_lab_access()
    path = PROMPT_COMPARISON_DIR / "ratings.jsonl"
    record = {**rating.model_dump(), "created_at": time.time()}
    with path.open("a", encoding="utf8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return {"saved": True}


@app.get("/api/preflight/items")
def preflight_items():
    _require_lab_access()
    audit_path = BENCHMARK_DIR / "detection-audit" / "detections.json"
    if not audit_path.exists():
        return {"items": [], "message": "尚未运行检测审计。"}
    audit = json.loads(audit_path.read_text("utf8"))
    saved_path = BENCHMARK_DIR / "preflight.json"
    saved = json.loads(saved_path.read_text("utf8")) if saved_path.exists() else {}
    items = []
    for record in audit["records"]:
        sample_id = record["sample_id"]
        existing = saved.get(sample_id)
        items.append({
            "sample_id": sample_id,
            "image_url": f"/media/uploads/sample-{sample_id}.jpg",
            "width": record["width"],
            "height": record["height"],
            "keep_people": record["keep_people"],
            "regions": existing["regions"] if existing else record["detections"],
            "reviewed": bool(existing),
        })
    return {"items": items, "reviewed": len(saved), "total": len(items)}


@app.post("/api/preflight/save")
def preflight_save(value: PreflightSave):
    _require_lab_access()
    path = BENCHMARK_DIR / "preflight.json"
    saved = json.loads(path.read_text("utf8")) if path.exists() else {}
    saved[value.sample_id] = {
        "sample_id": value.sample_id,
        "regions": [region.model_dump() for region in value.regions],
        "reviewed_at": time.time(),
    }
    path.write_text(json.dumps(saved, ensure_ascii=False, indent=2), "utf8")
    return {"saved": True, "reviewed": len(saved)}
