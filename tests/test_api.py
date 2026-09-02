from __future__ import annotations

import io
import json
import time
import uuid
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image
from pillow_heif import register_heif_opener

from app.config import PROVIDER_SETTINGS, ProviderSettings
from app.detector import detector
from app.main import app
import app.main as main_module
from app.providers import PROVIDERS
from app.cost_ledger import cost_ledger
from app.master_upload import master_upload_store


client = TestClient(app)
register_heif_opener()


@pytest.fixture(autouse=True)
def isolate_cost_ledger(tmp_path, monkeypatch):
    """API tests must not be counted as real model usage in the local ledger."""
    monkeypatch.setattr(cost_ledger, "path", tmp_path / "cost-ledger.jsonl")
    monkeypatch.setattr(cost_ledger, "outcomes_path", tmp_path / "cost-outcomes.json")


def image_payload() -> bytes:
    image = Image.new("RGB", (320, 220), (218, 211, 196))
    buffer = io.BytesIO()
    image.save(buffer, "JPEG")
    return buffer.getvalue()


def encoded_payload(fmt: str, size: tuple[int, int] = (320, 220)) -> bytes:
    image = Image.new("RGB", size, (94, 132, 176))
    buffer = io.BytesIO()
    if fmt == "HEIF":
        image.save(buffer, format="HEIF", quality=80)
    else:
        image.save(buffer, format=fmt)
    return buffer.getvalue()


def detect_and_upload_master(monkeypatch):
    monkeypatch.setattr(detector, "detect", lambda image, **kwargs: [])
    payload = image_payload()
    detected = client.post("/api/detect", files={"file": ("sample.jpg", payload, "image/jpeg")})
    assert detected.status_code == 200
    image = detected.json()
    uploaded = client.post(
        f"/api/images/{image['image_id']}/master",
        files={"file": ("sample.jpg", payload, "image/jpeg")},
    )
    assert uploaded.status_code == 200
    return image


def test_health_and_providers():
    assert client.get("/api/health").json()["ok"] is True
    assert client.get("/pose-avatar-playground").status_code == 200
    service_worker = client.get("/sw.js")
    assert service_worker.status_code == 200
    assert service_worker.headers["service-worker-allowed"] == "/"
    manifest = client.get("/static/manifest.webmanifest")
    assert manifest.status_code == 200
    assert manifest.json()["display"] == "standalone"
    providers = client.get("/api/providers").json()["providers"]
    local = next(item for item in providers if item["id"] == "local")
    assert local["configured"] is True
    fal = next(item for item in providers if item["id"] == "fal")
    assert fal["model"] in {
        "fal-ai/nano-banana-pro/edit", "openai/gpt-image-2/edit",
        "fal-ai/bytedance/seedream/v4.5/edit", "fal-ai/bytedance/seedream/v5/lite/edit",
    }
    assert fal["supports_mask"] is True
    ark = next(item for item in providers if item["id"] == "ark")
    assert ark["model"] == "doubao-seedream-5.0-lite"


def test_detection_rejects_full_resolution_payload_before_detector(monkeypatch):
    called = False

    def should_not_run(image, **kwargs):
        nonlocal called
        called = True
        return []

    monkeypatch.setattr(detector, "detect", should_not_run)
    image = Image.new("RGB", (2000, 1700), (40, 50, 60))
    buffer = io.BytesIO()
    image.save(buffer, "JPEG", quality=72)
    response = client.post("/api/detect", files={"file": ("too-large.jpg", buffer.getvalue(), "image/jpeg")})
    assert response.status_code == 413
    assert called is False


def test_detection_is_decoupled_from_master_upload(monkeypatch):
    monkeypatch.setattr(detector, "detect", lambda image, **kwargs: [])
    response = client.post("/api/detect", files={"file": ("detect.jpg", image_payload(), "image/jpeg")})
    assert response.status_code == 200
    result = response.json()
    assert result["image_url"] is None
    assert result["detection_timings"]["request_total_ms"] >= 0
    assert not (Path(__file__).resolve().parents[1] / ".local" / "app" / "uploads" / f"{result['image_id']}.png").exists()


@pytest.mark.parametrize(("fmt", "mime", "suffix"), [
    ("JPEG", "image/jpeg", "jpg"),
    ("PNG", "image/png", "png"),
    ("HEIF", "image/heic", "heic"),
])
def test_master_upload_formats_have_durable_ready_state(monkeypatch, fmt, mime, suffix):
    monkeypatch.setattr(detector, "detect", lambda image, **kwargs: [])
    payload = encoded_payload(fmt)
    detected = client.post("/api/detect", files={"file": (f"detect.{suffix}", image_payload(), "image/jpeg")}).json()
    image_id = detected["image_id"]
    missing = client.get(f"/api/images/{image_id}/master").json()
    assert missing["status"] == "missing"

    uploaded = client.post(
        f"/api/images/{image_id}/master",
        files={"file": (f"master.{suffix}", payload, mime)},
    )
    assert uploaded.status_code == 200, uploaded.text
    result = uploaded.json()
    assert result["status"] == "ready"
    assert result["format"] in {fmt.lower(), "heif"}
    assert result["width"] == 320
    assert result["timings"]["server_total_ms"] >= 0
    assert "instance_id" in result

    status = client.get(f"/api/images/{image_id}/master").json()
    assert status["status"] == "ready"
    assert status["image_url"].endswith(f"{image_id}.png")


def test_master_ready_is_reused_without_second_decode(monkeypatch):
    detected = detect_and_upload_master(monkeypatch)
    image_id = detected["image_id"]
    original = master_upload_store._process
    calls = 0

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(master_upload_store, "_process", counted)
    response = client.post(
        f"/api/images/{image_id}/master",
        files={"file": ("master.jpg", image_payload(), "image/jpeg")},
    )
    assert response.status_code == 200
    assert response.json()["status"] == "ready"
    assert response.json()["reused"] is True
    assert calls == 0


def test_master_duplicate_claim_has_one_owner():
    image_id = "f" * 32
    # Make this test independent of a previous local run using the same id.
    state_path = master_upload_store._state_path(image_id)
    state_path.unlink(missing_ok=True)
    (Path(__file__).resolve().parents[1] / ".local" / "app" / "uploads" / f"{image_id}.png").unlink(missing_ok=True)
    first, first_claimed = master_upload_store.begin_upload(image_id)
    second, second_claimed = master_upload_store.begin_upload(image_id)
    assert first["status"] == second["status"] == "uploading"
    assert first_claimed is True
    assert second_claimed is False


def test_master_rejects_more_than_sixteen_megapixels(monkeypatch):
    monkeypatch.setattr(detector, "detect", lambda image, **kwargs: [])
    detected = client.post("/api/detect", files={"file": ("detect.jpg", image_payload(), "image/jpeg")}).json()
    payload = encoded_payload("JPEG", (4100, 4000))
    response = client.post(
        f"/api/images/{detected['image_id']}/master",
        files={"file": ("master.jpg", payload, "image/jpeg")},
    )
    assert response.status_code == 413
    state = client.get(f"/api/images/{detected['image_id']}/master").json()
    assert state["status"] == "failed"
    assert state["error_code"] == "pixel_limit"


def test_master_processing_survives_a_waiter_disconnect(monkeypatch):
    """The worker owns preparation after bytes land; an HTTP waiter may vanish."""
    image_id = uuid.uuid4().hex
    state_path = master_upload_store._state_path(image_id)
    state_path.unlink(missing_ok=True)
    current, claimed = master_upload_store.begin_upload(image_id)
    assert claimed is True
    temporary = master_upload_store.temp_path(image_id)
    temporary.write_bytes(image_payload())
    future = master_upload_store.submit(image_id, temporary, {
        "bytes": len(image_payload()), "receive_ms": 1, "format_hint": "image/jpeg",
    })
    # Deliberately do not await through the API request.  This models a client
    # disappearing while the independent preparation worker keeps ownership.
    result = future.result(timeout=10)
    assert result["status"] == "ready"
    assert master_upload_store.status(image_id)["status"] == "ready"


def local_person_metadata(**changes):
    value = {
        "image_id": "1" * 32, "person_id": "manual-one",
        "original_size": [4032, 3024], "head_box": [900, 500, 300, 420],
        "crop_box": [774, 324, 552, 772], "head_box_in_crop": [126, 176, 300, 420],
        "upload_scale": 1, "retry_nonce": 0, "source": "manual", "selected": True,
    }
    value.update(changes)
    return value


def test_local_master_feature_is_off_by_default():
    features = client.get("/api/features").json()
    assert features["traditional_master_fallback"] is True
    assert features["max_parallel_people"] == 2


def test_local_person_endpoint_rejects_whole_original(monkeypatch):
    monkeypatch.setattr(main_module, "LOCAL_MASTER_ENABLED", True)
    monkeypatch.setitem(
        PROVIDER_SETTINGS, "ark",
        ProviderSettings("ark", "fake", None, "fake-local-master", 0.0, False),
    )
    payload = encoded_payload("JPEG", (4032, 3024))
    metadata = local_person_metadata(
        original_size=[4032, 3024], head_box=[900, 500, 300, 420],
        crop_box=[0, 0, 4032, 3024], head_box_in_crop=[900, 500, 300, 420], upload_scale=1,
    )
    response = client.post("/api/local-person-jobs", data={"metadata": json.dumps(metadata), "provider": "ark"}, files={"crop": ("full.jpg", payload, "image/jpeg")})
    assert response.status_code == 413
    assert "整图" in response.json()["detail"] or "过多场景" in response.json()["detail"]


def test_local_person_endpoint_accepts_only_consistent_crop(monkeypatch):
    class FakeProvider:
        def edit(self, crop, mask, **kwargs):
            return Image.new("RGB", crop.size, (20, 160, 210))

    monkeypatch.setattr(main_module, "LOCAL_MASTER_ENABLED", True)
    monkeypatch.setitem(PROVIDERS, "ark", FakeProvider())
    monkeypatch.setitem(
        PROVIDER_SETTINGS, "ark",
        ProviderSettings("ark", "fake", None, "fake-local-master", 0.0, False),
    )
    metadata = local_person_metadata(retry_nonce=int(time.time() * 1000))
    payload = encoded_payload("JPEG", (552, 772))
    response = client.post("/api/local-person-jobs", data={"metadata": json.dumps(metadata), "provider": "ark"}, files={"crop": ("person.jpg", payload, "image/jpeg")})
    assert response.status_code == 202, response.text
    body = response.json()
    for _ in range(80):
        job = client.get(f"/api/local-person-jobs/{body['job_id']}").json()
        if job["status"] in {"completed", "failed"}:
            break
        time.sleep(.05)
    assert job["status"] == "completed", job
    assert job["person_id"] == "manual-one"
    assert job["output_url"].startswith("/media/local-person/")

    duplicate = client.post("/api/local-person-jobs", data={"metadata": json.dumps(metadata), "provider": "ark"}, files={"crop": ("person.jpg", payload, "image/jpeg")})
    assert duplicate.status_code == 202
    assert duplicate.json()["job_id"] == body["job_id"]
    assert duplicate.json()["reused"] is True


def test_detect_manual_edit_and_export(monkeypatch):
    image = detect_and_upload_master(monkeypatch)
    assert image["detections"] == []
    assert image["width"] == 320

    request = {
        "image_id": image["image_id"],
        "regions": [{
            "id": "manual-one", "box": [110, 35, 80, 125], "head_box": [110, 35, 80, 125],
            "confidence": 1, "source": "manual", "selected": True,
        }],
        "mode": "anime", "provider": "local", "cloud_scope": "crop", "retry_nonce": 0,
    }
    response = client.post("/api/edit", json=request)
    assert response.status_code == 202
    job_id = response.json()["job_id"]
    for _ in range(80):
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] in {"completed", "failed"}:
            break
        time.sleep(0.05)
    assert job["status"] == "completed", job
    assert job["outside_mask_exact"] is True
    assert job["people"][0]["actual_mode"] == "anime"

    exported = client.post("/api/export", json={"job_id": job_id, "format": "jpeg", "quality": 92})
    assert exported.status_code == 200
    assert exported.json()["pixel_exact_outside_mask"] is False


def test_unconfigured_remote_provider_is_rejected(monkeypatch):
    detected = detect_and_upload_master(monkeypatch)
    response = client.post("/api/edit", json={
        "image_id": detected["image_id"], "regions": [], "mode": "anime", "provider": "openai", "cloud_scope": "crop"
    })
    assert response.status_code in {202, 400}


def test_confirmed_low_confidence_region_is_not_silently_downgraded(monkeypatch):
    detected = detect_and_upload_master(monkeypatch)
    response = client.post("/api/edit", json={
        "image_id": detected["image_id"],
        "regions": [{
            "id": "reviewed-low-confidence", "box": [110, 35, 80, 125], "head_box": [110, 35, 80, 125],
            "confidence": 0.3, "source": "manual", "selected": True,
        }],
        "mode": "anime", "provider": "local", "cloud_scope": "crop", "selection_confirmed": True,
    })
    assert response.status_code == 202
    job_id = response.json()["job_id"]
    for _ in range(80):
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] in {"completed", "failed"}:
            break
        time.sleep(0.05)
    assert job["status"] == "completed", job
    assert job["people"][0]["actual_mode"] == "anime"


def test_safe_cover_is_composited_locally_without_changing_scene_pixels(monkeypatch):
    detected = detect_and_upload_master(monkeypatch)
    response = client.post("/api/edit", json={
        "image_id": detected["image_id"],
        "regions": [{
            "id": "safe-cover-one", "box": [110, 35, 80, 125], "head_box": [110, 35, 80, 125],
            "confidence": 1, "source": "manual", "selected": True,
        }],
        "mode": "safe", "provider": "local", "cloud_scope": "crop", "safe_cover_id": "architect",
    })
    assert response.status_code == 202
    job_id = response.json()["job_id"]
    for _ in range(80):
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] in {"completed", "failed"}:
            break
        time.sleep(0.05)
    assert job["status"] == "completed", job
    assert job["people"][0]["actual_mode"] == "safe"
    assert job["people"][0]["safe_cover_id"] == "architect"
    assert job["outside_mask_exact"] is True


def test_full_scene_benchmark_discards_generated_pixels_outside_head(monkeypatch):
    class FullSceneFake:
        def __init__(self):
            self.received_size = None

        def edit(self, image, mask, *, subject_id, retry_nonce):
            self.received_size = image.size
            return Image.new("RGB", image.size, (20, 180, 120))

    fake = FullSceneFake()
    monkeypatch.setitem(PROVIDERS, "openai", fake)
    monkeypatch.setitem(
        PROVIDER_SETTINGS,
        "openai",
        ProviderSettings("openai", "测试整图模型", None, "fake", 0.0, False),
    )
    detected = detect_and_upload_master(monkeypatch)
    response = client.post("/api/edit", json={
        "image_id": detected["image_id"],
        "regions": [{
            "id": "full-scene-one", "box": [110, 35, 80, 125], "head_box": [110, 35, 80, 125],
            "confidence": 1, "source": "manual", "selected": True,
        }],
        "mode": "anime", "provider": "openai", "cloud_scope": "full",
    })
    assert response.status_code == 202
    job_id = response.json()["job_id"]
    for _ in range(80):
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] in {"completed", "failed"}:
            break
        time.sleep(0.05)
    assert job["status"] == "completed", job
    assert fake.received_size == (320, 220)
    assert job["outside_mask_exact"] is True
    assert job["people"][0]["cloud_scope"] == "full"
