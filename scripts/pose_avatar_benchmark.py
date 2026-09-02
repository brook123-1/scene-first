from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
import statistics
from collections import Counter
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import AVATAR_ASSET_DIR, MODEL_DIR  # noqa: E402
from app.detector import FaceDetector  # noqa: E402
from app.pose_avatar.adapters import detected_head_from_region  # noqa: E402
from app.pose_avatar.benchmark import (  # noqa: E402
    AdapterObservation,
    BenchmarkAnnotation,
    BenchmarkDocument,
    ReviewStatus,
    anonymized_distribution,
    benchmark_metrics,
)
from app.pose_avatar.calibration import calibration_proposal  # noqa: E402
from app.pose_avatar.debug import annotated_input  # noqa: E402
from app.pose_avatar.coverage import EllipseCoverageMaskProvider, LandmarkSilhouetteMaskProvider  # noqa: E402
from app.pose_avatar.pose_adapters import (  # noqa: E402
    OpenCvMediaPipePoseAdapter,
    PoseAdapterResult,
    YuNetHeuristicAdapter,
    YuNetSolvePnPAdapter,
)
from app.pose_avatar.registry import AssetRegistry  # noqa: E402
from app.pose_avatar.renderer import PoseAvatarRenderer, _composite_rgba, _warp_asset  # noqa: E402
from app.pose_avatar.safety import (  # noqa: E402
    P1_CANDIDATE_SAFETY_POLICY,
    conservative_head_envelope,
    coverage_proxies,
)
from app.pose_avatar.transform import solve_face_primary_transform, solve_two_stage_head_neck  # noqa: E402
from app.pose_avatar.human_baseline import analyze_human_baseline  # noqa: E402


LOCAL_ROOT = ROOT / ".local" / "app" / "pose-avatar-p1"
ANNOTATIONS = LOCAL_ROOT / "annotations.json"
PREFLIGHT = ROOT / ".local" / "app" / "benchmark" / "preflight.json"
MANIFEST = ROOT / "samples" / "manifest.csv"
INBOX = ROOT / "samples" / "inbox"
ADAPTER_IDS = (
    "yunet_5pt_heuristic",
    "yunet_5pt_heuristic_p1_safety",
    "yunet_5pt_solvepnp",
    "opencv_mediapipe_pose_hybrid",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local-only Pose-aware Avatar Overlay P1 benchmark")
    parser.add_argument("command", choices=["init", "run", "validate", "report", "gallery", "import-review", "calibrate", "human-baseline"])
    parser.add_argument("--count", type=int, default=48)
    parser.add_argument("--download-models", action="store_true")
    parser.add_argument("--require-reviewed", action="store_true")
    parser.add_argument("--review-file", type=Path)
    return parser.parse_args()


def _load_document() -> BenchmarkDocument:
    if not ANNOTATIONS.exists():
        raise RuntimeError("benchmark is not initialized; run the init command first")
    return BenchmarkDocument.model_validate_json(ANNOTATIONS.read_text("utf8"))


def _save_document(document: BenchmarkDocument) -> None:
    LOCAL_ROOT.mkdir(parents=True, exist_ok=True)
    ANNOTATIONS.write_text(document.model_dump_json(indent=2), "utf8")


def _manifest_rows() -> dict[str, dict]:
    return {
        row["id"].strip(): {key: (value or "").strip() for key, value in row.items()}
        for row in csv.DictReader(MANIFEST.open("r", encoding="utf-8-sig"))
    }


def _resolve_image(row: dict) -> Path:
    matches = list(INBOX.rglob(Path(row["filename"]).name))
    if len(matches) != 1:
        raise RuntimeError("a benchmark source image is missing or ambiguous")
    return matches[0]


def _size_bucket(edge: float) -> str:
    return "tiny" if edge < 48 else "small" if edge < 96 else "medium" if edge < 192 else "large"


def _stable_key(seed: str, *values: str) -> str:
    return hashlib.sha256((seed + "|" + "|".join(values)).encode()).hexdigest()


def init_benchmark(count: int) -> BenchmarkDocument:
    if not 30 <= count <= 50:
        raise ValueError("P1 benchmark count must be between 30 and 50")
    preflight = json.loads(PREFLIGHT.read_text("utf8"))
    rows = _manifest_rows()
    seed = "pose-avatar-p1-v1"
    candidates = []
    for sample_id, sample in sorted(preflight.items()):
        row = rows.get(sample_id)
        if not row or row.get("rights", "").lower() not in {"owned", "consented", "licensed"}:
            continue
        if row.get("contains_minors", "").lower() in {"yes", "true", "1"}:
            continue
        source_path = _resolve_image(row)
        for index, region in enumerate(sample.get("regions", [])):
            if not region.get("selected", True):
                continue
            box = region.get("head_box") or region.get("box")
            if not isinstance(box, list) or len(box) != 4 or min(box[2:]) <= 0:
                continue
            candidates.append({
                "sample_id": sample_id,
                "path": source_path,
                "region": region,
                "region_index": index,
                "edge": float(min(box[2], box[3])),
                "bucket": _size_bucket(float(min(box[2], box[3]))),
                "key": _stable_key(seed, sample_id, str(region.get("id", index))),
            })
    # First keep one deterministic instance per photo, then cycle through size
    # buckets so a dense group photo cannot dominate the benchmark.
    by_sample: dict[str, list[dict]] = defaultdict(list)
    by_bucket: dict[str, list[dict]] = defaultdict(list)
    for value in candidates:
        by_sample[value["sample_id"]].append(value)
        by_bucket[value["bucket"]].append(value)
    selected = [sorted(values, key=lambda value: value["key"])[0] for values in by_sample.values()]
    selected_keys = {value["key"] for value in selected}
    for values in by_bucket.values():
        values.sort(key=lambda value: value["key"])
    order = ["tiny", "small", "medium", "large"]
    while len(selected) < count:
        progressed = False
        for bucket in order:
            value = next((item for item in by_bucket[bucket] if item["key"] not in selected_keys), None)
            if value:
                selected.append(value)
                selected_keys.add(value["key"])
                progressed = True
                if len(selected) == count:
                    break
        if not progressed:
            break
    if len(selected) < count:
        raise RuntimeError(f"only {len(selected)} eligible head instances are available")
    cases = []
    for number, value in enumerate(sorted(selected, key=lambda item: (item["sample_id"], item["key"])), 1):
        region, box = value["region"], value["region"].get("head_box") or value["region"].get("box")
        cases.append(BenchmarkAnnotation(
            case_id=f"p1-{number:03d}",
            local_image_ref=str(value["path"].relative_to(ROOT)),
            source_sample_id=value["sample_id"],
            source_region_id=str(region.get("id") or value["region_index"]),
            head_bbox=tuple(float(item) for item in box),
            head_size_px=float(min(box[2], box[3])),
            detector_source=str(region.get("source") or "manual"),
        ))
    document = BenchmarkDocument(
        benchmark_id="pose-avatar-placement-p1",
        created_at=datetime.now(timezone.utc).isoformat(),
        sampling_seed=seed,
        cases=cases,
    )
    _save_document(document)
    (LOCAL_ROOT / "dataset-summary.json").write_text(json.dumps(anonymized_distribution(document), indent=2), "utf8")
    return document


def _overlap(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    left, top, right, bottom = max(ax, bx), max(ay, by), min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0, right - left) * max(0, bottom - top)
    return inter / max(1.0, min(aw * ah, bw * bh))


def _match_detection(case: BenchmarkAnnotation, detections: list[dict]) -> dict | None:
    scored = [(_overlap(case.head_bbox, item.get("head_box") or item["box"]), item) for item in detections]
    score, match = max(scored, default=(0.0, None), key=lambda item: item[0])
    return match if score >= 0.38 else None


def _local_image_path(case: BenchmarkAnnotation) -> Path:
    path = (ROOT / case.local_image_ref).resolve()
    try:
        path.relative_to(INBOX.resolve())
    except ValueError as exc:
        raise ValueError(f"{case.case_id}: local_image_ref must remain below samples/inbox") from exc
    if not path.is_file():
        raise FileNotFoundError(f"{case.case_id}: local benchmark image is missing")
    return path


def _crop_box(head_bbox, size):
    x, y, width, height = head_bbox
    pad_x, pad_top, pad_bottom = width * 0.55, height * 0.45, height * 0.75
    return (
        max(0, int(x - pad_x)), max(0, int(y - pad_top)),
        min(size[0], int(x + width + pad_x)), min(size[1], int(y + height + pad_bottom)),
    )


def _render_observation(
    image: Image.Image,
    head,
    adapter_result: PoseAdapterResult,
    renderer: PoseAvatarRenderer,
    adapter_id: str,
) -> tuple[AdapterObservation, object]:
    adapted = adapter_result.apply(head)
    started = time.perf_counter()
    result = renderer.render(image, [adapted], image_id="local-p1")
    render_elapsed = round((time.perf_counter() - started) * 1000)
    elapsed = adapter_result.elapsed_ms + render_elapsed
    decision, trace = result.decisions[0], result.traces[0]
    proxies = coverage_proxies(result.transformed_overlay.getchannel("A"), adapted)
    multiplier = 1.0
    if result.transforms[0] and trace.scene_id:
        multiplier = renderer.safety_policy.multiplier(trace.scene_id, adapted.pose.roll_deg)
    return AdapterObservation(
        adapter_id=adapter_id,
        available=adapter_result.available,
        elapsed_ms=elapsed,
        adapter_elapsed_ms=adapter_result.elapsed_ms,
        render_elapsed_ms=render_elapsed,
        yaw_deg=adapted.pose.yaw_deg,
        pitch_deg=adapted.pose.pitch_deg,
        roll_deg=adapted.pose.roll_deg,
        pose_confidence=adapted.pose.confidence,
        body_landmarks=adapted.body_landmarks,
        actual_scene=decision.scene_id,
        actual_route=decision.route_type,
        fallback_reason=decision.fallback_reason,
        selected_asset=trace.selected_asset,
        safety_multiplier=multiplier,
        proxies=proxies,
        residual_components=(result.transforms[0].residual_components if result.transforms[0] else {}),
        error=adapter_result.error,
    ), result


def run_benchmark(document: BenchmarkDocument, download_models: bool) -> BenchmarkDocument:
    registry = AssetRegistry(AVATAR_ASSET_DIR)
    baseline_renderer = PoseAvatarRenderer(registry)
    safety_renderer = PoseAvatarRenderer(registry, safety_policy=P1_CANDIDATE_SAFETY_POLICY)
    detector = FaceDetector()
    adapters = [
        YuNetHeuristicAdapter(),
        YuNetSolvePnPAdapter(),
        OpenCvMediaPipePoseAdapter(MODEL_DIR / "pose_estimation_mediapipe_2023mar.onnx", allow_download=download_models),
    ]
    by_sample: dict[str, list[BenchmarkAnnotation]] = defaultdict(list)
    for case in document.cases:
        by_sample[case.source_sample_id].append(case)
    case_updates = {}
    bundle_root = LOCAL_ROOT / "cases"
    bundle_root.mkdir(parents=True, exist_ok=True)
    for sample_number, cases in enumerate(sorted(by_sample.values(), key=lambda values: values[0].source_sample_id), 1):
        image_path = _local_image_path(cases[0])
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        detections = detector.detect(image, include_pose_data=True)
        for case in cases:
            match = _match_detection(case, detections)
            region = {
                "id": case.case_id,
                "head_box": list(case.head_bbox),
                "confidence": float(match.get("confidence", 0.0)) if match else 0.0,
                "face_landmarks": (match or {}).get("face_landmarks", {}),
                "anchor_provenance": {key: "yunet_full_image_match" for key in (match or {}).get("face_landmarks", {})},
            }
            head = detected_head_from_region(region, image.size)
            adapter_results = {adapter.adapter_id: adapter.estimate(image, head) for adapter in adapters}
            observations = {}
            baseline_observation, baseline_result = _render_observation(
                image, head, adapter_results["yunet_5pt_heuristic"], baseline_renderer, "yunet_5pt_heuristic",
            )
            safety_observation, safety_result = _render_observation(
                image, head, adapter_results["yunet_5pt_heuristic"], safety_renderer, "yunet_5pt_heuristic_p1_safety",
            )
            observations[baseline_observation.adapter_id] = baseline_observation
            observations[safety_observation.adapter_id] = safety_observation
            for adapter_id in ("yunet_5pt_solvepnp", "opencv_mediapipe_pose_hybrid"):
                observation, _ = _render_observation(image, head, adapter_results[adapter_id], baseline_renderer, adapter_id)
                observations[adapter_id] = observation
            case_updates[case.case_id] = case.model_copy(update={"observations": observations})

            case_dir = bundle_root / case.case_id
            case_dir.mkdir(exist_ok=True)
            crop = _crop_box(case.head_bbox, image.size)
            image.crop(crop).save(case_dir / "01-original-crop.jpg", quality=90)
            annotated_input(image, [head]).crop(crop).save(case_dir / "02-bbox-landmarks-pose.jpg", quality=90)
            baseline_result.transformed_overlay.crop(crop).save(case_dir / "03-overlay-baseline.png")
            conservative_head_envelope(image.size, head).crop(crop).save(case_dir / "04-safety-envelope.png")
            baseline_result.image.crop(crop).save(case_dir / "05-composite-baseline.jpg", quality=90)
            safety_result.image.crop(crop).save(case_dir / "06-composite-p1-safety.jpg", quality=90)
            coverage_payload = {}
            for provider in (EllipseCoverageMaskProvider(), LandmarkSilhouetteMaskProvider()):
                masks = provider.evaluate(image.size, head, baseline_result.transformed_overlay.getchannel("A"))
                prefix = "08" if provider.provider_id.startswith("bbox") else "09"
                masks.original_head_mask.crop(crop).save(case_dir / f"{prefix}-original-head-mask-{provider.provider_id}.png")
                masks.avatar_coverage_mask.crop(crop).save(case_dir / f"{prefix}-avatar-coverage-mask-{provider.provider_id}.png")
                masks.exposed_head_mask.crop(crop).save(case_dir / f"{prefix}-exposed-head-mask-{provider.provider_id}.png")
                coverage_payload[provider.provider_id] = {"exposed_head_ratio": masks.exposed_head_ratio}
            fitting_payload = {}
            scene = baseline_result.decisions[0].scene_id
            record = registry.select(scene) if scene else None
            if record is not None:
                try:
                    face_transform = solve_face_primary_transform(head, record.anchors)
                    face_overlay = _warp_asset(registry.image(record), face_transform.matrix, image.size)
                    _composite_rgba(image, face_overlay).crop(crop).save(case_dir / "10-composite-face-primary.jpg", quality=90)
                    hybrid_head = adapter_results["opencv_mediapipe_pose_hybrid"].apply(head)
                    head_transform, neck_transform = solve_two_stage_head_neck(hybrid_head, record.anchors)
                    asset = registry.image(record)
                    neck_y = int((record.anchors.anchors["neck_left"][1] + record.anchors.anchors["neck_right"][1]) / 2)
                    head_asset, neck_asset = asset.copy(), asset.copy()
                    rows = np.indices((asset.height, asset.width))[0]
                    alpha = np.asarray(asset.getchannel("A"), dtype=np.uint8)
                    head_asset.putalpha(Image.fromarray(((rows <= neck_y + 12) * alpha).astype(np.uint8), "L"))
                    neck_asset.putalpha(Image.fromarray(((rows >= neck_y - 18) * alpha).astype(np.uint8), "L"))
                    two_overlay = _warp_asset(head_asset, head_transform.matrix, image.size)
                    if neck_transform:
                        two_overlay = Image.alpha_composite(two_overlay, _warp_asset(neck_asset, neck_transform.matrix, image.size))
                    _composite_rgba(image, two_overlay).crop(crop).save(case_dir / "11-composite-two-stage.jpg", quality=90)
                    fitting_payload = {
                        "p1_baseline": baseline_result.transforms[0].model_dump(mode="json") if baseline_result.transforms[0] else None,
                        "face_primary": face_transform.model_dump(mode="json"),
                        "two_stage_head": head_transform.model_dump(mode="json"),
                        "two_stage_neck": neck_transform.model_dump(mode="json") if neck_transform else None,
                    }
                except (ValueError, TypeError) as exc:
                    fitting_payload = {"error": str(exc)}
            (case_dir / "07-trace.json").write_text(json.dumps({
                "case_id": case.case_id,
                "match_found": match is not None,
                "head": head.model_dump(mode="json"),
                "observations": {key: value.model_dump(mode="json") for key, value in observations.items()},
                "baseline_trace": baseline_result.traces[0].model_dump(mode="json"),
                "safety_trace": safety_result.traces[0].model_dump(mode="json"),
                "coverage": coverage_payload,
                "fitting_bakeoff": fitting_payload,
            }, indent=2), "utf8")
        print(f"sample {sample_number}/{len(by_sample)}: {len(cases)} cases", flush=True)
    updated = document.model_copy(update={"cases": [case_updates[case.case_id] for case in document.cases]})
    _save_document(updated)
    write_gallery(updated)
    write_report(updated)
    return updated


def write_gallery(document: BenchmarkDocument) -> Path:
    raw_document = json.dumps(document.model_dump(mode="json"), ensure_ascii=False).replace("<", "\\u003c")
    html = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Placement Review Console</title>
<style>
:root{--paper:#e9e3d5;--ink:#1c201e;--signal:#ed5b35;--night:#111716;--muted:#74786f;--line:#b8b1a4}*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font:13px Georgia,"Noto Serif SC",serif}.top{height:58px;background:var(--night);color:#f6f0e5;display:flex;align-items:center;padding:0 22px;gap:18px;position:sticky;top:0;z-index:5}.mark{font:700 19px Consolas,monospace;letter-spacing:.08em}.local{color:#ff9b78;font:11px Consolas,monospace}.progress{margin-left:auto}.bar{display:inline-block;width:140px;height:5px;background:#39413e;margin-left:10px}.bar i{display:block;height:100%;background:var(--signal)}button,select,textarea{font:inherit}.workspace{display:grid;grid-template-columns:248px minmax(0,1fr) 330px;min-height:calc(100vh - 58px)}aside{padding:20px;border-right:1px solid var(--line)}.filters label,.field{display:block;margin:0 0 13px}.filters select,.field select,.field textarea{width:100%;padding:9px;border:1px solid var(--line);background:#f5f0e6}.toggles{display:grid;gap:7px;margin:14px 0}.nav{display:grid;grid-template-columns:1fr 1fr;gap:7px}.nav button,.export{padding:10px;border:1px solid var(--ink);background:transparent;cursor:pointer}.export{width:100%;margin-top:12px;background:var(--ink);color:white}.stage{padding:22px;min-width:0}.casehead{display:flex;align-items:end;gap:18px;margin-bottom:14px}.casehead h1{font:700 clamp(28px,4vw,52px) Consolas,monospace;margin:0}.meta{color:var(--muted);line-height:1.55}.views{display:grid;grid-template-columns:1fr 1fr;gap:10px}.views figure{margin:0;background:#0c100f;border:1px solid #343a37;min-height:240px;position:relative}.views img{display:block;width:100%;height:clamp(260px,38vh,520px);object-fit:contain}.views figcaption{position:absolute;left:8px;bottom:8px;background:#111d;color:#fff;padding:5px 8px;font:10px Consolas,monospace}.thumbs{display:flex;gap:6px;overflow:auto;margin-top:8px}.thumbs button{border:1px solid var(--line);background:transparent;padding:6px 8px;white-space:nowrap}.thumbs button.on{background:var(--signal);color:white;border-color:var(--signal)}.review{padding:20px;border-left:1px solid var(--line);background:#f3eee4}.review h2{font:700 16px Consolas,monospace;margin:0 0 18px}.choices{display:grid;grid-template-columns:1fr 1fr;gap:6px}.choices label,.failures label{border:1px solid var(--line);padding:8px;cursor:pointer;background:#faf6ed}.choices label:has(input:checked),.failures label:has(input:checked){border-color:var(--signal);box-shadow:inset 3px 0 var(--signal)}.choices input,.failures input{margin-right:5px}.failures{display:grid;grid-template-columns:1fr 1fr;gap:5px;max-height:205px;overflow:auto}.failures label{font:10px Consolas,monospace}.status{padding:9px;background:#ded7ca;color:#575a54;margin:10px 0;font:11px Consolas,monospace}.trace{color:#9a321a}.keys{font:10px Consolas,monospace;color:var(--muted);line-height:1.6}@media(max-width:1050px){.workspace{grid-template-columns:210px 1fr}.review{grid-column:1/-1;border-left:0;border-top:1px solid var(--line)}.views figure{min-height:190px}}@media(max-width:680px){.workspace{display:block}aside{border-right:0;border-bottom:1px solid var(--line)}.views{grid-template-columns:1fr}.top{padding:0 12px}.bar{width:70px}}
</style><header class="top"><span class="mark">PLACEMENT / P1.5</span><span class="local">LOCAL ONLY · AUTOSAVE</span><span class="progress" id="progress"></span><span class="bar"><i id="bar"></i></span></header>
<main class="workspace"><aside class="filters"><strong>QUEUE CONTROL</strong><label>Route<select id="fRoute"><option value="">全部</option></select></label><label>Scene<select id="fScene"><option value="">全部</option></select></label><label>Placement<select id="fPlacement"><option value="">全部</option></select></label><label>Failure<select id="fFailure"><option value="">全部</option></select></label><label>Sort<select id="sort"><option value="case">Case ID</option><option value="residual-desc">Residual ↓</option><option value="residual-asc">Residual ↑</option></select></label><div class="toggles"><label><input id="onlyStandard" type="checkbox"> only STANDARD</label><label><input id="onlyFallback" type="checkbox"> only fallback</label></div><div class="nav"><button id="prev">← Previous</button><button id="next">Next →</button></div><button class="export" id="export">EXPORT REVIEW JSON</button><p class="keys">←/→ navigate · 1/2/3/4 placement<br>Q/W/E/R route · S mark reviewed<br>Changes save immediately in this browser.</p></aside>
<section class="stage"><div class="casehead"><h1 id="caseId"></h1><div class="meta" id="meta"></div></div><div class="views"><figure><img id="leftImage"><figcaption id="leftCaption"></figcaption></figure><figure><img id="rightImage"><figcaption id="rightCaption"></figcaption></figure></div><div class="thumbs" id="thumbs"></div></section>
<section class="review"><h2>HUMAN JUDGMENT</h2><div class="field">Placement<div class="choices" id="placement"></div></div><div class="field">Route judgment<div class="choices" id="route"></div></div><label class="field">Expected scene<select id="scene"></select></label><label class="field">View<select id="view"><option value="">—</option><option>FRONT</option><option>LEFT_34</option><option>RIGHT_34</option><option>LEFT_PROFILE</option><option>RIGHT_PROFILE</option><option>BACK</option></select></label><label class="field">Occlusion<select id="occlusion"><option>UNKNOWN</option><option>LOW</option><option>HIGH</option></select></label><div class="field">Failure taxonomy<div class="failures" id="failures"></div></div><label class="field">Primary failure<select id="primary"></select></label><label class="field">Reviewer note<textarea id="note" rows="3" maxlength="500"></textarea></label><label class="field"><input id="reviewed" type="checkbox"> Review complete</label><div class="status" id="status"></div><a class="trace" id="trace" target="_blank">OPEN DECISION TRACE ↗</a></section></main>
<script>const original=__DOCUMENT__;const key='pose-avatar-review:'+original.benchmark_id;let doc=structuredClone(original),queue=[],index=0,view=0;try{const saved=JSON.parse(localStorage.getItem(key));if(saved&&saved.benchmark_id===doc.benchmark_id)doc=saved}catch(e){}
const $=id=>document.getElementById(id),obs=c=>c.observations.yunet_5pt_heuristic||{},placements=['PASS','BORDERLINE','FAIL','NOT_APPLICABLE'],routes=['STANDARD_ELIGIBLE','SIMPLIFIED_ELIGIBLE','SHOULD_FALLBACK','UNSURE'],scenes=['','S01_FRONT_NEUTRAL','S04_L34_NEUTRAL','S07_R34_NEUTRAL','S12_BACK'],failures=['WRONG_SCENE','WRONG_SCALE','WRONG_ROLL','WRONG_POSITION','FLOATING_NECK','NECK_OVERLAP','EXPOSED_ORIGINAL_HEAD','WRONG_OCCLUSION','BAD_ASSET_GEOMETRY','SHOULD_HAVE_FALLBACK','FALSE_FALLBACK','UNSUPPORTED_SCENE','OTHER'];
const files=[['01-original-crop.jpg','ORIGINAL'],['05-composite-baseline.jpg','FINAL COMPOSITE'],['02-bbox-landmarks-pose.jpg','BBOX · FACE/BODY LANDMARKS'],['03-overlay-baseline.png','TRANSFORMED OVERLAY'],['04-safety-envelope.png','COVERAGE / SAFETY PROXY'],['06-composite-p1-safety.jpg','SAFETY CANDIDATE'],['08-exposed-head-mask-bbox_ellipse_v1.png','EXPOSED MASK · BASELINE'],['09-exposed-head-mask-landmark_silhouette_v1.png','EXPOSED MASK · CANDIDATE'],['10-composite-face-primary.jpg','FACE-PRIMARY FIT'],['11-composite-two-stage.jpg','TWO-STAGE HEAD + NECK']];
function persist(){localStorage.setItem(key,JSON.stringify(doc));$('status').textContent='autosaved · '+new Date().toLocaleTimeString()}
function options(id,values){$(id).innerHTML=values.map(v=>`<option value="${v}">${v||'—'}</option>`).join('')}
function choices(id,values,name){$(id).innerHTML=values.map((v,i)=>`<label><input type="radio" name="${name}" value="${v}">${i+1}. ${v}</label>`).join('')}
choices('placement',placements,'placement');choices('route',routes,'route');options('scene',scenes);options('primary',['',...failures]);$('failures').innerHTML=failures.map(v=>`<label><input type="checkbox" value="${v}">${v}</label>`).join('');options('fRoute',['',...new Set(doc.cases.map(c=>obs(c).actual_route).filter(Boolean))]);options('fScene',['',...new Set(doc.cases.map(c=>obs(c).actual_scene).filter(Boolean))]);options('fPlacement',['',...placements,'PENDING']);options('fFailure',['',...failures]);
function rebuild(){const current=queue[index]?.case_id;queue=doc.cases.filter(c=>{const o=obs(c),p=c.placement_result||'PENDING';return (!$('fRoute').value||o.actual_route===$('fRoute').value)&&(!$('fScene').value||o.actual_scene===$('fScene').value)&&(!$('fPlacement').value||p===$('fPlacement').value)&&(!$('fFailure').value||(c.failure_reasons||[]).includes($('fFailure').value))&&(!$('onlyStandard').checked||o.actual_route==='STANDARD_AVATAR')&&(!$('onlyFallback').checked||o.actual_route!=='STANDARD_AVATAR')});const sort=$('sort').value,res=c=>obs(c).residual_components?.final_normalized_residual??-1;queue.sort((a,b)=>sort==='residual-desc'?res(b)-res(a):sort==='residual-asc'?res(a)-res(b):a.case_id.localeCompare(b.case_id));index=Math.max(0,queue.findIndex(c=>c.case_id===current));render()}
function setRadio(name,value){document.querySelectorAll(`input[name=${name}]`).forEach(x=>x.checked=x.value===value)}
function render(){const reviewed=doc.cases.filter(c=>c.review_status==='REVIEWED').length;$('progress').textContent=`${reviewed}/${doc.cases.length} reviewed · ${queue.length} visible`;$('bar').style.width=(reviewed/doc.cases.length*100)+'%';if(!queue.length){$('caseId').textContent='NO CASES';return}const c=queue[index],o=obs(c),r=o.residual_components||{};$('caseId').textContent=c.case_id;$('meta').innerHTML=`route <b>${o.actual_route||'—'}</b> · scene <b>${o.actual_scene||'—'}</b><br>yaw ${o.yaw_deg?.toFixed(1)??'—'} · pitch ${o.pitch_deg?.toFixed(1)??'—'} · roll ${o.roll_deg?.toFixed(1)??'—'} · residual ${r.final_normalized_residual?.toFixed(4)??'—'}<br>asset ${o.selected_asset||'—'}`;showImages();setRadio('placement',c.placement_result);setRadio('route',c.route_judgment);$('scene').value=c.expected_scene||'';$('view').value=c.view||'';$('occlusion').value=c.occlusion_level||'UNKNOWN';document.querySelectorAll('#failures input').forEach(x=>x.checked=(c.failure_reasons||[]).includes(x.value));$('primary').value=c.primary_failure_reason||'';$('note').value=c.reviewer_note||'';$('reviewed').checked=c.review_status==='REVIEWED';$('trace').href=`cases/${c.case_id}/07-trace.json`;sessionStorage.setItem(key+':resume',c.case_id)}
function showImages(){const c=queue[index],left=files[view%files.length],right=files[(view+1)%files.length],root=`cases/${c.case_id}/`;$('leftImage').src=root+left[0];$('rightImage').src=root+right[0];$('leftCaption').textContent=left[1];$('rightCaption').textContent=right[1];$('thumbs').innerHTML=files.map((f,i)=>`<button class="${i===view?'on':''}" onclick="view=${i};showImages()">${f[1]}</button>`).join('')}
function update(){const c=queue[index];c.placement_result=document.querySelector('input[name=placement]:checked')?.value||null;c.route_judgment=document.querySelector('input[name=route]:checked')?.value||null;c.expected_scene=$('scene').value||null;c.view=$('view').value||null;c.occlusion_level=$('occlusion').value;c.failure_reasons=[...document.querySelectorAll('#failures input:checked')].map(x=>x.value);c.primary_failure_reason=$('primary').value||null;c.reviewer_note=$('note').value||null;c.review_status=$('reviewed').checked?'REVIEWED':'PENDING';if(c.review_status==='REVIEWED'&&(!c.placement_result||!c.route_judgment)){c.review_status='PENDING';$('reviewed').checked=false;$('status').textContent='needs placement + route judgment';return}if(c.placement_result==='FAIL'&&(!c.failure_reasons.length||!c.primary_failure_reason)){c.review_status='PENDING';$('reviewed').checked=false;$('status').textContent='FAIL needs taxonomy + primary reason';return}persist();render()}
document.querySelectorAll('.review input,.review select,.review textarea').forEach(x=>x.addEventListener('change',update));['fRoute','fScene','fPlacement','fFailure','sort','onlyStandard','onlyFallback'].forEach(id=>$(id).onchange=rebuild);$('prev').onclick=()=>{index=(index-1+queue.length)%queue.length;render()};$('next').onclick=()=>{index=(index+1)%queue.length;render()};$('export').onclick=()=>{const blob=new Blob([JSON.stringify(doc,null,2)],{type:'application/json'}),a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='reviewed-annotations.json';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),0)};window.onkeydown=e=>{if(['TEXTAREA','SELECT'].includes(document.activeElement.tagName))return;if(e.key==='ArrowLeft')$('prev').click();if(e.key==='ArrowRight')$('next').click();if('1234'.includes(e.key)){document.querySelectorAll('input[name=placement]')[+e.key-1].click()}if('qwer'.includes(e.key.toLowerCase())){document.querySelectorAll('input[name=route]')['qwer'.indexOf(e.key.toLowerCase())].click()}if(e.key.toLowerCase()==='s'){$('reviewed').click()}};const resume=sessionStorage.getItem(key+':resume');rebuild();if(resume){const i=queue.findIndex(c=>c.case_id===resume);if(i>=0){index=i;render()}}</script></html>'''.replace('__DOCUMENT__', raw_document)
    path = LOCAL_ROOT / "gallery.html"
    path.write_text(html, "utf8")
    return path


def write_report(document: BenchmarkDocument) -> dict:
    reports = {adapter: benchmark_metrics(document, adapter) for adapter in ADAPTER_IDS}
    reviewed = [case for case in document.cases if case.review_status == ReviewStatus.REVIEWED]
    baseline_id = "yunet_5pt_heuristic"
    blocked = [case for case in document.cases if case.observations.get(baseline_id) and case.observations[baseline_id].fallback_reason == "transform_residual_too_high"]
    components: dict[str, list[float]] = defaultdict(list)
    for case in blocked:
        for name, value in case.observations[baseline_id].residual_components.items():
            components[name].append(value)
    residual_analysis = {
        "blocked_cases": len(blocked),
        "components": {name: {"count": len(values), "mean": round(statistics.mean(values), 5), "median": round(statistics.median(values), 5)} for name, values in sorted(components.items())},
        "largest_normalized_components": [name for name, _ in sorted(
            ((name, statistics.mean(values)) for name, values in components.items() if name.endswith("_head_width")),
            key=lambda item: item[1], reverse=True,
        )[:5]],
    }
    segments = {}
    if len(reviewed) >= 30:
        segmenters = {
            "size": lambda c: "SMALL" if c.head_size_px < 96 else "MEDIUM" if c.head_size_px < 192 else "LARGE",
            "view": lambda c: c.view.value if c.view else "UNLABELED",
            "occlusion": lambda c: c.occlusion_level.value,
        }
        for dimension, classify in segmenters.items():
            segments[dimension] = {}
            for label in sorted({classify(case) for case in reviewed}):
                subset = [case for case in reviewed if classify(case) == label]
                subset_doc = document.model_copy(update={"cases": subset})
                segments[dimension][label] = benchmark_metrics(subset_doc, baseline_id)
    payload = {
        "distribution": anonymized_distribution(document),
        "formal_baseline": {"status": "ready" if len(reviewed) >= 30 else "insufficient_reviewed_cases", "minimum_reviewed": 30, "reviewed_cases": len(reviewed), "metrics": reports[baseline_id] if len(reviewed) >= 30 else None, "segments": segments},
        "metrics": reports,
        "residual_component_analysis": residual_analysis,
    }
    (LOCAL_ROOT / "metrics.json").write_text(json.dumps(payload, indent=2), "utf8")
    return payload


def validate(document: BenchmarkDocument, require_reviewed: bool) -> None:
    for case in document.cases:
        _local_image_path(case)
    if require_reviewed:
        pending = [case.case_id for case in document.cases if case.review_status != ReviewStatus.REVIEWED]
        if pending:
            raise RuntimeError(f"{len(pending)} cases still require human review")
    print(json.dumps(anonymized_distribution(document), ensure_ascii=False))


def import_review(path: Path | None) -> BenchmarkDocument:
    if path is None:
        raise ValueError("--review-file is required")
    reviewed = BenchmarkDocument.model_validate_json(path.read_text("utf8"))
    current = _load_document()
    if reviewed.benchmark_id != current.benchmark_id or {c.case_id for c in reviewed.cases} != {c.case_id for c in current.cases}:
        raise ValueError("review file does not match this benchmark")
    current_by_id = {case.case_id: case for case in current.cases}
    for case in reviewed.cases:
        original = current_by_id[case.case_id]
        immutable = ("local_image_ref", "source_sample_id", "source_region_id", "head_bbox", "head_size_px", "detector_source")
        if any(getattr(case, field) != getattr(original, field) for field in immutable):
            raise ValueError(f"review changed immutable geometry or local source for {case.case_id}")
    _save_document(reviewed)
    write_report(reviewed)
    return reviewed


def write_human_baseline(document: BenchmarkDocument) -> Path:
    report = analyze_human_baseline(document, LOCAL_ROOT / "cases", AVATAR_ASSET_DIR)
    destination = ROOT / "benchmarks" / "pose_avatar" / "reports" / "human-baseline-v0.1.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return destination


def main() -> int:
    args = parse_args()
    if args.command == "init":
        document = init_benchmark(args.count)
        print(json.dumps(anonymized_distribution(document), ensure_ascii=False))
    else:
        document = _load_document()
        if args.command == "run":
            document = run_benchmark(document, args.download_models)
        elif args.command == "validate":
            validate(document, args.require_reviewed)
        elif args.command == "report":
            print(json.dumps(write_report(document), ensure_ascii=False))
        elif args.command == "gallery":
            print(write_gallery(document))
        elif args.command == "import-review":
            validate(import_review(args.review_file), False)
        elif args.command == "calibrate":
            proposal = calibration_proposal(document)
            (LOCAL_ROOT / "calibration-proposal.json").write_text(json.dumps(proposal, indent=2), "utf8")
            print(json.dumps(proposal, ensure_ascii=False))
        elif args.command == "human-baseline":
            print(write_human_baseline(document))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
