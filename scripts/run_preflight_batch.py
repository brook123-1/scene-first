from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx
from PIL import Image, ImageDraw, ImageOps


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import BENCHMARK_DIR, OUTPUT_DIR  # noqa: E402


MANIFEST = ROOT / "samples" / "manifest.csv"
INBOX = ROOT / "samples" / "inbox"
PREFLIGHT = BENCHMARK_DIR / "preflight.json"
ALLOWED_RIGHTS = {"owned", "consented", "licensed"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one confirmed anime-protection pass across reviewed samples.")
    parser.add_argument("--server", default="http://127.0.0.1:8765")
    parser.add_argument("--provider", default="ark")
    parser.add_argument("--prompt-profile", choices=["default", "balanced_portrait", "balanced_painterly"], default="balanced_portrait")
    parser.add_argument("--mask-profile", choices=["standard", "neck_blend"], default="standard")
    parser.add_argument("--parallel-images", type=int, default=2, choices=[1, 2])
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--confirm-cloud", action="store_true", help="Required before a cloud provider receives reviewed head crops.")
    return parser.parse_args()


def reviewed_samples() -> tuple[list[dict], dict]:
    if not PREFLIGHT.exists():
        raise RuntimeError("No preflight annotations found. Review all samples first.")
    annotations = json.loads(PREFLIGHT.read_text("utf8"))
    samples = []
    for raw in csv.DictReader(MANIFEST.open("r", encoding="utf-8-sig")):
        row = {key: (value or "").strip() for key, value in raw.items()}
        sample_id = row["id"]
        if row.get("rights", "").lower() not in ALLOWED_RIGHTS or row.get("contains_minors", "").lower() in {"yes", "true", "1"}:
            continue
        if sample_id not in annotations:
            raise RuntimeError(f"Sample {sample_id} is not reviewed.")
        paths = list(INBOX.rglob(Path(row["filename"]).name))
        if len(paths) != 1:
            raise RuntimeError(f"Sample {sample_id} source image is missing or duplicated.")
        regions = annotations[sample_id]["regions"]
        samples.append({"id": sample_id, "path": paths[0], "regions": regions})
    return samples, annotations


def wait_job(client: httpx.Client, server: str, job_id: str, people: int) -> dict:
    deadline = time.monotonic() + max(900, people * 150)
    while time.monotonic() < deadline:
        job = client.get(f"{server}/api/jobs/{job_id}").json()
        if job["status"] in {"completed", "failed"}:
            return job
        time.sleep(1)
    return {"id": job_id, "status": "failed", "error": "batch job timeout"}


def run_sample(sample: dict, args: argparse.Namespace) -> tuple[dict, dict | None]:
    selected = [region for region in sample["regions"] if region.get("selected", True)]
    if not selected:
        return {"sample_id": sample["id"], "status": "skipped", "reason": "no confirmed regions"}, None
    started = time.perf_counter()
    with httpx.Client(timeout=180, trust_env=False) as client:
        with sample["path"].open("rb") as handle:
            upload = client.post(f"{args.server}/api/detect", files={"file": (sample["path"].name, handle)})
        upload.raise_for_status()
        detected = upload.json()
        payload = {
            "image_id": detected["image_id"], "regions": sample["regions"], "mode": "anime",
            "provider": args.provider, "cloud_scope": "crop", "retry_nonce": 0,
            "selection_confirmed": True, "prompt_profile": args.prompt_profile,
            "mask_profile": args.mask_profile,
        }
        response = client.post(f"{args.server}/api/edit", json=payload)
        response.raise_for_status()
        job = wait_job(client, args.server, response.json()["job_id"], len(selected))
    record = {
        "sample_id": sample["id"], "status": job.get("status"), "job_id": job.get("id"),
        "elapsed_ms": round((time.perf_counter() - started) * 1000),
        "confirmed_people": len(selected), "outside_mask_exact": job.get("outside_mask_exact"),
        "estimated_cost_cny": job.get("estimated_cost_cny", 0), "people": job.get("people", []),
        "error": job.get("error"),
    }
    item = None
    if job.get("status") == "completed":
        item = {
            "id": sample["id"], "sample_id": sample["id"], "blind_label": sample["id"],
            "original_url": detected["image_url"], "result_url": job["output_url"],
        }
    return record, item


def contact_sheet(items: list[dict], destination: Path) -> None:
    cells = []
    for item in sorted(items, key=lambda value: value["sample_id"]):
        job_name = Path(item["result_url"]).name
        source = OUTPUT_DIR / job_name
        with Image.open(source) as image:
            preview = ImageOps.contain(image.convert("RGB"), (420, 280), Image.Resampling.LANCZOS)
        cell = Image.new("RGB", (440, 330), "#f2f0e9")
        cell.paste(preview, ((440 - preview.width) // 2, 8))
        ImageDraw.Draw(cell).text((12, 298), f"sample {item['sample_id']} · anime protection", fill="#10221a")
        cells.append(cell)
    columns = 4
    rows = max(1, (len(cells) + columns - 1) // columns)
    sheet = Image.new("RGB", (columns * 440, rows * 330), "#dedfd9")
    for index, cell in enumerate(cells):
        sheet.paste(cell, ((index % columns) * 440, (index // columns) * 330))
    sheet.save(destination, "JPEG", quality=90, optimize=True)


def main() -> int:
    args = parse_args()
    if args.provider != "local" and not args.confirm_cloud:
        print("Refusing cloud batch without --confirm-cloud.")
        return 2
    run_dir = BENCHMARK_DIR / "runs" / args.run_id
    if run_dir.exists():
        print(f"Run directory already exists: {args.run_id}")
        return 2
    run_dir.mkdir(parents=True)
    samples, _ = reviewed_samples()
    started_at = time.time()
    manifest = {
        "run_id": args.run_id, "provider": args.provider, "prompt_profile": args.prompt_profile,
        "mask_profile": args.mask_profile, "started_at": started_at,
        "samples": len(samples), "confirmed_people": sum(sum(r.get("selected", True) for r in s["regions"]) for s in samples),
        "status": "running",
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), "utf8")
    results: list[dict] = []
    items: list[dict] = []
    with ThreadPoolExecutor(max_workers=args.parallel_images) as executor:
        futures = {executor.submit(run_sample, sample, args): sample["id"] for sample in samples}
        for future in as_completed(futures):
            sample_id = futures[future]
            try:
                record, item = future.result()
            except Exception as exc:
                record, item = {"sample_id": sample_id, "status": "failed", "error": str(exc)}, None
            results.append(record)
            if item:
                items.append(item)
            (run_dir / "results.json").write_text(json.dumps(sorted(results, key=lambda value: value["sample_id"]), ensure_ascii=False, indent=2), "utf8")
            print(f"{sample_id}: {record['status']} ({len(results)}/{len(samples)})", flush=True)
    (run_dir / "review.json").write_text(json.dumps(sorted(items, key=lambda value: value["sample_id"]), ensure_ascii=False, indent=2), "utf8")
    if items:
        contact_sheet(items, run_dir / "result-contact-sheet.jpg")
    manifest.update({"status": "completed", "completed_at": time.time(), "completed_images": len(items), "failed_images": sum(r["status"] == "failed" for r in results)})
    (run_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), "utf8")
    print(json.dumps(manifest, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
