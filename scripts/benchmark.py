from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import httpx


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "samples" / "manifest.csv"
INBOX = ROOT / "samples" / "inbox"
BENCHMARK_DIR = ROOT / ".local" / "app" / "benchmark"
PREFLIGHT = BENCHMARK_DIR / "preflight.json"
ALLOWED_RIGHTS = {"owned", "consented", "licensed"}


def parse_args():
    parser = argparse.ArgumentParser(description="Run a one-output-per-provider blind benchmark.")
    parser.add_argument("--server", default="http://127.0.0.1:8765")
    parser.add_argument("--providers", default="local,fal,openai,gemini,bfl,qwen")
    parser.add_argument("--cloud-scope", choices=["crop", "full"], default="crop")
    parser.add_argument("--confirm-cloud", action="store_true", help="Required before any non-local provider receives images.")
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--sample-ids", help="Comma-separated reviewed sample IDs for a small calibration run.")
    parser.add_argument("--prompt-profile", choices=["default", "balanced_portrait", "balanced_painterly"], default="default")
    parser.add_argument("--mask-profile", choices=["standard", "neck_blend"], default="standard")
    return parser.parse_args()


def load_samples(limit: int) -> list[dict]:
    rows = [
        {key: (value or "").strip() for key, value in row.items()}
        for row in csv.DictReader(MANIFEST.open("r", encoding="utf-8-sig"))
    ]
    valid = []
    for row in rows:
        if row["id"].startswith("example-"):
            continue
        if row.get("rights", "").strip().lower() not in ALLOWED_RIGHTS:
            print(f"Skip {row.get('id')}: rights must be owned, consented, or licensed.")
            continue
        if row.get("contains_minors", "").strip().lower() in {"yes", "true", "1"}:
            print(f"Skip {row.get('id')}: first-round samples containing minors are excluded.")
            continue
        matches = list(INBOX.rglob(Path(row["filename"]).name))
        if not matches:
            print(f"Skip {row.get('id')}: missing {Path(row['filename']).name}.")
            continue
        if len(matches) > 1:
            print(f"Skip {row.get('id')}: duplicate filename found in sample inbox.")
            continue
        path = matches[0]
        row["path"] = path
        valid.append(row)
    return valid[:limit]


def wait_job(client: httpx.Client, server: str, job_id: str) -> dict:
    for _ in range(600):
        job = client.get(f"{server}/api/jobs/{job_id}").json()
        if job["status"] in {"completed", "failed"}:
            return job
        time.sleep(0.75)
    return {"id": job_id, "status": "failed", "error": "benchmark timeout"}


def main() -> int:
    args = parse_args()
    providers = [item.strip() for item in args.providers.split(",") if item.strip()]
    if any(provider != "local" for provider in providers) and not args.confirm_cloud:
        print("Refusing cloud benchmark without --confirm-cloud. Review image rights and provider policies first.")
        return 2
    samples = load_samples(args.limit)
    if args.sample_ids:
        wanted = {value.strip() for value in args.sample_ids.split(",") if value.strip()}
        samples = [sample for sample in samples if sample["id"] in wanted]
        missing = wanted - {sample["id"] for sample in samples}
        if missing:
            print(f"Unknown or ineligible sample IDs: {', '.join(sorted(missing))}")
            return 2
    if not samples:
        print("No eligible samples. Put owned/consented photos in samples/inbox and edit samples/manifest.csv.")
        return 2

    if not PREFLIGHT.exists():
        print("No reviewed preflight annotations. Open /preflight and review every sample first.")
        return 2
    annotations = json.loads(PREFLIGHT.read_text("utf8"))
    unreviewed = [sample["id"] for sample in samples if sample["id"] not in annotations]
    if unreviewed:
        print(f"Refusing benchmark: unreviewed samples: {', '.join(unreviewed)}")
        return 2

    BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
    results_path = BENCHMARK_DIR / "results.jsonl"
    review_items = []
    with httpx.Client(timeout=180, trust_env=False) as client:
        available = {item["id"]: item for item in client.get(f"{args.server}/api/providers").json()["providers"]}
        unavailable = [provider for provider in providers if provider not in available or not available[provider]["configured"]]
        if unavailable:
            print(f"Requested providers are not configured: {', '.join(unavailable)}")
            return 2
        with results_path.open("a", encoding="utf8") as result_file:
            for sample in samples:
                with sample["path"].open("rb") as handle:
                    detected_response = client.post(f"{args.server}/api/detect", files={"file": (sample["path"].name, handle)})
                detected_response.raise_for_status()
                detected = detected_response.json()
                regions = annotations[sample["id"]]["regions"]
                if not any(region.get("selected", True) for region in regions):
                    print(f"{sample['id']}: no selected regions after preflight; skipped.")
                    continue
                for provider in providers:
                    started = time.perf_counter()
                    payload = {
                        "image_id": detected["image_id"], "regions": regions, "mode": "anime",
                        "provider": provider, "cloud_scope": args.cloud_scope, "retry_nonce": 0,
                        # Preflight annotations are the user's explicit one-time
                        # confirmation.  Selected regions therefore receive the
                        # requested anime route even when their original proposal
                        # came from a low-confidence detector.
                        "selection_confirmed": True,
                        "prompt_profile": args.prompt_profile,
                        "mask_profile": args.mask_profile,
                    }
                    response = client.post(f"{args.server}/api/edit", json=payload)
                    if not response.is_success:
                        record = {"sample_id": sample["id"], "provider": provider, "status": "failed", "error": response.text}
                        result_file.write(json.dumps(record, ensure_ascii=False) + "\n"); result_file.flush()
                        continue
                    job = wait_job(client, args.server, response.json()["job_id"])
                    record = {
                        "sample_id": sample["id"], "provider": provider, "cloud_scope": args.cloud_scope,
                        "status": job["status"], "job_id": job.get("id"), "elapsed_ms": round((time.perf_counter()-started)*1000),
                        "estimated_cost_cny": job.get("estimated_cost_cny", 0), "outside_mask_exact": job.get("outside_mask_exact"),
                        "people": job.get("people", []), "error": job.get("error"),
                    }
                    result_file.write(json.dumps(record, ensure_ascii=False) + "\n"); result_file.flush()
                    if job["status"] == "completed":
                        blind = hashlib.sha256(f"{sample['id']}:{provider}:scene-first".encode()).hexdigest()[:5].upper()
                        review_items.append({
                            "id": f"{sample['id']}-{blind}", "sample_id": sample["id"], "blind_label": blind,
                            "provider": provider, "original_url": detected["image_url"], "result_url": job["output_url"],
                        })
                    print(f"{sample['id']} / {provider}: {job['status']}")
    (BENCHMARK_DIR / "review.json").write_text(json.dumps(review_items, ensure_ascii=False, indent=2), "utf8")
    print(f"Done. Open {args.server}/review for blinded ratings.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
