from __future__ import annotations

import json
import threading
import time
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any

from .config import COST_LEDGER_PATH, COST_OUTCOME_PATH, ENV, PROVIDER_SETTINGS


# Planning references captured at the time of an API call. These do not replace
# the provider invoice, which remains the source of truth for actual billing.
FAL_REFERENCE_USD = {
    "fal-ai/bytedance/seedream/v5/lite/edit": 0.035,
    "fal-ai/nano-banana-pro/edit": 0.15,
    "openai/gpt-image-2/edit": 0.219,  # adapter requests high quality
}
USD_TO_CNY_REFERENCE = float(ENV.get("COST_LEDGER_USD_CNY", "7.2"))


def pricing_snapshot(provider: str, model: str) -> dict[str, Any]:
    """Keep the cost basis used for one attempt, without recording credentials."""
    if provider == "local":
        return {"basis": "local", "reference_cny": 0.0, "quota_units": 0, "billing_status": "no_charge"}
    if provider == "ark":
        if ENV.get("ARK_BILLING_MODE", "agent_plan").strip().lower() == "payg":
            return {"basis": "ark_payg", "reference_cny": 0.22, "quota_units": 0, "billing_status": "reference"}
        return {"basis": "ark_agent_plan", "reference_cny": 0.0, "quota_units": 1, "billing_status": "quota"}
    if provider == "fal" and model in FAL_REFERENCE_USD:
        usd = FAL_REFERENCE_USD[model]
        return {"basis": "fal_public_reference", "reference_usd": usd, "reference_cny": round(usd * USD_TO_CNY_REFERENCE, 4), "quota_units": 0, "billing_status": "reference"}
    configured = PROVIDER_SETTINGS[provider].estimated_cny
    if configured > 0:
        return {"basis": "local_config_reference", "reference_cny": configured, "quota_units": 0, "billing_status": "reference"}
    return {"basis": "unpriced", "reference_cny": 0.0, "quota_units": 0, "billing_status": "needs_reconciliation"}


class CostLedger:
    """Append-only local model-attempt accounting; no image bytes or prompts."""

    def __init__(self, path: Path = COST_LEDGER_PATH, outcomes_path: Path = COST_OUTCOME_PATH) -> None:
        self.path = path
        self.outcomes_path = outcomes_path
        self.lock = threading.Lock()

    def record_attempt(self, *, job_id: str, image_id: str, audit: dict[str, Any], generation_attempted: bool) -> dict[str, Any]:
        provider = audit.get("attempted_provider", audit["provider"]) if generation_attempted else audit["provider"]
        model = audit.get("attempted_model", audit["provider_model"]) if generation_attempted else audit["provider_model"]
        generated = generation_attempted and audit["actual_mode"] != "safe" and not audit.get("fallback_reason")
        pricing = pricing_snapshot(provider, model) if generated else pricing_snapshot("local", "local-illustration-v1")
        entry = {
            "entry_id": uuid.uuid4().hex, "recorded_at": time.time(), "job_id": job_id, "image_id": image_id,
            "subject_id": audit["id"], "provider": provider, "model": model,
            "requested_mode": audit["requested_mode"], "actual_mode": audit["actual_mode"],
            "generation_attempted": generation_attempted,
            "result": "generated" if generated else ("safe_after_failure" if generation_attempted else "safe_without_api"),
            "fallback_reason": audit.get("fallback_reason"), "retry_nonce": audit.get("retry_nonce", 0),
            "elapsed_ms": audit["elapsed_ms"], "pricing": pricing,
        }
        with self.lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf8") as handle:
                handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
        return entry

    def mark_publishable(self, job_id: str, publishable: bool) -> dict[str, Any]:
        with self.lock:
            outcomes = self._read_outcomes()
            outcomes[job_id] = {"publishable": publishable, "updated_at": time.time()}
            self.outcomes_path.write_text(json.dumps(outcomes, ensure_ascii=False, indent=2), "utf8")
        return outcomes[job_id]

    def summary(self) -> dict[str, Any]:
        entries, outcomes = self._read_entries(), self._read_outcomes()
        jobs: dict[str, list[dict[str, Any]]] = defaultdict(list)
        images: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for entry in entries:
            jobs[entry["job_id"]].append(entry)
            images[entry["image_id"]].append(entry)
        generated = [item for item in entries if item["result"] == "generated"]
        fallbacks = [item for item in entries if item["result"] == "safe_after_failure"]
        direct_safe = [item for item in entries if item["result"] == "safe_without_api"]
        rows: dict[tuple[str, str], dict[str, Any]] = {}
        for item in entries:
            key = (item["provider"], item["model"])
            row = rows.setdefault(key, {"provider": key[0], "model": key[1], "attempts": 0, "generated": 0, "fallbacks": 0, "elapsed_ms": 0, "reference_cny": 0.0, "reference_usd": 0.0, "quota_units": 0})
            row["attempts"] += 1
            row["elapsed_ms"] += item["elapsed_ms"]
            row["generated"] += int(item["result"] == "generated")
            row["fallbacks"] += int(item["result"] == "safe_after_failure")
            if item["result"] == "generated":
                row["reference_cny"] += float(item["pricing"].get("reference_cny", 0))
                row["reference_usd"] += float(item["pricing"].get("reference_usd", 0))
                row["quota_units"] += int(item["pricing"].get("quota_units", 0))
        providers = []
        for row in rows.values():
            row["reference_cny"] = round(row["reference_cny"], 4)
            row["reference_usd"] = round(row["reference_usd"], 4)
            row["average_elapsed_ms"] = round(row["elapsed_ms"] / row["attempts"]) if row["attempts"] else 0
            providers.append(row)
        published_images = {jobs[job_id][0]["image_id"] for job_id, outcome in outcomes.items() if outcome.get("publishable") is True and jobs.get(job_id)}
        publishable_cny = sum(sum(float(item["pricing"].get("reference_cny", 0)) for item in images[image_id] if item["result"] == "generated") for image_id in published_images)
        return {
            "generated_since_enabled": len(generated), "safe_fallbacks": len(fallbacks), "safe_without_api": len(direct_safe),
            "photos_processed": len(images), "jobs_completed": len(jobs), "retries": sum(int(item.get("retry_nonce", 0) > 0) for item in entries),
            "reference_cny": round(sum(float(item["pricing"].get("reference_cny", 0)) for item in generated), 4),
            "reference_usd": round(sum(float(item["pricing"].get("reference_usd", 0)) for item in generated), 4),
            "quota_units": sum(int(item["pricing"].get("quota_units", 0)) for item in generated),
            "average_generated_elapsed_ms": round(sum(item["elapsed_ms"] for item in generated) / len(generated)) if generated else 0,
            "publishable_photos": len(published_images),
            "reference_cny_per_publishable_photo": round(publishable_cny / len(published_images), 4) if published_images else None,
            "providers": sorted(providers, key=lambda item: item["generated"], reverse=True),
            "note": "金额为调用时的公开价格或本地参考价；实际账单以供应商后台为准。方舟 Agent Plan 只记录额度消耗，不把套餐月费伪装成单次现金费用。",
        }

    def _read_entries(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        records = []
        for line in self.path.read_text("utf8").splitlines():
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return records

    def _read_outcomes(self) -> dict[str, dict[str, Any]]:
        if not self.outcomes_path.exists():
            return {}
        try:
            value = json.loads(self.outcomes_path.read_text("utf8"))
            return value if isinstance(value, dict) else {}
        except json.JSONDecodeError:
            return {}


cost_ledger = CostLedger()
