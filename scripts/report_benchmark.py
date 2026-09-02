from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_DIR = ROOT / ".local" / "app" / "benchmark"


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text("utf8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 2) if values else None


def main() -> int:
    results = read_jsonl(BENCHMARK_DIR / "results.jsonl")
    ratings = read_jsonl(BENCHMARK_DIR / "ratings.jsonl")
    review_path = BENCHMARK_DIR / "review.json"
    review_items = json.loads(review_path.read_text("utf8")) if review_path.exists() else []
    provider_by_item = {item["id"]: item.get("provider", "unknown") for item in review_items}
    rating_groups: dict[str, list[dict]] = defaultdict(list)
    for rating in ratings:
        rating_groups[provider_by_item.get(rating["item_id"], "unknown")].append(rating)

    result_groups: dict[str, list[dict]] = defaultdict(list)
    for result in results:
        result_groups[result.get("provider", "unknown")].append(result)

    providers = sorted(set(result_groups) | set(rating_groups))
    rows = []
    for provider in providers:
        provider_results = result_groups[provider]
        completed = [row for row in provider_results if row.get("status") == "completed"]
        provider_ratings = rating_groups[provider]
        publishable = [row for row in provider_ratings if row.get("publishable")]
        exact = [row for row in completed if row.get("outside_mask_exact") is True]
        fallbacks = sum(
            1 for result in completed for person in result.get("people", []) if person.get("actual_mode") == "safe"
        )
        people = sum(len(result.get("people", [])) for result in completed)
        rows.append({
            "provider": provider,
            "runs": len(provider_results),
            "completed": len(completed),
            "rated": len(provider_ratings),
            "publishable_rate_pct": round(len(publishable) / len(provider_ratings) * 100, 1) if provider_ratings else None,
            "passes_70_pct_gate": len(provider_ratings) >= 20 and len(publishable) / len(provider_ratings) >= 0.70,
            "naturalness_avg": mean([row["naturalness"] for row in provider_ratings]),
            "privacy_avg": mean([row["privacy"] for row in provider_ratings]),
            "outside_mask_exact_rate_pct": round(len(exact) / len(completed) * 100, 1) if completed else None,
            "fallback_rate_pct": round(fallbacks / people * 100, 1) if people else None,
            "latency_avg_ms": mean([row.get("elapsed_ms", 0) for row in completed]),
            "estimated_cost_cny": round(sum(row.get("estimated_cost_cny", 0) for row in completed), 2),
        })

    BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
    (BENCHMARK_DIR / "summary.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), "utf8")
    csv_path = BENCHMARK_DIR / "summary.csv"
    if rows:
        with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps(rows, ensure_ascii=False, indent=2))
    if not rows:
        print("尚无竞赛结果。先运行 scripts/benchmark.py 并完成盲测评分。")
    else:
        print(f"汇总已写入 {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
