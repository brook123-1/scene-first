from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str) -> dict:
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


def test_staging_is_independent_and_cannot_claim_production_domain() -> None:
    production = _load("wrangler.jsonc")
    staging = _load("wrangler.staging.jsonc")

    assert staging["name"] != production["name"]
    assert staging["workers_dev"] is True
    assert "routes" not in staging
    assert staging["vars"]["SCENE_FIRST_LOCAL_MASTER"] == "1"
    assert "SCENE_FIRST_TEST_PASSWORD" not in staging["vars"]
    assert staging["containers"][0]["max_instances"] == 1


def test_checked_in_cloudflare_configs_are_author_neutral() -> None:
    production = _load("wrangler.jsonc")
    staging = _load("wrangler.staging.jsonc")

    assert "account_id" not in production
    assert "routes" not in production
    assert "account_id" not in staging


def test_production_local_master_is_explicitly_enabled() -> None:
    production = _load("wrangler.jsonc")

    assert production["vars"]["SCENE_FIRST_LOCAL_MASTER"] == "1"
