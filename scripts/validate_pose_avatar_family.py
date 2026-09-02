from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.pose_avatar.registry import AssetRegistry, AssetValidationError


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate one imported pose-aware avatar family.")
    parser.add_argument("family", type=Path, help="Family directory containing manifest.json")
    args = parser.parse_args()
    family = args.family.resolve()
    manifest = family / "manifest.json"
    if not manifest.is_file():
        raise SystemExit(f"missing {manifest}; copy and complete manifest.template.json first")
    try:
        registry = AssetRegistry(family.parent)
    except AssetValidationError as exc:
        raise SystemExit(f"avatar import failed: {exc}") from exc
    records = [record for record in registry.records if record.path.is_relative_to(family)]
    if not records:
        raise SystemExit(f"no enabled assets imported from {family}")
    scenes = ", ".join(record.scene.value for record in records)
    print(f"OK {records[0].family.family_id}@{records[0].family.version}: {len(records)} scenes [{scenes}]")


if __name__ == "__main__":
    main()
