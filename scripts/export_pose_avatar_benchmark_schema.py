from __future__ import annotations

import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.pose_avatar.benchmark import BenchmarkDocument  # noqa: E402


def main() -> None:
    destination = ROOT / "benchmarks" / "pose_avatar" / "annotation.schema.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(BenchmarkDocument.model_json_schema(), ensure_ascii=False, indent=2) + "\n",
        "utf8",
    )
    print(destination)


if __name__ == "__main__":
    main()
