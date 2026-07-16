"""Build the frozen TD-15 source-matched human/distilled cohort."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mvp_distillation_ab import build_ab_cohort


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/mvp_distillation_ab.yaml")
    args = parser.parse_args()
    result = build_ab_cohort(repository_root=ROOT, config_path=args.config.resolve())
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
