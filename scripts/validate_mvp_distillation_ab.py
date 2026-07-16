"""Validate and publish the frozen TD-15 A/B fairness record."""

from __future__ import annotations

import json
from pathlib import Path

from mvp_distillation_ab import validate_ab_release


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    result = validate_ab_release(
        repository_root=ROOT,
        config_path=ROOT / "configs/mvp_distillation_ab.yaml",
        human_recipe_path=ROOT / "configs/mvp_training_m2_human.yaml",
        distilled_recipe_path=ROOT / "configs/mvp_training_m2_distilled.yaml",
        report_path=ROOT / "artifacts/model-training/reports/m2/distillation-ab.json",
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
