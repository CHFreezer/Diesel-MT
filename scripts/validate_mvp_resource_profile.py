"""Publish the complete TD-14 selected-profile acceptance record."""

from __future__ import annotations

import json
from pathlib import Path

from mvp_resource_profile import build_td14_evidence


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    result = build_td14_evidence(
        repository_root=ROOT,
        profile_path=ROOT / "configs/mvp_training_m2_profile.yaml",
        benchmark_path=ROOT / "artifacts/model-training/reports/m2/resources/candidate-benchmark.json",
        soak_config_path=ROOT / "configs/mvp_training_td14_soak.yaml",
        soak_report_path=ROOT / "artifacts/model-training/reports/m2/resources/soak-run.json",
        resume_report_path=ROOT / "artifacts/model-training/reports/m2/resources/resume-probe.json",
        output_path=ROOT / "artifacts/model-training/reports/m2/resources/profile.json",
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
