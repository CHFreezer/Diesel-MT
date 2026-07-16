"""Run TD-14 real-distribution student resource candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mvp_resource_benchmark import benchmark_candidates


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/mvp_training_td14_candidates.yaml")
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=ROOT / "artifacts/model-training/reports/m2/resources/candidate-benchmark.json")
    args = parser.parse_args()
    result = benchmark_candidates(
        repository_root=ROOT,
        meta_path=args.config.resolve(),
        runtime_root=args.runtime_root.resolve(),
        report_path=args.report.resolve(),
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
