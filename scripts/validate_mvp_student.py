"""CLI for the TD-09 20-route student acceptance gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mvp_student import EncodingPolicy, _atomic_write_json, run_td09_acceptance


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "mvp_e8_d2_v48k.yaml",
    )
    parser.add_argument(
        "--fixture",
        type=Path,
        default=ROOT
        / "data"
        / "model"
        / "corpus"
        / "mvp"
        / "finalized"
        / "train.jsonl",
    )
    parser.add_argument(
        "--checkpoint-output",
        type=Path,
        default=ROOT
        / "artifacts"
        / "model-training"
        / "runtime"
        / "student-checkpoint",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT
        / "artifacts"
        / "model-training"
        / "reports"
        / "student"
        / "encoding-validation.json",
    )
    parser.add_argument("--max-source-length", type=int, default=128)
    parser.add_argument("--max-target-length", type=int, default=128)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = run_td09_acceptance(
        repository_root=ROOT,
        student_config_path=args.config.resolve(),
        fixture_path=args.fixture.resolve(),
        checkpoint_output=args.checkpoint_output.resolve(),
        policy=EncodingPolicy(
            max_source_length=args.max_source_length,
            max_target_length=args.max_target_length,
        ),
    )
    _atomic_write_json(args.report.resolve(), report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
