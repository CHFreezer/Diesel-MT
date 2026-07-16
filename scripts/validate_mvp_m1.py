"""CLI for the TD-12 M1 overfit and exact-resume acceptance."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from mvp_m1 import run_m1_acceptance
from mvp_training import _atomic_json


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--acceptance-config",
        type=Path,
        default=ROOT / "configs" / "mvp_m1_acceptance.yaml",
    )
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT
        / "artifacts"
        / "model-training"
        / "reports"
        / "student"
        / "m1-overfit.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    runtime_root = args.runtime_root.resolve()
    if runtime_root.exists():
        raise RuntimeError(f"M1 runtime root already exists: {runtime_root}")
    report = run_m1_acceptance(
        repository_root=ROOT,
        acceptance_path=args.acceptance_config.resolve(),
        runtime_root=runtime_root,
    )
    _atomic_json(args.report.resolve(), report)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
