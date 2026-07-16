"""Run or dry-run the bounded Diesel-MT student training loop."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mvp_training import _atomic_json, compare_training_runs, run_training


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "mvp_training_td10_smoke.yaml",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--compare-to", type=Path)
    parser.add_argument("--checkpoint-root", type=Path)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--stop-after-optimizer-step", type=int)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = run_training(
        config_path=args.config.resolve(),
        repository_root=ROOT,
        output_dir=args.output.resolve() if args.output else None,
        dry_run=args.dry_run,
        checkpoint_root=(
            args.checkpoint_root.resolve() if args.checkpoint_root else None
        ),
        resume_from=args.resume_from.resolve() if args.resume_from else None,
        stop_after_optimizer_steps=args.stop_after_optimizer_step,
    )
    if args.compare_to is not None and not args.dry_run:
        baseline = json.loads(args.compare_to.resolve().read_text(encoding="utf-8"))
        report = {
            **report,
            "determinism_replay": compare_training_runs(baseline, report),
        }
    if args.report is not None and not args.dry_run:
        _atomic_json(args.report.resolve(), report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
