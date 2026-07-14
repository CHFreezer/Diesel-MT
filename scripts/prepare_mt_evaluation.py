#!/usr/bin/env python3
"""Materialize the locked FLORES-200 external MT evaluation reference."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from model_data_pipeline import PipelineError
from mt_evaluation_reference import EvaluationReferenceError, dry_run_plan, load_lock, prepare_reference


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--lock", type=Path, default=Path("configs/mvp_mt_evaluation.lock.json"))
    result.add_argument("--out", type=Path, default=Path("data/model"))
    result.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("data/model/cache/flores200-original"),
    )
    result.add_argument("--dry-run", action="store_true")
    result.add_argument("--offline", action="store_true")
    result.add_argument("--timeout", type=int, default=120)
    result.add_argument("--retries", type=int, default=4)
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.timeout <= 0 or args.retries <= 0:
            raise EvaluationReferenceError("--timeout and --retries must be positive")
        lock = load_lock(args.lock)
        if args.dry_run:
            result = dry_run_plan(lock, args.out, args.cache_dir, offline=args.offline)
        else:
            result = {
                "status": "complete",
                **prepare_reference(
                    lock,
                    args.out,
                    args.cache_dir,
                    offline=args.offline,
                    timeout=args.timeout,
                    retries=args.retries,
                ),
            }
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    except (EvaluationReferenceError, PipelineError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
