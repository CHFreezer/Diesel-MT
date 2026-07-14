#!/usr/bin/env python3
"""Build canonical MVP parallel data from its immutable source lock."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from model_data_pipeline import PipelineError, build_model_data, dry_run_plan
from model_training_contract import ContractError, load_model_data_config, load_source_lock


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--config", type=Path, default=Path("configs/mvp_model_data.yaml"))
    result.add_argument("--lock", type=Path, default=Path("configs/mvp_model_data.lock.json"))
    result.add_argument("--out", type=Path, default=Path("data/model"))
    result.add_argument("--cache-dir", type=Path, help="shared archive cache (default: OUT/cache)")
    result.add_argument("--dry-run", action="store_true")
    result.add_argument("--use-cache", action="store_true", help="require a validated cache and prohibit network access")
    result.add_argument("--offline", action="store_true", help="prohibit all network access")
    result.add_argument("--resume", action="store_true", help="reuse verified per-locale normalization checkpoints")
    result.add_argument("--timeout", type=int, default=120, help="HTTP timeout in seconds")
    result.add_argument("--retries", type=int, default=4, help="locked archive download attempts")
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.timeout <= 0 or args.retries <= 0:
            raise PipelineError("--timeout and --retries must be positive")
        config = load_model_data_config(args.config)
        lock = load_source_lock(args.lock, config)
        cache_root = args.cache_dir if args.cache_dir is not None else args.out / "cache"
        if args.dry_run:
            result = dry_run_plan(
                config,
                lock,
                args.out,
                cache_root,
                offline=args.offline,
                use_cache=args.use_cache,
                resume=args.resume,
            )
        else:
            result = {
                "status": "complete",
                **build_model_data(
                    config,
                    lock,
                    args.out,
                    cache_root,
                    offline=args.offline,
                    use_cache=args.use_cache,
                    resume=args.resume,
                    timeout=args.timeout,
                    retries=args.retries,
                ),
            }
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    except (ContractError, PipelineError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

