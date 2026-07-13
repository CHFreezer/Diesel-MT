#!/usr/bin/env python3
"""Fetch and build deterministic tokenizer corpora from a source lock."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tokenizer_dataset_pipeline import (
    PipelineError,
    build_corpus,
    dry_run_plan,
    load_config,
    load_lock,
    resolve_lock,
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--config", type=Path, default=Path("configs/tokenizer_datasets_mvp.yaml"))
    result.add_argument("--lock", type=Path, default=Path("configs/tokenizer_datasets_mvp.lock.json"))
    result.add_argument("--out", type=Path, default=Path("data/tokenizer"))
    result.add_argument("--cache-dir", type=Path, help="shared cache directory (default: OUT/cache)")
    result.add_argument("--staging-dir", type=Path, help="fast local staging root for completed corpus files before verified background transfer")
    result.add_argument("--profile", choices=("smoke", "mvp"), default="smoke")
    result.add_argument("--seed", type=int, help="override the profile seed")
    result.add_argument("--concurrency", type=int, help="accepted for reproducibility tests; logical processing order remains locked")
    result.add_argument("--dry-run", action="store_true")
    result.add_argument("--use-cache", action="store_true", help="require validated cache and do not access the network")
    result.add_argument("--offline", action="store_true", help="prohibit all network access")
    result.add_argument("--resolve-lock", action="store_true", help="explicitly resolve remote maps and locked prefixes, then replace LOCK")
    result.add_argument("--resume", action="store_true", help="reuse completed languages in OUT/interim and restart only an incomplete language")
    result.add_argument("--max-memory-gib", type=float, help="RAM-first main-process RSS safety limit")
    result.add_argument("--min-available-memory-gib", type=float, help="stop safely if system available RAM falls below this value")
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        config = load_config(args.config)
        profile = config["profiles"][args.profile]
        seed = int(args.seed if args.seed is not None else profile["random_seed"])
        if args.concurrency is not None:
            if args.concurrency <= 0:
                raise PipelineError("--concurrency must be positive")
            profile["concurrency"] = args.concurrency
        cache_root = args.cache_dir if args.cache_dir else args.out / "cache"
        if args.resolve_lock:
            if args.offline or args.use_cache or args.dry_run:
                raise PipelineError("--resolve-lock requires network access and cannot be combined with --offline, --use-cache, or --dry-run")
            lock = resolve_lock(config, args.config, args.lock, cache_root, args.profile)
            print(json.dumps({"status": "lock-resolved", "lock": str(args.lock), "profile": args.profile, "sources": len(lock["sources"])}, ensure_ascii=False, sort_keys=True))
            return 0
        lock = load_lock(args.lock, config, args.profile, args.config)
        if args.dry_run:
            print(
                json.dumps(
                    dry_run_plan(
                        config,
                        lock,
                        args.out,
                        cache_root,
                        args.profile,
                        seed,
                        args.offline,
                        args.use_cache,
                        staging_root=args.staging_dir,
                        max_memory_gib=args.max_memory_gib,
                        min_available_memory_gib=args.min_available_memory_gib,
                    ),
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        result = build_corpus(
            config,
            args.config,
            lock,
            args.lock,
            args.out,
            cache_root,
            args.profile,
            seed,
            offline=args.offline,
            use_cache=args.use_cache,
            resume=args.resume,
            staging_root=args.staging_dir,
            max_memory_gib=args.max_memory_gib,
            min_available_memory_gib=args.min_available_memory_gib,
        )
        print(json.dumps({"status": "complete", **result}, ensure_ascii=False, sort_keys=True))
        return 0
    except PipelineError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return exc.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
