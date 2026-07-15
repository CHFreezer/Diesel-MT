#!/usr/bin/env python3
"""Generate, replay, and finalize bounded train-only Hy-MT2 teacher data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from hymt2_distillation import DistillationError
from hymt2_distillation_data import finalize_d0, generate_d0, replay_d0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "action",
        choices=("dry-run", "generate", "replay", "finalize"),
    )
    result.add_argument(
        "--config",
        type=Path,
        default=Path("configs/hymt2_distillation.yaml"),
    )
    result.add_argument("--repository-root", type=Path, default=Path("."))
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    repository_root = args.repository_root.resolve()
    config_path = args.config.resolve()
    try:
        if args.action == "dry-run":
            report = generate_d0(repository_root, config_path, dry_run=True)
        elif args.action == "generate":
            report = generate_d0(repository_root, config_path, dry_run=False)
        elif args.action == "replay":
            report = replay_d0(repository_root, config_path)
        else:
            report = finalize_d0(repository_root, config_path)
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    except (DistillationError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
