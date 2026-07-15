#!/usr/bin/env python3
"""Build the immutable 20-route D1 distilled composite."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from distillation_composite import CompositeError, build_composite


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--config",
        type=Path,
        default=Path("configs/hymt2_distillation_d1_20route_composite.yaml"),
    )
    result.add_argument("--repository-root", type=Path, default=Path("."))
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        result = build_composite(args.repository_root, args.config)
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    except (CompositeError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
