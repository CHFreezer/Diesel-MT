#!/usr/bin/env python3
"""Finalize TD-03 samples with deterministic split, dedup, and leakage checks."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from model_data_split_pipeline import (
    SplitPipelineError,
    dry_run_plan,
    load_contamination_registry,
    load_td03_samples,
    prepare_finalized_samples,
    publish_finalized_data,
    registry_is_complete,
    scan_registry_references,
)
from model_training_contract import ContractError, load_model_data_config


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--config", type=Path, default=Path("configs/mvp_model_data.yaml"))
    result.add_argument(
        "--registry",
        type=Path,
        default=Path("configs/mvp_model_contamination.yaml"),
    )
    result.add_argument("--input-root", type=Path, default=Path("data/model"))
    result.add_argument("--out", type=Path, default=Path("data/model"))
    result.add_argument("--repo-root", type=Path, default=Path("."))
    result.add_argument("--dry-run", action="store_true")
    result.add_argument(
        "--allow-incomplete-references",
        action="store_true",
        help="development only: permit publication before formal MT evaluation identity is locked",
    )
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        config = load_model_data_config(args.config)
        registry = load_contamination_registry(args.registry)
        if args.dry_run:
            result = dry_run_plan(config, registry, args.input_root, args.out)
        else:
            if not args.allow_incomplete_references and not registry_is_complete(registry):
                raise SplitPipelineError(
                    "reference registry is incomplete; lock formal MT evaluation before publication"
                )
            samples, input_identity = load_td03_samples(args.input_root, config)
            prepared = prepare_finalized_samples(
                samples,
                config,
                derived_links=registry["derived_sample_links"],
            )
            reference_scan = scan_registry_references(
                prepared["candidate_entries"], registry, args.repo_root.resolve()
            )
            result = {
                "status": "complete",
                **publish_finalized_data(
                    prepared,
                    input_identity,
                    reference_scan,
                    args.out,
                    require_complete_references=not args.allow_incomplete_references,
                ),
            }
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    except (ContractError, SplitPipelineError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
