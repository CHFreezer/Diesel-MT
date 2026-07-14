#!/usr/bin/env python3
"""Prepare manual review or publish the accepted TD-05 M0 model dataset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from m0_dataset_acceptance import (
    M0AcceptanceError,
    accept_m0,
    load_sampling_config,
    prepare_review,
)
from model_training_contract import ContractError, load_model_data_config


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--root", type=Path, default=Path("data/model"))
    result.add_argument(
        "--rebuild-root",
        type=Path,
        default=Path("data/model/interim/td05-rebuild"),
    )
    result.add_argument("--config", type=Path, default=Path("configs/mvp_model_data.yaml"))
    result.add_argument(
        "--sampling",
        type=Path,
        default=Path("configs/mvp_direction_sampling.yaml"),
    )
    result.add_argument(
        "--attestation",
        type=Path,
        default=Path("configs/m0_manual_review.yaml"),
    )
    result.add_argument("--prepare-review", action="store_true")
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        config = load_model_data_config(args.config)
        sampling = load_sampling_config(args.sampling)
        if args.prepare_review:
            result = prepare_review(args.root, config, sampling)
        else:
            result = accept_m0(
                args.root,
                args.rebuild_root,
                config,
                sampling,
                args.attestation,
            )
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    except (ContractError, M0AcceptanceError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
