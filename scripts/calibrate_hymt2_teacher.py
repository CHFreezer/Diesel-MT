#!/usr/bin/env python3
"""Calibrate the frozen Hy-MT2 teacher on bounded human dev references."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

from hymt2_distillation import (
    ROUTES,
    DistillationError,
    LlamaCppTeacher,
    atomic_write_json,
    atomic_write_jsonl,
    canonical_json_bytes,
    config_sha256,
    deterministic_route_sample,
    load_prompt_config,
    read_parallel_jsonl,
    route_gate_failures,
    run_profile,
    sha256_bytes,
    sha256_file,
    summarize_generation_records,
)


PIPELINE_VERSION = "td07-hymt2-calibration-v1"


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--config",
        type=Path,
        default=Path("configs/hymt2_teacher_prompt_decode.yaml"),
    )
    result.add_argument("--repository-root", type=Path, default=Path("."))
    result.add_argument("--dry-run", action="store_true")
    return result


def _replay_samples(samples: Sequence[Mapping[str, Any]], per_route: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for route in ROUTES:
        route_records = [sample for sample in samples if f"{sample['src_lang']}->{sample['tgt_lang']}" == route]
        selected.extend(dict(sample) for sample in route_records[:per_route])
    return selected


def _replay_result(
    original: Sequence[Mapping[str, Any]],
    replayed: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    original_by_key = {
        (str(record["profile"]), str(record["sample_id"])): record for record in original
    }
    mismatches: list[dict[str, str]] = []
    for record in replayed:
        key = (str(record["profile"]), str(record["sample_id"]))
        expected = original_by_key[key]
        if (
            record["raw_output_sha256"] != expected["raw_output_sha256"]
            or record["normalized_output_sha256"] != expected["normalized_output_sha256"]
        ):
            mismatches.append(
                {
                    "profile": key[0],
                    "sample_id": key[1],
                    "expected_raw_sha256": str(expected["raw_output_sha256"]),
                    "actual_raw_sha256": str(record["raw_output_sha256"]),
                }
            )
    return {
        "records": len(replayed),
        "exact": not mismatches,
        "mismatches": mismatches,
        "records_sha256": sha256_bytes(b"".join(canonical_json_bytes(record) for record in replayed)),
    }


def _decision(
    summaries: Mapping[str, Mapping[str, Any]],
    replays: Mapping[str, Mapping[str, Any]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    greedy = summaries["greedy-v1"]
    sampling = summaries["official-sampling-v1"]
    greedy_failures = route_gate_failures(greedy, config)
    sampling_failures = route_gate_failures(sampling, config)
    delta = float(sampling["macro"]["chrf"]) - float(greedy["macro"]["chrf"])
    minimum_delta = float(config["selection"]["sampling_minimum_macro_chrf_delta"])
    no_format_regression = (
        float(sampling["macro"]["accepted_rate"]) >= float(greedy["macro"]["accepted_rate"])
        and float(sampling["macro"]["script_compliance_rate"])
        >= float(greedy["macro"]["script_compliance_rate"])
    )
    sampling_eligible = (
        not sampling_failures
        and bool(replays["official-sampling-v1"]["exact"])
        and delta >= minimum_delta
        and no_format_regression
    )
    selected = "official-sampling-v1" if sampling_eligible else str(config["selection"]["tie_breaker"])
    selected_failures = sampling_failures if selected == "official-sampling-v1" else greedy_failures
    return {
        "selected_profile": selected,
        "configured_profile": config["selection"]["selected_profile"],
        "selection_matches_config": selected == config["selection"]["selected_profile"],
        "greedy_route_gate_failures": greedy_failures,
        "sampling_route_gate_failures": sampling_failures,
        "sampling_macro_chrf_delta": round(delta, 6),
        "sampling_minimum_macro_chrf_delta": minimum_delta,
        "sampling_no_format_regression": no_format_regression,
        "sampling_eligible": sampling_eligible,
        "selected_route_gate_failures": selected_failures,
        "selected_passes_all_gates": not selected_failures,
    }


def calibrate(repository_root: Path, config_path: Path, *, dry_run: bool) -> dict[str, Any]:
    repository_root = repository_root.resolve()
    config_path = (repository_root / config_path).resolve() if not config_path.is_absolute() else config_path.resolve()
    config = load_prompt_config(config_path)
    input_path = repository_root / PurePosixPath(str(config["calibration_input"]["path"]))
    records = read_parallel_jsonl(
        input_path,
        expected_split="dev",
        expected_sha256=str(config["calibration_input"]["file_sha256"]),
    )
    samples = deterministic_route_sample(
        records,
        per_route=int(config["calibration_input"]["samples_per_route"]),
        seed=str(config["calibration_input"]["selection_seed"]),
    )
    sample_identity = sha256_bytes(
        b"".join(canonical_json_bytes({"sample_id": sample["sample_id"]}) for sample in samples)
    )
    common = {
        "schema_version": 1,
        "pipeline_version": PIPELINE_VERSION,
        "prompt_config_sha256": config_sha256(config),
        "prompt_config_file_sha256": sha256_file(config_path),
        "input": {
            "path": config["calibration_input"]["path"],
            "sha256": sha256_file(input_path),
            "split": "dev",
            "available_records": len(records),
            "selected_records": len(samples),
            "selected_per_route": config["calibration_input"]["samples_per_route"],
            "selection_sha256": sample_identity,
        },
        "test_accessed": False,
    }
    if dry_run:
        return {**common, "status": "dry-run", "profiles": list(config["decode_profiles"])}

    started = time.perf_counter()
    profile_records: dict[str, list[dict[str, Any]]] = {}
    replay_reports: dict[str, dict[str, Any]] = {}
    replay_samples = _replay_samples(samples, int(config["calibration_gates"]["replay_samples_per_route"]))
    with LlamaCppTeacher(repository_root, config) as teacher:
        for profile_name in config["decode_profiles"]:
            generated = run_profile(
                teacher,
                samples,
                profile_name=str(profile_name),
                config=config,
            )
            profile_records[str(profile_name)] = generated
            replayed = run_profile(
                teacher,
                replay_samples,
                profile_name=str(profile_name),
                config=config,
            )
            replay_reports[str(profile_name)] = _replay_result(generated, replayed)
        runtime_evidence = {
            "command": teacher.command,
            "server_log_tail": teacher.logs[-80:],
            "model": str(teacher.paths["model"]),
            "model_bytes": teacher.paths["model"].stat().st_size,
            "model_sha256": sha256_file(teacher.paths["model"]),
        }

    summaries = {
        name: summarize_generation_records(generated, config)
        for name, generated in profile_records.items()
    }
    decision = _decision(summaries, replay_reports, config)
    all_records = [record for name in config["decode_profiles"] for record in profile_records[str(name)]]
    output_path = repository_root / PurePosixPath(str(config["outputs"]["records"]))
    atomic_write_jsonl(output_path, all_records)
    status = "complete"
    if not decision["selection_matches_config"]:
        status = "selection-mismatch"
    elif not decision["selected_passes_all_gates"]:
        status = "blocked-route-gates"
    elif config["calibration_gates"]["require_exact_replay"] and not replay_reports[decision["selected_profile"]]["exact"]:
        status = "blocked-replay"
    report = {
        **common,
        "status": status,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(time.perf_counter() - started, 6),
        "profiles": summaries,
        "replay": replay_reports,
        "decision": decision,
        "records": {
            "path": config["outputs"]["records"],
            "records": len(all_records),
            "bytes": output_path.stat().st_size,
            "sha256": sha256_file(output_path),
        },
        "runtime": runtime_evidence,
    }
    report_path = repository_root / PurePosixPath(str(config["outputs"]["report"]))
    atomic_write_json(report_path, report)
    if status != "complete":
        raise DistillationError(f"TD-07 calibration did not close: {status}")
    return report


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        report = calibrate(args.repository_root, args.config, dry_run=args.dry_run)
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    except (DistillationError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
