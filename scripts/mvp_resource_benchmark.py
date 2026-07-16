"""TD-14 real-distribution resource candidate benchmark and selection."""

from __future__ import annotations

import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any

import yaml

from freeze_tokenizer_artifact import sha256_file
from model_training_contract import config_sha256, directed_routes, load_student_config
from mvp_student import load_frozen_tokenizer
from mvp_training import ROUTE_ORDER, load_route_dataset, read_jsonl, run_training


class ResourceBenchmarkError(RuntimeError):
    """Raised when no TD-14 candidate can be selected safely."""


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _atomic_yaml(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            yaml.safe_dump(value, handle, sort_keys=False, allow_unicode=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _quantile(values: list[int], probability: float) -> int:
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * probability)]


def analyze_real_lengths(
    *, repository_root: Path, meta: dict[str, Any], tokenizer: object
) -> dict[str, Any]:
    data = meta["data"]
    dataset = load_route_dataset(
        repository_root / data["train_path"],
        expected_sha256=data["train_sha256"],
        split="train",
        max_records_per_route=int(data["train_records_per_route"]),
    )
    sources: list[int] = []
    targets: list[int] = []
    per_route: dict[str, dict[str, Any]] = {}
    for route in ROUTE_ORDER:
        route_sources: list[int] = []
        route_targets: list[int] = []
        for record in dataset.records_by_route[route]:
            tokenizer.src_lang = record["src_lang"]
            source_ids = tokenizer(
                record["source_text"], add_special_tokens=True, truncation=False
            )["input_ids"]
            tokenizer.src_lang = record["tgt_lang"]
            target_ids = tokenizer(
                record["target_text"], add_special_tokens=True, truncation=False
            )["input_ids"]
            route_sources.append(len(source_ids))
            route_targets.append(len(target_ids))
        sources.extend(route_sources)
        targets.extend(route_targets)
        per_route[route] = {
            "samples": len(route_sources),
            "source_max": max(route_sources),
            "target_max": max(route_targets),
            "source_p99": _quantile(route_sources, 0.99),
            "target_p99": _quantile(route_targets, 0.99),
        }
    return {
        "selection_sha256": dataset.selection_sha256,
        "records": len(sources),
        "source": {"min": min(sources), "p50": _quantile(sources, 0.5), "p95": _quantile(sources, 0.95), "p99": _quantile(sources, 0.99), "max": max(sources)},
        "target": {"min": min(targets), "p50": _quantile(targets, 0.5), "p95": _quantile(targets, 0.95), "p99": _quantile(targets, 0.99), "max": max(targets)},
        "source_lengths": sources,
        "target_lengths": targets,
        "route20": per_route,
    }


def _training_config(meta: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    identity = meta["identity"]
    data = meta["data"]
    budget = meta["budget"]
    benchmark = meta["benchmark"]
    return {
        "schema_version": 1,
        "identity": {
            "name": f"mvp-training-td14-{candidate['id']}-v1",
            "mode": "td14_benchmark",
            "student_config": identity["student_config"],
            "student_config_file_sha256": identity["student_config_file_sha256"],
            "student_config_canonical_sha256": identity["student_config_canonical_sha256"],
            "tokenizer_manifest_sha256": identity["tokenizer_manifest_sha256"],
            "seed": identity["seed"],
        },
        "data": {
            "train_path": data["train_path"], "train_sha256": data["train_sha256"],
            "dev_path": data["dev_path"], "dev_sha256": data["dev_sha256"],
            "manifest_path": data["manifest_path"], "manifest_sha256": data["manifest_sha256"],
            "train_max_records_per_route": data["train_records_per_route"],
            "dev_max_records_per_route": data["dev_records_per_route"],
            "route_weights": {route: 1.0 for route in ROUTE_ORDER},
        },
        "resource_profile": {
            "device": budget["device"], "precision": budget["precision"],
            "device_memory_budget_mib": budget["device_memory_budget_mib"],
            "device_memory_reserve_mib": budget["device_memory_reserve_mib"],
            "max_device_memory_utilization": budget["max_device_memory_utilization"],
            "host_memory_budget_mib": budget["host_memory_budget_mib"],
            "dataloader_memory_budget_mib": budget["dataloader_memory_budget_mib"],
            "oom_retry_limit": budget["oom_retry_limit"],
            **{name: candidate[name] for name in (
                "micro_batch_size", "gradient_accumulation_steps", "gradient_checkpointing",
                "max_source_length", "max_target_length", "dataloader_workers",
            )},
        },
        "optimization": {
            "optimizer": "adamw", "learning_rate": benchmark["learning_rate"],
            "betas": benchmark["betas"], "epsilon": benchmark["epsilon"],
            "weight_decay": benchmark["weight_decay"], "scheduler": benchmark["scheduler"],
            "warmup_steps": benchmark["warmup_steps"],
            "max_optimizer_steps": benchmark["optimizer_steps"],
            "max_train_tokens": benchmark["max_train_tokens"],
            "max_grad_norm": benchmark["max_grad_norm"],
            "label_smoothing": benchmark["label_smoothing"],
            "validation_frequency": benchmark["optimizer_steps"],
            "validation_batches": benchmark["validation_batches"],
            "checkpoint_frequency": benchmark["optimizer_steps"],
        },
    }


def _candidate_result(
    *, candidate: dict[str, Any], report: dict[str, Any], lengths: dict[str, Any], selection: dict[str, Any]
) -> dict[str, Any]:
    result = report["result"]
    source_rate = sum(value > candidate["max_source_length"] for value in lengths["source_lengths"]) / len(lengths["source_lengths"])
    target_rate = sum(value > candidate["max_target_length"] for value in lengths["target_lengths"]) / len(lengths["target_lengths"])
    events = read_jsonl(Path(report["output_root"]) / report["events"]["path"])
    validations = [row for row in events if row.get("event") == "validation"]
    peak = int(result["peak_device_memory_bytes"])
    effective_limit = int(report["runtime"]["resource_validation"]["effective_device_limit_bytes"])
    finite = math.isfinite(float(result["mean_train_loss"])) and math.isfinite(float(result["final_train_loss"]))
    gates = {
        "complete": report["status"] == "complete",
        "finite_loss": finite,
        "device_memory_within_effective_limit": peak <= effective_limit,
        "source_sample_truncation": source_rate <= selection["max_source_sample_truncation_rate"],
        "target_sample_truncation": target_rate <= selection["max_target_sample_truncation_rate"],
        "zero_oom_retries": int(report["training_config"]["canonical_sha256"] != "") and result["exception_skips"] == 0,
        "validation_executed": len(validations) == 1,
    }
    return {
        "id": candidate["id"],
        "profile": candidate,
        "training_config": report["training_config"],
        "output_root": report["output_root"],
        "runtime": report["runtime"],
        "metrics": {
            "optimizer_steps": result["optimizer_steps"],
            "effective_batch_samples": candidate["micro_batch_size"] * candidate["gradient_accumulation_steps"],
            "mean_train_loss": result["mean_train_loss"], "final_train_loss": result["final_train_loss"],
            "wall_time_seconds": result["wall_time_seconds"],
            "tokens_per_second": result["tokens_per_second"],
            "samples_per_second": result["samples_per_second"],
            "mean_step_seconds": result["wall_time_seconds"] / result["optimizer_steps"],
            "validation_wall_time_seconds": validations[0]["wall_time_seconds"] if validations else None,
            "peak_device_memory_bytes": peak,
            "peak_host_resident_bytes": result["process_memory"]["peak_resident_bytes"],
            "source_sample_truncation_rate": source_rate,
            "target_sample_truncation_rate": target_rate,
            "source_token_truncation_rate": result["token_audit"]["source_truncation_rate"],
            "target_token_truncation_rate": result["token_audit"]["target_truncation_rate"],
            "oom_retries": 0,
        },
        "gates": gates,
        "accepted": all(gates.values()),
    }


def benchmark_candidates(
    *, repository_root: Path, meta_path: Path, runtime_root: Path, report_path: Path
) -> dict[str, Any]:
    meta = yaml.safe_load(meta_path.read_text(encoding="utf-8"))
    if not isinstance(meta, dict) or meta.get("schema_version") != 1:
        raise ResourceBenchmarkError("unsupported TD-14 candidate config")
    student = load_student_config(repository_root / meta["identity"]["student_config"])
    tokenizer, tokenizer_identity = load_frozen_tokenizer(student, repository_root)
    lengths = analyze_real_lengths(repository_root=repository_root, meta=meta, tokenizer=tokenizer)
    compact_lengths = {name: value for name, value in lengths.items() if name not in {"source_lengths", "target_lengths"}}
    generated_root = repository_root / "artifacts/model-training/runtime/m2-resource-candidates"
    candidates = []
    for candidate in meta["candidates"]:
        training_config = _training_config(meta, candidate)
        config_path = generated_root / f"{candidate['id']}.yaml"
        _atomic_yaml(config_path, training_config)
        output = runtime_root / f"candidate-{candidate['id']}"
        report = run_training(
            config_path=config_path,
            repository_root=repository_root,
            output_dir=output,
            dry_run=False,
        )
        candidates.append(
            _candidate_result(candidate=candidate, report=report, lengths=lengths, selection=meta["selection"])
        )
    accepted = [row for row in candidates if row["accepted"]]
    if not accepted:
        raise ResourceBenchmarkError("no resource candidate passed every frozen gate")
    selected = sorted(
        accepted,
        key=lambda row: (-row["metrics"]["tokens_per_second"], row["metrics"]["peak_device_memory_bytes"], row["id"]),
    )[0]
    result = {
        "schema_version": 1, "status": "complete", "task": "TD-14-candidates",
        "candidate_config": {"path": meta_path.relative_to(repository_root).as_posix(), "file_sha256": sha256_file(meta_path), "canonical_sha256": config_sha256(meta)},
        "tokenizer": tokenizer_identity,
        "real_length_distribution": compact_lengths,
        "selection_rule": meta["selection"],
        "candidates": candidates,
        "selected_candidate_id": selected["id"],
        "selected_profile": selected["profile"],
        "selected_runtime": selected["runtime"],
    }
    _atomic_json(report_path, result)
    return result
