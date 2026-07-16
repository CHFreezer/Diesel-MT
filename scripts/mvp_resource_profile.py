"""Validation and evidence publication for the frozen TD-14 M2 profile."""

from __future__ import annotations

import json
import math
import os
import re
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from freeze_tokenizer_artifact import sha256_file
from model_training_contract import config_sha256
from mvp_checkpoint import validate_checkpoint
from mvp_training import load_training_config, read_jsonl


class ResourceProfileError(RuntimeError):
    """Raised when the selected profile or its soak evidence is inconsistent."""


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
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


def load_m2_profile(path: Path) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ResourceProfileError(f"cannot load M2 profile: {exc}") from exc
    if not isinstance(value, Mapping) or value.get("schema_version") != 1:
        raise ResourceProfileError("unsupported M2 profile schema")
    expected = {"schema_version", "identity", "resource_profile", "optimization", "storage", "soak_acceptance"}
    if set(value) != expected:
        raise ResourceProfileError("M2 profile fields are incomplete or unknown")
    identity = value["identity"]
    if identity.get("name") != "mvp-m2-resource-profile-v1" or identity.get("status") != "frozen_by_td14":
        raise ResourceProfileError("M2 profile identity is not frozen")
    resource = value["resource_profile"]
    required_resource = {
        "device", "precision", "device_memory_budget_mib", "device_memory_reserve_mib",
        "max_device_memory_utilization", "host_memory_budget_mib", "dataloader_memory_budget_mib",
        "oom_retry_limit", "micro_batch_size", "gradient_accumulation_steps",
        "gradient_checkpointing", "max_source_length", "max_target_length", "dataloader_workers",
    }
    if set(resource) != required_resource:
        raise ResourceProfileError("frozen resource fields are incomplete or unknown")
    if resource["oom_retry_limit"] != 0:
        raise ResourceProfileError("formal M2 profile may not retry OOM or mutate itself")
    serialized = json.dumps(value, ensure_ascii=False)
    if re.search(r"(?:^|[\"'\s])(?:[A-Za-z]:\\|[A-Za-z]:/)", serialized):
        raise ResourceProfileError("semantic profile contains a drive-letter path")
    return json.loads(json.dumps(value))


def validate_profile_binding(
    *, profile: Mapping[str, Any], benchmark: Mapping[str, Any], soak_config: Mapping[str, Any]
) -> dict[str, Any]:
    identity = profile["identity"]
    if benchmark.get("status") != "complete" or benchmark.get("selected_candidate_id") != identity["selected_candidate_id"]:
        raise ResourceProfileError("profile selected candidate differs from benchmark")
    selected = next(
        (row for row in benchmark["candidates"] if row["id"] == identity["selected_candidate_id"]),
        None,
    )
    if selected is None or not selected.get("accepted"):
        raise ResourceProfileError("profile candidate did not pass the frozen gates")
    candidate = selected["profile"]
    for field in (
        "micro_batch_size", "gradient_accumulation_steps", "gradient_checkpointing",
        "max_source_length", "max_target_length", "dataloader_workers",
    ):
        if profile["resource_profile"][field] != candidate[field]:
            raise ResourceProfileError(f"profile field differs from selected candidate: {field}")
    if dict(soak_config["resource_profile"]) != dict(profile["resource_profile"]):
        raise ResourceProfileError("soak resource profile differs from frozen M2 profile")
    profile_optimization = dict(profile["optimization"])
    profile_optimization.pop("checkpoint_retention")
    if dict(soak_config["optimization"]) != profile_optimization:
        raise ResourceProfileError("soak optimization differs from frozen M2 profile")
    return {
        "selected_candidate_accepted": True,
        "resource_profile_exact": True,
        "optimization_exact": True,
    }


def build_td14_evidence(
    *,
    repository_root: Path,
    profile_path: Path,
    benchmark_path: Path,
    soak_config_path: Path,
    soak_report_path: Path,
    resume_report_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    profile = load_m2_profile(profile_path)
    benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
    if sha256_file(benchmark_path) != profile["identity"]["candidate_benchmark_sha256"]:
        raise ResourceProfileError("candidate benchmark SHA-256 differs from frozen profile")
    if sha256_file(repository_root / "configs/mvp_training_td14_candidates.yaml") != profile["identity"]["candidate_config_sha256"]:
        raise ResourceProfileError("candidate config SHA-256 differs from frozen profile")
    soak_config = load_training_config(soak_config_path)
    binding = validate_profile_binding(profile=profile, benchmark=benchmark, soak_config=soak_config)
    soak = json.loads(soak_report_path.read_text(encoding="utf-8"))
    resume = json.loads(resume_report_path.read_text(encoding="utf-8"))
    acceptance = profile["soak_acceptance"]
    events_path = Path(soak["output_root"]) / soak["events"]["path"]
    events = read_jsonl(events_path)
    optimizer_events = [row for row in events if row.get("event") == "optimizer_step"]
    validations = [row for row in events if row.get("event") == "validation"]
    checkpoint_events = [row for row in events if row.get("event") == "checkpoint"]
    checkpoint_manifests = []
    for record in soak["checkpoints"]:
        checkpoint_path = Path(record["path"])
        manifest = validate_checkpoint(checkpoint_path)
        if sha256_file(checkpoint_path / "checkpoint-manifest.json") != record["manifest_sha256"]:
            raise ResourceProfileError("reported checkpoint manifest SHA-256 changed")
        checkpoint_manifests.append(
            {
                "path": str(checkpoint_path), "checkpoint_id": manifest["checkpoint_id"],
                "manifest_sha256": record["manifest_sha256"], "summary": manifest["summary"],
            }
        )
    peak = int(soak["result"]["peak_device_memory_bytes"])
    effective = int(soak["runtime"]["resource_validation"]["effective_device_limit_bytes"])
    warmup_boundary = int(acceptance["validation_steps"][0])
    boundary_peak = max(int(row["peak_device_memory_bytes"]) for row in optimizer_events if row["optimizer_step"] <= warmup_boundary)
    later_peak = max(int(row["peak_device_memory_bytes"]) for row in optimizer_events if row["optimizer_step"] > warmup_boundary)
    result = soak["result"]
    gates = {
        **binding,
        "intentional_soak_boundary": soak["status"] == "interrupted" and result["optimizer_steps"] == acceptance["optimizer_steps"],
        "finite_losses": math.isfinite(float(result["mean_train_loss"])) and math.isfinite(float(result["final_train_loss"])),
        "device_memory_within_budget": peak <= effective,
        "no_peak_memory_growth_after_warmup": later_peak <= boundary_peak,
        "source_truncation_within_limit": result["token_audit"]["source_truncation_rate"] <= acceptance["max_source_token_truncation_rate"],
        "target_truncation_within_limit": result["token_audit"]["target_truncation_rate"] <= acceptance["max_target_token_truncation_rate"],
        "zero_exception_skips": result["exception_skips"] == 0,
        "validation_schedule_exact": [row["optimizer_step"] for row in validations] == acceptance["validation_steps"],
        "checkpoint_schedule_exact": [row["optimizer_step"] for row in checkpoint_events] == acceptance["checkpoint_steps"],
        "checkpoint_publications_complete": len(checkpoint_manifests) == len(acceptance["checkpoint_steps"]),
        "resume_probe_advanced_one_step": resume["status"] == "interrupted" and resume["result"]["optimizer_steps"] == acceptance["resume_probe_step"],
        "resume_uses_final_soak_checkpoint": Path(resume["resume_from"]).resolve() == Path(checkpoint_manifests[-1]["path"]).resolve(),
        "hot_output_external_to_repository": not Path(soak["output_root"]).resolve().is_relative_to(repository_root.resolve()),
        "hot_checkpoints_external_to_repository": all(not Path(row["path"]).resolve().is_relative_to(repository_root.resolve()) for row in checkpoint_manifests),
    }
    if not all(gates.values()):
        failed = [name for name, passed in gates.items() if not passed]
        raise ResourceProfileError(f"TD-14 evidence gates failed: {failed}")
    evidence = {
        "schema_version": 1, "status": "complete", "task": "TD-14",
        "profile": {"path": profile_path.relative_to(repository_root).as_posix(), "file_sha256": sha256_file(profile_path), "canonical_sha256": config_sha256(profile)},
        "candidate_benchmark": {"path": benchmark_path.relative_to(repository_root).as_posix(), "file_sha256": sha256_file(benchmark_path), "selected_candidate_id": benchmark["selected_candidate_id"], "candidates": benchmark["candidates"]},
        "length_distribution": benchmark["real_length_distribution"],
        "runtime_hardware_manifest": soak["runtime"],
        "soak": {
            "config_path": soak_config_path.relative_to(repository_root).as_posix(),
            "config_file_sha256": sha256_file(soak_config_path),
            "report_file_sha256": sha256_file(soak_report_path),
            "events_sha256": soak["events"]["sha256"],
            "result": result,
            "validation_events": validations,
            "checkpoint_events": checkpoint_events,
            "checkpoints": checkpoint_manifests,
            "warmup_boundary_peak_bytes": boundary_peak,
            "post_warmup_peak_bytes": later_peak,
        },
        "resume_probe": {
            "report_file_sha256": sha256_file(resume_report_path),
            "from": resume["resume_from"], "optimizer_steps": resume["result"]["optimizer_steps"],
            "consumed_samples": resume["result"]["consumed_samples"],
            "consumed_tokens": resume["result"]["consumed_tokens"],
        },
        "gates": gates,
    }
    _atomic_json(output_path, evidence)
    return evidence
