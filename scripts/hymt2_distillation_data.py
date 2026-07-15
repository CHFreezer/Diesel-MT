"""TD-08 bounded train-only sequence-level distillation pipeline."""

from __future__ import annotations

import json
import os
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

from hymt2_distillation import (
    ROUTES,
    DistillationError,
    LlamaCppTeacher,
    atomic_write_bytes,
    atomic_write_json,
    atomic_write_jsonl,
    build_prompt,
    canonical_json_bytes,
    config_sha256,
    deterministic_route_sample,
    filter_output,
    generation_record,
    load_json,
    load_prompt_config,
    load_yaml,
    read_parallel_jsonl,
    route_id,
    route_limits,
    sha256_bytes,
    sha256_file,
)
from model_training_contract import validate_parallel_sample


PIPELINE_VERSION = "td08-hymt2-distillation-v1"
TEACHER_MODEL = "tencent/Hy-MT2-7B-GGUF"
TEACHER_REVISION = "ab8472660ac61fac25f1af43fac2599d52a8a775"
TEACHER_ARTIFACT_SHA256 = "58b3ad55dd6f6fa08c695cddc34fb5f8f708a844f78ae10508071914b0ed67c0"
BACKEND_ID = "llama.cpp-b10012-cuda13.3"
D0_IDENTITY = {
    "name": "hymt2-sequence-distillation-d0-v1",
    "status": "frozen",
    "scope": "bounded-train-only-eighteen-route-mvp",
}
D1_IDENTITY = {
    "name": "hymt2-sequence-distillation-d1-v1",
    "status": "frozen",
    "scope": "mvp-train-only-eighteen-route-distilled-corpus",
}


def _exact_keys(value: Mapping[str, Any], required: set[str], context: str) -> None:
    missing = sorted(required - set(value))
    unknown = sorted(set(value) - required)
    if missing:
        raise DistillationError(f"{context} missing fields: {', '.join(missing)}")
    if unknown:
        raise DistillationError(f"{context} unknown fields: {', '.join(unknown)}")


def _mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise DistillationError(f"{context} must be a mapping")
    return value


def _positive_integer(value: Any, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise DistillationError(f"{context} must be a positive integer")
    return value


def _rate(value: Any, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1:
        raise DistillationError(f"{context} must be in [0, 1]")
    return float(value)


def _repo_path(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise DistillationError(f"{context} must be a repository-relative POSIX path")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "." in path.parts:
        raise DistillationError(f"{context} must not escape the repository")
    return path.as_posix()


def validate_distillation_config(config: Mapping[str, Any]) -> dict[str, Any]:
    base_keys = {
        "schema_version",
        "identity",
        "prompt_decode",
        "input",
        "sampling",
        "generation",
        "manual_review",
        "replay",
        "acceptance_gates",
        "runtime",
        "outputs",
    }
    identity = _mapping(config.get("identity"), "identity")
    _exact_keys(identity, {"name", "status", "scope"}, "identity")
    if dict(identity) == D0_IDENTITY:
        maturity = "d0"
        _exact_keys(config, base_keys, "distillation config")
    elif dict(identity) == D1_IDENTITY:
        maturity = "d1"
        _exact_keys(config, base_keys | {"reuse"}, "distillation config")
    else:
        raise DistillationError("distillation identity changed")
    if config["schema_version"] != 1:
        raise DistillationError("distillation schema_version must be 1")

    prompt = _mapping(config["prompt_decode"], "prompt_decode")
    _exact_keys(
        prompt,
        {"path", "file_sha256", "calibration_report", "calibration_report_sha256", "selected_profile"},
        "prompt_decode",
    )
    for field in ("path", "calibration_report"):
        _repo_path(prompt[field], f"prompt_decode.{field}")
    for field in ("file_sha256", "calibration_report_sha256"):
        if not isinstance(prompt[field], str) or len(prompt[field]) != 64:
            raise DistillationError(f"prompt_decode.{field} must be a SHA-256")
    if prompt["selected_profile"] != "greedy-v1":
        raise DistillationError("TD-08 must consume the TD-07 selected greedy profile")

    input_config = _mapping(config["input"], "input")
    _exact_keys(
        input_config,
        {
            "path",
            "file_sha256",
            "m0_manifest",
            "m0_manifest_sha256",
            "split",
            "dev_access",
            "test_access",
        },
        "input",
    )
    _repo_path(input_config["path"], "input.path")
    _repo_path(input_config["m0_manifest"], "input.m0_manifest")
    if input_config["split"] != "train" or input_config["dev_access"] != "prohibited" or input_config["test_access"] != "prohibited":
        raise DistillationError("TD-08 input must be train-only and prohibit dev/test")

    sampling = _mapping(config["sampling"], "sampling")
    _exact_keys(
        sampling,
        {"unit", "records_per_route", "total_records", "selection_seed", "order", "replacement"},
        "sampling",
    )
    per_route = _positive_integer(sampling["records_per_route"], "sampling.records_per_route")
    if sampling["total_records"] != per_route * len(ROUTES):
        raise DistillationError("sampling.total_records must equal records_per_route * 18")
    if sampling["unit"] != "directed_sample" or sampling["replacement"] is not False:
        raise DistillationError("D0 sampling must use directed samples without replacement")
    if sampling["order"] != "frozen-route-order-then-selection-hash":
        raise DistillationError("D0 output order changed")
    if not isinstance(sampling["selection_seed"], str) or not sampling["selection_seed"]:
        raise DistillationError("sampling.selection_seed must be non-empty")
    expected_sampling = {
        "d0": (128, 2_304, "diesel-mt-td08-d0-v1"),
        "d1": (2_224, 40_032, "diesel-mt-td08-d0-v1"),
    }[maturity]
    if (
        per_route,
        sampling["total_records"],
        sampling["selection_seed"],
    ) != expected_sampling:
        raise DistillationError(f"{maturity.upper()} frozen sampling identity changed")

    generation = _mapping(config["generation"], "generation")
    _exact_keys(
        generation,
        {
            "shard_unit",
            "checkpoint_granularity",
            "resume_identity",
            "raw_and_normalized_separate",
            "request_failure",
            "external_network",
        },
        "generation",
    )
    if dict(generation) != {
        "shard_unit": "route",
        "checkpoint_granularity": "sample",
        "resume_identity": "generation-contract-sha256",
        "raw_and_normalized_separate": True,
        "request_failure": "stop-without-publish",
        "external_network": "prohibited",
    }:
        raise DistillationError("generation/resume boundary changed")

    manual = _mapping(config["manual_review"], "manual_review")
    _exact_keys(
        manual,
        {"accepted_per_route", "rejected_per_route", "rejected_if_fewer", "traditional_targets_extra_review", "attestation"},
        "manual_review",
    )
    _positive_integer(manual["accepted_per_route"], "manual_review.accepted_per_route")
    _positive_integer(manual["rejected_per_route"], "manual_review.rejected_per_route")
    if manual["rejected_if_fewer"] != "inspect-all" or manual["traditional_targets_extra_review"] is not True:
        raise DistillationError("manual review boundary changed")
    _repo_path(manual["attestation"], "manual_review.attestation")

    replay = _mapping(config["replay"], "replay")
    _exact_keys(replay, {"samples_per_route", "exact_raw_output_required", "exact_normalized_output_required"}, "replay")
    _positive_integer(replay["samples_per_route"], "replay.samples_per_route")
    if replay["exact_raw_output_required"] is not True or replay["exact_normalized_output_required"] is not True:
        raise DistillationError("replay must require exact raw and normalized output")

    gates = _mapping(config["acceptance_gates"], "acceptance_gates")
    _exact_keys(
        gates,
        {
            "minimum_accepted_per_route",
            "minimum_route_accepted_rate",
            "minimum_route_script_compliance_rate",
            "maximum_route_retry_rate",
            "require_all_eighteen_routes",
            "require_manual_review",
            "require_exact_replay",
            "require_zero_test_records",
        },
        "acceptance_gates",
    )
    minimum = _positive_integer(gates["minimum_accepted_per_route"], "minimum_accepted_per_route")
    if minimum > per_route:
        raise DistillationError("minimum accepted exceeds the route sample budget")
    for field in ("minimum_route_accepted_rate", "minimum_route_script_compliance_rate", "maximum_route_retry_rate"):
        _rate(gates[field], f"acceptance_gates.{field}")
    for field in (
        "require_all_eighteen_routes",
        "require_manual_review",
        "require_exact_replay",
        "require_zero_test_records",
    ):
        if gates[field] is not True:
            raise DistillationError(f"acceptance_gates.{field} must be true")
    expected_minimum = 100 if maturity == "d0" else 2_000
    if minimum != expected_minimum:
        raise DistillationError(f"{maturity.upper()} minimum accepted route count changed")

    runtime = _mapping(config["runtime"], "runtime")
    _exact_keys(runtime, {"work_root_override_env", "default_work_root"}, "runtime")
    if not isinstance(runtime["work_root_override_env"], str) or not runtime["work_root_override_env"].startswith("DIESEL_MT_"):
        raise DistillationError("runtime override environment variable is invalid")
    _repo_path(runtime["default_work_root"], "runtime.default_work_root")

    outputs = _mapping(config["outputs"], "outputs")
    _exact_keys(
        outputs,
        {
            "root",
            "generation_contract",
            "raw_subdir",
            "accepted",
            "filtered",
            "manual_review_queue",
            "replay_report",
            "quality_report",
            "manifest",
            "evidence",
        },
        "outputs",
    )
    for field, value in outputs.items():
        if field == "raw_subdir":
            if value != "raw":
                raise DistillationError("outputs.raw_subdir must be raw")
        else:
            _repo_path(value, f"outputs.{field}")
    if maturity == "d1":
        reuse = _mapping(config["reuse"], "reuse")
        _exact_keys(
            reuse,
            {
                "source_config",
                "source_config_file_sha256",
                "source_manifest",
                "source_manifest_sha256",
                "source_generation_contract_sha256",
                "required_records_per_route",
                "require_selected_prefix",
            },
            "reuse",
        )
        for field in ("source_config", "source_manifest"):
            _repo_path(reuse[field], f"reuse.{field}")
        for field in (
            "source_config_file_sha256",
            "source_manifest_sha256",
            "source_generation_contract_sha256",
        ):
            if not isinstance(reuse[field], str) or len(reuse[field]) != 64:
                raise DistillationError(f"reuse.{field} must be a SHA-256")
        if reuse["required_records_per_route"] != 128 or reuse["require_selected_prefix"] is not True:
            raise DistillationError("D1 must reuse the verified 128-record D0 prefix per route")
    return dict(config)


def load_distillation_config(path: Path) -> dict[str, Any]:
    return validate_distillation_config(load_yaml(path))


def resolve_work_root(repository_root: Path, config: Mapping[str, Any]) -> Path:
    runtime = config["runtime"]
    override = os.environ.get(str(runtime["work_root_override_env"]))
    if override:
        path = Path(override).expanduser()
        if not path.is_absolute():
            raise DistillationError(f"{runtime['work_root_override_env']} must be absolute")
        return path.resolve()
    return (repository_root / PurePosixPath(str(runtime["default_work_root"]))).resolve()


def bound_configs(
    repository_root: Path,
    distillation_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    config = load_distillation_config(distillation_path)
    prompt_path = repository_root / PurePosixPath(str(config["prompt_decode"]["path"]))
    report_path = repository_root / PurePosixPath(str(config["prompt_decode"]["calibration_report"]))
    if sha256_file(prompt_path) != config["prompt_decode"]["file_sha256"]:
        raise DistillationError("prompt/decode file differs from the TD-08 lock")
    if sha256_file(report_path) != config["prompt_decode"]["calibration_report_sha256"]:
        raise DistillationError("TD-07 calibration report differs from the TD-08 lock")
    report = load_json(report_path)
    if (
        report.get("status") != "complete"
        or report.get("test_accessed") is not False
        or report.get("decision", {}).get("selected_profile") != config["prompt_decode"]["selected_profile"]
    ):
        raise DistillationError("TD-07 evidence is not complete or selected profile differs")
    prompt_config = load_prompt_config(prompt_path)
    if prompt_config["selection"]["selected_profile"] != config["prompt_decode"]["selected_profile"]:
        raise DistillationError("prompt config selected profile differs from TD-08")
    return config, prompt_config


def _route_filename(route: str) -> str:
    return route.replace("->", "--") + ".json"


def _raw_shard_filename(route: str) -> str:
    return route.replace("->", "--") + ".jsonl"


def _selected_identity(samples: Sequence[Mapping[str, Any]]) -> str:
    return sha256_bytes(
        b"".join(canonical_json_bytes({"sample_id": sample["sample_id"]}) for sample in samples)
    )


def generation_contract(
    config: Mapping[str, Any],
    prompt_config: Mapping[str, Any],
    samples: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    counts = Counter(route_id(str(sample["src_lang"]), str(sample["tgt_lang"])) for sample in samples)
    return {
        "schema_version": 1,
        "pipeline_version": PIPELINE_VERSION,
        "identity": config["identity"],
        "distillation_config_sha256": config_sha256(config),
        "prompt_config_sha256": config_sha256(prompt_config),
        "selected_profile": config["prompt_decode"]["selected_profile"],
        "teacher": {
            "model": TEACHER_MODEL,
            "revision": TEACHER_REVISION,
            "artifact_sha256": TEACHER_ARTIFACT_SHA256,
            "backend": BACKEND_ID,
        },
        "input": {
            "path": config["input"]["path"],
            "sha256": config["input"]["file_sha256"],
            "split": "train",
            "dev_access": "prohibited",
            "test_access": "prohibited",
        },
        "sampling": {
            **config["sampling"],
            "selected_sample_ids_sha256": _selected_identity(samples),
            "records_by_route": {route: int(counts[route]) for route in ROUTES},
        },
    }


def prepare_inputs(
    repository_root: Path,
    config: Mapping[str, Any],
    prompt_config: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any], str]:
    input_path = repository_root / PurePosixPath(str(config["input"]["path"]))
    manifest_path = repository_root / PurePosixPath(str(config["input"]["m0_manifest"]))
    if sha256_file(manifest_path) != config["input"]["m0_manifest_sha256"]:
        raise DistillationError("M0 manifest differs from the TD-08 lock")
    records = read_parallel_jsonl(
        input_path,
        expected_split="train",
        expected_sha256=str(config["input"]["file_sha256"]),
    )
    selected = deterministic_route_sample(
        records,
        per_route=int(config["sampling"]["records_per_route"]),
        seed=str(config["sampling"]["selection_seed"]),
    )
    contract = generation_contract(config, prompt_config, selected)
    digest = sha256_bytes(canonical_json_bytes(contract))
    return selected, contract, digest


def write_or_verify_generation_contract(
    repository_root: Path,
    config: Mapping[str, Any],
    contract: Mapping[str, Any],
    digest: str,
) -> Path:
    path = repository_root / PurePosixPath(str(config["outputs"]["generation_contract"]))
    if path.exists():
        if sha256_file(path) != digest:
            raise DistillationError("existing generation contract differs; refuse to mix runs")
    else:
        atomic_write_bytes(path, canonical_json_bytes(contract))
    return path


def _load_route_state(
    path: Path,
    *,
    route: str,
    contract_sha256: str,
    samples: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    state = load_json(path)
    _exact_keys(state, {"schema_version", "generation_contract_sha256", "route", "sample_ids", "results"}, "route state")
    expected_ids = [str(sample["sample_id"]) for sample in samples]
    if (
        state["schema_version"] != 1
        or state["generation_contract_sha256"] != contract_sha256
        or state["route"] != route
        or state["sample_ids"] != expected_ids
        or not isinstance(state["results"], list)
    ):
        raise DistillationError(f"route checkpoint identity differs: {path}")
    results = [dict(record) for record in state["results"]]
    if [record.get("sample_id") for record in results] != expected_ids[: len(results)]:
        raise DistillationError(f"route checkpoint is not a valid sample prefix: {path}")
    return results


def _write_route_state(
    path: Path,
    *,
    route: str,
    contract_sha256: str,
    samples: Sequence[Mapping[str, Any]],
    results: Sequence[Mapping[str, Any]],
) -> None:
    atomic_write_json(
        path,
        {
            "schema_version": 1,
            "generation_contract_sha256": contract_sha256,
            "route": route,
            "sample_ids": [sample["sample_id"] for sample in samples],
            "results": list(results),
        },
    )


def _route_checkpoint_root(state_root: Path, route: str) -> Path:
    return state_root / route.replace("->", "--")


def _load_route_checkpoints(
    state_root: Path,
    *,
    route: str,
    contract_sha256: str,
    samples: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    root = _route_checkpoint_root(state_root, route)
    identity_path = root / "identity.json"
    expected_identity = {
        "schema_version": 1,
        "generation_contract_sha256": contract_sha256,
        "route": route,
        "sample_ids": [sample["sample_id"] for sample in samples],
    }
    if identity_path.is_file():
        if load_json(identity_path) != expected_identity:
            raise DistillationError(f"route checkpoint identity differs: {identity_path}")
    else:
        atomic_write_json(identity_path, expected_identity)
    result_root = root / "results"
    paths = sorted(result_root.glob("*.json")) if result_root.is_dir() else []
    expected_names = [f"{index:08d}.json" for index in range(len(paths))]
    if [path.name for path in paths] != expected_names or len(paths) > len(samples):
        raise DistillationError(f"route checkpoints are not a contiguous sample prefix: {root}")
    results = [load_json(path) for path in paths]
    expected_ids = [str(sample["sample_id"]) for sample in samples]
    if [record.get("sample_id") for record in results] != expected_ids[: len(results)]:
        raise DistillationError(f"route checkpoints contain a wrong sample prefix: {root}")
    return results


def _write_sample_checkpoint(
    state_root: Path,
    *,
    route: str,
    index: int,
    record: Mapping[str, Any],
) -> None:
    path = _route_checkpoint_root(state_root, route) / "results" / f"{index:08d}.json"
    if path.is_file():
        if load_json(path) != dict(record):
            raise DistillationError(f"sample checkpoint differs: {path}")
        return
    atomic_write_json(path, dict(record))


def _enrich_raw_record(
    record: Mapping[str, Any],
    *,
    sample: Mapping[str, Any],
    prompt_config: Mapping[str, Any],
    generation_contract_sha256: str,
) -> dict[str, Any]:
    return {
        **record,
        "source_id": sample["source_id"],
        "source_version": sample["source_version"],
        "license": sample["license"],
        "teacher_model": TEACHER_MODEL,
        "teacher_revision": TEACHER_REVISION,
        "teacher_artifact_sha256": TEACHER_ARTIFACT_SHA256,
        "backend": BACKEND_ID,
        "prompt_version": prompt_config["prompt"]["version"],
        "decode_config_sha256": config_sha256(prompt_config["decode_profiles"][record["profile"]]),
        "generation_manifest_sha256": generation_contract_sha256,
    }


def load_reused_results(
    repository_root: Path,
    config: Mapping[str, Any],
    prompt_config: Mapping[str, Any],
    samples: Sequence[Mapping[str, Any]],
    target_contract_sha256: str,
) -> dict[str, dict[str, Any]]:
    if "reuse" not in config:
        return {}
    reuse = config["reuse"]
    source_config_path = repository_root / PurePosixPath(str(reuse["source_config"]))
    if sha256_file(source_config_path) != reuse["source_config_file_sha256"]:
        raise DistillationError("D0 source config differs from the D1 reuse lock")
    source_config, source_prompt_config = bound_configs(repository_root, source_config_path)
    if source_config["identity"] != D0_IDENTITY:
        raise DistillationError("D1 reuse source is not the frozen D0 identity")
    if config_sha256(source_prompt_config) != config_sha256(prompt_config):
        raise DistillationError("D0 and D1 prompt/decode identities differ")
    source_samples, _, source_contract_sha256 = prepare_inputs(
        repository_root,
        source_config,
        source_prompt_config,
    )
    if source_contract_sha256 != reuse["source_generation_contract_sha256"]:
        raise DistillationError("D0 generation contract differs from the D1 reuse lock")
    source_contract_path = repository_root / PurePosixPath(
        str(source_config["outputs"]["generation_contract"])
    )
    if sha256_file(source_contract_path) != source_contract_sha256:
        raise DistillationError("D0 generation contract artifact is missing or changed")
    source_manifest_path = repository_root / PurePosixPath(str(reuse["source_manifest"]))
    if sha256_file(source_manifest_path) != reuse["source_manifest_sha256"]:
        raise DistillationError("D0 manifest differs from the D1 reuse lock")
    source_manifest = load_json(source_manifest_path)
    if (
        source_manifest.get("status") != "complete"
        or source_manifest.get("identity") != D0_IDENTITY
        or source_manifest.get("generation_contract_sha256") != source_contract_sha256
    ):
        raise DistillationError("D0 manifest is incomplete or binds to another run")
    source_raw_root = (
        repository_root
        / PurePosixPath(str(source_config["outputs"]["root"]))
        / str(source_config["outputs"]["raw_subdir"])
    )
    if source_manifest.get("outputs", {}).get("raw_shards") != route_file_identities(source_raw_root):
        raise DistillationError("D0 raw shards differ from the frozen D0 manifest")
    source_raw = load_raw_results(repository_root, source_config)

    source_samples_by_route: dict[str, list[dict[str, Any]]] = defaultdict(list)
    target_samples_by_route: dict[str, list[dict[str, Any]]] = defaultdict(list)
    raw_by_sample = {str(record["sample_id"]): record for record in source_raw}
    for sample in source_samples:
        source_samples_by_route[route_id(str(sample["src_lang"]), str(sample["tgt_lang"]))].append(sample)
    for sample in samples:
        target_samples_by_route[route_id(str(sample["src_lang"]), str(sample["tgt_lang"]))].append(sample)

    required = int(reuse["required_records_per_route"])
    rebound: dict[str, dict[str, Any]] = {}
    profile_name = str(config["prompt_decode"]["selected_profile"])
    expected_decode_sha256 = config_sha256(prompt_config["decode_profiles"][profile_name])
    for route in ROUTES:
        source_route = source_samples_by_route[route]
        target_prefix = target_samples_by_route[route][:required]
        if len(source_route) != required or [sample["sample_id"] for sample in source_route] != [
            sample["sample_id"] for sample in target_prefix
        ]:
            raise DistillationError(f"D0 is not the required selected D1 prefix for {route}")
        for sample in source_route:
            sample_id = str(sample["sample_id"])
            raw = raw_by_sample.get(sample_id)
            if raw is None:
                raise DistillationError(f"D0 raw result is missing {sample_id}")
            expected_fields = {
                "route": route,
                "sample_group_id": sample["sample_group_id"],
                "split": "train",
                "src_lang": sample["src_lang"],
                "tgt_lang": sample["tgt_lang"],
                "source_text": sample["source_text"],
                "reference_text": sample["target_text"],
                "source_id": sample["source_id"],
                "source_version": sample["source_version"],
                "license": sample["license"],
                "teacher_model": TEACHER_MODEL,
                "teacher_revision": TEACHER_REVISION,
                "teacher_artifact_sha256": TEACHER_ARTIFACT_SHA256,
                "backend": BACKEND_ID,
                "prompt_version": prompt_config["prompt"]["version"],
                "profile": profile_name,
                "decode_config_sha256": expected_decode_sha256,
                "generation_manifest_sha256": source_contract_sha256,
            }
            if any(raw.get(field) != value for field, value in expected_fields.items()):
                raise DistillationError(f"D0 raw result identity differs for {sample_id}")
            raw_output = str(raw.get("raw_output", ""))
            normalized_output = str(raw.get("normalized_output", ""))
            if (
                raw.get("raw_output_sha256") != sha256_bytes(raw_output.encode("utf-8"))
                or raw.get("normalized_output_sha256")
                != sha256_bytes(normalized_output.encode("utf-8"))
            ):
                raise DistillationError(f"D0 output hash differs for {sample_id}")
            rebound[sample_id] = {
                **raw,
                "generation_manifest_sha256": target_contract_sha256,
                "reuse_provenance": {
                    "source_generation_contract_sha256": source_contract_sha256,
                    "source_manifest_sha256": reuse["source_manifest_sha256"],
                    "verification": "byte-identical-output-rebound-to-d1-contract",
                },
            }
    if len(rebound) != required * len(ROUTES):
        raise DistillationError("D1 reused record count differs from the frozen contract")
    return rebound


def generate_d0(
    repository_root: Path,
    config_path: Path,
    *,
    dry_run: bool,
) -> dict[str, Any]:
    repository_root = repository_root.resolve()
    config_path = config_path.resolve()
    config, prompt_config = bound_configs(repository_root, config_path)
    samples, contract, contract_sha256 = prepare_inputs(repository_root, config, prompt_config)
    reused_results = load_reused_results(
        repository_root,
        config,
        prompt_config,
        samples,
        contract_sha256,
    )
    common = {
        "schema_version": 1,
        "pipeline_version": PIPELINE_VERSION,
        "status": "dry-run" if dry_run else "generated-awaiting-review",
        "generation_contract_sha256": contract_sha256,
        "selected_records": len(samples),
        "records_per_route": config["sampling"]["records_per_route"],
        "selected_sample_ids_sha256": contract["sampling"]["selected_sample_ids_sha256"],
        "reused_records": len(reused_results),
        "test_accessed": False,
        "dev_accessed": False,
    }
    if dry_run:
        return common
    write_or_verify_generation_contract(repository_root, config, contract, contract_sha256)
    work_root = resolve_work_root(repository_root, config)
    state_root = work_root / "route-state"
    raw_root = repository_root / PurePosixPath(str(config["outputs"]["root"])) / str(config["outputs"]["raw_subdir"])
    by_route: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        by_route[route_id(str(sample["src_lang"]), str(sample["tgt_lang"]))].append(sample)
    limits = route_limits(prompt_config)
    profile_name = str(config["prompt_decode"]["selected_profile"])
    profile = prompt_config["decode_profiles"][profile_name]
    started = time.perf_counter()
    all_results: list[dict[str, Any]] = []
    with LlamaCppTeacher(repository_root, prompt_config) as teacher:
        for route in ROUTES:
            route_samples = by_route[route]
            results = _load_route_checkpoints(
                state_root,
                route=route,
                contract_sha256=contract_sha256,
                samples=route_samples,
            )
            if results:
                print(f"resume {route}: {len(results)}/{len(route_samples)}", file=sys.stderr, flush=True)
            while len(results) < len(route_samples):
                sample = route_samples[len(results)]
                reused = reused_results.get(str(sample["sample_id"]))
                if reused is None:
                    break
                results.append(reused)
                _write_sample_checkpoint(
                    state_root,
                    route=route,
                    index=len(results) - 1,
                    record=reused,
                )
            if results and len(results) == int(config.get("reuse", {}).get("required_records_per_route", 0)):
                print(f"reuse {route}: {len(results)}/{len(route_samples)}", file=sys.stderr, flush=True)
            limit = limits[route]
            for sample in route_samples[len(results) :]:
                if len(str(sample["source_text"])) > int(limit["max_source_characters"]):
                    raise DistillationError(f"{sample['sample_id']} exceeds the frozen source limit")
                response = teacher.generate(
                    prompt=build_prompt(prompt_config, str(sample["source_text"]), str(sample["tgt_lang"])),
                    profile=profile,
                    sample_id=str(sample["sample_id"]),
                    max_tokens=int(limit["max_output_tokens"]),
                    stop=limit["stop"],
                )
                raw = generation_record(
                    sample,
                    profile_name=profile_name,
                    config=prompt_config,
                    response=response,
                )
                results.append(
                    _enrich_raw_record(
                        raw,
                        sample=sample,
                        prompt_config=prompt_config,
                        generation_contract_sha256=contract_sha256,
                    )
                )
                _write_sample_checkpoint(
                    state_root,
                    route=route,
                    index=len(results) - 1,
                    record=results[-1],
                )
                if len(results) % 100 == 0:
                    print(f"progress {route}: {len(results)}/{len(route_samples)}", file=sys.stderr, flush=True)
            shard_path = raw_root / _raw_shard_filename(route)
            atomic_write_jsonl(shard_path, results)
            all_results.extend(results)
            print(f"complete {route}: {len(results)}/{len(route_samples)}", file=sys.stderr, flush=True)
        runtime_evidence = {
            "command": teacher.command,
            "server_log_tail": teacher.logs[-80:],
            "model": str(teacher.paths["model"]),
            "model_sha256": sha256_file(teacher.paths["model"]),
        }
    queue = prepare_review_queue(all_results, config)
    queue_path = repository_root / PurePosixPath(str(config["outputs"]["manual_review_queue"]))
    atomic_write_jsonl(queue_path, queue)
    return {
        **common,
        "elapsed_seconds": round(time.perf_counter() - started, 6),
        "work_root": str(work_root),
        "raw_shards": route_file_identities(raw_root),
        "review_queue": {
            "path": config["outputs"]["manual_review_queue"],
            "records": len(queue),
            "bytes": queue_path.stat().st_size,
            "sha256": sha256_file(queue_path),
        },
        "runtime": runtime_evidence,
    }


def load_raw_results(repository_root: Path, config: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw_root = repository_root / PurePosixPath(str(config["outputs"]["root"])) / str(config["outputs"]["raw_subdir"])
    results: list[dict[str, Any]] = []
    for route in ROUTES:
        path = raw_root / _raw_shard_filename(route)
        if not path.is_file():
            raise DistillationError(f"raw route shard is missing: {path}")
        with path.open("r", encoding="utf-8") as handle:
            route_records = [json.loads(line) for line in handle if line.strip()]
        if len(route_records) != int(config["sampling"]["records_per_route"]):
            raise DistillationError(f"raw route shard has the wrong record count: {path}")
        if any(record.get("route") != route or record.get("split") != "train" for record in route_records):
            raise DistillationError(f"raw route shard contains wrong route/split records: {path}")
        results.extend(route_records)
    return results


def _review_score(kind: str, record: Mapping[str, Any]) -> str:
    return sha256_bytes(f"td08-review-v1\0{kind}\0{record['record_id']}".encode("utf-8"))


def prepare_review_queue(
    records: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    by_route: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        by_route[str(record["route"])].append(record)
    queue: list[dict[str, Any]] = []
    accepted_limit = int(config["manual_review"]["accepted_per_route"])
    rejected_limit = int(config["manual_review"]["rejected_per_route"])
    for route in ROUTES:
        route_records = by_route[route]
        accepted = sorted(
            (record for record in route_records if record["accepted"]),
            key=lambda record: (_review_score("accepted", record), str(record["record_id"])),
        )[:accepted_limit]
        rejected = sorted(
            (record for record in route_records if not record["accepted"]),
            key=lambda record: (_review_score("rejected", record), str(record["record_id"])),
        )[:rejected_limit]
        selected_ids = {str(record["record_id"]) for record in accepted + rejected}
        extra: list[Mapping[str, Any]] = []
        if route.endswith("->zho_Hant"):
            traditional_candidates = sorted(
                (
                    record
                    for record in route_records
                    if record["accepted"] and str(record["record_id"]) not in selected_ids
                ),
                key=lambda record: (
                    int(record["chinese_script_evidence"]["traditional"]),
                    _review_score("traditional", record),
                ),
            )[:5]
            extra.extend(traditional_candidates)
        for selection_tag, selected in (
            ("accepted", accepted),
            ("rejected", rejected),
            ("traditional-extra", extra),
        ):
            for record in selected:
                review_id = sha256_bytes(
                    f"td08-review-record-v1\0{selection_tag}\0{record['record_id']}".encode("utf-8")
                )
                queue.append(
                    {
                        "review_id": review_id,
                        "record_id": record["record_id"],
                        "selection_tag": selection_tag,
                        "route": route,
                        "sample_id": record["sample_id"],
                        "sample_group_id": record["sample_group_id"],
                        "source_text": record["source_text"],
                        "human_reference": record["reference_text"],
                        "teacher_output": record["normalized_output"],
                        "automated_accepted": record["accepted"],
                        "automated_rejection_reasons": record["rejection_reasons"],
                        "script_counts": record["script_counts"],
                        "chinese_script_evidence": record["chinese_script_evidence"],
                    }
                )
    return queue


def route_file_identities(root: Path) -> list[dict[str, Any]]:
    identities: list[dict[str, Any]] = []
    for route in ROUTES:
        path = root / _raw_shard_filename(route)
        identities.append(
            {
                "route": route,
                "path": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return identities


def replay_d0(repository_root: Path, config_path: Path) -> dict[str, Any]:
    repository_root = repository_root.resolve()
    config, prompt_config = bound_configs(repository_root, config_path.resolve())
    samples, contract, contract_sha256 = prepare_inputs(repository_root, config, prompt_config)
    contract_path = write_or_verify_generation_contract(repository_root, config, contract, contract_sha256)
    raw = load_raw_results(repository_root, config)
    raw_by_sample = {str(record["sample_id"]): record for record in raw}
    by_route: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        by_route[route_id(str(sample["src_lang"]), str(sample["tgt_lang"]))].append(sample)
    replay_samples = [
        sample
        for route in ROUTES
        for sample in by_route[route][: int(config["replay"]["samples_per_route"])]
    ]
    profile_name = str(config["prompt_decode"]["selected_profile"])
    profile = prompt_config["decode_profiles"][profile_name]
    limits = route_limits(prompt_config)
    mismatches: list[dict[str, Any]] = []
    replay_records: list[dict[str, Any]] = []
    started = time.perf_counter()
    with LlamaCppTeacher(repository_root, prompt_config) as teacher:
        for sample in replay_samples:
            route = route_id(str(sample["src_lang"]), str(sample["tgt_lang"]))
            limit = limits[route]
            response = teacher.generate(
                prompt=build_prompt(prompt_config, str(sample["source_text"]), str(sample["tgt_lang"])),
                profile=profile,
                sample_id=str(sample["sample_id"]),
                max_tokens=int(limit["max_output_tokens"]),
                stop=limit["stop"],
            )
            replayed = generation_record(
                sample,
                profile_name=profile_name,
                config=prompt_config,
                response=response,
            )
            expected = raw_by_sample[str(sample["sample_id"])]
            raw_match = replayed["raw_output_sha256"] == expected["raw_output_sha256"]
            normalized_match = replayed["normalized_output_sha256"] == expected["normalized_output_sha256"]
            replay_records.append(
                {
                    "sample_id": sample["sample_id"],
                    "route": route,
                    "raw_match": raw_match,
                    "normalized_match": normalized_match,
                    "expected_raw_sha256": expected["raw_output_sha256"],
                    "actual_raw_sha256": replayed["raw_output_sha256"],
                }
            )
            if not raw_match or not normalized_match:
                mismatches.append(replay_records[-1])
    path = repository_root / PurePosixPath(str(config["outputs"]["replay_report"]))
    record_identities_sha256 = sha256_bytes(
        b"".join(canonical_json_bytes(record) for record in replay_records)
    )
    existing = load_json(path) if path.is_file() else {}
    preserve_canonical_run_evidence = (
        not mismatches
        and existing.get("status") == "complete"
        and existing.get("generation_contract_sha256") == sha256_file(contract_path)
        and existing.get("record_identities_sha256") == record_identities_sha256
        and existing.get("exact_raw") is True
        and existing.get("exact_normalized") is True
    )
    created_at = (
        existing["created_at"]
        if preserve_canonical_run_evidence
        else datetime.now(timezone.utc).isoformat()
    )
    elapsed_seconds = (
        existing["elapsed_seconds"]
        if preserve_canonical_run_evidence
        else round(time.perf_counter() - started, 6)
    )
    report = {
        "schema_version": 1,
        "pipeline_version": PIPELINE_VERSION,
        "status": "complete" if not mismatches else "mismatch",
        "created_at": created_at,
        "generation_contract_sha256": sha256_file(contract_path),
        "records": len(replay_records),
        "records_per_route": config["replay"]["samples_per_route"],
        "exact_raw": all(record["raw_match"] for record in replay_records),
        "exact_normalized": all(record["normalized_match"] for record in replay_records),
        "mismatches": mismatches,
        "elapsed_seconds": elapsed_seconds,
        "test_accessed": False,
        "dev_accessed": False,
        "record_identities_sha256": record_identities_sha256,
    }
    atomic_write_json(path, report)
    if mismatches:
        raise DistillationError("TD-08 replay differs from original generation")
    return report


def _load_review_queue(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    if not records:
        raise DistillationError("manual review queue is empty")
    return records


def validate_manual_attestation(
    path: Path,
    queue_path: Path,
    queue: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    attestation = load_yaml(path)
    _exact_keys(attestation, {"schema_version", "status", "queue", "reviewer", "method", "decisions", "notes"}, "manual attestation")
    if attestation["schema_version"] != 1 or attestation["status"] != "complete":
        raise DistillationError("manual attestation is not complete")
    queue_identity = _mapping(attestation["queue"], "manual attestation queue")
    _exact_keys(queue_identity, {"path", "records", "bytes", "sha256"}, "manual attestation queue")
    if queue_identity != {
        "path": config["outputs"]["manual_review_queue"],
        "records": len(queue),
        "bytes": queue_path.stat().st_size,
        "sha256": sha256_file(queue_path),
    }:
        raise DistillationError("manual attestation queue identity differs")
    decisions = _mapping(attestation["decisions"], "manual attestation decisions")
    _exact_keys(
        decisions,
        {
            "accepted_reviewed_by_route",
            "rejected_reviewed_by_route",
            "traditional_extra_reviewed_by_route",
            "manual_rejections",
            "rejected_rule_mismatches",
            "systemic_blocker",
        },
        "manual attestation decisions",
    )
    expected_counts: dict[str, Counter[str]] = defaultdict(Counter)
    queue_by_id = {str(record["review_id"]): record for record in queue}
    for record in queue:
        expected_counts[str(record["route"])][str(record["selection_tag"])] += 1
    for field, tag in (
        ("accepted_reviewed_by_route", "accepted"),
        ("rejected_reviewed_by_route", "rejected"),
        ("traditional_extra_reviewed_by_route", "traditional-extra"),
    ):
        values = _mapping(decisions[field], f"decisions.{field}")
        expected = {route: int(expected_counts[route][tag]) for route in ROUTES}
        if dict(values) != expected:
            raise DistillationError(f"manual review counts differ for {field}")
    if decisions["systemic_blocker"] is not False:
        raise DistillationError("manual review contains a systemic blocker")
    manual_rejections = decisions["manual_rejections"]
    if not isinstance(manual_rejections, list):
        raise DistillationError("manual_rejections must be a list")
    rejected_rule_mismatches = decisions["rejected_rule_mismatches"]
    if not isinstance(rejected_rule_mismatches, list):
        raise DistillationError("rejected_rule_mismatches must be a list")
    seen: set[str] = set()
    for index, raw_rejection in enumerate(manual_rejections):
        rejection = _mapping(raw_rejection, f"manual_rejections[{index}]")
        _exact_keys(rejection, {"review_id", "reason"}, f"manual_rejections[{index}]")
        review_id = str(rejection["review_id"])
        if review_id in seen or review_id not in queue_by_id:
            raise DistillationError(f"manual rejection has an unknown/duplicate review_id: {review_id}")
        if not queue_by_id[review_id]["automated_accepted"]:
            raise DistillationError("manual rejection must refer to an automated-accepted record")
        if not isinstance(rejection["reason"], str) or not str(rejection["reason"]).strip():
            raise DistillationError("manual rejection reason must be non-empty")
        seen.add(review_id)
    for index, raw_mismatch in enumerate(rejected_rule_mismatches):
        mismatch = _mapping(raw_mismatch, f"rejected_rule_mismatches[{index}]")
        _exact_keys(
            mismatch,
            {"review_id", "rule", "reason"},
            f"rejected_rule_mismatches[{index}]",
        )
        review_id = str(mismatch["review_id"])
        if review_id in seen or review_id not in queue_by_id:
            raise DistillationError(
                f"manual filter override has an unknown/duplicate review_id: {review_id}"
            )
        reviewed = queue_by_id[review_id]
        if reviewed["automated_accepted"]:
            raise DistillationError("manual filter override must refer to an automated-rejected record")
        if mismatch["rule"] != "source_copy" or reviewed["automated_rejection_reasons"] != ["source_copy"]:
            raise DistillationError(
                "manual filter override is restricted to source_copy-only false positives"
            )
        if not isinstance(mismatch["reason"], str) or not str(mismatch["reason"]).strip():
            raise DistillationError("manual filter override reason must be non-empty")
        seen.add(review_id)
    return dict(attestation)


def _accepted_sample(
    raw: Mapping[str, Any],
    *,
    generation_contract_sha256: str,
    prompt_config: Mapping[str, Any],
) -> dict[str, Any]:
    target_hash = str(raw["normalized_output_sha256"])
    sample = {
        "sample_id": "teacher-sha256:"
        + sha256_bytes(
            f"{raw['sample_id']}\0{generation_contract_sha256}\0{target_hash}".encode("utf-8")
        ),
        "sample_group_id": raw["sample_group_id"],
        "source_id": raw.get("source_id", "massive-1.1"),
        "source_version": raw.get("source_version", "1.1"),
        "license": raw.get("license", "CC-BY-4.0"),
        "src_lang": raw["src_lang"],
        "tgt_lang": raw["tgt_lang"],
        "source_text": raw["source_text"],
        "target_text": raw["normalized_output"],
        "split": "train",
        "provenance": {
            "kind": "teacher_synthetic",
            "teacher_model": TEACHER_MODEL,
            "teacher_revision": TEACHER_REVISION,
            "prompt_version": prompt_config["prompt"]["version"],
            "decode_config_sha256": raw["decode_config_sha256"],
            "generation_manifest_sha256": generation_contract_sha256,
        },
    }
    return sample


def _quality_summary(
    raw_records: Sequence[Mapping[str, Any]],
    accepted_records: Sequence[Mapping[str, Any]],
    manual_rejected_ids: set[str],
    manual_accepted_ids: set[str],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    raw_by_route: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    accepted_by_route = Counter(f"{record['src_lang']}->{record['tgt_lang']}" for record in accepted_records)
    for record in raw_records:
        raw_by_route[str(record["route"])].append(record)
    routes: dict[str, Any] = {}
    failures: list[str] = []
    gates = config["acceptance_gates"]
    for route in ROUTES:
        records = raw_by_route[route]
        count = len(records)
        reason_counts = Counter(reason for record in records for reason in record["rejection_reasons"])
        accepted = int(accepted_by_route[route])
        retry_records = sum(int(record["request_attempts"]) > 1 for record in records)
        script_rate = sum(bool(record["script_compliant"]) for record in records) / count
        accepted_rate = accepted / count
        retry_rate = retry_records / count
        record = {
            "input": count,
            "accepted": accepted,
            "filtered": count - accepted,
            "accepted_rate": round(accepted_rate, 6),
            "script_compliance_rate": round(script_rate, 6),
            "retry_rate": round(retry_rate, 6),
            "manual_rejections": sum(
                str(item["record_id"]) in manual_rejected_ids for item in records
            ),
            "manual_acceptances": sum(
                str(item["record_id"]) in manual_accepted_ids for item in records
            ),
            "rejection_reasons": dict(sorted(reason_counts.items())),
            "completion_tokens": sum(int(item["completion_tokens"]) for item in records),
            "latency_seconds": round(sum(float(item["latency_seconds"]) for item in records), 6),
        }
        routes[route] = record
        checks = {
            "minimum_accepted": accepted >= int(gates["minimum_accepted_per_route"]),
            "accepted_rate": accepted_rate >= float(gates["minimum_route_accepted_rate"]),
            "script_compliance": script_rate >= float(gates["minimum_route_script_compliance_rate"]),
            "retry_rate": retry_rate <= float(gates["maximum_route_retry_rate"]),
        }
        failures.extend(f"{route}:{name}" for name, passed in checks.items() if not passed)
    return {"routes": routes, "gate_failures": failures}


def finalize_d0(repository_root: Path, config_path: Path) -> dict[str, Any]:
    repository_root = repository_root.resolve()
    config, prompt_config = bound_configs(repository_root, config_path.resolve())
    samples, contract, contract_sha256 = prepare_inputs(repository_root, config, prompt_config)
    contract_path = write_or_verify_generation_contract(repository_root, config, contract, contract_sha256)
    raw_records = load_raw_results(repository_root, config)
    if len(raw_records) != len(samples):
        raise DistillationError("raw generation record count differs from the frozen sample set")
    if any(record.get("generation_manifest_sha256") != contract_sha256 for record in raw_records):
        raise DistillationError("raw records do not bind to the generation contract")

    queue_path = repository_root / PurePosixPath(str(config["outputs"]["manual_review_queue"]))
    queue = _load_review_queue(queue_path)
    attestation_path = repository_root / PurePosixPath(str(config["manual_review"]["attestation"]))
    attestation = validate_manual_attestation(attestation_path, queue_path, queue, config)
    queue_by_id = {str(record["review_id"]): record for record in queue}
    manual_rejected_ids = {
        str(queue_by_id[str(record["review_id"])]["record_id"])
        for record in attestation["decisions"]["manual_rejections"]
    }
    manual_accepted_ids = {
        str(queue_by_id[str(record["review_id"])]["record_id"])
        for record in attestation["decisions"]["rejected_rule_mismatches"]
    }

    replay_path = repository_root / PurePosixPath(str(config["outputs"]["replay_report"]))
    replay = load_json(replay_path)
    if replay.get("status") != "complete" or replay.get("exact_raw") is not True or replay.get("exact_normalized") is not True:
        raise DistillationError("TD-08 replay evidence is missing or not exact")
    if replay.get("generation_contract_sha256") != contract_sha256:
        raise DistillationError("replay evidence binds to a different generation contract")

    accepted_raw = [
        record
        for record in raw_records
        if (
            (record["accepted"] and str(record["record_id"]) not in manual_rejected_ids)
            or str(record["record_id"]) in manual_accepted_ids
        )
    ]
    accepted = [
        _accepted_sample(
            record,
            generation_contract_sha256=contract_sha256,
            prompt_config=prompt_config,
        )
        for record in accepted_raw
    ]
    data_config = load_yaml(repository_root / "configs" / "mvp_model_data.yaml")
    for sample in accepted:
        validate_parallel_sample(sample, data_config)
    filtered = [
        {
            **record,
            "manual_rejected": str(record["record_id"]) in manual_rejected_ids,
            "manual_rejection_reason": next(
                (
                    str(item["reason"])
                    for item in attestation["decisions"]["manual_rejections"]
                    if str(queue_by_id[str(item["review_id"])]["record_id"]) == str(record["record_id"])
                ),
                None,
            ),
        }
        for record in raw_records
        if (
            (not record["accepted"] and str(record["record_id"]) not in manual_accepted_ids)
            or str(record["record_id"]) in manual_rejected_ids
        )
    ]
    quality = _quality_summary(
        raw_records,
        accepted,
        manual_rejected_ids,
        manual_accepted_ids,
        config,
    )
    if quality["gate_failures"]:
        raise DistillationError(
            "TD-08 quality gates failed: " + ", ".join(quality["gate_failures"])
        )

    accepted_path = repository_root / PurePosixPath(str(config["outputs"]["accepted"]))
    filtered_path = repository_root / PurePosixPath(str(config["outputs"]["filtered"]))
    quality_path = repository_root / PurePosixPath(str(config["outputs"]["quality_report"]))
    atomic_write_jsonl(accepted_path, accepted)
    atomic_write_jsonl(filtered_path, filtered)
    quality_report = {
        "schema_version": 1,
        "pipeline_version": PIPELINE_VERSION,
        "status": "complete",
        "generation_contract_sha256": contract_sha256,
        "records": len(raw_records),
        "accepted": len(accepted),
        "filtered": len(filtered),
        "manual_rejections": len(manual_rejected_ids),
        "manual_acceptances": len(manual_accepted_ids),
        "test_accessed": False,
        "dev_accessed": False,
        **quality,
    }
    atomic_write_json(quality_path, quality_report)
    outputs = {
        "generation_contract": {
            "path": config["outputs"]["generation_contract"],
            "bytes": contract_path.stat().st_size,
            "sha256": sha256_file(contract_path),
        },
        "raw_shards": route_file_identities(
            repository_root / PurePosixPath(str(config["outputs"]["root"])) / str(config["outputs"]["raw_subdir"])
        ),
        "accepted": {
            "path": config["outputs"]["accepted"],
            "records": len(accepted),
            "bytes": accepted_path.stat().st_size,
            "sha256": sha256_file(accepted_path),
        },
        "filtered": {
            "path": config["outputs"]["filtered"],
            "records": len(filtered),
            "bytes": filtered_path.stat().st_size,
            "sha256": sha256_file(filtered_path),
        },
        "manual_review_queue": {
            "path": config["outputs"]["manual_review_queue"],
            "records": len(queue),
            "bytes": queue_path.stat().st_size,
            "sha256": sha256_file(queue_path),
        },
        "manual_review_attestation": {
            "path": config["manual_review"]["attestation"],
            "bytes": attestation_path.stat().st_size,
            "sha256": sha256_file(attestation_path),
        },
        "replay_report": {
            "path": config["outputs"]["replay_report"],
            "bytes": replay_path.stat().st_size,
            "sha256": sha256_file(replay_path),
        },
        "quality_report": {
            "path": config["outputs"]["quality_report"],
            "bytes": quality_path.stat().st_size,
            "sha256": sha256_file(quality_path),
        },
    }
    is_d1 = config["identity"] == D1_IDENTITY
    reused_records = sum("reuse_provenance" in record for record in raw_records)
    scope = {
        "routes": 18,
        "input": len(raw_records),
        "accepted": len(accepted),
        "filtered": len(filtered),
        "teacher_synthetic": len(accepted),
        "test_records": 0,
        "dev_records": 0,
    }
    if is_d1:
        scope["reused_d0_inputs"] = reused_records
    manifest = {
        "schema_version": 1,
        "pipeline_version": PIPELINE_VERSION,
        "status": "complete",
        "created_at": replay["created_at"],
        "identity": config["identity"],
        "distillation_config_sha256": config_sha256(config),
        "prompt_config_sha256": config_sha256(prompt_config),
        "generation_contract_sha256": contract_sha256,
        "teacher": contract["teacher"],
        "scope": scope,
        "quality": quality,
        "outputs": outputs,
    }
    manifest_path = repository_root / PurePosixPath(str(config["outputs"]["manifest"]))
    atomic_write_json(manifest_path, manifest)
    evidence = {
        "schema_version": 1,
        "pipeline_version": PIPELINE_VERSION,
        "status": "complete",
        "release": "d1-hymt2-distillation-mvp" if is_d1 else "d0-hymt2-bounded-distillation",
        "corpus_maturity": "mvp" if is_d1 else "smoke",
        "date": str(replay["created_at"])[:10],
        "scope": manifest["scope"],
        "identities": {
            "distillation_config_sha256": config_sha256(config),
            "prompt_config_file_sha256": config["prompt_decode"]["file_sha256"],
            "td07_calibration_report_sha256": config["prompt_decode"]["calibration_report_sha256"],
            "generation_contract_sha256": contract_sha256,
            "manifest_sha256": sha256_file(manifest_path),
            "accepted_sha256": outputs["accepted"]["sha256"],
            "quality_report_sha256": outputs["quality_report"]["sha256"],
            "manual_review_sha256": outputs["manual_review_attestation"]["sha256"],
            "replay_report_sha256": outputs["replay_report"]["sha256"],
        },
        "quality": quality,
        "manual_review": {
            "queue_records": len(queue),
            "manual_rejections": len(manual_rejected_ids),
            "manual_acceptances": len(manual_accepted_ids),
            "systemic_blocker": False,
        },
        "replay": {
            "records": replay["records"],
            "exact_raw": True,
            "exact_normalized": True,
        },
        "test_accessed": False,
        "dev_accessed": False,
        "downstream_consumer": "TD-15" if is_d1 else None,
        "td08_completed": is_d1,
        "td09_started": False,
    }
    evidence_path = repository_root / PurePosixPath(str(config["outputs"]["evidence"]))
    atomic_write_json(evidence_path, evidence)
    return evidence
