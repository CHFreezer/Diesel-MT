"""TD-12 frozen M1 overfit, generation, resume, and HF reload acceptance."""

from __future__ import annotations

import gc
import json
import math
import os
import re
import shutil
import tempfile
import unicodedata
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from freeze_tokenizer_artifact import sha256_file
from model_data_pipeline import wrong_script_dominates
from model_training_contract import config_sha256, load_student_config
from mvp_checkpoint import CHECKPOINT_MANIFEST, validate_checkpoint
from mvp_student import (
    DirectionAwareCollator,
    EncodingPolicy,
    build_student,
    load_frozen_tokenizer,
    model_inputs,
    state_dict_sha256,
    validate_student_alignment,
)
from mvp_training import (
    ROUTE_ORDER,
    _atomic_json,
    load_route_dataset,
    load_training_config,
    read_jsonl,
    run_training,
    semantic_trace_sha256,
)
from tokenizer_utils import forced_bos_token_id, verify_tokenizer


M1_SCHEMA_VERSION = 1


class M1AcceptanceError(RuntimeError):
    """Raised when the pre-frozen M1 acceptance contract is not met."""


def _expect_keys(value: Mapping[str, Any], expected: set[str], context: str) -> None:
    missing = sorted(expected - set(value))
    unknown = sorted(set(value) - expected)
    if missing or unknown:
        raise M1AcceptanceError(
            f"{context} fields differ: missing={missing}, unknown={unknown}"
        )


def load_m1_acceptance(path: Path, repository_root: Path) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise M1AcceptanceError(f"cannot load M1 acceptance config: {exc}") from exc
    if not isinstance(value, Mapping):
        raise M1AcceptanceError("M1 acceptance config must be a mapping")
    _expect_keys(
        value,
        {"schema_version", "identity", "fixture", "thresholds", "generation", "resume"},
        "M1 acceptance",
    )
    if value["schema_version"] != M1_SCHEMA_VERSION:
        raise M1AcceptanceError("M1 acceptance schema_version changed")
    identity = value["identity"]
    _expect_keys(
        identity,
        {
            "name",
            "training_config",
            "training_config_file_sha256",
            "training_config_canonical_sha256",
            "initial_state_dict_sha256",
            "tokenizer_manifest_sha256",
        },
        "M1 identity",
    )
    training_path = repository_root / identity["training_config"]
    training = load_training_config(training_path)
    if sha256_file(training_path) != identity["training_config_file_sha256"]:
        raise M1AcceptanceError("M1 training config file SHA-256 changed")
    if config_sha256(training) != identity["training_config_canonical_sha256"]:
        raise M1AcceptanceError("M1 training config canonical SHA-256 changed")
    if training["identity"]["mode"] != "m1":
        raise M1AcceptanceError("M1 acceptance requires an m1 training config")
    if training["identity"]["tokenizer_manifest_sha256"] != identity["tokenizer_manifest_sha256"]:
        raise M1AcceptanceError("M1 tokenizer identity changed")
    _expect_keys(
        value["fixture"],
        {"split", "records_per_route", "required_routes", "required_exposures_per_route"},
        "M1 fixture",
    )
    if dict(value["fixture"]) != {
        "split": "train",
        "records_per_route": 1,
        "required_routes": 20,
        "required_exposures_per_route": int(training["optimization"]["max_optimizer_steps"]),
    }:
        raise M1AcceptanceError("M1 fixture contract changed")
    _expect_keys(
        value["thresholds"],
        {
            "final_eval_loss_to_initial_ratio_max",
            "normalized_exact_match_routes",
            "target_language_control_routes",
            "empty_output_max",
            "cross_language_source_copy_max",
        },
        "M1 thresholds",
    )
    if dict(value["thresholds"]) != {
        "final_eval_loss_to_initial_ratio_max": 0.10,
        "normalized_exact_match_routes": 20,
        "target_language_control_routes": 20,
        "empty_output_max": 0,
        "cross_language_source_copy_max": 0,
    }:
        raise M1AcceptanceError("M1 thresholds changed")
    _expect_keys(
        value["generation"],
        {
            "decoding",
            "do_sample",
            "num_beams",
            "max_new_tokens",
            "length_penalty",
            "early_stopping",
            "normalization",
        },
        "M1 generation",
    )
    if value["generation"] != {
        "decoding": "greedy",
        "do_sample": False,
        "num_beams": 1,
        "max_new_tokens": 64,
        "length_penalty": 1.0,
        "early_stopping": False,
        "normalization": "unicode_nfc_strip_collapse_whitespace",
    }:
        raise M1AcceptanceError("M1 generation contract changed")
    _expect_keys(
        value["resume"],
        {"interruption_optimizer_step", "final_optimizer_step", "comparison"},
        "M1 resume",
    )
    if value["resume"]["comparison"] != "exact":
        raise M1AcceptanceError("M1 resume comparison must be exact")
    if value["resume"]["final_optimizer_step"] != training["optimization"]["max_optimizer_steps"]:
        raise M1AcceptanceError("M1 final step differs from the training config")
    if not 0 < value["resume"]["interruption_optimizer_step"] < value["resume"]["final_optimizer_step"]:
        raise M1AcceptanceError("M1 interruption step is invalid")
    return dict(value)


def normalize_generation(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", text).strip())


def _ordered_records(dataset: object) -> list[dict[str, Any]]:
    return [dataset.records_by_route[route][0] for route in ROUTE_ORDER]


def fixture_loss(
    model: object,
    tokenizer: object,
    records: list[dict[str, Any]],
    resource: Mapping[str, Any],
) -> float:
    import torch

    policy = EncodingPolicy(
        max_source_length=int(resource["max_source_length"]),
        max_target_length=int(resource["max_target_length"]),
    )
    batch = DirectionAwareCollator(tokenizer, policy)(records)
    device = torch.device(resource["device"])
    model.to(device)
    model.eval()
    moved = {name: tensor.to(device) for name, tensor in model_inputs(batch).items()}
    context = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device.type == "cuda" and resource["precision"] == "bf16"
        else __import__("contextlib").nullcontext()
    )
    with torch.no_grad(), context:
        loss = model(**moved).loss
    if loss is None or not bool(torch.isfinite(loss).item()):
        raise M1AcceptanceError("M1 fixture evaluation produced a non-finite loss")
    return float(loss.detach().cpu().item())


def generate_fixture(
    model: object,
    tokenizer: object,
    records: list[dict[str, Any]],
    generation: Mapping[str, Any],
) -> list[dict[str, Any]]:
    import torch

    device = next(model.parameters()).device
    model.eval()
    original_language = tokenizer.src_lang
    outputs = []
    try:
        for record in records:
            tokenizer.src_lang = record["src_lang"]
            encoded = tokenizer(
                record["source_text"],
                return_tensors="pt",
                truncation=True,
                max_length=64,
            )
            encoded = {name: tensor.to(device) for name, tensor in encoded.items()}
            target_id = forced_bos_token_id(tokenizer, record["tgt_lang"])
            with torch.no_grad():
                generated = model.generate(
                    **encoded,
                    forced_bos_token_id=target_id,
                    do_sample=bool(generation["do_sample"]),
                    num_beams=int(generation["num_beams"]),
                    max_new_tokens=int(generation["max_new_tokens"]),
                    length_penalty=float(generation["length_penalty"]),
                )
            ids = generated[0].detach().cpu().tolist()
            text = tokenizer.decode(ids, skip_special_tokens=True)
            normalized = normalize_generation(text)
            reference = normalize_generation(record["target_text"])
            source = normalize_generation(record["source_text"])
            target_control = len(ids) > 1 and ids[1] == target_id
            target_control = target_control and not wrong_script_dominates(
                normalized, record["tgt_lang"]
            )
            cross_language = not {
                record["src_lang"],
                record["tgt_lang"],
            } <= {"zho_Hans", "zho_Hant"}
            outputs.append(
                {
                    "sample_id": record["sample_id"],
                    "sample_group_id": record["sample_group_id"],
                    "route": f"{record['src_lang']}->{record['tgt_lang']}",
                    "source": record["source_text"],
                    "reference": record["target_text"],
                    "generated": text,
                    "generated_token_ids": ids,
                    "normalized_exact_match": normalized == reference,
                    "target_language_control": target_control,
                    "empty_output": not normalized,
                    "cross_language_source_copy": cross_language and normalized == source,
                }
            )
    finally:
        tokenizer.src_lang = original_language
    return outputs


def _payload_hashes(checkpoint: Path) -> dict[str, str]:
    manifest = json.loads((checkpoint / CHECKPOINT_MANIFEST).read_text(encoding="utf-8"))
    return {record["path"]: record["sha256"] for record in manifest["files"]}


def save_m1_hf_checkpoint(
    output_dir: Path,
    *,
    model: object,
    tokenizer: object,
    student_config: Mapping[str, Any],
    source_checkpoint: Path,
) -> dict[str, Any]:
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    if output_dir.exists():
        raise M1AcceptanceError(f"M1 HF output already exists: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        model.to("cpu")
        model.save_pretrained(staging, safe_serialization=True)
        tokenizer.save_pretrained(staging)
        state_hash = state_dict_sha256(model)
        files = [
            {
                "path": path.relative_to(staging).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in sorted(item for item in staging.rglob("*") if item.is_file())
        ]
        manifest = {
            "schema_version": 1,
            "status": "complete",
            "purpose": "TD-12 M1 fixture memorization only; not translation quality",
            "student_config_sha256": config_sha256(student_config),
            "source_checkpoint_manifest_sha256": sha256_file(
                source_checkpoint / CHECKPOINT_MANIFEST
            ),
            "state_dict_sha256": state_hash,
            "tokenizer_manifest_sha256": student_config["tokenizer"][
                "artifact_manifest_sha256"
            ],
            "files": files,
        }
        _atomic_json(staging / "m1-manifest.json", manifest)
        reloaded_tokenizer = AutoTokenizer.from_pretrained(staging, local_files_only=True)
        verify_tokenizer(reloaded_tokenizer, expected_vocab_size=49_152)
        if reloaded_tokenizer.get_vocab() != tokenizer.get_vocab():
            raise M1AcceptanceError("M1 tokenizer changed across offline reload")
        reloaded_model = AutoModelForSeq2SeqLM.from_pretrained(
            staging, local_files_only=True
        )
        validate_student_alignment(reloaded_model, reloaded_tokenizer, student_config)
        if state_dict_sha256(reloaded_model) != state_hash:
            raise M1AcceptanceError("M1 model state changed across offline reload")
        os.replace(staging, output_dir)
        return {
            "path": str(output_dir),
            "manifest_sha256": sha256_file(output_dir / "m1-manifest.json"),
            "state_dict_sha256": state_hash,
            "files": files,
            "offline_model_reload": True,
            "offline_tokenizer_reload": True,
        }
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def run_m1_acceptance(
    *,
    repository_root: Path,
    acceptance_path: Path,
    runtime_root: Path,
) -> dict[str, Any]:
    import torch

    acceptance = load_m1_acceptance(acceptance_path, repository_root)
    training_path = repository_root / acceptance["identity"]["training_config"]
    training = load_training_config(training_path)
    student_path = repository_root / training["identity"]["student_config"]
    student = load_student_config(student_path)
    tokenizer, tokenizer_identity = load_frozen_tokenizer(student, repository_root)
    data = training["data"]
    fixture = load_route_dataset(
        repository_root / data["train_path"],
        expected_sha256=data["train_sha256"],
        split="train",
        max_records_per_route=1,
    )
    records = _ordered_records(fixture)

    initial_model, _ = build_student(student, tokenizer)
    initial_state_hash = state_dict_sha256(initial_model)
    if initial_state_hash != acceptance["identity"]["initial_state_dict_sha256"]:
        raise M1AcceptanceError("M1 initial state-dict identity changed")
    initial_loss = fixture_loss(
        initial_model, tokenizer, records, training["resource_profile"]
    )
    del initial_model
    gc.collect()
    torch.cuda.empty_cache()

    final_step = int(acceptance["resume"]["final_optimizer_step"])
    interruption_step = int(acceptance["resume"]["interruption_optimizer_step"])
    baseline = run_training(
        config_path=training_path,
        repository_root=repository_root,
        output_dir=runtime_root / "uninterrupted-run",
        dry_run=False,
        checkpoint_root=runtime_root / "uninterrupted-checkpoints",
    )
    gc.collect()
    torch.cuda.empty_cache()
    interrupted = run_training(
        config_path=training_path,
        repository_root=repository_root,
        output_dir=runtime_root / "interrupted-run",
        dry_run=False,
        checkpoint_root=runtime_root / "resumed-checkpoints",
        stop_after_optimizer_steps=interruption_step,
    )
    gc.collect()
    torch.cuda.empty_cache()
    resume_checkpoint = runtime_root / "resumed-checkpoints" / f"step-{interruption_step:08d}"
    resumed = run_training(
        config_path=training_path,
        repository_root=repository_root,
        output_dir=runtime_root / "resumed-run",
        dry_run=False,
        checkpoint_root=runtime_root / "resumed-checkpoints",
        resume_from=resume_checkpoint,
    )
    final_baseline = runtime_root / "uninterrupted-checkpoints" / f"step-{final_step:08d}"
    final_resumed = runtime_root / "resumed-checkpoints" / f"step-{final_step:08d}"
    validate_checkpoint(final_baseline)
    validate_checkpoint(final_resumed)
    baseline_trace = semantic_trace_sha256(
        read_jsonl(Path(baseline["output_root"]) / baseline["events"]["path"])
    )
    resumed_trace = semantic_trace_sha256(
        [
            *read_jsonl(Path(interrupted["output_root"]) / interrupted["events"]["path"]),
            *read_jsonl(Path(resumed["output_root"]) / resumed["events"]["path"]),
        ]
    )
    resume_checks = {
        "semantic_trace": baseline_trace == resumed_trace,
        "final_loss": baseline["result"]["final_train_loss"]
        == resumed["result"]["final_train_loss"],
        "mean_loss": baseline["result"]["mean_train_loss"]
        == resumed["result"]["mean_train_loss"],
        "sampler_state": config_sha256(baseline["result"]["sampler_state"])
        == config_sha256(resumed["result"]["sampler_state"]),
        "checkpoint_payloads": _payload_hashes(final_baseline)
        == _payload_hashes(final_resumed),
    }
    if not all(resume_checks.values()):
        raise M1AcceptanceError(f"M1 resume comparison failed: {resume_checks}")

    final_model, _ = build_student(student, tokenizer)
    final_model.load_state_dict(
        torch.load(final_resumed / "model.pt", map_location="cpu", weights_only=True),
        strict=True,
    )
    final_loss = fixture_loss(
        final_model, tokenizer, records, training["resource_profile"]
    )
    ratio = final_loss / initial_loss
    generated = generate_fixture(
        final_model, tokenizer, records, acceptance["generation"]
    )
    counts = {
        "normalized_exact_match_routes": sum(
            bool(row["normalized_exact_match"]) for row in generated
        ),
        "target_language_control_routes": sum(
            bool(row["target_language_control"]) for row in generated
        ),
        "empty_outputs": sum(bool(row["empty_output"]) for row in generated),
        "cross_language_source_copies": sum(
            bool(row["cross_language_source_copy"]) for row in generated
        ),
    }
    thresholds = acceptance["thresholds"]
    route_counts = resumed["result"]["route_counts"]
    gates = {
        "loss_ratio": ratio <= thresholds["final_eval_loss_to_initial_ratio_max"],
        "normalized_exact_match": counts["normalized_exact_match_routes"]
        >= thresholds["normalized_exact_match_routes"],
        "target_language_control": counts["target_language_control_routes"]
        >= thresholds["target_language_control_routes"],
        "empty_output": counts["empty_outputs"] <= thresholds["empty_output_max"],
        "cross_language_source_copy": counts["cross_language_source_copies"]
        <= thresholds["cross_language_source_copy_max"],
        "route_exposure": set(route_counts) == set(ROUTE_ORDER)
        and min(route_counts.values())
        >= acceptance["fixture"]["required_exposures_per_route"],
        "resume_exact": all(resume_checks.values()),
    }
    hf = save_m1_hf_checkpoint(
        runtime_root / "m1-hf",
        model=final_model,
        tokenizer=tokenizer,
        student_config=student,
        source_checkpoint=final_resumed,
    )
    tokenizer_manifest_after = sha256_file(
        repository_root
        / student["tokenizer"]["path"]
        / "artifact_manifest.json"
    )
    gates["tokenizer_unchanged"] = (
        tokenizer_manifest_after == acceptance["identity"]["tokenizer_manifest_sha256"]
    )
    report = {
        "schema_version": M1_SCHEMA_VERSION,
        "status": "complete" if all(gates.values()) else "failed",
        "task": "TD-12",
        "acceptance_config": {
            "path": acceptance_path.relative_to(repository_root).as_posix(),
            "file_sha256": sha256_file(acceptance_path),
            "canonical_sha256": config_sha256(acceptance),
        },
        "training_config": {
            "path": training_path.relative_to(repository_root).as_posix(),
            "file_sha256": sha256_file(training_path),
            "canonical_sha256": config_sha256(training),
        },
        "tokenizer": tokenizer_identity,
        "fixture": {
            "selection_sha256": fixture.selection_sha256,
            "records": len(records),
            "routes": list(ROUTE_ORDER),
        },
        "loss": {
            "initial_eval": initial_loss,
            "final_eval": final_loss,
            "final_to_initial_ratio": ratio,
            "threshold_ratio_max": thresholds[
                "final_eval_loss_to_initial_ratio_max"
            ],
        },
        "generation": {"counts": counts, "records": generated},
        "route_exposure": route_counts,
        "resume": {
            "interruption_step": interruption_step,
            "final_step": final_step,
            "checks": resume_checks,
            "semantic_trace_sha256": baseline_trace,
            "final_checkpoint_payloads": _payload_hashes(final_resumed),
        },
        "resources": {
            "uninterrupted": baseline["result"],
            "resumed": resumed["result"],
        },
        "hf_checkpoint": hf,
        "gates": gates,
        "warning": "M1 fixture memorization and resume acceptance only; not translation quality.",
    }
    if report["status"] != "complete":
        raise M1AcceptanceError(f"M1 frozen gates failed: {gates}")
    return report
