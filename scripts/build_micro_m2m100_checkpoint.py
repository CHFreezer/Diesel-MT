#!/usr/bin/env python3
"""Build the deterministic local micro M2M100 checkpoint used by CT2 tests.

The checkpoint contains a copy of the frozen tokenizer because the
CTranslate2 Transformers converter loads the model and tokenizer from the
same local directory. The immutable source tokenizer is only read and
verified; generated weights live under a Git-ignored runtime directory.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import shutil
import struct
import tempfile
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in os.sys.path:
    os.sys.path.insert(0, str(SCRIPTS_DIR))

from freeze_tokenizer_artifact import sha256_file, verify_artifact_files  # noqa: E402
from tokenizer_utils import (  # noqa: E402
    PROJECT_LANGUAGES,
    REQUIRED_SPECIAL_IDS,
    atomic_write_json,
    build_language_mapping,
    reload_tokenizer,
    replace_file_with_retry,
    verify_tokenizer,
)


SCHEMA_VERSION = 1
CHECKPOINT_MANIFEST = "checkpoint_manifest.json"
DEFAULT_SPEC = Path("configs/micro_m2m100_deployment.json")
DEFAULT_OUTPUT = Path("artifacts/ctranslate2/runtime/hf-micro-checkpoint")
DEFAULT_REPORT = Path("artifacts/ctranslate2/deployment-validation.json")
REPORT_WORKFLOW = "ctranslate2-deployment"
REPORT_PHASES = (
    "td_01_hf_checkpoint",
    "td_02_conversion",
    "td_03_vocab_integrity",
    "td_04_cpu_inference",
    "td_05_offline_package",
)
SMOKE_SOURCE = "A local checkpoint verifies the deployment interface."


class CheckpointBuildError(RuntimeError):
    """Raised when a checkpoint build or validation invariant fails."""


def read_object(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CheckpointBuildError(f"cannot read JSON object {path}: {error}") from error
    if not isinstance(value, dict):
        raise CheckpointBuildError(f"expected a JSON object in {path}")
    return value


def update_consolidated_report(path: Path, updates: Mapping[str, Mapping]) -> dict:
    """Merge successful phase records into the workflow's single JSON report."""
    unknown = set(updates) - set(REPORT_PHASES)
    if unknown:
        raise CheckpointBuildError(f"unknown deployment report phases: {sorted(unknown)}")

    if path.exists():
        report = read_object(path)
        if (
            report.get("schema_version") != SCHEMA_VERSION
            or report.get("workflow") != REPORT_WORKFLOW
            or not isinstance(report.get("phases"), dict)
        ):
            raise CheckpointBuildError(f"invalid consolidated deployment report: {path}")
        phases = dict(report["phases"])
        unknown_existing = set(phases) - set(REPORT_PHASES)
        if unknown_existing:
            raise CheckpointBuildError(
                f"unknown phases in consolidated deployment report: {sorted(unknown_existing)}"
            )
    else:
        phases = {}

    generated_at = datetime.now(timezone.utc).isoformat()
    for name, record in updates.items():
        payload = dict(record)
        payload["generated_at_utc"] = generated_at
        phases[name] = payload

    statuses = [phase.get("status") for phase in phases.values()]
    if len(phases) == len(REPORT_PHASES) and all(status == "passed" for status in statuses):
        status = "passed"
    elif any(status != "passed" for status in statuses):
        status = "failed"
    else:
        status = "partial"

    report = {
        "schema_version": SCHEMA_VERSION,
        "workflow": REPORT_WORKFLOW,
        "status": status,
        "updated_at_utc": generated_at,
        "phases": {name: phases[name] for name in REPORT_PHASES if name in phases},
    }
    atomic_write_json(path, report)
    return report


def portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return resolved.name


def canonical_json_sha256(value: Mapping) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def validate_spec(spec: Mapping) -> tuple[dict, dict]:
    if spec.get("schema_version") != SCHEMA_VERSION:
        raise CheckpointBuildError("unsupported micro-checkpoint specification schema")
    tokenizer_spec = spec.get("tokenizer")
    model_config = spec.get("model_config")
    if not isinstance(tokenizer_spec, dict) or not isinstance(model_config, dict):
        raise CheckpointBuildError("specification requires tokenizer and model_config objects")
    required_tokenizer = {
        "path",
        "expected_vocab_size",
        "artifact_manifest_sha256",
    }
    missing_tokenizer = required_tokenizer - set(tokenizer_spec)
    if missing_tokenizer:
        raise CheckpointBuildError(
            f"tokenizer specification is missing: {sorted(missing_tokenizer)}"
        )
    required_model = {
        "d_model",
        "encoder_layers",
        "decoder_layers",
        "encoder_ffn_dim",
        "decoder_ffn_dim",
        "encoder_attention_heads",
        "decoder_attention_heads",
        "max_position_embeddings",
        "tie_word_embeddings",
    }
    missing_model = required_model - set(model_config)
    if missing_model:
        raise CheckpointBuildError(
            f"model configuration is missing: {sorted(missing_model)}"
        )
    seed = spec.get("random_seed")
    if not isinstance(seed, int) or seed < 0:
        raise CheckpointBuildError("random_seed must be a non-negative integer")
    if model_config.get("tie_word_embeddings") is not True:
        raise CheckpointBuildError("micro checkpoint requires tied word embeddings")
    return dict(tokenizer_spec), dict(model_config)


def verify_frozen_tokenizer(
    tokenizer_dir: Path,
    *,
    expected_vocab_size: int,
    expected_manifest_sha256: str,
) -> tuple[object, dict]:
    manifest, manifest_sha256 = verify_artifact_files(tokenizer_dir)
    if manifest_sha256 != expected_manifest_sha256:
        raise CheckpointBuildError(
            "frozen tokenizer manifest SHA-256 changed: "
            f"{manifest_sha256} != {expected_manifest_sha256}"
        )
    tokenizer = reload_tokenizer(tokenizer_dir)
    verify_tokenizer(tokenizer, expected_vocab_size=expected_vocab_size)
    language_ids = build_language_mapping(tokenizer)
    if tuple(language_ids) != PROJECT_LANGUAGES:
        raise CheckpointBuildError("tokenizer language order differs from the project contract")
    return tokenizer, {
        "path": portable_path(tokenizer_dir),
        "artifact_manifest_sha256": manifest_sha256,
        "artifact_files_verified": len(manifest["files"]),
        "vocab_size": len(tokenizer),
        "language_token_ids": language_ids,
        "special_token_ids": {
            token: tokenizer.convert_tokens_to_ids(token)
            for token in REQUIRED_SPECIAL_IDS
        },
    }


def model_dimensions(model: object) -> dict[str, int | bool]:
    dimensions: dict[str, int | bool] = {
        "config_vocab_size": model.config.vocab_size,
        "shared_embedding_rows": model.model.shared.num_embeddings,
        "encoder_embedding_rows": model.model.encoder.embed_tokens.num_embeddings,
        "decoder_embedding_rows": model.model.decoder.embed_tokens.num_embeddings,
        "lm_head_rows": model.lm_head.out_features,
        "input_output_embeddings_tied": (
            model.model.shared.weight.data_ptr() == model.lm_head.weight.data_ptr()
        ),
    }
    expected = int(model.config.vocab_size)
    for name in (
        "shared_embedding_rows",
        "encoder_embedding_rows",
        "decoder_embedding_rows",
        "lm_head_rows",
    ):
        if dimensions[name] != expected:
            raise CheckpointBuildError(
                f"model vocabulary dimension mismatch for {name}: {dimensions[name]} != {expected}"
            )
    if dimensions["input_output_embeddings_tied"] is not True:
        raise CheckpointBuildError("input and output word embeddings are not tied")
    return dimensions


def state_dict_sha256(model: object) -> str:
    """Hash tensor names, dtypes, shapes, and canonical CPU bytes."""
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        name_bytes = name.encode("utf-8")
        dtype_bytes = str(value.dtype).encode("ascii")
        digest.update(struct.pack("<I", len(name_bytes)))
        digest.update(name_bytes)
        digest.update(struct.pack("<I", len(dtype_bytes)))
        digest.update(dtype_bytes)
        digest.update(struct.pack("<I", value.ndim))
        for dimension in value.shape:
            digest.update(struct.pack("<Q", dimension))
        digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def build_model(tokenizer: object, model_config: Mapping, seed: int) -> object:
    import torch
    from transformers import M2M100Config, M2M100ForConditionalGeneration

    kwargs = dict(model_config)
    kwargs.update(
        {
            "vocab_size": len(tokenizer),
            "bos_token_id": tokenizer.bos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "decoder_start_token_id": tokenizer.eos_token_id,
        }
    )
    random_state = random.getstate()
    try:
        random.seed(seed)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            model = M2M100ForConditionalGeneration(M2M100Config(**kwargs))
    finally:
        random.setstate(random_state)
    model.eval()
    model_dimensions(model)
    return model


def smoke_forward(model: object, tokenizer: object) -> dict:
    import torch

    original_language = tokenizer.src_lang
    try:
        tokenizer.src_lang = "eng_Latn"
        inputs = tokenizer(SMOKE_SOURCE, return_tensors="pt")
        decoder_input_ids = torch.tensor(
            [[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("zho_Hans")]],
            dtype=torch.long,
        )
        with torch.no_grad():
            output = model(**inputs, decoder_input_ids=decoder_input_ids)
        if output.logits.shape != (1, 2, len(tokenizer)):
            raise CheckpointBuildError(
                f"unexpected smoke logits shape: {tuple(output.logits.shape)}"
            )
        if not bool(torch.isfinite(output.logits).all().item()):
            raise CheckpointBuildError("smoke forward produced non-finite logits")
        return {
            "source_language": "eng_Latn",
            "target_language": "zho_Hans",
            "source_token_ids": inputs["input_ids"][0].tolist(),
            "decoder_input_ids": decoder_input_ids[0].tolist(),
            "logits_shape": list(output.logits.shape),
            "logits_finite": True,
        }
    finally:
        tokenizer.src_lang = original_language


def file_records(directory: Path, *, excluded: frozenset[str] = frozenset()) -> list[dict]:
    records = []
    for path in sorted(item for item in directory.rglob("*") if item.is_file()):
        relative = path.relative_to(directory).as_posix()
        if relative in excluded:
            continue
        records.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return records


def package_versions() -> dict[str, str]:
    result = {"python": platform.python_version()}
    for distribution in (
        "torch",
        "transformers",
        "tokenizers",
        "safetensors",
        "ctranslate2",
    ):
        try:
            result[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            result[distribution] = "not-installed"
    return result


def save_and_reload_checkpoint(
    output_dir: Path,
    *,
    tokenizer: object,
    model: object,
    spec: Mapping,
    tokenizer_record: Mapping,
    overwrite: bool,
) -> dict:
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, NllbTokenizer

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent)
    )
    try:
        model.save_pretrained(staging, safe_serialization=True)
        tokenizer.save_pretrained(staging)
        saved_files = file_records(staging)
        if not any(row["path"].endswith(".safetensors") for row in saved_files):
            raise CheckpointBuildError("checkpoint did not save safetensors weights")
        original_state_sha256 = state_dict_sha256(model)
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "status": "complete",
            "purpose": spec["purpose"],
            "random_seed": spec["random_seed"],
            "spec_sha256": canonical_json_sha256(spec),
            "state_dict_sha256": original_state_sha256,
            "tokenizer": dict(tokenizer_record),
            "files": saved_files,
            "warning": "Random weights for deployment-interface validation only; no translation quality.",
        }
        atomic_write_json(staging / CHECKPOINT_MANIFEST, manifest)

        reloaded_tokenizer = AutoTokenizer.from_pretrained(
            staging, local_files_only=True
        )
        if not isinstance(reloaded_tokenizer, NllbTokenizer):
            raise CheckpointBuildError(
                "offline checkpoint reload returned the wrong tokenizer class: "
                f"{type(reloaded_tokenizer).__name__}"
            )
        verify_tokenizer(
            reloaded_tokenizer,
            expected_vocab_size=int(tokenizer_record["vocab_size"]),
        )
        if reloaded_tokenizer.get_vocab() != tokenizer.get_vocab():
            raise CheckpointBuildError("checkpoint tokenizer vocabulary differs from frozen source")

        reloaded_model = AutoModelForSeq2SeqLM.from_pretrained(
            staging, local_files_only=True
        )
        reloaded_model.eval()
        dimensions = model_dimensions(reloaded_model)
        reloaded_state_sha256 = state_dict_sha256(reloaded_model)
        if reloaded_state_sha256 != original_state_sha256:
            raise CheckpointBuildError("model state changed across save and offline reload")
        smoke = smoke_forward(reloaded_model, reloaded_tokenizer)
        manifest_sha256 = sha256_file(staging / CHECKPOINT_MANIFEST)

        backup: Path | None = None
        if output_dir.exists():
            if not overwrite:
                raise CheckpointBuildError(
                    f"output directory already exists (use --overwrite): {output_dir}"
                )
            backup = Path(
                tempfile.mkdtemp(
                    prefix=f".{output_dir.name}.backup.", dir=output_dir.parent
                )
            )
            backup.rmdir()
            output_dir.replace(backup)
        try:
            staging.replace(output_dir)
        except BaseException:
            if backup is not None and backup.exists() and not output_dir.exists():
                backup.replace(output_dir)
            raise
        else:
            if backup is not None:
                shutil.rmtree(backup)
        return {
            "checkpoint": portable_path(output_dir),
            "checkpoint_manifest_sha256": manifest_sha256,
            "files": saved_files,
            "dimensions": dimensions,
            "parameter_count": sum(parameter.numel() for parameter in reloaded_model.parameters()),
            "state_dict_sha256": reloaded_state_sha256,
            "offline_model_reload": True,
            "offline_tokenizer_reload": True,
            "smoke_forward": smoke,
        }
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def build_checkpoint(
    *,
    spec_path: Path,
    output_dir: Path,
    tokenizer_dir: Path | None = None,
    overwrite: bool = False,
) -> dict:
    spec = read_object(spec_path)
    tokenizer_spec, model_config = validate_spec(spec)
    if tokenizer_dir is None:
        configured = Path(str(tokenizer_spec["path"]))
        tokenizer_dir = configured if configured.is_absolute() else ROOT / configured
    tokenizer_dir = tokenizer_dir.resolve()
    output_dir = output_dir.resolve()

    if output_dir.exists() and not overwrite:
        raise CheckpointBuildError(
            f"output directory already exists (use --overwrite): {output_dir}"
        )

    tokenizer, tokenizer_record = verify_frozen_tokenizer(
        tokenizer_dir,
        expected_vocab_size=int(tokenizer_spec["expected_vocab_size"]),
        expected_manifest_sha256=str(tokenizer_spec["artifact_manifest_sha256"]),
    )
    model = build_model(tokenizer, model_config, int(spec["random_seed"]))
    record = save_and_reload_checkpoint(
        output_dir,
        tokenizer=tokenizer,
        model=model,
        spec=spec,
        tokenizer_record=tokenizer_record,
        overwrite=overwrite,
    )
    record.update(
        {
            "schema_version": SCHEMA_VERSION,
            "status": "passed",
            "spec": portable_path(spec_path),
            "spec_file_sha256": sha256_file(spec_path),
            "spec_sha256": canonical_json_sha256(spec),
            "random_seed": spec["random_seed"],
            "model_config": model_config,
            "tokenizer": tokenizer_record,
            "versions": package_versions(),
            "warning": "Random weights validate deployment interfaces only and have no translation quality.",
        }
    )
    record["reproduction_command"] = (
        ".\\.conda\\python.exe scripts\\build_micro_m2m100_checkpoint.py "
        f"--spec {portable_path(spec_path)} --output-dir {portable_path(output_dir)} --overwrite"
    )
    return record


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        replace_file_with_retry(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--tokenizer-dir", type=Path)
    parser.add_argument("--report-json", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    record = build_checkpoint(
        spec_path=args.spec.resolve(),
        output_dir=args.output_dir,
        tokenizer_dir=args.tokenizer_dir,
        overwrite=args.overwrite,
    )
    update_consolidated_report(args.report_json, {"td_01_hf_checkpoint": record})
    print(
        json.dumps(
            {
                "status": record["status"],
                "checkpoint": record["checkpoint"],
                "state_dict_sha256": record["state_dict_sha256"],
                "report": portable_path(args.report_json),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
