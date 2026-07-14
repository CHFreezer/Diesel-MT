#!/usr/bin/env python3
"""Validate the Diesel-MT Hugging Face to CTranslate2 deployment path."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in os.sys.path:
    os.sys.path.insert(0, str(SCRIPTS_DIR))

from build_micro_m2m100_checkpoint import (  # noqa: E402
    DEFAULT_REPORT,
    atomic_write_text,
    file_records,
    model_dimensions,
    package_versions,
    portable_path,
    read_object,
    update_consolidated_report,
)
from freeze_tokenizer_artifact import sha256_file, verify_artifact_files  # noqa: E402
from tokenizer_utils import (  # noqa: E402
    PROJECT_LANGUAGES,
    REQUIRED_SPECIAL_IDS,
    atomic_write_json,
    build_language_mapping,
    reload_tokenizer,
    verify_tokenizer,
)


SCHEMA_VERSION = 1
CONVERSION_MANIFEST = "conversion_manifest.json"
DEPLOYMENT_MANIFEST = "deployment_manifest.json"
DEFAULT_TOKENIZER = Path("artifacts/tokenizers/mvp-tokenizer-v0")
DEFAULT_HF_CHECKPOINT = Path("artifacts/ctranslate2/runtime/hf-micro-checkpoint")
DEFAULT_FLOAT32 = Path("artifacts/ctranslate2/runtime/ct2-float32")
DEFAULT_INT8 = Path("artifacts/ctranslate2/runtime/ct2-int8")
DEFAULT_PACKAGE = Path("artifacts/ctranslate2/runtime/offline-package-int8")
DEFAULT_FAILURE_LOG = Path("artifacts/ctranslate2/runtime/logs/last-failure.json")
EXPECTED_TOKENIZER_MANIFEST_SHA256 = (
    "eb79ae22f523f1d9c9fcf75b80f2b322e3c2882a8fddb7545b5933dd4053fa7f"
)
REQUIRED_CT2_FILES = frozenset({"config.json", "model.bin", "shared_vocabulary.json"})
SMOKE_CASES = (
    ("eng_Latn", "A local model validates the CPU translation interface.", "zho_Hans"),
    ("zho_Hans", "本地模型验证离线推理接口。", "zho_Hant"),
    ("zho_Hant", "本機模型驗證離線推理介面。", "jpn_Jpan"),
    ("jpn_Jpan", "ローカルモデルでオフライン推論を確認します。", "kor_Hang"),
    ("kor_Hang", "로컬 모델로 오프라인 추론을 확인합니다.", "eng_Latn"),
)


class DeploymentValidationError(RuntimeError):
    """Raised when a CTranslate2 deployment invariant fails."""


def validate_manifest_relative_path(relative: object) -> str:
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise DeploymentValidationError(f"invalid manifest path: {relative!r}")
    path = PurePosixPath(relative)
    if (
        path.is_absolute()
        or path.as_posix() != relative
        or any(part in ("", ".", "..") for part in path.parts)
        or Path(relative).is_absolute()
    ):
        raise DeploymentValidationError(f"invalid manifest path: {relative!r}")
    return relative


def enable_offline_environment() -> dict[str, str]:
    values = {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
    }
    os.environ.update(values)
    return values


def verify_payload_manifest(
    root: Path,
    manifest_name: str,
    *,
    required_files: frozenset[str] = frozenset(),
) -> tuple[dict, str]:
    manifest_path = root / manifest_name
    manifest = read_object(manifest_path)
    records = manifest.get("files")
    if manifest.get("schema_version") != SCHEMA_VERSION or not isinstance(records, list):
        raise DeploymentValidationError(f"unsupported manifest schema: {manifest_path}")
    if manifest.get("status") != "complete":
        raise DeploymentValidationError(f"manifest is not complete: {manifest_path}")
    expected: set[str] = set()
    for record in records:
        if not isinstance(record, Mapping):
            raise DeploymentValidationError(f"non-object file record in {manifest_path}")
        relative = validate_manifest_relative_path(record.get("path"))
        if relative in expected:
            raise DeploymentValidationError(f"invalid manifest path: {relative!r}")
        expected.add(relative)
        path = root / relative
        if path.is_symlink():
            raise DeploymentValidationError(f"manifest file cannot be a symlink: {relative}")
        if not path.is_file():
            raise DeploymentValidationError(f"manifest file is missing: {path}")
        if path.stat().st_size != record.get("bytes"):
            raise DeploymentValidationError(f"manifest byte count mismatch: {relative}")
        if sha256_file(path) != record.get("sha256"):
            raise DeploymentValidationError(f"manifest SHA-256 mismatch: {relative}")
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.relative_to(root).as_posix() != manifest_name
    }
    if actual != expected:
        raise DeploymentValidationError(
            f"manifest file set mismatch in {root}: "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )
    if not required_files.issubset(expected):
        raise DeploymentValidationError(
            f"required files are missing from {root}: {sorted(required_files - expected)}"
        )
    return manifest, sha256_file(manifest_path)


def publish_directory(staging: Path, output: Path, *, overwrite: bool) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    backup: Path | None = None
    if output.exists():
        if not overwrite:
            raise DeploymentValidationError(
                f"output directory already exists (use --overwrite): {output}"
            )
        backup = Path(
            tempfile.mkdtemp(prefix=f".{output.name}.backup.", dir=output.parent)
        )
        backup.rmdir()
        output.replace(backup)
    try:
        staging.replace(output)
    except BaseException:
        if backup is not None and backup.exists() and not output.exists():
            backup.replace(output)
        raise
    else:
        if backup is not None:
            shutil.rmtree(backup)


def verify_hf_checkpoint(checkpoint: Path) -> tuple[dict, str]:
    manifest, manifest_sha256 = verify_payload_manifest(
        checkpoint, "checkpoint_manifest.json"
    )
    state_sha256 = manifest.get("state_dict_sha256")
    if (
        not isinstance(state_sha256, str)
        or len(state_sha256) != 64
        or any(character not in "0123456789abcdef" for character in state_sha256)
    ):
        raise DeploymentValidationError("HF checkpoint has no valid state dict SHA-256")
    return manifest, manifest_sha256


def convert_one(
    checkpoint: Path,
    output: Path,
    *,
    quantization: str,
    requested_compute_type: str,
    checkpoint_manifest_sha256: str,
    overwrite: bool,
) -> dict:
    import ctranslate2
    from ctranslate2.converters import TransformersConverter

    output.parent.mkdir(parents=True, exist_ok=True)
    work_root = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.convert.", dir=output.parent)
    )
    staging = work_root / "model"
    started = time.perf_counter()
    try:
        converter = TransformersConverter(
            str(checkpoint),
            revision=None,
            low_cpu_mem_usage=False,
            trust_remote_code=False,
        )
        converter.convert(str(staging), quantization=quantization, force=False)
        payload = file_records(staging)
        paths = {record["path"] for record in payload}
        if not REQUIRED_CT2_FILES.issubset(paths):
            raise DeploymentValidationError(
                f"CTranslate2 {quantization} output is incomplete: "
                f"{sorted(REQUIRED_CT2_FILES - paths)}"
            )
        translator = ctranslate2.Translator(
            str(staging), device="cpu", compute_type=requested_compute_type
        )
        runtime_compute_type = translator.compute_type
        del translator
        gc.collect()
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "status": "complete",
            "format": "CTranslate2",
            "quantization_requested": quantization,
            "cpu_compute_type_requested": requested_compute_type,
            "cpu_compute_type_actual": runtime_compute_type,
            "input_checkpoint_manifest_sha256": checkpoint_manifest_sha256,
            "files": payload,
            "warning": "Random weights for deployment-interface validation only.",
        }
        atomic_write_json(staging / CONVERSION_MANIFEST, manifest)
        conversion_manifest_sha256 = sha256_file(staging / CONVERSION_MANIFEST)
        publish_directory(staging, output, overwrite=overwrite)
        return {
            "output": portable_path(output),
            "quantization_requested": quantization,
            "cpu_compute_type_requested": requested_compute_type,
            "cpu_compute_type_actual": runtime_compute_type,
            "elapsed_seconds": round(time.perf_counter() - started, 6),
            "conversion_manifest_sha256": conversion_manifest_sha256,
            "files": payload,
            "equivalent_cli": (
                ".\\.conda\\Scripts\\ct2-transformers-converter.exe "
                f"--model {portable_path(checkpoint)} --output_dir {portable_path(output)} "
                f"--quantization {quantization} --force"
            ),
        }
    finally:
        if work_root.exists():
            shutil.rmtree(work_root)


def convert_models(
    checkpoint: Path,
    float32_dir: Path,
    int8_dir: Path,
    *,
    overwrite: bool,
) -> dict:
    import ctranslate2

    checkpoint = checkpoint.resolve()
    float32_dir = float32_dir.resolve()
    int8_dir = int8_dir.resolve()
    for output in (float32_dir, int8_dir):
        if output.exists() and not overwrite:
            raise DeploymentValidationError(
                f"output directory already exists (use --overwrite): {output}"
            )
    checkpoint_manifest, checkpoint_manifest_sha256 = verify_hf_checkpoint(checkpoint)
    environment = enable_offline_environment()
    models = {
        "float32": convert_one(
            checkpoint,
            float32_dir,
            quantization="float32",
            requested_compute_type="float32",
            checkpoint_manifest_sha256=checkpoint_manifest_sha256,
            overwrite=overwrite,
        ),
        "int8": convert_one(
            checkpoint,
            int8_dir,
            quantization="int8",
            requested_compute_type="int8",
            checkpoint_manifest_sha256=checkpoint_manifest_sha256,
            overwrite=overwrite,
        ),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "passed",
        "phase": "TD-02",
        "input_checkpoint": portable_path(checkpoint),
        "input_checkpoint_manifest_sha256": checkpoint_manifest_sha256,
        "input_checkpoint_state_dict_sha256": checkpoint_manifest.get(
            "state_dict_sha256"
        ),
        "environment": environment,
        "trust_remote_code": False,
        "revision": None,
        "supported_cpu_compute_types": sorted(
            ctranslate2.get_supported_compute_types("cpu")
        ),
        "models": models,
        "versions": package_versions(),
    }


def ordered_vocabulary(tokenizer: object) -> list[str]:
    vocabulary = tokenizer.get_vocab()
    size = len(tokenizer)
    if len(vocabulary) != size:
        raise DeploymentValidationError(
            f"tokenizer get_vocab size differs from len(tokenizer): {len(vocabulary)} != {size}"
        )
    tokens: list[str | None] = [None] * size
    for token, token_id in vocabulary.items():
        if not isinstance(token_id, int) or token_id < 0 or token_id >= size:
            raise DeploymentValidationError(f"token ID is out of range: {token!r}={token_id}")
        if tokens[token_id] is not None:
            raise DeploymentValidationError(
                f"duplicate token ID {token_id}: {tokens[token_id]!r}, {token!r}"
            )
        tokens[token_id] = token
    if any(token is None for token in tokens):
        holes = [index for index, token in enumerate(tokens) if token is None]
        raise DeploymentValidationError(f"tokenizer ID space has holes: {holes[:10]}")
    result = [str(token) for token in tokens]
    if len(set(result)) != size:
        raise DeploymentValidationError("tokenizer vocabulary contains duplicate token strings")
    return result


def token_sequence_sha256(tokens: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for token in tokens:
        payload = token.encode("utf-8")
        digest.update(len(payload).to_bytes(8, "little"))
        digest.update(payload)
    return digest.hexdigest()


def require_same_vocabulary(label: str, expected: Sequence[str], actual: object) -> list[str]:
    if not isinstance(actual, list) or not all(isinstance(token, str) for token in actual):
        raise DeploymentValidationError(f"{label} vocabulary is not a string list")
    if len(actual) != len(expected):
        raise DeploymentValidationError(
            f"{label} vocabulary length mismatch: {len(actual)} != {len(expected)}"
        )
    for token_id, (expected_token, actual_token) in enumerate(zip(expected, actual)):
        if actual_token != expected_token:
            raise DeploymentValidationError(
                f"{label} vocabulary mismatch at ID {token_id}: "
                f"{actual_token!r} != {expected_token!r}"
            )
    if len(set(actual)) != len(actual):
        raise DeploymentValidationError(f"{label} vocabulary contains duplicate tokens")
    return actual


def validate_ct2_config(path: Path, tokenizer: object) -> dict:
    config = read_object(path)
    expected = {
        "bos_token": tokenizer.bos_token,
        "eos_token": tokenizer.eos_token,
        "unk_token": tokenizer.unk_token,
        "decoder_start_token": tokenizer.eos_token,
        "add_source_bos": False,
        "add_source_eos": False,
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise DeploymentValidationError(
                f"CT2 config {path} has {key}={config.get(key)!r}, expected {value!r}"
            )
    return expected


def validate_vocabulary_integrity(
    tokenizer_dir: Path,
    checkpoint: Path,
    float32_dir: Path,
    int8_dir: Path,
) -> dict:
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    tokenizer_dir = tokenizer_dir.resolve()
    checkpoint = checkpoint.resolve()
    float32_dir = float32_dir.resolve()
    int8_dir = int8_dir.resolve()
    _, tokenizer_manifest_sha256 = verify_artifact_files(tokenizer_dir)
    if tokenizer_manifest_sha256 != EXPECTED_TOKENIZER_MANIFEST_SHA256:
        raise DeploymentValidationError("frozen tokenizer manifest root changed")
    tokenizer = reload_tokenizer(tokenizer_dir)
    verify_tokenizer(tokenizer, expected_vocab_size=49_152)
    expected_tokens = ordered_vocabulary(tokenizer)
    sequence_sha256 = token_sequence_sha256(expected_tokens)

    verify_hf_checkpoint(checkpoint)
    checkpoint_tokenizer = AutoTokenizer.from_pretrained(
        checkpoint, local_files_only=True
    )
    require_same_vocabulary(
        "HF checkpoint tokenizer", expected_tokens, ordered_vocabulary(checkpoint_tokenizer)
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint, local_files_only=True)
    dimensions = model_dimensions(model)
    del model
    gc.collect()

    conversions: dict[str, dict] = {}
    for label, directory in (("float32", float32_dir), ("int8", int8_dir)):
        manifest, manifest_sha256 = verify_payload_manifest(
            directory,
            CONVERSION_MANIFEST,
            required_files=REQUIRED_CT2_FILES,
        )
        vocabulary_path = directory / "shared_vocabulary.json"
        vocabulary = require_same_vocabulary(
            f"CT2 {label}",
            expected_tokens,
            json.loads(vocabulary_path.read_text(encoding="utf-8")),
        )
        config_semantics = validate_ct2_config(directory / "config.json", tokenizer)
        conversions[label] = {
            "path": portable_path(directory),
            "conversion_manifest_sha256": manifest_sha256,
            "quantization_requested": manifest["quantization_requested"],
            "cpu_compute_type_actual": manifest["cpu_compute_type_actual"],
            "vocabulary_file_sha256": sha256_file(vocabulary_path),
            "vocabulary_sequence_sha256": token_sequence_sha256(vocabulary),
            "config_semantics": config_semantics,
        }

    language_ids = build_language_mapping(tokenizer)
    special_ids = {
        token: tokenizer.convert_tokens_to_ids(token) for token in REQUIRED_SPECIAL_IDS
    }
    for token, token_id in {**special_ids, **language_ids}.items():
        if expected_tokens[token_id] != token or token_id == tokenizer.unk_token_id and token != "<unk>":
            raise DeploymentValidationError(
                f"special token ID contract failed: {token!r}={token_id}"
            )
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "passed",
        "phase": "TD-03",
        "vocab_size": len(expected_tokens),
        "vocabulary_sequence_sha256": sequence_sha256,
        "tokenizer_manifest_sha256": tokenizer_manifest_sha256,
        "hf_dimensions": dimensions,
        "hf_tokenizer_exact_match": True,
        "special_token_ids": special_ids,
        "language_token_ids": language_ids,
        "conversions": conversions,
    }


def run_cpu_inference_smoke(
    tokenizer_dir: Path,
    float32_dir: Path,
    int8_dir: Path,
) -> dict:
    import ctranslate2

    tokenizer = reload_tokenizer(tokenizer_dir.resolve())
    verify_tokenizer(tokenizer, expected_vocab_size=49_152)
    models = (
        ("float32", float32_dir.resolve(), "float32"),
        ("int8", int8_dir.resolve(), "int8"),
    )
    results: dict[str, dict] = {}
    original_language = tokenizer.src_lang
    try:
        for label, directory, compute_type in models:
            verify_payload_manifest(
                directory,
                CONVERSION_MANIFEST,
                required_files=REQUIRED_CT2_FILES,
            )
            translator = ctranslate2.Translator(
                str(directory), device="cpu", compute_type=compute_type
            )
            cases = []
            model_started = time.perf_counter()
            for source_language, text, target_language in SMOKE_CASES:
                tokenizer.src_lang = source_language
                source_ids = tokenizer(text, add_special_tokens=True)["input_ids"]
                source_tokens = tokenizer.convert_ids_to_tokens(source_ids)
                if source_tokens[0] != source_language or source_tokens[-1] != "</s>":
                    raise DeploymentValidationError(
                        f"source token boundary failed for {source_language}: {source_tokens}"
                    )
                started = time.perf_counter()
                result = translator.translate_batch(
                    [source_tokens],
                    target_prefix=[[target_language]],
                    beam_size=1,
                    max_decoding_length=8,
                    return_end_token=True,
                )[0]
                elapsed = time.perf_counter() - started
                hypothesis = result.hypotheses[0]
                if not hypothesis or hypothesis[0] != target_language:
                    raise DeploymentValidationError(
                        f"target prefix failed for {source_language}->{target_language}: {hypothesis}"
                    )
                if tokenizer.convert_tokens_to_ids(target_language) == tokenizer.unk_token_id:
                    raise DeploymentValidationError(
                        f"target language token degraded to unknown: {target_language}"
                    )
                hypothesis_ids = tokenizer.convert_tokens_to_ids(hypothesis[1:])
                roundtrip_tokens = tokenizer.convert_ids_to_tokens(hypothesis_ids)
                if roundtrip_tokens != hypothesis[1:]:
                    raise DeploymentValidationError(
                        f"hypothesis token/ID roundtrip failed: {hypothesis[1:]}"
                    )
                decoded = tokenizer.decode(hypothesis_ids, skip_special_tokens=True)
                cases.append(
                    {
                        "source_language": source_language,
                        "target_language": target_language,
                        "source_text": text,
                        "source_token_ids": source_ids,
                        "source_tokens": source_tokens,
                        "hypothesis_tokens": hypothesis,
                        "decoded_text": decoded,
                        "elapsed_seconds": round(elapsed, 6),
                    }
                )
            results[label] = {
                "path": portable_path(directory),
                "compute_type_requested": compute_type,
                "compute_type_actual": translator.compute_type,
                "elapsed_seconds": round(time.perf_counter() - model_started, 6),
                "cases": cases,
            }
            del translator
    finally:
        tokenizer.src_lang = original_language
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "passed",
        "phase": "TD-04",
        "models": results,
        "source_languages_covered": list(PROJECT_LANGUAGES),
        "target_languages_covered": [case[2] for case in SMOKE_CASES],
        "warning": "Random hypotheses validate interfaces only; decoded text has no quality meaning.",
    }


def deployment_readme() -> str:
    return """# Diesel-MT offline CTranslate2 smoke package

This package contains an independent tokenizer directory and an INT8
CTranslate2 model directory. Run the included smoke test with the same Python
environment used to build the package:

```pwsh
python offline_smoke.py --deployment-root .
```

The model is randomly initialized. This package validates file integrity,
token IDs, CPU loading, target prefixes, and decoding only. It is not a
translation model and its output must not be evaluated for semantic quality.
"""


def run_offline_subprocess(package_dir: Path) -> dict:
    with tempfile.TemporaryDirectory(prefix="diesel-mt-offline-process-") as directory:
        clean_root = Path(directory)
        environment = os.environ.copy()
        environment.update(enable_offline_environment())
        environment["HF_HOME"] = str(clean_root / "hf-home")
        environment["HTTP_PROXY"] = "http://127.0.0.1:9"
        environment["HTTPS_PROXY"] = "http://127.0.0.1:9"
        environment["ALL_PROXY"] = "http://127.0.0.1:9"
        environment["NO_PROXY"] = ""
        environment["PYTHONUTF8"] = "1"
        environment["PYTHONIOENCODING"] = "utf-8"
        environment.pop("PYTHONPATH", None)
        command = [
            sys.executable,
            str(package_dir / "offline_smoke.py"),
            "--deployment-root",
            str(package_dir),
        ]
        started = time.perf_counter()
        result = subprocess.run(
            command,
            cwd=clean_root,
            env=environment,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=120,
        )
        elapsed = time.perf_counter() - started
        if result.returncode != 0:
            raise DeploymentValidationError(
                "offline subprocess failed:\n"
                f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
            )
        lines = [line for line in result.stdout.splitlines() if line.strip()]
        if not lines:
            raise DeploymentValidationError("offline subprocess produced no JSON output")
        try:
            payload = json.loads(lines[-1])
        except json.JSONDecodeError as error:
            raise DeploymentValidationError(
                f"offline subprocess output is not JSON: {lines[-1]!r}"
            ) from error
        if payload.get("status") != "passed":
            raise DeploymentValidationError(f"offline subprocess did not pass: {payload}")
        return {
            "returncode": result.returncode,
            "elapsed_seconds": round(elapsed, 6),
            "clean_working_directory": True,
            "environment": {
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "HF_DATASETS_OFFLINE": "1",
                "HTTP_PROXY": "http://127.0.0.1:9",
                "HTTPS_PROXY": "http://127.0.0.1:9",
                "ALL_PROXY": "http://127.0.0.1:9",
                "NO_PROXY": "",
                "PYTHONUTF8": "1",
                "PYTHONIOENCODING": "utf-8",
                "PYTHONPATH": "removed",
            },
            "result": payload,
            "stderr": result.stderr.strip(),
        }


def build_offline_package(
    tokenizer_dir: Path,
    int8_dir: Path,
    package_dir: Path,
    runner_source: Path,
    *,
    overwrite: bool,
) -> dict:
    tokenizer_dir = tokenizer_dir.resolve()
    int8_dir = int8_dir.resolve()
    package_dir = package_dir.resolve()
    runner_source = runner_source.resolve()
    if package_dir.exists() and not overwrite:
        raise DeploymentValidationError(
            f"output directory already exists (use --overwrite): {package_dir}"
        )
    _, tokenizer_manifest_sha256 = verify_artifact_files(tokenizer_dir)
    if tokenizer_manifest_sha256 != EXPECTED_TOKENIZER_MANIFEST_SHA256:
        raise DeploymentValidationError("frozen tokenizer manifest root changed")
    conversion_manifest, conversion_manifest_sha256 = verify_payload_manifest(
        int8_dir,
        CONVERSION_MANIFEST,
        required_files=REQUIRED_CT2_FILES,
    )
    if conversion_manifest.get("quantization_requested") != "int8":
        raise DeploymentValidationError("offline package input is not the int8 conversion")

    package_dir.parent.mkdir(parents=True, exist_ok=True)
    work_root = Path(
        tempfile.mkdtemp(prefix=f".{package_dir.name}.package.", dir=package_dir.parent)
    )
    staging = work_root / "deployment"
    try:
        (staging / "tokenizer").mkdir(parents=True)
        (staging / "model").mkdir(parents=True)
        for name in ("tokenizer.json", "tokenizer_config.json", "language_map.json"):
            shutil.copy2(tokenizer_dir / name, staging / "tokenizer" / name)
        for name in (
            "config.json",
            "model.bin",
            "shared_vocabulary.json",
            CONVERSION_MANIFEST,
        ):
            shutil.copy2(int8_dir / name, staging / "model" / name)
        shutil.copy2(runner_source, staging / "offline_smoke.py")
        atomic_write_text(staging / "README.md", deployment_readme())
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "status": "complete",
            "layout": {"tokenizer": "tokenizer/", "model": "model/"},
            "tokenizer_manifest_sha256": tokenizer_manifest_sha256,
            "int8_conversion_manifest_sha256": conversion_manifest_sha256,
            "files": file_records(staging),
            "warning": "Random model package for offline deployment-interface validation only.",
        }
        atomic_write_json(staging / DEPLOYMENT_MANIFEST, manifest)
        deployment_manifest_sha256 = sha256_file(staging / DEPLOYMENT_MANIFEST)
        verify_payload_manifest(
            staging,
            DEPLOYMENT_MANIFEST,
            required_files=frozenset(
                {
                    "tokenizer/tokenizer.json",
                    "tokenizer/tokenizer_config.json",
                    "model/config.json",
                    "model/model.bin",
                    "model/shared_vocabulary.json",
                    "offline_smoke.py",
                    "README.md",
                }
            ),
        )
        publish_directory(staging, package_dir, overwrite=overwrite)
    finally:
        if work_root.exists():
            shutil.rmtree(work_root)

    subprocess_record = run_offline_subprocess(package_dir)
    manifest, verified_manifest_sha256 = verify_payload_manifest(
        package_dir, DEPLOYMENT_MANIFEST
    )
    if verified_manifest_sha256 != deployment_manifest_sha256:
        raise DeploymentValidationError("deployment manifest changed after publication")
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "passed",
        "phase": "TD-05",
        "package": portable_path(package_dir),
        "deployment_manifest_sha256": deployment_manifest_sha256,
        "tokenizer_manifest_sha256": tokenizer_manifest_sha256,
        "int8_conversion_manifest_sha256": conversion_manifest_sha256,
        "layout": manifest["layout"],
        "files": manifest["files"],
        "offline_subprocess": subprocess_record,
        "versions": package_versions(),
        "reproduction_command": (
            ".\\.conda\\python.exe scripts\\validate_ctranslate2_deployment.py "
            "--phase package --overwrite"
        ),
        "known_limitations": [
            "Random weights do not provide translation quality.",
            "This smoke test is not a production latency or throughput benchmark.",
            "The package requires compatible Python, Transformers, and CTranslate2 runtimes.",
        ],
        "troubleshooting": [
            "A manifest mismatch means a package file is missing, added, or modified; rebuild the package.",
            "A model-load failure should be checked against the recorded CTranslate2 version and CPU compute types.",
            "A prefix or decode failure should be investigated with TD-03 ordered-vocabulary and special-token hashes.",
        ],
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase", choices=("convert", "vocab", "smoke", "package", "all"), default="all"
    )
    parser.add_argument("--tokenizer-dir", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--hf-checkpoint", type=Path, default=DEFAULT_HF_CHECKPOINT)
    parser.add_argument("--float32-dir", type=Path, default=DEFAULT_FLOAT32)
    parser.add_argument("--int8-dir", type=Path, default=DEFAULT_INT8)
    parser.add_argument("--package-dir", type=Path, default=DEFAULT_PACKAGE)
    parser.add_argument("--report-json", type=Path, default=DEFAULT_REPORT)
    parser.add_argument(
        "--offline-runner",
        type=Path,
        default=Path("scripts/run_offline_ctranslate2_smoke.py"),
    )
    parser.add_argument("--failure-log", type=Path, default=DEFAULT_FAILURE_LOG)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    enable_offline_environment()
    phase = args.phase
    report_updates: dict[str, Mapping] = {}
    try:
        if phase in ("convert", "all"):
            record = convert_models(
                args.hf_checkpoint, args.float32_dir, args.int8_dir, overwrite=args.overwrite
            )
            report_updates["td_02_conversion"] = record
        if phase in ("vocab", "all"):
            record = validate_vocabulary_integrity(
                args.tokenizer_dir,
                args.hf_checkpoint,
                args.float32_dir,
                args.int8_dir,
            )
            report_updates["td_03_vocab_integrity"] = record
        if phase in ("smoke", "all"):
            record = run_cpu_inference_smoke(
                args.tokenizer_dir, args.float32_dir, args.int8_dir
            )
            report_updates["td_04_cpu_inference"] = record
        if phase in ("package", "all"):
            record = build_offline_package(
                args.tokenizer_dir,
                args.int8_dir,
                args.package_dir,
                args.offline_runner,
                overwrite=args.overwrite,
            )
            report_updates["td_05_offline_package"] = record
        update_consolidated_report(args.report_json, report_updates)
    except BaseException as error:
        failure = {
            "schema_version": SCHEMA_VERSION,
            "status": "failed",
            "phase_requested": phase,
            "failed_at_utc": datetime.now(timezone.utc).isoformat(),
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
        }
        atomic_write_json(args.failure_log, failure)
        raise
    else:
        args.failure_log.unlink(missing_ok=True)
    print(json.dumps({"status": "passed", "phase": phase}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
