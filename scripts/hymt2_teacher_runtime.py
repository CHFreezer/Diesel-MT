"""Locked artifact and offline runtime helpers for the Hy-MT2 7B teacher."""

from __future__ import annotations

import contextlib
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path, PurePosixPath
from typing import Any, Iterator, Mapping

import yaml

from model_training_contract import LANGUAGE_TAGS, canonical_json_bytes


class TeacherRuntimeError(RuntimeError):
    """Raised when the locked teacher runtime contract is violated."""


def _mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TeacherRuntimeError(f"{context} must be a mapping")
    return value


def _exact_keys(
    value: Mapping[str, Any], required: set[str], context: str, optional: set[str] | None = None
) -> None:
    optional = optional or set()
    missing = sorted(required - set(value))
    unknown = sorted(set(value) - required - optional)
    if missing:
        raise TeacherRuntimeError(f"{context} missing fields: {', '.join(missing)}")
    if unknown:
        raise TeacherRuntimeError(f"{context} unknown fields: {', '.join(unknown)}")


def load_yaml_mapping(path: Path) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise TeacherRuntimeError(f"cannot load {path}: {exc}") from exc
    return dict(_mapping(value, str(path)))


def load_json_mapping(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TeacherRuntimeError(f"cannot load {path}: {exc}") from exc
    return dict(_mapping(value, str(path)))


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def file_identity(path: Path) -> dict[str, Any]:
    return {"bytes": path.stat().st_size, "sha256": sha256_file(path)}


def validate_contract(profile: Mapping[str, Any], lock: Mapping[str, Any]) -> None:
    _exact_keys(
        profile,
        {"schema_version", "identity", "runtime", "environment", "loading", "smoke", "acceptance"},
        "teacher profile",
    )
    if profile["schema_version"] != 1 or lock.get("schema_version") != 1:
        raise TeacherRuntimeError("unsupported teacher profile or lock schema")
    identity = _mapping(profile["identity"], "identity")
    selected = _mapping(lock.get("selected"), "lock.selected")
    for profile_key, lock_key in (
        ("source_model", "source_model"),
        ("selected_artifact", "repo_id"),
        ("selected_revision", "revision"),
        ("artifact_format", "artifact_format"),
        ("license", "license"),
    ):
        if identity.get(profile_key) != selected.get(lock_key):
            raise TeacherRuntimeError(f"profile {profile_key} does not match artifact lock")
    revision = str(selected.get("revision", ""))
    if len(revision) != 40 or any(char not in "0123456789abcdef" for char in revision):
        raise TeacherRuntimeError("artifact revision must be an immutable 40-character commit SHA")

    files = selected.get("files")
    if not isinstance(files, list) or not files:
        raise TeacherRuntimeError("artifact lock must contain files")
    paths: set[str] = set()
    total = 0
    for index, raw in enumerate(files):
        item = _mapping(raw, f"lock file {index}")
        _exact_keys(item, {"path", "bytes", "sha256", "role", "runtime_required"}, f"lock file {index}")
        path = str(item["path"])
        posix = PurePosixPath(path)
        if posix.is_absolute() or ".." in posix.parts or path != posix.as_posix():
            raise TeacherRuntimeError(f"unsafe artifact path: {path}")
        if path in paths:
            raise TeacherRuntimeError(f"duplicate artifact path: {path}")
        paths.add(path)
        size = item["bytes"]
        digest = str(item["sha256"])
        if not isinstance(size, int) or size < 0:
            raise TeacherRuntimeError(f"invalid artifact size for {path}")
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise TeacherRuntimeError(f"invalid SHA-256 for {path}")
        total += size
    if selected.get("file_count") != len(files) or selected.get("total_bytes") != total:
        raise TeacherRuntimeError("artifact lock totals do not match its file list")

    audit = _mapping(lock.get("remote_code_audit"), "remote_code_audit")
    if audit.get("python_files") != [] or audit.get("config_auto_map") is not None:
        raise TeacherRuntimeError("locked teacher must not contain or request remote Python code")
    if audit.get("locked_runtime_trust_remote_code") is not False:
        raise TeacherRuntimeError("locked teacher must disable trust_remote_code")
    loading = _mapping(profile["loading"], "loading")
    if loading.get("local_files_only") is not True or loading.get("trust_remote_code") is not False:
        raise TeacherRuntimeError("teacher loading must be local-only with trust_remote_code=false")

    smoke = _mapping(profile["smoke"], "smoke")
    if smoke.get("diagnostic_only") is not True:
        raise TeacherRuntimeError("TD-06 smoke profile must remain diagnostic-only")
    if tuple(smoke.get("required_tags", [])) != LANGUAGE_TAGS:
        raise TeacherRuntimeError("smoke required_tags must use the frozen five-tag order")
    probes = smoke.get("probes")
    if not isinstance(probes, list) or [item.get("target_tag") for item in probes] != list(LANGUAGE_TAGS):
        raise TeacherRuntimeError("smoke probes must cover each frozen tag exactly once")


def resolve_runtime_root(
    profile: Mapping[str, Any], repository_root: Path, environ: Mapping[str, str] | None = None
) -> Path:
    runtime = _mapping(profile["runtime"], "runtime")
    environ = os.environ if environ is None else environ
    env_name = str(runtime["override_env"])
    override = environ.get(env_name)
    if override:
        base = Path(override).expanduser()
        if not base.is_absolute():
            raise TeacherRuntimeError(f"{env_name} must be an absolute path")
        return (base / PurePosixPath(str(runtime["override_subdir"]))).resolve()
    configured = PurePosixPath(str(runtime["default_root"]))
    if configured.is_absolute() or ".." in configured.parts:
        raise TeacherRuntimeError("default teacher runtime must be repository-relative")
    if not configured.as_posix().startswith("artifacts/model-training/runtime/"):
        raise TeacherRuntimeError("default teacher runtime escapes the Git-ignored runtime root")
    return (repository_root / configured).resolve()


def runtime_paths(profile: Mapping[str, Any], root: Path) -> dict[str, Path]:
    runtime = _mapping(profile["runtime"], "runtime")
    return {
        "root": root,
        "snapshot": root / str(runtime["snapshot_subdir"]),
        "overlay": root / str(runtime["overlay_subdir"]),
        "reports": root / str(runtime["reports_subdir"]),
    }


def verify_snapshot(snapshot: Path, lock: Mapping[str, Any], *, reject_unexpected: bool = True) -> list[dict[str, Any]]:
    selected = _mapping(lock["selected"], "lock.selected")
    locked = {str(item["path"]): item for item in selected["files"]}
    if not snapshot.is_dir():
        raise TeacherRuntimeError(f"teacher snapshot does not exist: {snapshot}")
    found = {
        path.relative_to(snapshot).as_posix()
        for path in snapshot.rglob("*")
        if path.is_file() and ".cache" not in path.relative_to(snapshot).parts
    }
    missing = sorted(set(locked) - found)
    unexpected = sorted(found - set(locked))
    if missing:
        raise TeacherRuntimeError(f"teacher snapshot missing files: {', '.join(missing)}")
    if reject_unexpected and unexpected:
        raise TeacherRuntimeError(f"teacher snapshot has unexpected files: {', '.join(unexpected)}")
    verified: list[dict[str, Any]] = []
    for relative, expected in locked.items():
        path = snapshot / PurePosixPath(relative)
        actual = file_identity(path)
        if actual["bytes"] != expected["bytes"]:
            raise TeacherRuntimeError(f"size mismatch for {relative}")
        if actual["sha256"] != expected["sha256"]:
            raise TeacherRuntimeError(f"SHA-256 mismatch for {relative}")
        verified.append({"path": relative, **actual, "role": expected["role"]})
    return verified


def inspect_snapshot(snapshot: Path, lock: Mapping[str, Any]) -> dict[str, Any]:
    config = load_json_mapping(snapshot / "config.json")
    generation = load_json_mapping(snapshot / "generation_config.json")
    if config.get("auto_map") is not None:
        raise TeacherRuntimeError("snapshot config unexpectedly requests remote code")
    forbidden = [path.name for path in snapshot.iterdir() if path.suffix.lower() in {".py", ".pyc", ".pyd", ".dll", ".so"}]
    if forbidden:
        raise TeacherRuntimeError(f"snapshot contains executable code: {', '.join(sorted(forbidden))}")
    audit = _mapping(lock["remote_code_audit"], "remote_code_audit")
    if config.get("model_type") != audit.get("model_type"):
        raise TeacherRuntimeError("model_type does not match remote-code audit")
    if config.get("architectures") != [audit.get("architecture")]:
        raise TeacherRuntimeError("model architecture does not match remote-code audit")
    return {
        "model_type": config.get("model_type"),
        "architecture": config.get("architectures", [None])[0],
        "auto_map": config.get("auto_map"),
        "quantization_method": config.get("quantization_config", {}).get("quant_method"),
        "generation_config": generation,
        "chat_template_sha256": sha256_file(snapshot / "chat_template.jinja"),
        "executable_files": forbidden,
    }


def download_locked_snapshot(snapshot: Path, lock: Mapping[str, Any], *, max_workers: int = 2) -> Path:
    from huggingface_hub import snapshot_download

    selected = _mapping(lock["selected"], "lock.selected")
    required_bytes = int(selected["total_bytes"])
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(snapshot.parent).free < required_bytes + 2 * 1024**3:
        raise TeacherRuntimeError("insufficient free disk for locked teacher snapshot plus safety margin")
    downloaded = snapshot_download(
        repo_id=str(selected["repo_id"]),
        revision=str(selected["revision"]),
        local_dir=snapshot,
        allow_patterns=[str(item["path"]) for item in selected["files"]],
        max_workers=max_workers,
    )
    return Path(downloaded).resolve()


def create_overlay(
    overlay: Path, base_python: Path, requirements: Path, *, env: Mapping[str, str] | None = None
) -> None:
    overlay.parent.mkdir(parents=True, exist_ok=True)
    if not (overlay / "pyvenv.cfg").exists():
        subprocess.run(
            [str(base_python), "-m", "venv", "--system-site-packages", str(overlay)],
            check=True,
            env=dict(env or os.environ),
        )
    python = overlay / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    subprocess.run(
        [str(python), "-m", "pip", "install", "-r", str(requirements)],
        check=True,
        env=dict(env or os.environ),
    )


def overlay_python(overlay: Path) -> Path:
    return overlay / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(canonical_json_bytes(value))
    os.replace(temporary, path)


@contextlib.contextmanager
def blocked_network() -> Iterator[list[str]]:
    attempts: list[str] = []
    original_connect = socket.socket.connect
    original_connect_ex = socket.socket.connect_ex
    original_create_connection = socket.create_connection

    def deny_connect(_socket: socket.socket, address: Any) -> None:
        attempts.append(repr(address))
        raise TeacherRuntimeError(f"network access blocked during offline teacher run: {address!r}")

    def deny_connect_ex(_socket: socket.socket, address: Any) -> int:
        deny_connect(_socket, address)
        return 1

    def deny_create_connection(address: Any, *args: Any, **kwargs: Any) -> socket.socket:
        attempts.append(repr(address))
        raise TeacherRuntimeError(f"network access blocked during offline teacher run: {address!r}")

    socket.socket.connect = deny_connect  # type: ignore[method-assign]
    socket.socket.connect_ex = deny_connect_ex  # type: ignore[method-assign]
    socket.create_connection = deny_create_connection
    try:
        yield attempts
    finally:
        socket.socket.connect = original_connect  # type: ignore[method-assign]
        socket.socket.connect_ex = original_connect_ex  # type: ignore[method-assign]
        socket.create_connection = original_create_connection


def package_versions(names: Iterator[str] | list[str]) -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def validate_environment(profile: Mapping[str, Any]) -> dict[str, Any]:
    environment = _mapping(profile["environment"], "environment")
    expected = {**dict(environment["inherited_packages"]), **dict(environment["overlay_packages"])}
    actual = package_versions(list(expected))
    mismatches = {
        name: {"expected": str(version), "actual": actual.get(name)}
        for name, version in expected.items()
        if actual.get(name) != str(version)
    }
    if platform.python_version() != str(environment["python_version"]):
        mismatches["python"] = {
            "expected": str(environment["python_version"]),
            "actual": platform.python_version(),
        }
    if sys.flags.utf8_mode != 1:
        mismatches["python_utf8_mode"] = {
            "expected": "1 (launch with PYTHONUTF8=1)",
            "actual": str(sys.flags.utf8_mode),
        }
    prefix = Path(sys.prefix).resolve()
    local_packages: dict[str, str] = {}
    for distribution in importlib.metadata.distributions():
        location = Path(distribution.locate_file("")).resolve()
        if location == prefix or prefix in location.parents:
            name = str(distribution.metadata["Name"])
            local_packages[name.lower().replace("_", "-")] = distribution.version
    expected_local = {
        str(name).lower().replace("_", "-"): str(version)
        for name, version in environment["overlay_packages"].items()
    }
    unexpected_local = sorted(set(local_packages) - set(expected_local) - {"pip", "setuptools"})
    missing_local = sorted(set(expected_local) - set(local_packages))
    if unexpected_local or missing_local:
        mismatches["overlay_local_packages"] = {
            "unexpected": unexpected_local,
            "missing": missing_local,
        }
    if mismatches:
        raise TeacherRuntimeError(f"teacher environment mismatch: {json.dumps(mismatches, sort_keys=True)}")
    return {
        "python": platform.python_version(),
        "python_utf8_mode": sys.flags.utf8_mode,
        "executable": str(Path(sys.executable).resolve()),
        "platform": platform.platform(),
        "packages": actual,
        "overlay_local_packages": dict(sorted(local_packages.items())),
    }


def _set_profile_environment(profile: Mapping[str, Any]) -> None:
    variables = _mapping(_mapping(profile["environment"], "environment")["variables"], "variables")
    for name, value in variables.items():
        os.environ[str(name)] = str(value)


def _device_map(loading: Mapping[str, Any]) -> str | dict[str, int | str]:
    strategy = str(loading["device_map"])
    if strategy != "hybrid":
        return strategy
    offloaded = {int(index) for index in loading.get("cpu_offload_layers", [])}
    if not offloaded or min(offloaded) < 0 or max(offloaded) >= 32:
        raise TeacherRuntimeError("hybrid device map requires valid cpu_offload_layers")
    mapping: dict[str, int | str] = {"model.embed_tokens": 0}
    for index in range(32):
        mapping[f"model.layers.{index}"] = "cpu" if index in offloaded else 0
    mapping["model.norm"] = 0
    mapping["lm_head"] = 0
    return mapping


def run_offline_smoke(snapshot: Path, profile: Mapping[str, Any]) -> dict[str, Any]:
    _set_profile_environment(profile)
    import psutil
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    if not torch.cuda.is_available():
        raise TeacherRuntimeError("CUDA is required by the selected FP8 teacher profile")
    loading = _mapping(profile["loading"], "loading")
    smoke = _mapping(profile["smoke"], "smoke")
    process = psutil.Process()
    rss_before = process.memory_info().rss
    gpu_free_before, gpu_total = torch.cuda.mem_get_info(0)
    torch.cuda.reset_peak_memory_stats(0)
    load_started = time.perf_counter()
    quantization_config = None
    if loading.get("quantization") == "bitsandbytes-llm-int8":
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=float(loading["llm_int8_threshold"]),
            llm_int8_enable_fp32_cpu_offload=bool(loading["llm_int8_enable_fp32_cpu_offload"]),
        )
    model_kwargs: dict[str, Any] = {
        "dtype": torch.bfloat16,
        "device_map": _device_map(loading),
        "local_files_only": True,
        "trust_remote_code": False,
    }
    if "gpu_max_memory" in loading or "cpu_max_memory" in loading:
        model_kwargs["max_memory"] = {
            0: str(loading["gpu_max_memory"]),
            "cpu": str(loading["cpu_max_memory"]),
        }
    if "low_cpu_mem_usage" in loading:
        model_kwargs["low_cpu_mem_usage"] = bool(loading["low_cpu_mem_usage"])
    if "attention_implementation" in loading:
        model_kwargs["attn_implementation"] = str(loading["attention_implementation"])
    if quantization_config is not None:
        model_kwargs["quantization_config"] = quantization_config
    with blocked_network() as network_attempts:
        tokenizer = AutoTokenizer.from_pretrained(
            snapshot,
            local_files_only=True,
            trust_remote_code=False,
        )
        model = AutoModelForCausalLM.from_pretrained(snapshot, **model_kwargs)
        model.eval()
        torch.cuda.synchronize()
        load_seconds = time.perf_counter() - load_started
        pre_forward_footprint = int(model.get_memory_footprint())
        had_decompress_hook = hasattr(model, "ct_decompress_hook")
        device_map = getattr(model, "hf_device_map", {})
        non_cuda_modules = {
            name: str(device)
            for name, device in device_map.items()
            if str(device) not in {"0", "cuda", "cuda:0"}
        }
        acceptance = _mapping(profile["acceptance"], "acceptance")
        if acceptance.get("require_all_parameters_on_cuda") and non_cuda_modules:
            raise TeacherRuntimeError(f"selected FP8 model was offloaded from CUDA: {non_cuda_modules}")
        required_offload = {f"model.layers.{index}" for index in acceptance.get("required_cpu_offload_layers", [])}
        actual_offload = {name for name, device in device_map.items() if str(device) == "cpu"}
        if required_offload and actual_offload != required_offload:
            raise TeacherRuntimeError(
                f"teacher CPU offload mismatch: expected {sorted(required_offload)}, got {sorted(actual_offload)}"
            )
        quantized_modules = sum(
            1 for module in model.modules() if module.__class__.__name__ == "Linear8bitLt"
        )
        if acceptance.get("require_bitsandbytes_8bit") and (
            not getattr(model, "is_loaded_in_8bit", False) or quantized_modules == 0
        ):
            raise TeacherRuntimeError("teacher did not load through bitsandbytes LLM.int8")
        input_device = model.get_input_embeddings().weight.device
        probes: list[dict[str, Any]] = []
        all_outputs: dict[str, list[str]] = {}
        for probe in smoke["probes"]:
            tag = str(probe["target_tag"])
            messages = [{"role": "user", "content": str(probe["prompt"])}]
            encoded = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
            encoded = {name: tensor.to(input_device) for name, tensor in encoded.items()}
            input_tokens = int(encoded["input_ids"].shape[-1])
            all_outputs[tag] = []
            for repeat in range(int(smoke["repeats"])):
                torch.manual_seed(int(smoke["seed"]))
                torch.cuda.manual_seed_all(int(smoke["seed"]))
                started = time.perf_counter()
                with torch.inference_mode():
                    output = model.generate(
                        **encoded,
                        max_new_tokens=int(smoke["max_new_tokens"]),
                        do_sample=bool(smoke["do_sample"]),
                    )
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - started
                generated = output[0, input_tokens:]
                text = tokenizer.decode(generated, skip_special_tokens=True).strip()
                new_tokens = int(generated.shape[-1])
                if not text:
                    raise TeacherRuntimeError(f"empty teacher output for {tag}")
                all_outputs[tag].append(text)
                probes.append(
                    {
                        "target_tag": tag,
                        "repeat": repeat,
                        "input_tokens": input_tokens,
                        "new_tokens": new_tokens,
                        "latency_seconds": elapsed,
                        "tokens_per_second": new_tokens / elapsed,
                        "output": text,
                        "output_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                    }
                )
        unstable = {tag: values for tag, values in all_outputs.items() if len(set(values)) != 1}
        if unstable:
            raise TeacherRuntimeError(f"teacher smoke output is not repeat-stable: {sorted(unstable)}")
        reference_mismatches = {
            str(probe["target_tag"]): {
                "expected": str(probe["expected_output"]),
                "actual": all_outputs[str(probe["target_tag"])][0],
            }
            for probe in smoke["probes"]
            if all_outputs[str(probe["target_tag"])][0] != str(probe["expected_output"])
        }
        if reference_mismatches:
            raise TeacherRuntimeError(
                f"teacher smoke output changed from the BF16-equivalent reference: {reference_mismatches}"
            )
        capacity = _mapping(smoke["capacity_probe"], "capacity_probe")
        capacity_messages = [{"role": "user", "content": str(capacity["prompt"])}]
        capacity_inputs = tokenizer.apply_chat_template(
            capacity_messages,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        capacity_inputs = {name: tensor.to(input_device) for name, tensor in capacity_inputs.items()}
        capacity_input_tokens = int(capacity_inputs["input_ids"].shape[-1])
        capacity_started = time.perf_counter()
        with torch.inference_mode():
            capacity_output = model.generate(
                **capacity_inputs,
                max_new_tokens=int(capacity["max_new_tokens"]),
                do_sample=False,
            )
        torch.cuda.synchronize()
        capacity_elapsed = time.perf_counter() - capacity_started
        capacity_generated = capacity_output[0, capacity_input_tokens:]
        capacity_text = tokenizer.decode(capacity_generated, skip_special_tokens=True).strip()
        if not capacity_text:
            raise TeacherRuntimeError("empty output for M0 maximum-length capacity probe")
        capacity_result = {
            "scope": capacity["scope"],
            "source_characters": int(capacity["source_characters"]),
            "target_tag": capacity["target_tag"],
            "input_tokens": capacity_input_tokens,
            "new_tokens": int(capacity_generated.shape[-1]),
            "latency_seconds": capacity_elapsed,
            "tokens_per_second": int(capacity_generated.shape[-1]) / capacity_elapsed,
            "output": capacity_text,
            "output_sha256": hashlib.sha256(capacity_text.encode("utf-8")).hexdigest(),
        }
        post_forward_footprint = int(model.get_memory_footprint())
        checkpoint_decompressed = had_decompress_hook and not hasattr(model, "ct_decompress_hook")

    gpu_free_after, _ = torch.cuda.mem_get_info(0)
    minimum_gpu_free = int(_mapping(profile["acceptance"], "acceptance").get("minimum_gpu_free_bytes", 0))
    if gpu_free_after < minimum_gpu_free:
        raise TeacherRuntimeError(
            f"teacher GPU safety margin too small: {gpu_free_after} < {minimum_gpu_free} bytes"
        )
    return {
        "status": "pass",
        "profile_id": smoke["profile_id"],
        "diagnostic_only": True,
        "offline": {
            "environment_flags": dict(_mapping(profile["environment"], "environment")["variables"]),
            "socket_attempts": network_attempts,
            "socket_attempt_count": len(network_attempts),
            "local_files_only": True,
            "trust_remote_code": False,
        },
        "hardware": {
            "gpu": torch.cuda.get_device_name(0),
            "compute_capability": list(torch.cuda.get_device_capability(0)),
            "cuda_runtime": torch.version.cuda,
            "gpu_total_bytes": gpu_total,
        },
        "loading": {
            "seconds": load_seconds,
            "device_map": {name: str(device) for name, device in device_map.items()},
            "non_cuda_modules": non_cuda_modules,
            "quantization": loading.get("quantization"),
            "is_loaded_in_8bit": bool(getattr(model, "is_loaded_in_8bit", False)),
            "linear_8bit_module_count": quantized_modules,
            "pre_forward_model_memory_footprint_bytes": pre_forward_footprint,
            "post_forward_model_memory_footprint_bytes": post_forward_footprint,
            "had_compressed_tensors_decompress_hook": had_decompress_hook,
            "checkpoint_decompressed_on_first_forward": checkpoint_decompressed,
        },
        "resources": {
            "process_rss_before_bytes": rss_before,
            "process_rss_after_bytes": process.memory_info().rss,
            "torch_peak_allocated_bytes": torch.cuda.max_memory_allocated(0),
            "torch_peak_reserved_bytes": torch.cuda.max_memory_reserved(0),
            "gpu_used_before_bytes": gpu_total - gpu_free_before,
            "gpu_used_after_bytes": gpu_total - gpu_free_after,
        },
        "probes": probes,
        "stable_outputs": {tag: values[0] for tag, values in all_outputs.items()},
        "reference_outputs_match": True,
        "capacity_probe": capacity_result,
    }
