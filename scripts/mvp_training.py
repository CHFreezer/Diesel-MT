"""TD-10 deterministic sampling, resource validation, and training loop."""

from __future__ import annotations

import contextlib
import ctypes
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import shutil
import struct
import subprocess
import sys
import tempfile
import threading
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

from artifact_io import atomic_write_json
from freeze_tokenizer_artifact import sha256_file
from model_training_contract import (
    config_sha256,
    directed_routes,
    load_student_config,
)
from mvp_checkpoint import CHECKPOINT_MANIFEST, validate_checkpoint
from mvp_student import (
    DirectionAwareCollator,
    EncodedSample,
    EncodingPolicy,
    StudentContractError,
    build_student,
    encode_language_text,
    encode_parallel_sample,
    encoded_sample_from_sequences,
    load_frozen_tokenizer,
    model_inputs,
)
from tokenizer_utils import build_language_mapping, reload_tokenizer


TRAINING_SCHEMA_VERSION = 1
MIB = 1024 * 1024
ROUTE_ORDER = tuple(f"{source}->{target}" for source, target in directed_routes())
SHA256_LENGTH = 64
TEXT_CACHE_SCHEMA_VERSION = 1
TEXT_CACHE_MANIFEST = "text-cache-manifest.json"
TEXT_CACHE_PAYLOAD = "text-cache.npz"


class TrainingContractError(RuntimeError):
    """Raised when TD-10 cannot preserve its training or resource contract."""


def _expect_keys(value: Mapping[str, Any], expected: set[str], context: str) -> None:
    missing = sorted(expected - set(value))
    unknown = sorted(set(value) - expected)
    if missing:
        raise TrainingContractError(f"{context} missing fields: {', '.join(missing)}")
    if unknown:
        raise TrainingContractError(f"{context} unknown fields: {', '.join(unknown)}")


def _mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TrainingContractError(f"{context} must be a mapping")
    return value


def _positive_int(value: Any, context: str, *, allow_zero: bool = False) -> int:
    minimum = 0 if allow_zero else 1
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise TrainingContractError(f"{context} must be an integer >= {minimum}")
    return value


def _finite_number(
    value: Any,
    context: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TrainingContractError(f"{context} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise TrainingContractError(f"{context} must be finite")
    if minimum is not None and result < minimum:
        raise TrainingContractError(f"{context} must be >= {minimum}")
    if maximum is not None and result > maximum:
        raise TrainingContractError(f"{context} must be <= {maximum}")
    return result


def _sha256(value: Any, context: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != SHA256_LENGTH
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise TrainingContractError(f"{context} must be a lowercase SHA-256")
    return value


def _repo_path(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise TrainingContractError(f"{context} must be a repository-relative POSIX path")
    path = PurePosixPath(value)
    if path.is_absolute() or "." in path.parts or ".." in path.parts:
        raise TrainingContractError(f"{context} escapes the repository boundary")
    if path.as_posix() != value:
        raise TrainingContractError(f"{context} is not normalized")
    return value


def validate_training_config(config: Mapping[str, Any]) -> dict[str, Any]:
    expected_top_level = {
        "schema_version",
        "identity",
        "data",
        "resource_profile",
        "optimization",
    }
    for optional in ("input_pipeline", "gpu_optimization", "logging"):
        if optional in config:
            expected_top_level.add(optional)
    _expect_keys(
        config,
        expected_top_level,
        "training config",
    )
    if config["schema_version"] != TRAINING_SCHEMA_VERSION:
        raise TrainingContractError("unsupported training schema_version")

    identity = _mapping(config["identity"], "training.identity")
    _expect_keys(
        identity,
        {
            "name",
            "mode",
            "student_config",
            "student_config_file_sha256",
            "student_config_canonical_sha256",
            "tokenizer_manifest_sha256",
            "seed",
        },
        "training.identity",
    )
    if not isinstance(identity["name"], str) or not identity["name"]:
        raise TrainingContractError("training identity name must be non-empty")
    if identity["mode"] not in {"td10_smoke", "m1", "td14_benchmark", "m2"}:
        raise TrainingContractError("training mode is unsupported")
    _repo_path(identity["student_config"], "training.identity.student_config")
    for field in (
        "student_config_file_sha256",
        "student_config_canonical_sha256",
        "tokenizer_manifest_sha256",
    ):
        _sha256(identity[field], f"training.identity.{field}")
    _positive_int(identity["seed"], "training.identity.seed", allow_zero=True)

    data = _mapping(config["data"], "training.data")
    _expect_keys(
        data,
        {
            "train_path",
            "train_sha256",
            "dev_path",
            "dev_sha256",
            "manifest_path",
            "manifest_sha256",
            "train_max_records_per_route",
            "dev_max_records_per_route",
            "route_weights",
        },
        "training.data",
    )
    for field in ("train_path", "dev_path", "manifest_path"):
        _repo_path(data[field], f"training.data.{field}")
    for field in ("train_sha256", "dev_sha256", "manifest_sha256"):
        _sha256(data[field], f"training.data.{field}")
    _positive_int(
        data["train_max_records_per_route"],
        "training.data.train_max_records_per_route",
    )
    _positive_int(
        data["dev_max_records_per_route"],
        "training.data.dev_max_records_per_route",
    )
    route_weights = _mapping(data["route_weights"], "training.data.route_weights")
    if set(route_weights) != set(ROUTE_ORDER):
        missing = sorted(set(ROUTE_ORDER) - set(route_weights))
        extra = sorted(set(route_weights) - set(ROUTE_ORDER))
        raise TrainingContractError(
            f"route weights must cover exactly 20 routes: missing={missing}, extra={extra}"
        )
    for route, weight in route_weights.items():
        _finite_number(weight, f"route weight {route}", minimum=0.000000001)

    resource = _mapping(config["resource_profile"], "training.resource_profile")
    _expect_keys(
        resource,
        {
            "device",
            "precision",
            "device_memory_budget_mib",
            "device_memory_reserve_mib",
            "max_device_memory_utilization",
            "host_memory_budget_mib",
            "dataloader_memory_budget_mib",
            "oom_retry_limit",
            "micro_batch_size",
            "gradient_accumulation_steps",
            "gradient_checkpointing",
            "max_source_length",
            "max_target_length",
            "dataloader_workers",
        },
        "training.resource_profile",
    )
    if resource["device"] not in {"cpu", "cuda"}:
        raise TrainingContractError("resource device must be cpu or cuda")
    if resource["precision"] not in {"fp32", "fp16", "bf16"}:
        raise TrainingContractError("resource precision must be fp32, fp16, or bf16")
    for field in (
        "device_memory_budget_mib",
        "host_memory_budget_mib",
        "dataloader_memory_budget_mib",
        "micro_batch_size",
        "gradient_accumulation_steps",
        "max_source_length",
        "max_target_length",
    ):
        _positive_int(resource[field], f"training.resource_profile.{field}")
    for field in ("device_memory_reserve_mib", "oom_retry_limit", "dataloader_workers"):
        _positive_int(
            resource[field], f"training.resource_profile.{field}", allow_zero=True
        )
    _finite_number(
        resource["max_device_memory_utilization"],
        "training.resource_profile.max_device_memory_utilization",
        minimum=0.000000001,
        maximum=1.0,
    )
    if not isinstance(resource["gradient_checkpointing"], bool):
        raise TrainingContractError("gradient_checkpointing must be boolean")
    EncodingPolicy(
        max_source_length=resource["max_source_length"],
        max_target_length=resource["max_target_length"],
    )
    if resource["dataloader_memory_budget_mib"] > resource["host_memory_budget_mib"]:
        raise TrainingContractError("dataloader memory budget exceeds host memory budget")
    if identity["mode"] != "td14_benchmark" and resource["oom_retry_limit"] != 0:
        raise TrainingContractError("OOM retries are allowed only in TD-14 benchmark mode")

    if "input_pipeline" in config:
        pipeline = _mapping(config["input_pipeline"], "training.input_pipeline")
        pipeline_fields = {
            "mode",
            "preencode_workers",
            "memory_budget_mib",
            "pin_memory",
            "non_blocking_transfer",
        }
        if "cache_mode" in pipeline:
            pipeline_fields.add("cache_mode")
        if "length_bucket_pool_batches" in pipeline:
            pipeline_fields.add("length_bucket_pool_batches")
        _expect_keys(
            pipeline,
            pipeline_fields,
            "training.input_pipeline",
        )
        if pipeline["mode"] not in {"on_demand", "preencode_memory"}:
            raise TrainingContractError("input pipeline mode is unsupported")
        if pipeline.get("cache_mode", "memory") not in {"memory", "persistent"}:
            raise TrainingContractError("input cache mode is unsupported")
        if (
            pipeline.get("cache_mode") == "persistent"
            and pipeline["mode"] != "preencode_memory"
        ):
            raise TrainingContractError(
                "persistent input cache requires preencode_memory mode"
            )
        if "length_bucket_pool_batches" in pipeline:
            _positive_int(
                pipeline["length_bucket_pool_batches"],
                "training.input_pipeline.length_bucket_pool_batches",
            )
            if pipeline["length_bucket_pool_batches"] < 2:
                raise TrainingContractError(
                    "length bucket pool must contain at least two batches"
                )
            if pipeline.get("cache_mode") != "persistent":
                raise TrainingContractError(
                    "length bucketing requires the persistent text cache"
                )
        _positive_int(
            pipeline["preencode_workers"],
            "training.input_pipeline.preencode_workers",
            allow_zero=True,
        )
        _positive_int(
            pipeline["memory_budget_mib"],
            "training.input_pipeline.memory_budget_mib",
        )
        for field in ("pin_memory", "non_blocking_transfer"):
            if not isinstance(pipeline[field], bool):
                raise TrainingContractError(f"input pipeline {field} must be boolean")
        if pipeline["memory_budget_mib"] > resource["dataloader_memory_budget_mib"]:
            raise TrainingContractError(
                "input pipeline memory budget exceeds dataloader memory budget"
            )
        if pipeline["pin_memory"] and resource["device"] != "cuda":
            raise TrainingContractError("pinned input memory requires CUDA")
        if pipeline["non_blocking_transfer"] and not pipeline["pin_memory"]:
            raise TrainingContractError(
                "non-blocking input transfer requires pinned input memory"
            )

    if "gpu_optimization" in config:
        gpu_optimization = _mapping(
            config["gpu_optimization"], "training.gpu_optimization"
        )
        gpu_fields = {"gradient_validation", "fused_adamw"}
        if "allocator_backend" in gpu_optimization:
            gpu_fields.add("allocator_backend")
        _expect_keys(
            gpu_optimization,
            gpu_fields,
            "training.gpu_optimization",
        )
        if gpu_optimization["gradient_validation"] not in {
            "per_parameter",
            "clip_error",
        }:
            raise TrainingContractError("gradient validation mode is unsupported")
        if not isinstance(gpu_optimization["fused_adamw"], bool):
            raise TrainingContractError("fused_adamw must be boolean")
        if gpu_optimization["fused_adamw"] and resource["device"] != "cuda":
            raise TrainingContractError("fused AdamW requires CUDA")
        allocator_backend = gpu_optimization.get("allocator_backend", "native")
        if allocator_backend not in {"native", "cudaMallocAsync"}:
            raise TrainingContractError("CUDA allocator backend is unsupported")
        if allocator_backend != "native" and resource["device"] != "cuda":
            raise TrainingContractError("asynchronous CUDA allocation requires CUDA")

    if "logging" in config:
        logging = _mapping(config["logging"], "training.logging")
        _expect_keys(logging, {"mode", "flush_frequency"}, "training.logging")
        if logging["mode"] not in {"full", "compact", "performance"}:
            raise TrainingContractError("training logging mode is unsupported")
        _positive_int(
            logging["flush_frequency"], "training.logging.flush_frequency"
        )

    optimization = _mapping(config["optimization"], "training.optimization")
    optimization_fields = {
        "optimizer",
        "learning_rate",
        "betas",
        "epsilon",
        "weight_decay",
        "scheduler",
        "warmup_steps",
        "max_optimizer_steps",
        "max_train_tokens",
        "max_grad_norm",
        "label_smoothing",
        "validation_frequency",
        "validation_batches",
        "checkpoint_frequency",
    }
    if "checkpoint_retention" in optimization:
        optimization_fields.add("checkpoint_retention")
    _expect_keys(
        optimization,
        optimization_fields,
        "training.optimization",
    )
    if optimization["optimizer"] != "adamw":
        raise TrainingContractError("TD-10 supports only the frozen AdamW optimizer")
    if optimization["scheduler"] != "linear":
        raise TrainingContractError("TD-10 supports only the frozen linear scheduler")
    _finite_number(optimization["learning_rate"], "learning_rate", minimum=0.0)
    betas = optimization["betas"]
    if not isinstance(betas, list) or len(betas) != 2:
        raise TrainingContractError("optimizer betas must contain two numbers")
    for index, beta in enumerate(betas):
        _finite_number(beta, f"optimizer beta[{index}]", minimum=0.0, maximum=0.999999999)
    _finite_number(optimization["epsilon"], "epsilon", minimum=0.0)
    _finite_number(optimization["weight_decay"], "weight_decay", minimum=0.0)
    _finite_number(optimization["max_grad_norm"], "max_grad_norm", minimum=0.000000001)
    _finite_number(
        optimization["label_smoothing"],
        "label_smoothing",
        minimum=0.0,
        maximum=0.999999999,
    )
    for field in (
        "warmup_steps",
        "max_optimizer_steps",
        "max_train_tokens",
        "validation_frequency",
        "validation_batches",
        "checkpoint_frequency",
    ):
        _positive_int(
            optimization[field],
            f"training.optimization.{field}",
            allow_zero=field == "warmup_steps",
        )
    if optimization["warmup_steps"] >= optimization["max_optimizer_steps"]:
        raise TrainingContractError("warmup_steps must be less than max_optimizer_steps")
    pool_batches = int(
        config.get("input_pipeline", {}).get("length_bucket_pool_batches", 1)
    )
    total_micro_steps = (
        int(optimization["max_optimizer_steps"])
        * int(resource["gradient_accumulation_steps"])
    )
    if total_micro_steps % pool_batches:
        raise TrainingContractError(
            "total micro steps must end on a length-bucket pool boundary"
        )
    if "checkpoint_retention" in optimization:
        _positive_int(
            optimization["checkpoint_retention"],
            "training.optimization.checkpoint_retention",
        )
    return dict(config)


def load_training_config(path: Path) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise TrainingContractError(f"cannot load training config {path}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise TrainingContractError("training config must contain a mapping")
    return validate_training_config(value)


def host_memory() -> tuple[int, int]:
    """Return total and currently available host memory without extra packages."""

    if os.name == "nt":
        class MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatus()
        status.dwLength = ctypes.sizeof(status)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            raise TrainingContractError("GlobalMemoryStatusEx failed")
        return int(status.ullTotalPhys), int(status.ullAvailPhys)
    if Path("/proc/meminfo").is_file():
        values: dict[str, int] = {}
        for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines():
            name, raw = line.split(":", 1)
            values[name] = int(raw.strip().split()[0]) * 1024
        return values["MemTotal"], values.get("MemAvailable", values["MemFree"])
    page_size = os.sysconf("SC_PAGE_SIZE")
    return (
        int(page_size * os.sysconf("SC_PHYS_PAGES")),
        int(page_size * os.sysconf("SC_AVPHYS_PAGES")),
    )


def process_memory() -> dict[str, int | None]:
    """Return current and OS-observed peak resident memory when available."""

    if os.name == "nt":
        size_type = ctypes.c_size_t

        class ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_ulong),
                ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", size_type),
                ("WorkingSetSize", size_type),
                ("QuotaPeakPagedPoolUsage", size_type),
                ("QuotaPagedPoolUsage", size_type),
                ("QuotaPeakNonPagedPoolUsage", size_type),
                ("QuotaNonPagedPoolUsage", size_type),
                ("PagefileUsage", size_type),
                ("PeakPagefileUsage", size_type),
            ]

        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        psapi.GetProcessMemoryInfo.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ProcessMemoryCounters),
            ctypes.c_ulong,
        ]
        psapi.GetProcessMemoryInfo.restype = ctypes.c_bool
        ok = psapi.GetProcessMemoryInfo(
            kernel32.GetCurrentProcess(),
            ctypes.byref(counters),
            counters.cb,
        )
        if ok:
            return {
                "resident_bytes": int(counters.WorkingSetSize),
                "peak_resident_bytes": int(counters.PeakWorkingSetSize),
            }
        return {"resident_bytes": None, "peak_resident_bytes": None}
    try:
        import resource

        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform != "darwin":
            peak *= 1024
        return {"resident_bytes": None, "peak_resident_bytes": int(peak)}
    except (ImportError, OSError, ValueError):
        return {"resident_bytes": None, "peak_resident_bytes": None}


def package_versions() -> dict[str, str]:
    versions = {"python": platform.python_version()}
    for distribution in ("torch", "transformers", "tokenizers", "safetensors"):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = "not-installed"
    return versions


def configure_cuda_allocator(config: Mapping[str, Any]) -> str | None:
    """Configure an explicit allocator before the first torch/CUDA import."""
    gpu_optimization = config.get("gpu_optimization", {})
    if "allocator_backend" not in gpu_optimization:
        return None
    backend = str(gpu_optimization["allocator_backend"])
    setting = f"backend:{backend}"
    current = os.environ.get("PYTORCH_ALLOC_CONF")
    if current:
        configured = next(
            (
                item.split(":", 1)[1]
                for item in current.split(",")
                if item.strip().startswith("backend:")
            ),
            None,
        )
        if configured is not None and configured != backend:
            raise TrainingContractError(
                "PYTORCH_ALLOC_CONF allocator conflicts with training config"
            )
        if configured is None:
            raise TrainingContractError(
                "explicit allocator config cannot inherit unrelated "
                "PYTORCH_ALLOC_CONF options"
            )
    else:
        os.environ["PYTORCH_ALLOC_CONF"] = setting
    return backend


def probe_runtime(resource_profile: Mapping[str, Any]) -> dict[str, Any]:
    import torch

    total_host, available_host = host_memory()
    requested_device = str(resource_profile["device"])
    precision = str(resource_profile["precision"])
    devices = []
    for index in range(torch.cuda.device_count()):
        properties = torch.cuda.get_device_properties(index)
        devices.append(
            {
                "index": index,
                "name": properties.name,
                "total_memory_bytes": int(properties.total_memory),
                "capability": list(torch.cuda.get_device_capability(index)),
            }
        )
    if requested_device == "cuda":
        if not torch.cuda.is_available() or not devices:
            raise TrainingContractError("CUDA was requested but is unavailable")
        if precision == "bf16" and not torch.cuda.is_bf16_supported():
            raise TrainingContractError("BF16 was requested but is unsupported")
        device_total = devices[0]["total_memory_bytes"]
        device_available = int(torch.cuda.mem_get_info(0)[0])
    else:
        if precision != "fp32":
            raise TrainingContractError("CPU training requires fp32 in the frozen TD-10 stack")
        device_total = total_host
        device_available = available_host
    driver_version = None
    if devices:
        try:
            completed = subprocess.run(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                check=True,
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=5,
            )
            driver_version = completed.stdout.splitlines()[0].strip() or None
        except (OSError, subprocess.SubprocessError, IndexError):
            driver_version = None
    runtime = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "host_logical_processors": os.cpu_count(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_driver": driver_version,
        "cudnn": torch.backends.cudnn.version(),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": len(devices),
        "cuda_devices": devices,
        "selected_device": requested_device,
        "selected_precision": precision,
        "device_total_bytes": int(device_total),
        "device_available_bytes": int(device_available),
        "host_total_bytes": total_host,
        "host_available_bytes": available_host,
        "packages": package_versions(),
    }
    if requested_device == "cuda":
        runtime["cuda_allocator_backend"] = torch.cuda.memory.get_allocator_backend()
    runtime["resource_validation"] = validate_resource_budget(resource_profile, runtime)
    return runtime


def validate_resource_budget(
    resource_profile: Mapping[str, Any], runtime: Mapping[str, Any]
) -> dict[str, Any]:
    budget = int(resource_profile["device_memory_budget_mib"]) * MIB
    reserve = int(resource_profile["device_memory_reserve_mib"]) * MIB
    utilization_limit = int(
        int(runtime["device_total_bytes"])
        * float(resource_profile["max_device_memory_utilization"])
    )
    reserve_limit = int(runtime["device_total_bytes"]) - reserve
    effective_limit = min(budget, utilization_limit, reserve_limit)
    if reserve_limit <= 0 or effective_limit < budget:
        raise TrainingContractError(
            "device capacity cannot satisfy the configured absolute budget, reserve, "
            "and utilization limit"
        )
    host_budget = int(resource_profile["host_memory_budget_mib"]) * MIB
    if host_budget > int(runtime["host_available_bytes"]):
        raise TrainingContractError("available host memory is below the configured budget")
    loader_budget = int(resource_profile["dataloader_memory_budget_mib"]) * MIB
    estimated_loader_bytes = (
        max(1, int(resource_profile["dataloader_workers"]))
        * int(resource_profile["micro_batch_size"])
        * (
            int(resource_profile["max_source_length"])
            + int(resource_profile["max_target_length"])
        )
        * 8
        * 4
    )
    if estimated_loader_bytes > loader_budget:
        raise TrainingContractError("estimated dataloader batch memory exceeds its budget")
    return {
        "absolute_budget_bytes": budget,
        "utilization_limit_bytes": utilization_limit,
        "reserve_limit_bytes": reserve_limit,
        "effective_device_limit_bytes": effective_limit,
        "host_budget_bytes": host_budget,
        "dataloader_budget_bytes": loader_budget,
        "estimated_dataloader_bytes": estimated_loader_bytes,
    }


@dataclass(frozen=True)
class RouteDataset:
    split: str
    records_by_route: dict[str, tuple[dict[str, Any], ...]]
    file_sha256: str
    selection_sha256: str

    @property
    def records(self) -> int:
        return sum(len(records) for records in self.records_by_route.values())


@dataclass(frozen=True)
class EncodedSampleCache:
    """Identity-bound in-memory encodings for deterministic GPU feeding."""

    samples_by_id: dict[str, EncodedSample]
    identities: tuple[str, ...]
    estimated_bytes: int
    identity_sha256: str

    def samples(self, records: Sequence[Mapping[str, Any]]) -> list[EncodedSample]:
        selected: list[EncodedSample] = []
        for record in records:
            sample_id = record.get("sample_id")
            if not isinstance(sample_id, str) or sample_id not in self.samples_by_id:
                raise TrainingContractError(
                    "encoded input cache is missing a selected sample identity"
                )
            selected.append(self.samples_by_id[sample_id])
        return selected

    @classmethod
    def merge(cls, *caches: "EncodedSampleCache") -> "EncodedSampleCache":
        samples: dict[str, EncodedSample] = {}
        for cache in caches:
            overlap = set(samples).intersection(cache.samples_by_id)
            if overlap:
                raise TrainingContractError("encoded input caches contain duplicate samples")
            samples.update(cache.samples_by_id)
        identities = tuple(cache.identity_sha256 for cache in caches)
        return cls(
            samples_by_id=samples,
            identities=identities,
            estimated_bytes=sum(cache.estimated_bytes for cache in caches),
            identity_sha256=config_sha256({"component_caches": identities}),
        )


@dataclass(frozen=True)
class TextEncodingCache:
    """Reusable full token sequences backed by one validated persistent payload."""

    encodings: dict[tuple[str, str], tuple[int, ...]]
    identity_sha256: str
    directory: Path | None
    source: str
    payload_bytes: int
    token_ids: int
    estimated_bytes: int


def _text_encoding_cache_bytes(
    encodings: Mapping[tuple[str, str], Sequence[int]],
) -> int:
    return sum(
        256
        + len(language.encode("utf-8"))
        + len(text_value.encode("utf-8"))
        + 36 * len(token_ids)
        for (language, text_value), token_ids in encodings.items()
    )


def _canonical_records(datasets: Sequence[RouteDataset]) -> list[dict[str, Any]]:
    return [
        record
        for dataset in datasets
        for route in ROUTE_ORDER
        for record in dataset.records_by_route[route]
    ]


def _canonical_text_keys(
    datasets: Sequence[RouteDataset],
) -> tuple[tuple[str, str], ...]:
    keys: set[tuple[str, str]] = set()
    for dataset in datasets:
        for route in ROUTE_ORDER:
            for record in dataset.records_by_route[route]:
                keys.add((str(record["src_lang"]), str(record["source_text"])))
                keys.add((str(record["tgt_lang"]), str(record["target_text"])))
    return tuple(sorted(keys))


def _text_key_digest(key: tuple[str, str]) -> bytes:
    language = key[0].encode("utf-8")
    text_value = key[1].encode("utf-8")
    digest = hashlib.sha256()
    digest.update(struct.pack("<I", len(language)))
    digest.update(language)
    digest.update(struct.pack("<Q", len(text_value)))
    digest.update(text_value)
    return digest.digest()


def _text_cache_identity(
    *,
    datasets: Sequence[RouteDataset],
    tokenizer_manifest_sha256: str,
    keys: Sequence[tuple[str, str]],
) -> str:
    key_digest = hashlib.sha256()
    for key in keys:
        key_digest.update(_text_key_digest(key))
    return config_sha256(
        {
            "schema_version": TEXT_CACHE_SCHEMA_VERSION,
            "datasets": [
                {
                    "split": dataset.split,
                    "file_sha256": dataset.file_sha256,
                    "selection_sha256": dataset.selection_sha256,
                    "records": dataset.records,
                }
                for dataset in datasets
            ],
            "tokenizer_manifest_sha256": tokenizer_manifest_sha256,
            "unique_language_texts": len(keys),
            "unique_language_texts_sha256": key_digest.hexdigest(),
        }
    )


def _load_text_encoding_cache(
    *,
    directory: Path,
    expected_identity_sha256: str,
    keys: Sequence[tuple[str, str]],
    source: str,
) -> TextEncodingCache:
    import numpy

    manifest_path = directory / TEXT_CACHE_MANIFEST
    payload_path = directory / TEXT_CACHE_PAYLOAD
    if not directory.is_dir() or directory.is_symlink():
        raise TrainingContractError("persistent text cache must be a real directory")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TrainingContractError("persistent text cache manifest is unreadable") from exc
    if manifest.get("status") != "complete":
        raise TrainingContractError("persistent text cache is not complete")
    if manifest.get("identity_sha256") != expected_identity_sha256:
        raise TrainingContractError("persistent text cache identity changed")
    if int(manifest.get("unique_language_texts", -1)) != len(keys):
        raise TrainingContractError("persistent text cache key count changed")
    payload = manifest.get("payload")
    if not isinstance(payload, Mapping):
        raise TrainingContractError("persistent text cache payload metadata is invalid")
    if set(payload) != {"path", "bytes", "sha256"}:
        raise TrainingContractError("persistent text cache payload fields changed")
    if payload["path"] != TEXT_CACHE_PAYLOAD or not payload_path.is_file():
        raise TrainingContractError("persistent text cache payload is missing")
    if payload_path.is_symlink() or payload_path.stat().st_size != int(payload["bytes"]):
        raise TrainingContractError("persistent text cache payload size changed")
    if sha256_file(payload_path) != payload["sha256"]:
        raise TrainingContractError("persistent text cache payload SHA-256 changed")
    try:
        with numpy.load(payload_path, allow_pickle=False) as arrays:
            if set(arrays.files) != {"key_hashes", "offsets", "token_ids"}:
                raise TrainingContractError(
                    "persistent text cache array set changed"
                )
            key_hashes = arrays["key_hashes"]
            offsets = arrays["offsets"]
            token_ids = arrays["token_ids"]
    except (OSError, ValueError) as exc:
        raise TrainingContractError("persistent text cache payload is invalid") from exc
    expected_hashes = numpy.frombuffer(
        b"".join(_text_key_digest(key) for key in keys), dtype=numpy.uint8
    ).reshape(len(keys), 32)
    if (
        key_hashes.dtype != numpy.uint8
        or key_hashes.shape != expected_hashes.shape
        or not numpy.array_equal(key_hashes, expected_hashes)
    ):
        raise TrainingContractError("persistent text cache key identities changed")
    if offsets.dtype != numpy.int64 or offsets.shape != (len(keys) + 1,):
        raise TrainingContractError("persistent text cache offsets are invalid")
    if token_ids.dtype != numpy.int32 or token_ids.ndim != 1:
        raise TrainingContractError("persistent text cache token IDs are invalid")
    if (
        int(offsets[0]) != 0
        or int(offsets[-1]) != len(token_ids)
        or bool(numpy.any(offsets[1:] < offsets[:-1]))
    ):
        raise TrainingContractError("persistent text cache offsets are inconsistent")
    encodings = {
        key: tuple(int(value) for value in token_ids[int(offsets[index]) : int(offsets[index + 1])])
        for index, key in enumerate(keys)
    }
    return TextEncodingCache(
        encodings=encodings,
        identity_sha256=expected_identity_sha256,
        directory=directory,
        source=source,
        payload_bytes=payload_path.stat().st_size,
        token_ids=len(token_ids),
        estimated_bytes=_text_encoding_cache_bytes(encodings),
    )


_PROCESS_TEXT_TOKENIZER: object | None = None
_PROCESS_TEXT_MAPPING: Mapping[str, int] | None = None
_PROCESS_TEXT_VOCAB_SIZE: int | None = None


def _initialize_text_encoder(tokenizer_path: str) -> None:
    global _PROCESS_TEXT_TOKENIZER
    global _PROCESS_TEXT_MAPPING
    global _PROCESS_TEXT_VOCAB_SIZE
    _PROCESS_TEXT_TOKENIZER = reload_tokenizer(Path(tokenizer_path))
    _PROCESS_TEXT_MAPPING = build_language_mapping(_PROCESS_TEXT_TOKENIZER)
    _PROCESS_TEXT_VOCAB_SIZE = len(_PROCESS_TEXT_TOKENIZER)


def _process_encode_text(key: tuple[str, str]) -> tuple[int, ...]:
    if (
        _PROCESS_TEXT_TOKENIZER is None
        or _PROCESS_TEXT_MAPPING is None
        or _PROCESS_TEXT_VOCAB_SIZE is None
    ):
        raise TrainingContractError("text encoding worker was not initialized")
    return encode_language_text(
        _PROCESS_TEXT_TOKENIZER,
        key[1],
        key[0],
        language_mapping=_PROCESS_TEXT_MAPPING,
        vocab_size=_PROCESS_TEXT_VOCAB_SIZE,
    )


def load_or_build_text_encoding_cache(
    *,
    datasets: Sequence[RouteDataset],
    tokenizer: object,
    tokenizer_path: Path,
    tokenizer_manifest_sha256: str,
    workers: int,
    cache_root: Path | None,
) -> TextEncodingCache:
    """Load or atomically publish one deduplicated, identity-bound token payload."""

    import numpy

    keys = _canonical_text_keys(datasets)
    identity_sha256 = _text_cache_identity(
        datasets=datasets,
        tokenizer_manifest_sha256=tokenizer_manifest_sha256,
        keys=keys,
    )
    target = (
        cache_root / f"text-cache-{identity_sha256}"
        if cache_root is not None
        else None
    )
    if target is not None and target.exists():
        return _load_text_encoding_cache(
            directory=target,
            expected_identity_sha256=identity_sha256,
            keys=keys,
            source="persistent",
        )
    if workers:
        chunksize = max(1, len(keys) // max(1, workers * 16))
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_initialize_text_encoder,
            initargs=(str(tokenizer_path),),
        ) as executor:
            values = list(executor.map(_process_encode_text, keys, chunksize=chunksize))
    else:
        mapping = build_language_mapping(tokenizer)
        vocab_size = len(tokenizer)
        values = [
            encode_language_text(
                tokenizer,
                key[1],
                key[0],
                language_mapping=mapping,
                vocab_size=vocab_size,
            )
            for key in keys
        ]
    if target is None:
        return TextEncodingCache(
            encodings=(encodings := dict(zip(keys, values, strict=True))),
            identity_sha256=identity_sha256,
            directory=None,
            source="memory",
            payload_bytes=0,
            token_ids=sum(len(value) for value in values),
            estimated_bytes=_text_encoding_cache_bytes(encodings),
        )
    cache_root.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{target.name}.staging-",
            dir=cache_root,
        )
    )
    try:
        offsets = numpy.zeros(len(values) + 1, dtype=numpy.int64)
        for index, value in enumerate(values, start=1):
            offsets[index] = offsets[index - 1] + len(value)
        token_ids = numpy.fromiter(
            (token for value in values for token in value),
            dtype=numpy.int32,
            count=int(offsets[-1]),
        )
        key_hashes = numpy.frombuffer(
            b"".join(_text_key_digest(key) for key in keys), dtype=numpy.uint8
        ).reshape(len(keys), 32)
        payload_path = staging / TEXT_CACHE_PAYLOAD
        with payload_path.open("wb") as handle:
            numpy.savez(
                handle,
                key_hashes=key_hashes,
                offsets=offsets,
                token_ids=token_ids,
            )
            handle.flush()
            os.fsync(handle.fileno())
        _atomic_json(
            staging / TEXT_CACHE_MANIFEST,
            {
                "schema_version": TEXT_CACHE_SCHEMA_VERSION,
                "status": "complete",
                "identity_sha256": identity_sha256,
                "unique_language_texts": len(keys),
                "token_ids": len(token_ids),
                "payload": {
                    "path": TEXT_CACHE_PAYLOAD,
                    "bytes": payload_path.stat().st_size,
                    "sha256": sha256_file(payload_path),
                },
            },
        )
        try:
            os.replace(staging, target)
        except OSError:
            if not target.exists():
                raise
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return _load_text_encoding_cache(
        directory=target,
        expected_identity_sha256=identity_sha256,
        keys=keys,
        source="built",
    )


_PROCESS_ENCODER_TOKENIZER: object | None = None
_PROCESS_ENCODER_POLICY: EncodingPolicy | None = None
_PROCESS_ENCODER_MAPPING: Mapping[str, int] | None = None
_PROCESS_ENCODER_VOCAB_SIZE: int | None = None


def _initialize_process_encoder(
    tokenizer_path: str, policy_values: Mapping[str, Any]
) -> None:
    global _PROCESS_ENCODER_TOKENIZER
    global _PROCESS_ENCODER_POLICY
    global _PROCESS_ENCODER_MAPPING
    global _PROCESS_ENCODER_VOCAB_SIZE
    _PROCESS_ENCODER_TOKENIZER = reload_tokenizer(Path(tokenizer_path))
    _PROCESS_ENCODER_POLICY = EncodingPolicy(**dict(policy_values))
    _PROCESS_ENCODER_MAPPING = build_language_mapping(_PROCESS_ENCODER_TOKENIZER)
    _PROCESS_ENCODER_VOCAB_SIZE = len(_PROCESS_ENCODER_TOKENIZER)


def _process_encode_sample(record: Mapping[str, Any]) -> EncodedSample:
    if (
        _PROCESS_ENCODER_TOKENIZER is None
        or _PROCESS_ENCODER_POLICY is None
        or _PROCESS_ENCODER_MAPPING is None
        or _PROCESS_ENCODER_VOCAB_SIZE is None
    ):
        raise TrainingContractError("pre-encoding worker was not initialized")
    return encode_parallel_sample(
        _PROCESS_ENCODER_TOKENIZER,
        record,
        _PROCESS_ENCODER_POLICY,
        language_mapping=_PROCESS_ENCODER_MAPPING,
        vocab_size=_PROCESS_ENCODER_VOCAB_SIZE,
    )


def _encoded_sample_bytes(sample: EncodedSample) -> int:
    return (
        256
        + 8 * (len(sample.input_ids) + len(sample.labels))
        + len(sample.sample_id.encode("utf-8"))
        + len(sample.sample_group_id.encode("utf-8"))
        + len(sample.source_language.encode("ascii"))
        + len(sample.target_language.encode("ascii"))
    )


def build_encoded_sample_cache(
    *,
    dataset: RouteDataset,
    tokenizer: object,
    tokenizer_path: Path,
    tokenizer_manifest_sha256: str,
    policy: EncodingPolicy,
    workers: int,
    memory_budget_mib: int,
    text_cache: TextEncodingCache | None = None,
) -> EncodedSampleCache:
    """Pre-encode a frozen split in canonical order with bounded host memory."""

    records = [
        record
        for route in ROUTE_ORDER
        for record in dataset.records_by_route[route]
    ]
    if text_cache is not None:
        mapping = build_language_mapping(tokenizer)
        vocab_size = len(tokenizer)
        try:
            encoded = [
                encoded_sample_from_sequences(
                    tokenizer,
                    record,
                    policy,
                    source_ids=text_cache.encodings[
                        (str(record["src_lang"]), str(record["source_text"]))
                    ],
                    target_ids=text_cache.encodings[
                        (str(record["tgt_lang"]), str(record["target_text"]))
                    ],
                    language_mapping=mapping,
                    vocab_size=vocab_size,
                )
                for record in records
            ]
        except KeyError as exc:
            raise TrainingContractError(
                "text encoding cache is missing a selected language/text identity"
            ) from exc
    elif workers:
        chunksize = max(1, len(records) // max(1, workers * 16))
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_initialize_process_encoder,
            initargs=(str(tokenizer_path), asdict(policy)),
        ) as executor:
            encoded = list(
                executor.map(_process_encode_sample, records, chunksize=chunksize)
            )
    else:
        mapping = build_language_mapping(tokenizer)
        vocab_size = len(tokenizer)
        encoded = [
            encode_parallel_sample(
                tokenizer,
                record,
                policy,
                language_mapping=mapping,
                vocab_size=vocab_size,
            )
            for record in records
        ]
    samples_by_id = {sample.sample_id: sample for sample in encoded}
    if len(samples_by_id) != len(records):
        raise TrainingContractError("pre-encoded input cache contains duplicate samples")
    estimated_bytes = sum(_encoded_sample_bytes(sample) for sample in encoded)
    if estimated_bytes > memory_budget_mib * MIB:
        raise TrainingContractError("pre-encoded input cache exceeds its memory budget")
    identity = {
        "schema_version": 1,
        "split": dataset.split,
        "file_sha256": dataset.file_sha256,
        "selection_sha256": dataset.selection_sha256,
        "tokenizer_manifest_sha256": tokenizer_manifest_sha256,
        "encoding_policy": asdict(policy),
        "text_cache_identity_sha256": (
            text_cache.identity_sha256 if text_cache is not None else None
        ),
        "records": len(encoded),
    }
    return EncodedSampleCache(
        samples_by_id=samples_by_id,
        identities=(config_sha256(identity),),
        estimated_bytes=estimated_bytes,
        identity_sha256=config_sha256(identity),
    )


def load_route_dataset(
    path: Path,
    *,
    expected_sha256: str,
    split: str,
    max_records_per_route: int,
) -> RouteDataset:
    actual_sha256 = sha256_file(path)
    if actual_sha256 != expected_sha256:
        raise TrainingContractError(f"{split} data SHA-256 changed")
    selected: dict[str, list[dict[str, Any]]] = {route: [] for route in ROUTE_ORDER}
    seen_ids: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise TrainingContractError(
                    f"invalid {split} JSON at line {line_number}: {exc}"
                ) from exc
            if record.get("split") != split:
                raise TrainingContractError(
                    f"{split} loader encountered split={record.get('split')!r}"
                )
            route = f"{record.get('src_lang')}->{record.get('tgt_lang')}"
            if route not in selected:
                raise TrainingContractError(f"{split} data contains unsupported route {route}")
            sample_id = record.get("sample_id")
            if not isinstance(sample_id, str) or not sample_id or sample_id in seen_ids:
                raise TrainingContractError(f"{split} data contains a missing/duplicate sample_id")
            seen_ids.add(sample_id)
            if len(selected[route]) < max_records_per_route:
                selected[route].append(record)
            if all(len(rows) >= max_records_per_route for rows in selected.values()):
                break
    missing = [route for route, rows in selected.items() if not rows]
    if missing:
        raise TrainingContractError(f"{split} data is empty for routes: {missing}")
    selection = {
        route: [record["sample_id"] for record in selected[route]]
        for route in ROUTE_ORDER
    }
    return RouteDataset(
        split=split,
        records_by_route={route: tuple(selected[route]) for route in ROUTE_ORDER},
        file_sha256=actual_sha256,
        selection_sha256=config_sha256(selection),
    )


@dataclass(frozen=True)
class SampleSelection:
    route: str
    route_epoch: int
    route_position: int
    record: dict[str, Any]


class DeterministicRouteSampler:
    """Smooth weighted round-robin route sampling with resumable shuffles."""

    def __init__(
        self,
        dataset: RouteDataset,
        route_weights: Mapping[str, float],
        seed: int,
    ) -> None:
        if set(route_weights) != set(ROUTE_ORDER):
            raise TrainingContractError("sampler route weights do not cover the route matrix")
        self.dataset = dataset
        self.weights = {route: float(route_weights[route]) for route in ROUTE_ORDER}
        if any(not math.isfinite(weight) or weight <= 0 for weight in self.weights.values()):
            raise TrainingContractError("sampler route weights must be finite and positive")
        self.total_weight = sum(self.weights.values())
        self.rng = random.Random(seed)
        self.scores = {route: 0.0 for route in ROUTE_ORDER}
        self.orders: dict[str, list[int]] = {}
        self.positions = {route: 0 for route in ROUTE_ORDER}
        self.epochs = {route: 0 for route in ROUTE_ORDER}
        for route in ROUTE_ORDER:
            order = list(range(len(dataset.records_by_route[route])))
            self.rng.shuffle(order)
            self.orders[route] = order
        self.samples_emitted = 0

    def _next_route(self) -> str:
        for route in ROUTE_ORDER:
            self.scores[route] += self.weights[route]
        route = max(ROUTE_ORDER, key=lambda item: (self.scores[item], -ROUTE_ORDER.index(item)))
        self.scores[route] -= self.total_weight
        return route

    def next_sample(self) -> SampleSelection:
        route = self._next_route()
        order = self.orders[route]
        position = self.positions[route]
        if position >= len(order):
            self.epochs[route] += 1
            order = list(range(len(self.dataset.records_by_route[route])))
            self.rng.shuffle(order)
            self.orders[route] = order
            self.positions[route] = 0
            position = 0
        record_index = order[position]
        self.positions[route] = position + 1
        self.samples_emitted += 1
        return SampleSelection(
            route=route,
            route_epoch=self.epochs[route],
            route_position=position,
            record=self.dataset.records_by_route[route][record_index],
        )

    def next_batch(self, batch_size: int) -> list[SampleSelection]:
        if batch_size < 1:
            raise TrainingContractError("batch size must be positive")
        return [self.next_sample() for _ in range(batch_size)]

    def state_dict(self) -> dict[str, Any]:
        return {
            "dataset_selection_sha256": self.dataset.selection_sha256,
            "weights": dict(self.weights),
            "scores": dict(self.scores),
            "orders": {route: list(order) for route, order in self.orders.items()},
            "positions": dict(self.positions),
            "epochs": dict(self.epochs),
            "samples_emitted": self.samples_emitted,
            "rng_state": self.rng.getstate(),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if state.get("dataset_selection_sha256") != self.dataset.selection_sha256:
            raise TrainingContractError("sampler state belongs to a different dataset")
        if state.get("weights") != self.weights:
            raise TrainingContractError("sampler state route weights changed")
        for field in ("scores", "orders", "positions", "epochs"):
            value = state.get(field)
            if not isinstance(value, Mapping) or set(value) != set(ROUTE_ORDER):
                raise TrainingContractError(f"sampler state {field} is incomplete")
        self.scores = {route: float(state["scores"][route]) for route in ROUTE_ORDER}
        self.orders = {
            route: [int(value) for value in state["orders"][route]]
            for route in ROUTE_ORDER
        }
        self.positions = {route: int(state["positions"][route]) for route in ROUTE_ORDER}
        self.epochs = {route: int(state["epochs"][route]) for route in ROUTE_ORDER}
        self.samples_emitted = int(state.get("samples_emitted", -1))
        if self.samples_emitted < 0:
            raise TrainingContractError("sampler samples_emitted is invalid")
        try:
            self.rng.setstate(state["rng_state"])
        except (KeyError, TypeError, ValueError) as exc:
            raise TrainingContractError("sampler RNG state is invalid") from exc


class DeterministicLengthBucketSampler:
    """Sort a bounded draw pool by cached length with exact pending-state resume."""

    def __init__(
        self,
        sampler: DeterministicRouteSampler,
        text_cache: TextEncodingCache,
        pool_batches: int,
    ) -> None:
        if pool_batches < 2:
            raise TrainingContractError("length bucket pool must contain two batches")
        self.sampler = sampler
        self.text_cache = text_cache
        self.pool_batches = pool_batches
        self.batch_size: int | None = None
        self.pending: list[list[SampleSelection]] = []
        self.records_by_id: dict[str, dict[str, Any]] = {}
        for route in ROUTE_ORDER:
            for record in sampler.dataset.records_by_route[route]:
                sample_id = str(record.get("sample_id", ""))
                if not sample_id or sample_id in self.records_by_id:
                    raise TrainingContractError(
                        "length bucket dataset sample identities are invalid"
                    )
                self.records_by_id[sample_id] = record

    @property
    def epochs(self) -> dict[str, int]:
        return self.sampler.epochs

    def _length_key(self, selection: SampleSelection) -> tuple[int, int, str, int, int]:
        record = selection.record
        try:
            source = self.text_cache.encodings[
                (str(record["src_lang"]), str(record["source_text"]))
            ]
            target = self.text_cache.encodings[
                (str(record["tgt_lang"]), str(record["target_text"]))
            ]
        except KeyError as exc:
            raise TrainingContractError(
                "length bucket cache is missing a selected language/text identity"
            ) from exc
        return (
            max(len(source), len(target)),
            len(source) + len(target),
            selection.route,
            selection.route_epoch,
            selection.route_position,
        )

    def next_batch(self, batch_size: int) -> list[SampleSelection]:
        if batch_size < 1:
            raise TrainingContractError("batch size must be positive")
        if self.batch_size is None:
            self.batch_size = batch_size
        elif self.batch_size != batch_size:
            raise TrainingContractError("length bucket batch size changed")
        if not self.pending:
            pool = self.sampler.next_batch(batch_size * self.pool_batches)
            ordered = sorted(pool, key=self._length_key)
            self.pending = [
                ordered[index : index + batch_size]
                for index in range(0, len(ordered), batch_size)
            ]
        return self.pending.pop(0)

    @staticmethod
    def _selection_state(selection: SampleSelection) -> dict[str, Any]:
        return {
            "route": selection.route,
            "route_epoch": selection.route_epoch,
            "route_position": selection.route_position,
            "sample_id": selection.record["sample_id"],
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "kind": "deterministic_length_bucket",
            "pool_batches": self.pool_batches,
            "batch_size": self.batch_size,
            "pending": [
                [self._selection_state(selection) for selection in batch]
                for batch in self.pending
            ],
            "sampler": self.sampler.state_dict(),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if (
            state.get("schema_version") != 1
            or state.get("kind") != "deterministic_length_bucket"
        ):
            raise TrainingContractError("length bucket sampler state is unsupported")
        if int(state.get("pool_batches", -1)) != self.pool_batches:
            raise TrainingContractError("length bucket pool size changed")
        batch_size = state.get("batch_size")
        if batch_size is not None and (not isinstance(batch_size, int) or batch_size < 1):
            raise TrainingContractError("length bucket batch size state is invalid")
        pending_value = state.get("pending")
        if not isinstance(pending_value, list):
            raise TrainingContractError("length bucket pending state is invalid")
        pending: list[list[SampleSelection]] = []
        for batch in pending_value:
            if not isinstance(batch, list):
                raise TrainingContractError("length bucket pending batch is invalid")
            selections: list[SampleSelection] = []
            for value in batch:
                if not isinstance(value, Mapping):
                    raise TrainingContractError("length bucket selection state is invalid")
                sample_id = str(value.get("sample_id", ""))
                record = self.records_by_id.get(sample_id)
                route = str(value.get("route", ""))
                if record is None or f"{record['src_lang']}->{record['tgt_lang']}" != route:
                    raise TrainingContractError(
                        "length bucket pending sample identity changed"
                    )
                selections.append(
                    SampleSelection(
                        route=route,
                        route_epoch=int(value["route_epoch"]),
                        route_position=int(value["route_position"]),
                        record=record,
                    )
                )
            pending.append(selections)
        if batch_size is not None and any(len(batch) != batch_size for batch in pending):
            raise TrainingContractError("length bucket pending batch size changed")
        self.sampler.load_state_dict(_mapping(state.get("sampler"), "sampler state"))
        self.batch_size = batch_size
        self.pending = pending


class BatchEncoder:
    """Bounded, ordered tokenizer workers without sampler prefetch."""

    def __init__(
        self,
        *,
        tokenizer: object,
        tokenizer_path: Path,
        policy: EncodingPolicy,
        workers: int,
        encoded_cache: EncodedSampleCache | None = None,
        text_cache: TextEncodingCache | None = None,
        pin_memory: bool = False,
        tokenizer_loader: Callable[[Path], object] = reload_tokenizer,
    ) -> None:
        self.collator = DirectionAwareCollator(tokenizer, policy)
        self.tokenizer_path = tokenizer_path
        self.policy = policy
        self.workers = workers
        self.encoded_cache = encoded_cache
        self.text_cache = text_cache
        self.pin_memory = pin_memory
        self.tokenizer_loader = tokenizer_loader
        self.local = threading.local()
        self.executor = (
            ThreadPoolExecutor(max_workers=workers)
            if workers and encoded_cache is None and text_cache is None
            else None
        )
        if encoded_cache is not None and text_cache is not None:
            raise TrainingContractError(
                "batch encoder accepts only one encoded cache representation"
            )

    def _encode(self, record: Mapping[str, Any]) -> EncodedSample:
        tokenizer = getattr(self.local, "tokenizer", None)
        if tokenizer is None:
            tokenizer = self.tokenizer_loader(self.tokenizer_path)
            self.local.tokenizer = tokenizer
            self.local.language_mapping = build_language_mapping(tokenizer)
            self.local.vocab_size = len(tokenizer)
        return encode_parallel_sample(
            tokenizer,
            record,
            self.policy,
            language_mapping=self.local.language_mapping,
            vocab_size=self.local.vocab_size,
        )

    def _pin_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        if self.pin_memory:
            for name in model_inputs(batch):
                batch[name] = batch[name].pin_memory()
        return batch

    def _finish(self, samples: Sequence[EncodedSample]) -> dict[str, Any]:
        return self._pin_batch(self.collator.collate_encoded(samples))

    def _from_text_cache(self, record: Mapping[str, Any]) -> EncodedSample:
        if self.text_cache is None:
            raise TrainingContractError("text encoding cache is unavailable")
        try:
            source_ids = self.text_cache.encodings[
                (str(record["src_lang"]), str(record["source_text"]))
            ]
            target_ids = self.text_cache.encodings[
                (str(record["tgt_lang"]), str(record["target_text"]))
            ]
        except KeyError as exc:
            raise TrainingContractError(
                "text encoding cache is missing a selected language/text identity"
            ) from exc
        return encoded_sample_from_sequences(
            self.collator.tokenizer,
            record,
            self.policy,
            source_ids=source_ids,
            target_ids=target_ids,
            language_mapping=self.collator.language_mapping,
            vocab_size=self.collator.vocab_size,
        )

    def __call__(self, records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        if self.encoded_cache is not None:
            return self._finish(self.encoded_cache.samples(records))
        if self.text_cache is not None:
            return self._finish([self._from_text_cache(record) for record in records])
        if self.executor is None:
            return self._pin_batch(self.collator(records))
        encoded = list(self.executor.map(self._encode, records))
        return self._finish(encoded)

    def close(self) -> None:
        if self.executor is not None:
            self.executor.shutdown(wait=True, cancel_futures=True)

    def __enter__(self) -> "BatchEncoder":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()


class JsonlRunLogger:
    def __init__(self, path: Path | None = None, *, flush_frequency: int = 1) -> None:
        if flush_frequency < 1:
            raise TrainingContractError("logger flush frequency must be positive")
        self.path = path
        self.flush_frequency = flush_frequency
        self.unflushed_events = 0
        self.events: list[dict[str, Any]] = []
        self.handle = None
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            self.handle = path.open("x", encoding="utf-8", newline="\n")

    def write(self, event: Mapping[str, Any]) -> None:
        row = dict(event)
        self.events.append(row)
        if self.handle is not None:
            self.handle.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                + "\n"
            )
            self.unflushed_events += 1
            if self.unflushed_events >= self.flush_frequency:
                self.handle.flush()
                self.unflushed_events = 0

    def close(self) -> None:
        if self.handle is not None:
            self.flush()
            os.fsync(self.handle.fileno())
            self.handle.close()
            self.handle = None

    def flush(self) -> None:
        if self.handle is not None:
            self.handle.flush()
            self.unflushed_events = 0

    def __enter__(self) -> "JsonlRunLogger":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()


def semantic_trace_sha256(events: Sequence[Mapping[str, Any]]) -> str:
    """Hash only deterministic training semantics, excluding timing/resource noise."""

    excluded = {
        "wall_time_seconds",
        "tokens_per_second",
        "samples_per_second",
        "peak_device_memory_bytes",
        "peak_device_reserved_bytes",
    }
    semantic = [
        {name: value for name, value in event.items() if name not in excluded}
        for event in events
        if event.get("event") not in {"checkpoint", "input_cache"}
    ]
    return config_sha256(semantic)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise TrainingContractError(
                    f"invalid event JSON at line {line_number}: {exc}"
                ) from exc
            if not isinstance(value, dict):
                raise TrainingContractError("event log contains a non-object row")
            events.append(value)
    return events


def compare_training_runs(
    baseline: Mapping[str, Any], candidate: Mapping[str, Any]
) -> dict[str, Any]:
    """Compare two completed runs while allowing resource/timing measurements to vary."""

    for name, report in (("baseline", baseline), ("candidate", candidate)):
        if report.get("status") != "complete":
            raise TrainingContractError(f"{name} run is not complete")
    baseline_events = read_jsonl(
        Path(str(baseline["output_root"])) / str(baseline["events"]["path"])
    )
    candidate_events = read_jsonl(
        Path(str(candidate["output_root"])) / str(candidate["events"]["path"])
    )
    baseline_trace = semantic_trace_sha256(baseline_events)
    candidate_trace = semantic_trace_sha256(candidate_events)
    comparisons = {
        "semantic_trace_sha256": baseline_trace == candidate_trace,
        "final_train_loss": (
            baseline["result"]["final_train_loss"]
            == candidate["result"]["final_train_loss"]
        ),
        "mean_train_loss": (
            baseline["result"]["mean_train_loss"]
            == candidate["result"]["mean_train_loss"]
        ),
        "sampler_state": (
            config_sha256(baseline["result"]["sampler_state"])
            == config_sha256(candidate["result"]["sampler_state"])
        ),
        "optimizer_steps": (
            baseline["result"]["optimizer_steps"]
            == candidate["result"]["optimizer_steps"]
        ),
        "micro_steps": baseline["result"]["micro_steps"] == candidate["result"]["micro_steps"],
    }
    if not all(comparisons.values()):
        raise TrainingContractError(f"deterministic training replay differs: {comparisons}")
    return {
        "status": "exact",
        "comparisons": comparisons,
        "semantic_trace_sha256": baseline_trace,
        "baseline": {
            "output_root": baseline["output_root"],
            "events_sha256": baseline["events"]["sha256"],
        },
        "candidate": {
            "output_root": candidate["output_root"],
            "events_sha256": candidate["events"]["sha256"],
        },
    }


def validate_resume_equivalence(
    *,
    config_path: Path,
    repository_root: Path,
    runtime_root: Path,
    report_path: Path | None = None,
) -> dict[str, Any]:
    """Run the TD-11 uninterrupted versus interrupted/resumed acceptance gate."""

    if runtime_root.exists():
        raise TrainingContractError(
            f"TD-11 runtime root already exists: {runtime_root}"
        )
    config = load_training_config(config_path)
    maximum = int(config["optimization"]["max_optimizer_steps"])
    if maximum < 2:
        raise TrainingContractError(
            "TD-11 acceptance requires at least two optimizer steps"
        )
    interruption_step = maximum // 2

    baseline = run_training(
        config_path=config_path,
        repository_root=repository_root,
        output_dir=runtime_root / "uninterrupted-run",
        dry_run=False,
        checkpoint_root=runtime_root / "uninterrupted-checkpoints",
    )
    interrupted = run_training(
        config_path=config_path,
        repository_root=repository_root,
        output_dir=runtime_root / "interrupted-run",
        dry_run=False,
        checkpoint_root=runtime_root / "resumed-checkpoints",
        stop_after_optimizer_steps=interruption_step,
    )
    resume_checkpoint = (
        runtime_root / "resumed-checkpoints" / f"step-{interruption_step:08d}"
    )
    resumed = run_training(
        config_path=config_path,
        repository_root=repository_root,
        output_dir=runtime_root / "resumed-run",
        dry_run=False,
        checkpoint_root=runtime_root / "resumed-checkpoints",
        resume_from=resume_checkpoint,
    )
    baseline_checkpoint = (
        runtime_root / "uninterrupted-checkpoints" / f"step-{maximum:08d}"
    )
    resumed_checkpoint = (
        runtime_root / "resumed-checkpoints" / f"step-{maximum:08d}"
    )
    baseline_manifest = validate_checkpoint(baseline_checkpoint)
    resumed_manifest = validate_checkpoint(resumed_checkpoint)

    def events(report: Mapping[str, Any]) -> list[dict[str, Any]]:
        return read_jsonl(Path(str(report["output_root"])) / report["events"]["path"])

    def payload_hashes(checkpoint: Path) -> dict[str, str]:
        manifest = json.loads(
            (checkpoint / CHECKPOINT_MANIFEST).read_text(encoding="utf-8")
        )
        return {
            str(record["path"]): str(record["sha256"])
            for record in manifest["files"]
        }

    baseline_trace = semantic_trace_sha256(events(baseline))
    resumed_trace = semantic_trace_sha256([*events(interrupted), *events(resumed)])
    comparisons = {
        "semantic_trace": baseline_trace == resumed_trace,
        "final_train_loss": baseline["result"]["final_train_loss"]
        == resumed["result"]["final_train_loss"],
        "mean_train_loss": baseline["result"]["mean_train_loss"]
        == resumed["result"]["mean_train_loss"],
        "optimizer_steps": baseline["result"]["optimizer_steps"]
        == resumed["result"]["optimizer_steps"],
        "micro_steps": baseline["result"]["micro_steps"]
        == resumed["result"]["micro_steps"],
        "sampler_state": config_sha256(baseline["result"]["sampler_state"])
        == config_sha256(resumed["result"]["sampler_state"]),
        "checkpoint_payloads": payload_hashes(baseline_checkpoint)
        == payload_hashes(resumed_checkpoint),
    }
    if not all(comparisons.values()):
        raise TrainingContractError(
            f"TD-11 exact resume comparison failed: {comparisons}"
        )
    report = {
        "schema_version": 1,
        "status": "complete",
        "task": "TD-11",
        "training_config": {
            "path": config_path.relative_to(repository_root).as_posix(),
            "file_sha256": sha256_file(config_path),
            "canonical_sha256": config_sha256(config),
        },
        "interruption_step": interruption_step,
        "final_step": maximum,
        "runs": {
            "uninterrupted": {
                "output_root": baseline["output_root"],
                "events_sha256": baseline["events"]["sha256"],
                "final_loss": baseline["result"]["final_train_loss"],
            },
            "interrupted": {
                "output_root": interrupted["output_root"],
                "events_sha256": interrupted["events"]["sha256"],
                "status": interrupted["status"],
            },
            "resumed": {
                "output_root": resumed["output_root"],
                "events_sha256": resumed["events"]["sha256"],
                "final_loss": resumed["result"]["final_train_loss"],
            },
        },
        "comparison": {
            "status": "exact",
            "checks": comparisons,
            "semantic_trace_sha256": baseline_trace,
        },
        "final_checkpoints": {
            "uninterrupted": {
                "path": str(baseline_checkpoint),
                "manifest_sha256": sha256_file(
                    baseline_checkpoint / CHECKPOINT_MANIFEST
                ),
                "identity_sha256": baseline_manifest["identity_sha256"],
                "payloads": payload_hashes(baseline_checkpoint),
            },
            "resumed": {
                "path": str(resumed_checkpoint),
                "manifest_sha256": sha256_file(
                    resumed_checkpoint / CHECKPOINT_MANIFEST
                ),
                "identity_sha256": resumed_manifest["identity_sha256"],
                "payloads": payload_hashes(resumed_checkpoint),
            },
        },
        "automated_gates": {
            "fault_injection_points": [
                "after_model",
                "after_optimizer",
                "before_manifest",
                "after_manifest_before_publish",
            ],
            "corrupt_incomplete_extra_identity_path_link_rejection": True,
            "retention_requires_newest_validation": True,
        },
    }
    if report_path is not None:
        _atomic_json(report_path, report)
    return report


def _linear_scheduler(optimizer: object, *, warmup_steps: int, total_steps: int) -> object:
    import torch

    def scale(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        return max(
            0.0,
            float(total_steps - step) / float(max(1, total_steps - warmup_steps)),
        )

    return torch.optim.lr_scheduler.LambdaLR(optimizer, scale)


def _autocast_context(device: str, precision: str):
    import torch

    if device != "cuda" or precision == "fp32":
        return contextlib.nullcontext()
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def _loss(output: object, labels: object, label_smoothing: float) -> object:
    import torch.nn.functional as functional

    if label_smoothing == 0.0:
        return output.loss
    return functional.cross_entropy(
        output.logits.reshape(-1, output.logits.shape[-1]),
        labels.reshape(-1),
        ignore_index=-100,
        label_smoothing=label_smoothing,
    )


def _move_batch(
    batch: Mapping[str, Any], device: object, *, non_blocking: bool = False
) -> dict[str, Any]:
    return {
        name: tensor.to(device, non_blocking=non_blocking)
        for name, tensor in model_inputs(batch).items()
    }


def _validate_gradients(model: object) -> None:
    import torch

    found = False
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        found = True
        if not bool(torch.isfinite(parameter.grad).all().item()):
            raise TrainingContractError("training produced a NaN/Inf gradient")
    if not found:
        raise TrainingContractError("training produced no gradients")


def _clip_and_validate_gradients(
    model: object, maximum: float, *, mode: str
) -> object:
    import torch

    if mode == "per_parameter":
        _validate_gradients(model)
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), maximum)
        if not bool(torch.isfinite(gradient_norm).item()):
            raise TrainingContractError("gradient norm is NaN/Inf")
        return gradient_norm
    parameters = [parameter for parameter in model.parameters() if parameter.grad is not None]
    if not parameters:
        raise TrainingContractError("training produced no gradients")
    try:
        return torch.nn.utils.clip_grad_norm_(
            parameters,
            maximum,
            error_if_nonfinite=True,
            foreach=True,
        )
    except RuntimeError as exc:
        if "non-finite" in str(exc).lower():
            raise TrainingContractError("training produced a NaN/Inf gradient") from exc
        raise


def seed_training(seed: int) -> None:
    """Seed every RNG used by the locked training stack."""

    import numpy
    import torch

    random.seed(seed)
    numpy.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = False


def evaluate_dev(
    *,
    model: object,
    dataset: RouteDataset,
    encoder: BatchEncoder,
    config: Mapping[str, Any],
    device: object,
) -> dict[str, Any]:
    import torch

    resource = config["resource_profile"]
    optimization = config["optimization"]
    pipeline = config.get("input_pipeline", {})
    sampler = DeterministicRouteSampler(
        dataset,
        config["data"]["route_weights"],
        int(config["identity"]["seed"]) + 1,
    )
    losses: list[float] = []
    routes: Counter[str] = Counter()
    samples = 0
    tokens = 0
    start = time.perf_counter()
    model.eval()
    with torch.no_grad():
        for _ in range(int(optimization["validation_batches"])):
            selections = sampler.next_batch(int(resource["micro_batch_size"]))
            batch = encoder([selection.record for selection in selections])
            moved = _move_batch(
                batch,
                device,
                non_blocking=bool(pipeline.get("non_blocking_transfer", False)),
            )
            with _autocast_context(resource["device"], resource["precision"]):
                output = model(**moved)
                loss = _loss(output, moved["labels"], float(optimization["label_smoothing"]))
            if loss is None or not bool(torch.isfinite(loss).item()):
                raise TrainingContractError("dev evaluation produced a NaN/Inf loss")
            losses.append(float(loss.detach().cpu().item()))
            routes.update(batch["routes"])
            samples += len(batch["routes"])
            tokens += int(batch["attention_mask"].sum().item())
            tokens += int((batch["labels"] != -100).sum().item())
    model.train()
    return {
        "loss": sum(losses) / len(losses),
        "batches": len(losses),
        "samples": samples,
        "tokens": tokens,
        "route_counts": dict(sorted(routes.items())),
        "wall_time_seconds": time.perf_counter() - start,
    }


def execute_training(
    *,
    model: object,
    tokenizer: object,
    tokenizer_path: Path,
    train_dataset: RouteDataset,
    dev_dataset: RouteDataset,
    config: Mapping[str, Any],
    logger: JsonlRunLogger,
    resume_loader: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None = None,
    checkpoint_callback: Callable[[Mapping[str, Any]], None] | None = None,
    stop_after_optimizer_steps: int | None = None,
    input_cache_root: Path | None = None,
) -> dict[str, Any]:
    """Execute the bounded TD-10 loop without checkpoint persistence."""

    import torch

    resource = config["resource_profile"]
    optimization = config["optimization"]
    pipeline = config.get("input_pipeline", {})
    gpu_optimization = config.get("gpu_optimization", {})
    logging_mode = str(config.get("logging", {}).get("mode", "full"))
    seed_training(int(config["identity"]["seed"]))
    policy = EncodingPolicy(
        max_source_length=int(resource["max_source_length"]),
        max_target_length=int(resource["max_target_length"]),
    )
    encoded_cache = None
    text_cache = None
    input_pipeline_report: dict[str, Any] = {
        "mode": str(pipeline.get("mode", "on_demand")),
        "cache_mode": str(pipeline.get("cache_mode", "memory")),
        "mapping_cache": "per_tokenizer_instance",
    }
    if pipeline.get("mode") == "preencode_memory":
        cache_start = time.perf_counter()
        cache_mode = str(pipeline.get("cache_mode", "memory"))
        if cache_mode == "persistent":
            if input_cache_root is None:
                raise TrainingContractError(
                    "persistent input cache requires an input cache root"
                )
            text_cache = load_or_build_text_encoding_cache(
                datasets=(train_dataset, dev_dataset),
                tokenizer=tokenizer,
                tokenizer_path=tokenizer_path,
                tokenizer_manifest_sha256=config["identity"][
                    "tokenizer_manifest_sha256"
                ],
                workers=int(pipeline["preencode_workers"]),
                cache_root=input_cache_root,
            )
        if text_cache is not None:
            if text_cache.estimated_bytes > int(pipeline["memory_budget_mib"]) * MIB:
                raise TrainingContractError("text input cache exceeds its memory budget")
            input_pipeline_report.update(
                {
                    "workers": int(pipeline["preencode_workers"]),
                    "records": train_dataset.records + dev_dataset.records,
                    "estimated_bytes": text_cache.estimated_bytes,
                    "identity_sha256": text_cache.identity_sha256,
                    "component_identity_sha256": [text_cache.identity_sha256],
                    "wall_time_seconds": time.perf_counter() - cache_start,
                }
            )
            input_pipeline_report["text_cache"] = {
                "identity_sha256": text_cache.identity_sha256,
                "source": text_cache.source,
                "directory": str(text_cache.directory.resolve()),
                "unique_language_texts": len(text_cache.encodings),
                "token_ids": text_cache.token_ids,
                "payload_bytes": text_cache.payload_bytes,
                "estimated_bytes": text_cache.estimated_bytes,
            }
        else:
            cache_arguments = {
                "tokenizer": tokenizer,
                "tokenizer_path": tokenizer_path,
                "tokenizer_manifest_sha256": config["identity"][
                    "tokenizer_manifest_sha256"
                ],
                "policy": policy,
                "workers": int(pipeline["preencode_workers"]),
                "memory_budget_mib": int(pipeline["memory_budget_mib"]),
            }
            train_cache = build_encoded_sample_cache(
                dataset=train_dataset,
                **cache_arguments,
            )
            dev_cache = build_encoded_sample_cache(
                dataset=dev_dataset,
                **cache_arguments,
            )
            encoded_cache = EncodedSampleCache.merge(train_cache, dev_cache)
            if encoded_cache.estimated_bytes > int(pipeline["memory_budget_mib"]) * MIB:
                raise TrainingContractError("merged input cache exceeds its memory budget")
            input_pipeline_report.update(
                {
                    "workers": int(pipeline["preencode_workers"]),
                    "records": len(encoded_cache.samples_by_id),
                    "estimated_bytes": encoded_cache.estimated_bytes,
                    "identity_sha256": encoded_cache.identity_sha256,
                    "component_identity_sha256": list(encoded_cache.identities),
                    "wall_time_seconds": time.perf_counter() - cache_start,
                }
            )
        logger.write({"event": "input_cache", **input_pipeline_report})
    device = torch.device(resource["device"])
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model.to(device)
    if resource["gradient_checkpointing"]:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    model.train()
    optimizer_arguments = {
        "lr": float(optimization["learning_rate"]),
        "betas": tuple(float(value) for value in optimization["betas"]),
        "eps": float(optimization["epsilon"]),
        "weight_decay": float(optimization["weight_decay"]),
    }
    if gpu_optimization.get("fused_adamw", False):
        optimizer_arguments["fused"] = True
    optimizer = torch.optim.AdamW(model.parameters(), **optimizer_arguments)
    scheduler = _linear_scheduler(
        optimizer,
        warmup_steps=int(optimization["warmup_steps"]),
        total_steps=int(optimization["max_optimizer_steps"]),
    )
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=device.type == "cuda" and resource["precision"] == "fp16",
    )
    sampler = DeterministicRouteSampler(
        train_dataset,
        config["data"]["route_weights"],
        int(config["identity"]["seed"]),
    )
    pool_batches = int(pipeline.get("length_bucket_pool_batches", 1))
    if pool_batches > 1:
        if text_cache is None:
            raise TrainingContractError(
                "length bucketing requires a loaded persistent text cache"
            )
        sampler = DeterministicLengthBucketSampler(
            sampler,
            text_cache,
            pool_batches,
        )
    optimizer_step = 0
    micro_step = 0
    consumed_samples = 0
    consumed_tokens = 0
    route_counts: Counter[str] = Counter()
    token_audit: Counter[str] = Counter()
    losses: list[float] = []
    if resume_loader is not None:
        restored = resume_loader(
            {
                "model": model,
                "optimizer": optimizer,
                "scheduler": scheduler,
                "scaler": scaler,
                "sampler": sampler,
            }
        )
        optimizer_step = int(restored["global_step"])
        micro_step = int(restored["micro_step"])
        consumed_samples = int(restored["consumed_samples"])
        consumed_tokens = int(restored["consumed_tokens"])
        if int(restored["accumulation_phase"]) != 0:
            raise TrainingContractError(
                "the TD-10 loop resumes checkpoints only at optimizer boundaries"
            )
        losses = [float(value) for value in restored.get("loss_history", [])]
        route_counts.update(restored.get("route_counts", {}))
        token_audit.update(restored.get("token_audit", {}))
    if optimizer_step > int(optimization["max_optimizer_steps"]):
        raise TrainingContractError("checkpoint step exceeds max_optimizer_steps")
    if stop_after_optimizer_steps is not None:
        if stop_after_optimizer_steps <= optimizer_step:
            raise TrainingContractError("stop-after step must be after the starting step")
        if stop_after_optimizer_steps > int(optimization["max_optimizer_steps"]):
            raise TrainingContractError("stop-after step exceeds max_optimizer_steps")
    start = time.perf_counter()
    interrupted = False
    with BatchEncoder(
        tokenizer=tokenizer,
        tokenizer_path=tokenizer_path,
        policy=policy,
        workers=int(resource["dataloader_workers"]),
        encoded_cache=encoded_cache,
        text_cache=text_cache,
        pin_memory=bool(pipeline.get("pin_memory", False)),
    ) as encoder:
        while optimizer_step < int(optimization["max_optimizer_steps"]):
            optimizer.zero_grad(set_to_none=True)
            step_losses: list[float] = []
            step_tokens = 0
            step_samples = 0
            step_route_counts: Counter[str] = Counter()
            step_sample_ids: list[str] = []
            for accumulation_phase in range(int(resource["gradient_accumulation_steps"])):
                selections = sampler.next_batch(int(resource["micro_batch_size"]))
                records = [selection.record for selection in selections]
                batch = encoder(records)
                if not batch["routes"]:
                    raise TrainingContractError("training produced an empty batch")
                tokens = int(batch["attention_mask"].sum().item())
                tokens += int((batch["labels"] != -100).sum().item())
                if consumed_tokens + step_tokens + tokens > int(optimization["max_train_tokens"]):
                    raise TrainingContractError(
                        "max_train_tokens would be exceeded before max_optimizer_steps"
                    )
                moved = _move_batch(
                    batch,
                    device,
                    non_blocking=bool(pipeline.get("non_blocking_transfer", False)),
                )
                try:
                    with _autocast_context(resource["device"], resource["precision"]):
                        output = model(**moved)
                        loss = _loss(
                            output,
                            moved["labels"],
                            float(optimization["label_smoothing"]),
                        )
                        scaled_loss = loss / int(resource["gradient_accumulation_steps"])
                    loss_value = float(loss.detach().float().cpu().item())
                    if not math.isfinite(loss_value):
                        raise TrainingContractError("training produced a NaN/Inf loss")
                    scaler.scale(scaled_loss).backward()
                except torch.cuda.OutOfMemoryError as exc:
                    raise TrainingContractError(
                        "CUDA OOM; formal training does not retry or mutate the profile"
                    ) from exc
                step_losses.append(loss_value)
                step_tokens += tokens
                step_samples += len(batch["routes"])
                step_route_counts.update(batch["routes"])
                if logging_mode != "performance":
                    step_sample_ids.extend(batch["sample_ids"])
                for statistics in batch["route_statistics"].values():
                    for name in (
                        "source_original_tokens",
                        "source_used_tokens",
                        "source_truncated_tokens",
                        "target_original_tokens",
                        "target_used_tokens",
                        "target_truncated_tokens",
                    ):
                        token_audit[name] += int(statistics[name])
                micro_event = {
                    "event": "micro_step",
                    "micro_step": micro_step,
                    "optimizer_step": optimizer_step,
                    "accumulation_phase": accumulation_phase,
                    "loss": loss_value,
                    "tokens": tokens,
                    "samples": len(batch["routes"]),
                }
                if logging_mode == "compact":
                    selection_trace = [
                        {
                            "route": selection.route,
                            "route_epoch": selection.route_epoch,
                            "route_position": selection.route_position,
                        }
                        for selection in selections
                    ]
                    micro_event.update(
                        {
                            "route_statistics": batch["route_statistics"],
                            "route_counts": dict(Counter(batch["routes"])),
                            "batch_trace_sha256": config_sha256(
                                {
                                    "sample_ids": batch["sample_ids"],
                                    "sample_group_ids": batch["sample_group_ids"],
                                    "sampler": selection_trace,
                                    "route_statistics": batch["route_statistics"],
                                }
                            ),
                        }
                    )
                elif logging_mode == "full":
                    selection_trace = [
                        {
                            "route": selection.route,
                            "route_epoch": selection.route_epoch,
                            "route_position": selection.route_position,
                        }
                        for selection in selections
                    ]
                    micro_event.update(
                        {
                            "route_statistics": batch["route_statistics"],
                            "routes": batch["routes"],
                            "sample_ids": batch["sample_ids"],
                            "sample_group_ids": batch["sample_group_ids"],
                            "sampler": selection_trace,
                        }
                    )
                logger.write(micro_event)
                micro_step += 1
            if scaler.is_enabled():
                scaler.unscale_(optimizer)
            gradient_norm = _clip_and_validate_gradients(
                model,
                float(optimization["max_grad_norm"]),
                mode=str(
                    gpu_optimization.get("gradient_validation", "per_parameter")
                ),
            )
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer_step += 1
            consumed_samples += step_samples
            consumed_tokens += step_tokens
            route_counts.update(step_route_counts)
            mean_loss = sum(step_losses) / len(step_losses)
            losses.append(mean_loss)
            elapsed = time.perf_counter() - start
            step_event = {
                "event": "optimizer_step",
                "optimizer_step": optimizer_step,
                "micro_step": micro_step,
                "loss": mean_loss,
                "learning_rate": float(scheduler.get_last_lr()[0]),
                "gradient_norm": float(gradient_norm.detach().cpu().item()),
                "tokens": step_tokens,
                "samples": step_samples,
                "consumed_tokens": consumed_tokens,
                "consumed_samples": consumed_samples,
                "tokens_per_second": consumed_tokens / max(elapsed, 1e-9),
                "samples_per_second": consumed_samples / max(elapsed, 1e-9),
                "wall_time_seconds": elapsed,
                "route_counts": dict(sorted(step_route_counts.items())),
                "checkpoint_due": optimizer_step % int(optimization["checkpoint_frequency"]) == 0,
            }
            if logging_mode == "full":
                step_event["sample_ids"] = step_sample_ids
            elif logging_mode == "compact":
                step_event["sample_ids_sha256"] = config_sha256(step_sample_ids)
            if device.type == "cuda":
                step_event["peak_device_memory_bytes"] = int(
                    torch.cuda.max_memory_allocated(device)
                )
                step_event["peak_device_reserved_bytes"] = int(
                    torch.cuda.max_memory_reserved(device)
                )
            logger.write(step_event)
            if optimizer_step % int(optimization["validation_frequency"]) == 0:
                dev = evaluate_dev(
                    model=model,
                    dataset=dev_dataset,
                    encoder=encoder,
                    config=config,
                    device=device,
                )
                logger.write(
                    {"event": "validation", "optimizer_step": optimizer_step, **dev}
                )
            if (
                checkpoint_callback is not None
                and optimizer_step % int(optimization["checkpoint_frequency"]) == 0
            ):
                checkpoint_start = time.perf_counter()
                checkpoint_callback(
                    {
                        "model": model,
                        "optimizer": optimizer,
                        "scheduler": scheduler,
                        "scaler": scaler,
                        "sampler": sampler,
                        "trainer_state": {
                            "global_step": optimizer_step,
                            "micro_step": micro_step,
                            "epoch": max(sampler.epochs.values()),
                            "consumed_samples": consumed_samples,
                            "consumed_tokens": consumed_tokens,
                            "accumulation_phase": 0,
                            "loss_history": list(losses),
                            "route_counts": dict(route_counts),
                            "token_audit": dict(token_audit),
                        },
                    }
                )
                logger.write(
                    {
                        "event": "checkpoint",
                        "optimizer_step": optimizer_step,
                        "wall_time_seconds": time.perf_counter() - checkpoint_start,
                    }
                )
                logger.flush()
            if (
                stop_after_optimizer_steps is not None
                and optimizer_step >= stop_after_optimizer_steps
            ):
                interrupted = optimizer_step < int(optimization["max_optimizer_steps"])
                break
    elapsed = time.perf_counter() - start
    return {
        "status": "interrupted" if interrupted else "complete",
        "optimizer_steps": optimizer_step,
        "micro_steps": micro_step,
        "consumed_samples": consumed_samples,
        "consumed_tokens": consumed_tokens,
        "mean_train_loss": sum(losses) / len(losses),
        "final_train_loss": losses[-1],
        "route_counts": dict(sorted(route_counts.items())),
        "token_audit": {
            **dict(token_audit),
            "source_truncation_rate": (
                token_audit["source_truncated_tokens"]
                / max(1, token_audit["source_original_tokens"])
            ),
            "target_truncation_rate": (
                token_audit["target_truncated_tokens"]
                / max(1, token_audit["target_original_tokens"])
            ),
        },
        "sampler_state": sampler.state_dict(),
        "wall_time_seconds": elapsed,
        "tokens_per_second": consumed_tokens / max(elapsed, 1e-9),
        "samples_per_second": consumed_samples / max(elapsed, 1e-9),
        "peak_device_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
        ),
        "peak_device_reserved_bytes": (
            int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else None
        ),
        "process_memory": process_memory(),
        "exception_skips": 0,
        "input_pipeline": input_pipeline_report,
        "semantic_trace_sha256": semantic_trace_sha256(logger.events),
    }


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    atomic_write_json(path, payload, sort_keys=True, allow_nan=True)


def git_identity(repository_root: Path) -> dict[str, Any]:
    def run(*arguments: str) -> str:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        return completed.stdout.strip()

    try:
        return {
            "commit": run("rev-parse", "HEAD"),
            "dirty": bool(run("status", "--porcelain")),
        }
    except (OSError, subprocess.CalledProcessError) as exc:
        raise TrainingContractError(f"cannot record Git identity: {exc}") from exc


def validate_run_inputs(
    config: Mapping[str, Any], repository_root: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    identity = config["identity"]
    student_path = repository_root / identity["student_config"]
    if sha256_file(student_path) != identity["student_config_file_sha256"]:
        raise TrainingContractError("student config file SHA-256 changed")
    student_config = load_student_config(student_path)
    if config_sha256(student_config) != identity["student_config_canonical_sha256"]:
        raise TrainingContractError("student config canonical SHA-256 changed")
    if student_config["tokenizer"]["artifact_manifest_sha256"] != identity["tokenizer_manifest_sha256"]:
        raise TrainingContractError("training tokenizer identity differs from student config")
    data = config["data"]
    verified = {}
    for path_field, hash_field in (
        ("train_path", "train_sha256"),
        ("dev_path", "dev_sha256"),
        ("manifest_path", "manifest_sha256"),
    ):
        path = repository_root / data[path_field]
        digest = sha256_file(path)
        if digest != data[hash_field]:
            raise TrainingContractError(f"{path_field} SHA-256 changed")
        verified[path_field] = {"path": data[path_field], "sha256": digest, "bytes": path.stat().st_size}
    return student_config, verified


def prepare_training_run(
    *, config_path: Path, repository_root: Path
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    config = load_training_config(config_path)
    expected_allocator = configure_cuda_allocator(config)
    student_config, inputs = validate_run_inputs(config, repository_root)
    runtime = probe_runtime(config["resource_profile"])
    if (
        expected_allocator is not None
        and runtime.get("cuda_allocator_backend") != expected_allocator
    ):
        raise TrainingContractError(
            "configured CUDA allocator was not active before runtime probing"
        )
    pipeline = config.get("input_pipeline", {})
    logical_processors = runtime.get("host_logical_processors")
    if (
        pipeline.get("mode") == "preencode_memory"
        and isinstance(logical_processors, int)
        and int(pipeline["preencode_workers"]) > logical_processors
    ):
        raise TrainingContractError(
            "pre-encoding worker count exceeds available logical processors"
        )
    report = {
        "schema_version": TRAINING_SCHEMA_VERSION,
        "status": "validated",
        "training_config": {
            "path": config_path.relative_to(repository_root).as_posix(),
            "file_sha256": sha256_file(config_path),
            "canonical_sha256": config_sha256(config),
        },
        "student_config_canonical_sha256": config_sha256(student_config),
        "inputs": inputs,
        "runtime": runtime,
        "git": git_identity(repository_root),
        "command": list(sys.argv),
    }
    return config, student_config, report


def run_training(
    *,
    config_path: Path,
    repository_root: Path,
    output_dir: Path | None,
    dry_run: bool,
    checkpoint_root: Path | None = None,
    resume_from: Path | None = None,
    stop_after_optimizer_steps: int | None = None,
    input_cache_root: Path | None = None,
) -> dict[str, Any]:
    config, student_config, report = prepare_training_run(
        config_path=config_path, repository_root=repository_root
    )
    if dry_run:
        return {**report, "status": "dry_run_complete"}
    if output_dir is None:
        raise TrainingContractError("a non-dry training run requires an output directory")
    if output_dir.exists():
        raise TrainingContractError(f"training output already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    manifest_path = output_dir / "run-manifest.json"
    report = {**report, "output_root": str(output_dir.resolve())}
    if checkpoint_root is not None:
        report["checkpoint_root"] = str(checkpoint_root.resolve())
    if resume_from is not None:
        report["resume_from"] = str(resume_from.resolve())
    if input_cache_root is not None:
        report["input_cache_root"] = str(input_cache_root.resolve())
    running = {**report, "status": "running"}
    _atomic_json(manifest_path, running)
    try:
        data = config["data"]
        train_dataset = load_route_dataset(
            repository_root / data["train_path"],
            expected_sha256=data["train_sha256"],
            split="train",
            max_records_per_route=int(data["train_max_records_per_route"]),
        )
        dev_dataset = load_route_dataset(
            repository_root / data["dev_path"],
            expected_sha256=data["dev_sha256"],
            split="dev",
            max_records_per_route=int(data["dev_max_records_per_route"]),
        )
        tokenizer, tokenizer_identity = load_frozen_tokenizer(student_config, repository_root)
        model, alignment = build_student(student_config, tokenizer)
        tokenizer_path = repository_root / student_config["tokenizer"]["path"]
        checkpoint_identity = None
        created_checkpoints: list[Path] = []
        resume_loader = None
        checkpoint_callback = None
        if checkpoint_root is not None or resume_from is not None:
            from mvp_checkpoint import (
                build_checkpoint_identity,
                load_checkpoint,
                prune_after_validated_publish,
                save_checkpoint,
            )

            checkpoint_identity = build_checkpoint_identity(
                repository_root=repository_root,
                training_report=report,
                training_config=config,
            )
            if checkpoint_root is not None:
                def publish_checkpoint(context: Mapping[str, Any]) -> None:
                    newest = save_checkpoint(
                        checkpoint_root,
                        model=context["model"],
                        optimizer=context["optimizer"],
                        scheduler=context["scheduler"],
                        scaler=context["scaler"],
                        sampler=context["sampler"],
                        trainer_state=context["trainer_state"],
                        identity=checkpoint_identity,
                    )
                    created_checkpoints.append(newest)
                    keep_last = config["optimization"].get(
                        "checkpoint_retention"
                    )
                    if keep_last is not None:
                        removed = set(
                            prune_after_validated_publish(
                                checkpoint_root,
                                newest_checkpoint=newest,
                                expected_identity=checkpoint_identity,
                                keep_last=int(keep_last),
                            )
                        )
                        created_checkpoints[:] = [
                            path for path in created_checkpoints if path not in removed
                        ]

                checkpoint_callback = publish_checkpoint
            if resume_from is not None:
                def restore_checkpoint(context: Mapping[str, Any]) -> Mapping[str, Any]:
                    return load_checkpoint(
                        resume_from,
                        model=context["model"],
                        optimizer=context["optimizer"],
                        scheduler=context["scheduler"],
                        scaler=context["scaler"],
                        sampler=context["sampler"],
                        expected_identity=checkpoint_identity,
                    )

                resume_loader = restore_checkpoint
        logging = config.get("logging", {})
        with JsonlRunLogger(
            output_dir / "events.jsonl",
            flush_frequency=int(logging.get("flush_frequency", 1)),
        ) as logger:
            result = execute_training(
                model=model,
                tokenizer=tokenizer,
                tokenizer_path=tokenizer_path,
                train_dataset=train_dataset,
                dev_dataset=dev_dataset,
                config=config,
                logger=logger,
                resume_loader=resume_loader,
                checkpoint_callback=checkpoint_callback,
                stop_after_optimizer_steps=stop_after_optimizer_steps,
                input_cache_root=input_cache_root,
            )
        for path_field, hash_field in (("train_path", "train_sha256"), ("dev_path", "dev_sha256")):
            if sha256_file(repository_root / data[path_field]) != data[hash_field]:
                raise TrainingContractError(f"{path_field} changed during training")
        complete = {
            **report,
            "status": result["status"],
            "tokenizer": tokenizer_identity,
            "model_alignment": alignment,
            "datasets": {
                "train_records": train_dataset.records,
                "train_selection_sha256": train_dataset.selection_sha256,
                "dev_records": dev_dataset.records,
                "dev_selection_sha256": dev_dataset.selection_sha256,
            },
            "result": result,
            "checkpoints": [
                {
                    "path": str(path.resolve()),
                    "manifest_sha256": sha256_file(path / "checkpoint-manifest.json"),
                }
                for path in created_checkpoints
            ],
            "events": {
                "path": "events.jsonl",
                "bytes": (output_dir / "events.jsonl").stat().st_size,
                "sha256": sha256_file(output_dir / "events.jsonl"),
            },
        }
        _atomic_json(manifest_path, complete)
        return complete
    except BaseException as exc:
        failed = {
            **report,
            "status": "failed",
            "error": {"type": type(exc).__name__, "message": str(exc)},
        }
        _atomic_json(manifest_path, failed)
        raise
