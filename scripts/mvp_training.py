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
import subprocess
import sys
import tempfile
import threading
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

from freeze_tokenizer_artifact import sha256_file
from model_training_contract import (
    config_sha256,
    directed_routes,
    load_student_config,
)
from mvp_student import (
    DirectionAwareCollator,
    EncodedSample,
    EncodingPolicy,
    StudentContractError,
    build_student,
    encode_parallel_sample,
    load_frozen_tokenizer,
    model_inputs,
)
from tokenizer_utils import reload_tokenizer


TRAINING_SCHEMA_VERSION = 1
MIB = 1024 * 1024
ROUTE_ORDER = tuple(f"{source}->{target}" for source, target in directed_routes())
SHA256_LENGTH = 64


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
    _expect_keys(
        config,
        {"schema_version", "identity", "data", "resource_profile", "optimization"},
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

    optimization = _mapping(config["optimization"], "training.optimization")
    _expect_keys(
        optimization,
        {
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
        },
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


class BatchEncoder:
    """Bounded, ordered tokenizer workers without sampler prefetch."""

    def __init__(
        self,
        *,
        tokenizer: object,
        tokenizer_path: Path,
        policy: EncodingPolicy,
        workers: int,
        tokenizer_loader: Callable[[Path], object] = reload_tokenizer,
    ) -> None:
        self.collator = DirectionAwareCollator(tokenizer, policy)
        self.tokenizer_path = tokenizer_path
        self.policy = policy
        self.workers = workers
        self.tokenizer_loader = tokenizer_loader
        self.local = threading.local()
        self.executor = ThreadPoolExecutor(max_workers=workers) if workers else None

    def _encode(self, record: Mapping[str, Any]) -> EncodedSample:
        tokenizer = getattr(self.local, "tokenizer", None)
        if tokenizer is None:
            tokenizer = self.tokenizer_loader(self.tokenizer_path)
            self.local.tokenizer = tokenizer
        return encode_parallel_sample(tokenizer, record, self.policy)

    def __call__(self, records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        if self.executor is None:
            return self.collator(records)
        encoded = list(self.executor.map(self._encode, records))
        return self.collator.collate_encoded(encoded)

    def close(self) -> None:
        if self.executor is not None:
            self.executor.shutdown(wait=True, cancel_futures=True)

    def __enter__(self) -> "BatchEncoder":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()


class JsonlRunLogger:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path
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
            self.handle.flush()

    def close(self) -> None:
        if self.handle is not None:
            self.handle.flush()
            os.fsync(self.handle.fileno())
            self.handle.close()
            self.handle = None

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
    }
    semantic = [
        {name: value for name, value in event.items() if name not in excluded}
        for event in events
        if event.get("event") != "checkpoint"
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


def _move_batch(batch: Mapping[str, Any], device: object) -> dict[str, Any]:
    return {name: tensor.to(device) for name, tensor in model_inputs(batch).items()}


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
            moved = _move_batch(batch, device)
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
) -> dict[str, Any]:
    """Execute the bounded TD-10 loop without checkpoint persistence."""

    import torch

    resource = config["resource_profile"]
    optimization = config["optimization"]
    seed_training(int(config["identity"]["seed"]))
    device = torch.device(resource["device"])
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model.to(device)
    if resource["gradient_checkpointing"]:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(optimization["learning_rate"]),
        betas=tuple(float(value) for value in optimization["betas"]),
        eps=float(optimization["epsilon"]),
        weight_decay=float(optimization["weight_decay"]),
    )
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
    policy = EncodingPolicy(
        max_source_length=int(resource["max_source_length"]),
        max_target_length=int(resource["max_target_length"]),
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
                moved = _move_batch(batch, device)
                try:
                    with _autocast_context(resource["device"], resource["precision"]):
                        output = model(**moved)
                        loss = _loss(
                            output,
                            moved["labels"],
                            float(optimization["label_smoothing"]),
                        )
                        scaled_loss = loss / int(resource["gradient_accumulation_steps"])
                    if loss is None or not bool(torch.isfinite(loss).item()):
                        raise TrainingContractError("training produced a NaN/Inf loss")
                    scaler.scale(scaled_loss).backward()
                except torch.cuda.OutOfMemoryError as exc:
                    raise TrainingContractError(
                        "CUDA OOM; formal training does not retry or mutate the profile"
                    ) from exc
                loss_value = float(loss.detach().cpu().item())
                step_losses.append(loss_value)
                step_tokens += tokens
                step_samples += len(batch["routes"])
                step_route_counts.update(batch["routes"])
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
                logger.write(
                    {
                        "event": "micro_step",
                        "micro_step": micro_step,
                        "optimizer_step": optimizer_step,
                        "accumulation_phase": accumulation_phase,
                        "loss": loss_value,
                        "tokens": tokens,
                        "samples": len(batch["routes"]),
                        "routes": batch["routes"],
                        "sample_ids": batch["sample_ids"],
                        "sample_group_ids": batch["sample_group_ids"],
                        "sampler": [
                            {
                                "route": selection.route,
                                "route_epoch": selection.route_epoch,
                                "route_position": selection.route_position,
                            }
                            for selection in selections
                        ],
                        "route_statistics": batch["route_statistics"],
                    }
                )
                micro_step += 1
            if scaler.is_enabled():
                scaler.unscale_(optimizer)
            _validate_gradients(model)
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(optimization["max_grad_norm"])
            )
            if not bool(torch.isfinite(gradient_norm).item()):
                raise TrainingContractError("gradient norm is NaN/Inf")
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
                "sample_ids": step_sample_ids,
                "checkpoint_due": optimizer_step % int(optimization["checkpoint_frequency"]) == 0,
            }
            if device.type == "cuda":
                step_event["peak_device_memory_bytes"] = int(
                    torch.cuda.max_memory_allocated(device)
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
        "process_memory": process_memory(),
        "exception_skips": 0,
        "semantic_trace_sha256": semantic_trace_sha256(logger.events),
    }


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


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
    student_config, inputs = validate_run_inputs(config, repository_root)
    runtime = probe_runtime(config["resource_profile"])
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
                save_checkpoint,
            )

            checkpoint_identity = build_checkpoint_identity(
                repository_root=repository_root,
                training_report=report,
                training_config=config,
            )
            if checkpoint_root is not None:
                def publish_checkpoint(context: Mapping[str, Any]) -> None:
                    created_checkpoints.append(
                        save_checkpoint(
                            checkpoint_root,
                            model=context["model"],
                            optimizer=context["optimizer"],
                            scheduler=context["scheduler"],
                            scaler=context["scaler"],
                            sampler=context["sampler"],
                            trainer_state=context["trainer_state"],
                            identity=checkpoint_identity,
                        )
                    )

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
        with JsonlRunLogger(output_dir / "events.jsonl") as logger:
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
