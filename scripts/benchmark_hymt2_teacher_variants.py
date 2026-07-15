"""Benchmark locked Hy-MT2 teacher variants with common prompts and decode settings."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping

import yaml


ROOT = Path(__file__).resolve().parents[1]


class BenchmarkError(RuntimeError):
    """Raised when a teacher benchmark contract or runtime check fails."""


def load_config(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise BenchmarkError("unsupported teacher benchmark config")
    variants = value.get("variants")
    required_variants = {"original-bf16", "bnb-int8", "gguf-q8"}
    if not isinstance(variants, dict) or set(variants) != required_variants:
        raise BenchmarkError(
            "benchmark config must define original-bf16, bnb-int8, and gguf-q8"
        )
    benchmark = value.get("benchmark")
    if not isinstance(benchmark, dict) or len(benchmark.get("probes", [])) != 5:
        raise BenchmarkError("benchmark must define five common probes")
    if benchmark.get("quality_reference_variant") != "original-bf16":
        raise BenchmarkError("quality reference must be the original BF16 model")
    return value


def load_lock(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise BenchmarkError("unsupported teacher benchmark lock")
    artifacts = value.get("artifacts")
    required = {"original-source", "gguf-q8", "llama.cpp-cuda"}
    if not isinstance(artifacts, dict) or set(artifacts) != required:
        raise BenchmarkError("benchmark lock has an unexpected artifact set")
    return value


def resolve_runtime_base(config: Mapping[str, Any]) -> Path:
    runtime = config["runtime"]
    override = os.environ.get(str(runtime["override_env"]))
    if override:
        root = Path(override).expanduser()
        if not root.is_absolute():
            raise BenchmarkError(f"{runtime['override_env']} must be absolute")
        return root.resolve()
    relative = PurePosixPath(str(runtime["default_root"]))
    if relative.is_absolute() or ".." in relative.parts:
        raise BenchmarkError("default benchmark runtime root must be repository-relative")
    return (ROOT / relative).resolve()


def variant_root(runtime_base: Path, variant: Mapping[str, Any]) -> Path:
    relative = PurePosixPath(str(variant["runtime_subdir"]))
    if relative.is_absolute() or ".." in relative.parts:
        raise BenchmarkError("variant runtime_subdir must be relative")
    return (runtime_base / relative).resolve()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def verify_locked_artifact(snapshot: Path, artifact: Mapping[str, Any]) -> None:
    files = artifact.get("files")
    if not isinstance(files, list) or len(files) != int(artifact.get("file_count", -1)):
        raise BenchmarkError("locked artifact file count is invalid")
    expected = {str(item["path"]): item for item in files}
    found = {path.name for path in snapshot.iterdir() if path.is_file()}
    if found != set(expected):
        raise BenchmarkError(
            f"artifact file set mismatch: missing={sorted(set(expected) - found)}, "
            f"unexpected={sorted(found - set(expected))}"
        )
    total = 0
    for name, item in expected.items():
        path = snapshot / name
        size = path.stat().st_size
        total += size
        if size != int(item["bytes"]) or sha256_file(path) != str(item["sha256"]):
            raise BenchmarkError(f"artifact identity mismatch for {name}")
    if total != int(artifact["total_bytes"]):
        raise BenchmarkError("locked artifact total bytes mismatch")


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


def gpu_used_mib() -> int:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return int(result.stdout.strip().splitlines()[0])


def summarize_probes(probes: list[dict[str, Any]]) -> dict[str, Any]:
    latencies = [float(item["latency_seconds"]) for item in probes]
    rates = [float(item["tokens_per_second"]) for item in probes]
    return {
        "runs": len(probes),
        "nonempty": sum(bool(str(item.get("output", "")).strip()) for item in probes),
        "mean_latency_seconds": statistics.fmean(latencies),
        "median_latency_seconds": statistics.median(latencies),
        "mean_tokens_per_second": statistics.fmean(rates),
        "median_tokens_per_second": statistics.median(rates),
    }


def _probe_record(
    target_tag: str,
    repeat: int,
    input_tokens: int,
    new_tokens: int,
    elapsed: float,
    output: str,
    reference: str | None,
) -> dict[str, Any]:
    text = output.strip()
    if not text or new_tokens <= 0 or elapsed <= 0:
        raise BenchmarkError(f"invalid benchmark output for {target_tag}")
    return {
        "target_tag": target_tag,
        "repeat": repeat,
        "input_tokens": input_tokens,
        "new_tokens": new_tokens,
        "latency_seconds": elapsed,
        "tokens_per_second": new_tokens / elapsed,
        "output": text,
        "output_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "matches_expected_smoke_output": (
            text == reference if reference is not None else None
        ),
    }


def benchmark_original_bf16(
    config: Mapping[str, Any], lock: Mapping[str, Any], runtime_base: Path
) -> dict[str, Any]:
    """Benchmark the official unquantized BF16 model used as quality reference."""

    variant = config["variants"]["original-bf16"]
    benchmark = config["benchmark"]
    if str(variant["dtype"]) != "bfloat16":
        raise BenchmarkError("original quality reference must load as torch.bfloat16")
    os.environ.update(
        {
            "PYTHONUTF8": "1",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )

    import psutil
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise BenchmarkError("CUDA is unavailable for original BF16 benchmark")
    torch.cuda.set_device(0)
    torch.cuda.init()
    if transformers.__version__ != str(variant["transformers_version"]):
        raise BenchmarkError("Transformers version does not match benchmark config")
    if torch.__version__ != str(variant["torch_version"]):
        raise BenchmarkError("PyTorch version does not match benchmark config")

    root = variant_root(runtime_base, variant)
    snapshot = root / str(variant["snapshot_subdir"])
    if not snapshot.is_dir():
        raise BenchmarkError(f"base teacher snapshot missing: {snapshot}")
    verify_locked_artifact(snapshot, lock["artifacts"]["original-source"])
    process = psutil.Process()
    baseline_gpu_mib = gpu_used_mib()
    rss_before = process.memory_info().rss
    sampler = ProcessSampler(os.getpid())
    sampler.start()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(0)
    load_started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(
        snapshot,
        local_files_only=bool(variant["local_files_only"]),
        trust_remote_code=bool(variant["trust_remote_code"]),
    )
    model = AutoModelForCausalLM.from_pretrained(
        snapshot,
        dtype=torch.bfloat16,
        device_map=str(variant["device_map"]),
        local_files_only=bool(variant["local_files_only"]),
        trust_remote_code=bool(variant["trust_remote_code"]),
    )
    model.eval()
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - load_started
    static_gpu_mib = gpu_used_mib()
    load_peak_allocated = int(torch.cuda.max_memory_allocated(0))
    load_peak_reserved = int(torch.cuda.max_memory_reserved(0))
    device_map = getattr(model, "hf_device_map", {}) or {}
    serializable_device_map = {
        str(name): str(device) for name, device in device_map.items()
    }
    non_cuda_modules = {
        name: device
        for name, device in serializable_device_map.items()
        if device not in {"0", "cuda", "cuda:0"} and not device.startswith("cuda:")
    }
    parameter_devices = sorted({str(parameter.device) for parameter in model.parameters()})
    input_device = model.get_input_embeddings().weight.device

    def generate(prompt: str, max_new_tokens: int) -> tuple[int, int, float, str]:
        encoded = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        encoded = {name: tensor.to(input_device) for name, tensor in encoded.items()}
        input_tokens = int(encoded["input_ids"].shape[-1])
        torch.manual_seed(int(benchmark["seed"]))
        torch.cuda.manual_seed_all(int(benchmark["seed"]))
        started = time.perf_counter()
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        tokens = generated[0, input_tokens:]
        return input_tokens, int(tokens.shape[-1]), elapsed, tokenizer.decode(
            tokens, skip_special_tokens=True
        ).strip()

    warmup = generate(
        str(benchmark["probes"][0]["prompt"]), int(benchmark["warmup_new_tokens"])
    )
    torch.cuda.reset_peak_memory_stats(0)
    probes: list[dict[str, Any]] = []
    for probe in benchmark["probes"]:
        for repeat in range(int(benchmark["repeats"])):
            record = generate(str(probe["prompt"]), int(benchmark["max_new_tokens"]))
            probes.append(
                _probe_record(
                    str(probe["target_tag"]),
                    repeat,
                    *record,
                    str(probe["expected_smoke_output"]),
                )
            )
    capacity = benchmark["capacity_probe"]
    capacity_values = generate(str(capacity["prompt"]), int(capacity["max_new_tokens"]))
    capacity_record = _probe_record(
        str(capacity["target_tag"]), 0, *capacity_values, None
    )
    generation_peak_allocated = int(torch.cuda.max_memory_allocated(0))
    generation_peak_reserved = int(torch.cuda.max_memory_reserved(0))
    final_gpu_mib = gpu_used_mib()
    sampler.stop()
    return {
        "schema_version": 1,
        "status": "pass",
        "benchmark_id": config["benchmark_id"],
        "variant": "original-bf16",
        "quality_reference": True,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "artifact": {
            "repo_id": variant["repo_id"],
            "revision": variant["revision"],
            "format": variant["artifact_format"],
        },
        "backend": {
            "name": variant["backend"],
            "transformers": transformers.__version__,
            "torch": torch.__version__,
            "dtype": str(model.dtype).removeprefix("torch."),
            "configured_device_map": variant["device_map"],
            "device_map": serializable_device_map,
            "non_cuda_modules": non_cuda_modules,
            "parameter_devices": parameter_devices,
        },
        "load": {"seconds": load_seconds},
        "memory": {
            "model_footprint_bytes": int(model.get_memory_footprint()),
            "process_rss_before_bytes": rss_before,
            "process_rss_after_bytes": process.memory_info().rss,
            "process_peak_rss_bytes": sampler.peak_rss_bytes,
            "gpu_baseline_mib": baseline_gpu_mib,
            "gpu_static_after_load_mib": static_gpu_mib,
            "gpu_static_delta_mib": static_gpu_mib - baseline_gpu_mib,
            "gpu_final_mib": final_gpu_mib,
            "gpu_peak_mib": sampler.peak_gpu_mib,
            "gpu_peak_delta_mib": sampler.peak_gpu_mib - baseline_gpu_mib,
            "samples": sampler.samples,
            "load_peak_allocated_bytes": load_peak_allocated,
            "load_peak_reserved_bytes": load_peak_reserved,
            "generation_peak_allocated_bytes": generation_peak_allocated,
            "generation_peak_reserved_bytes": generation_peak_reserved,
        },
        "warmup": {
            "input_tokens": warmup[0],
            "new_tokens": warmup[1],
            "latency_seconds": warmup[2],
            "output": warmup[3],
        },
        "probes": probes,
        "summary": summarize_probes(probes),
        "capacity_probe": {
            "scope": capacity["scope"],
            "source_characters": capacity["source_characters"],
            **capacity_record,
        },
    }


def benchmark_bnb(
    config: Mapping[str, Any], lock: Mapping[str, Any], runtime_base: Path
) -> dict[str, Any]:
    variant = config["variants"]["bnb-int8"]
    benchmark = config["benchmark"]
    expected_override = str(variant["bnb_cuda_version"])
    if os.environ.get("BNB_CUDA_VERSION") != expected_override:
        raise BenchmarkError(f"BNB_CUDA_VERSION must be {expected_override} before Python starts")
    os.environ.update(
        {
            "PYTHONUTF8": "1",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )

    import bitsandbytes as bnb
    import psutil
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    if not torch.cuda.is_available():
        raise BenchmarkError("CUDA is unavailable for bitsandbytes benchmark")
    if bnb.__version__ != str(variant["bitsandbytes_version"]):
        raise BenchmarkError("bitsandbytes version does not match benchmark config")
    if transformers.__version__ != str(variant["transformers_version"]):
        raise BenchmarkError("Transformers version does not match benchmark config")

    root = variant_root(runtime_base, variant)
    snapshot = root / str(variant["snapshot_subdir"])
    if not snapshot.is_dir():
        raise BenchmarkError(f"base teacher snapshot missing: {snapshot}")
    verify_locked_artifact(snapshot, lock["artifacts"]["original-source"])
    process = psutil.Process()
    baseline_gpu_mib = gpu_used_mib()
    rss_before = process.memory_info().rss
    sampler = ProcessSampler(os.getpid())
    sampler.start()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(0)
    load_started = time.perf_counter()
    quantization = BitsAndBytesConfig(load_in_8bit=bool(variant["load_in_8bit"]))
    tokenizer = AutoTokenizer.from_pretrained(
        snapshot,
        local_files_only=bool(variant["local_files_only"]),
        trust_remote_code=bool(variant["trust_remote_code"]),
    )
    model = AutoModelForCausalLM.from_pretrained(
        snapshot,
        dtype=torch.bfloat16,
        device_map=str(variant["device_map"]),
        quantization_config=quantization,
        local_files_only=bool(variant["local_files_only"]),
        trust_remote_code=bool(variant["trust_remote_code"]),
    )
    model.eval()
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - load_started
    static_gpu_mib = gpu_used_mib()
    load_peak_allocated = int(torch.cuda.max_memory_allocated(0))
    load_peak_reserved = int(torch.cuda.max_memory_reserved(0))
    quantized_modules = sum(
        module.__class__.__name__ == "Linear8bitLt" for module in model.modules()
    )
    if not getattr(model, "is_loaded_in_8bit", False) or quantized_modules == 0:
        raise BenchmarkError("model did not load through bitsandbytes LLM.int8")
    device_map = getattr(model, "hf_device_map", {})
    non_cuda_modules = {
        name: str(device)
        for name, device in device_map.items()
        if str(device) not in {"0", "cuda", "cuda:0"}
    }
    if non_cuda_modules:
        raise BenchmarkError(f"bitsandbytes model used CPU/disk offload: {non_cuda_modules}")
    input_device = model.get_input_embeddings().weight.device

    def generate(prompt: str, max_new_tokens: int) -> tuple[int, int, float, str]:
        encoded = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        encoded = {name: tensor.to(input_device) for name, tensor in encoded.items()}
        input_tokens = int(encoded["input_ids"].shape[-1])
        torch.manual_seed(int(benchmark["seed"]))
        torch.cuda.manual_seed_all(int(benchmark["seed"]))
        started = time.perf_counter()
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        tokens = generated[0, input_tokens:]
        return input_tokens, int(tokens.shape[-1]), elapsed, tokenizer.decode(
            tokens, skip_special_tokens=True
        ).strip()

    warmup = generate(
        str(benchmark["probes"][0]["prompt"]), int(benchmark["warmup_new_tokens"])
    )
    torch.cuda.reset_peak_memory_stats(0)
    probes: list[dict[str, Any]] = []
    for probe in benchmark["probes"]:
        for repeat in range(int(benchmark["repeats"])):
            record = generate(str(probe["prompt"]), int(benchmark["max_new_tokens"]))
            probes.append(
                _probe_record(
                    str(probe["target_tag"]),
                    repeat,
                    *record,
                    str(probe["expected_smoke_output"]),
                )
            )
    capacity = benchmark["capacity_probe"]
    capacity_values = generate(str(capacity["prompt"]), int(capacity["max_new_tokens"]))
    capacity_record = _probe_record(
        str(capacity["target_tag"]), 0, *capacity_values, None
    )
    generation_peak_allocated = int(torch.cuda.max_memory_allocated(0))
    generation_peak_reserved = int(torch.cuda.max_memory_reserved(0))
    final_gpu_mib = gpu_used_mib()
    sampler.stop()
    native_library = getattr(getattr(bnb, "cextension", None), "lib", None)
    native_cdll = getattr(native_library, "_lib", None)
    lib_name = Path(str(getattr(native_cdll, "_name", "unknown"))).name
    expected_library = f"libbitsandbytes_cuda{expected_override}.dll"
    if lib_name.lower() != expected_library.lower():
        raise BenchmarkError(
            f"bitsandbytes loaded {lib_name!r}, expected CUDA override library {expected_library!r}"
        )
    return {
        "schema_version": 1,
        "status": "pass",
        "benchmark_id": config["benchmark_id"],
        "variant": "bnb-int8",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "artifact": {
            "repo_id": variant["repo_id"],
            "revision": variant["revision"],
            "format": variant["artifact_format"],
        },
        "backend": {
            "name": variant["backend"],
            "transformers": transformers.__version__,
            "torch": torch.__version__,
            "bitsandbytes": bnb.__version__,
            "bnb_cuda_version": os.environ["BNB_CUDA_VERSION"],
            "loaded_library": lib_name,
            "load_in_8bit": True,
            "linear_8bit_module_count": quantized_modules,
            "is_loaded_in_8bit": bool(getattr(model, "is_loaded_in_8bit", False)),
            "device_map": {name: str(device) for name, device in device_map.items()},
            "non_cuda_modules": non_cuda_modules,
        },
        "load": {"seconds": load_seconds},
        "memory": {
            "model_footprint_bytes": int(model.get_memory_footprint()),
            "process_rss_before_bytes": rss_before,
            "process_rss_after_bytes": process.memory_info().rss,
            "process_peak_rss_bytes": sampler.peak_rss_bytes,
            "gpu_baseline_mib": baseline_gpu_mib,
            "gpu_static_after_load_mib": static_gpu_mib,
            "gpu_static_delta_mib": static_gpu_mib - baseline_gpu_mib,
            "gpu_final_mib": final_gpu_mib,
            "gpu_peak_mib": sampler.peak_gpu_mib,
            "gpu_peak_delta_mib": sampler.peak_gpu_mib - baseline_gpu_mib,
            "samples": sampler.samples,
            "load_peak_allocated_bytes": load_peak_allocated,
            "load_peak_reserved_bytes": load_peak_reserved,
            "generation_peak_allocated_bytes": generation_peak_allocated,
            "generation_peak_reserved_bytes": generation_peak_reserved,
        },
        "warmup": {
            "input_tokens": warmup[0],
            "new_tokens": warmup[1],
            "latency_seconds": warmup[2],
            "output": warmup[3],
        },
        "probes": probes,
        "summary": summarize_probes(probes),
        "capacity_probe": {
            "scope": capacity["scope"],
            "source_characters": capacity["source_characters"],
            **capacity_record,
        },
    }


class ProcessSampler:
    def __init__(self, process_id: int) -> None:
        self.process_id = process_id
        self.stop_event = threading.Event()
        self.peak_rss_bytes = 0
        self.peak_gpu_mib = 0
        self.samples = 0
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        import psutil

        tracked = psutil.Process(self.process_id)
        while not self.stop_event.wait(0.2):
            with contextlib.suppress(psutil.Error):
                processes = [tracked, *tracked.children(recursive=True)]
                rss = sum(item.memory_info().rss for item in processes if item.is_running())
                self.peak_rss_bytes = max(self.peak_rss_bytes, rss)
            with contextlib.suppress(Exception):
                self.peak_gpu_mib = max(self.peak_gpu_mib, gpu_used_mib())
            self.samples += 1

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=5)


def _http_json(
    url: str, payload: Mapping[str, Any] | None = None, timeout: float = 10
) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="GET" if data is None else "POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        value = json.loads(response.read().decode("utf-8"))
    if not isinstance(value, dict):
        raise BenchmarkError(f"non-object response from {url}")
    return value


def _find_executable(directory: Path, name: str) -> Path:
    matches = list(directory.rglob(name))
    if len(matches) != 1:
        raise BenchmarkError(f"expected exactly one {name} below {directory}, found {len(matches)}")
    return matches[0]


def benchmark_gguf(
    config: Mapping[str, Any], lock: Mapping[str, Any], runtime_base: Path
) -> dict[str, Any]:
    variant = config["variants"]["gguf-q8"]
    benchmark = config["benchmark"]
    root = variant_root(runtime_base, variant)
    model = root / PurePosixPath(str(variant["model_subpath"]))
    verify_locked_artifact(model.parent, lock["artifacts"]["gguf-q8"])
    llama_root = root / PurePosixPath(str(variant["llama_subdir"]))
    server = _find_executable(llama_root, "llama-server.exe")
    host = str(variant["host"])
    port = int(variant["port"])
    base_url = f"http://{host}:{port}"
    baseline_gpu_mib = gpu_used_mib()
    command = [
        str(server),
        "--model",
        str(model),
        "--host",
        host,
        "--port",
        str(port),
        "--ctx-size",
        str(benchmark["context_size"]),
        "--n-gpu-layers",
        str(variant["n_gpu_layers"]),
        "--flash-attn",
        "on" if variant["flash_attention"] else "off",
        "--jinja",
        "--metrics",
    ]
    creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    started = time.perf_counter()
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        creationflags=creationflags,
    )
    logs: list[str] = []

    def collect_logs() -> None:
        assert process.stdout is not None
        for line in process.stdout:
            logs.append(line.rstrip())

    log_thread = threading.Thread(target=collect_logs, daemon=True)
    log_thread.start()
    sampler = ProcessSampler(process.pid)
    sampler.start()
    try:
        deadline = time.monotonic() + 180
        while True:
            if process.poll() is not None:
                raise BenchmarkError(
                    "llama-server exited before readiness: " + "\n".join(logs[-40:])
                )
            try:
                health = _http_json(base_url + "/health", timeout=1)
                if health.get("status") == "ok":
                    break
            except (OSError, urllib.error.URLError, json.JSONDecodeError):
                pass
            if time.monotonic() >= deadline:
                raise BenchmarkError("timed out waiting for llama-server readiness")
            time.sleep(0.2)
        load_seconds = time.perf_counter() - started
        static_gpu_mib = gpu_used_mib()

        def generate(prompt: str, max_new_tokens: int) -> tuple[int, int, float, str]:
            payload = {
                "model": "hymt2-q8",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_new_tokens,
                "temperature": 0,
                "top_p": 1,
                "seed": int(benchmark["seed"]),
                "stream": False,
            }
            request_started = time.perf_counter()
            response = _http_json(base_url + "/v1/chat/completions", payload, timeout=180)
            elapsed = time.perf_counter() - request_started
            try:
                output = str(response["choices"][0]["message"]["content"]).strip()
                usage = response["usage"]
                input_tokens = int(usage["prompt_tokens"])
                new_tokens = int(usage["completion_tokens"])
            except (KeyError, IndexError, TypeError, ValueError) as exc:
                raise BenchmarkError(f"invalid llama-server response: {response}") from exc
            return input_tokens, new_tokens, elapsed, output

        warmup = generate(
            str(benchmark["probes"][0]["prompt"]), int(benchmark["warmup_new_tokens"])
        )
        probes: list[dict[str, Any]] = []
        for probe in benchmark["probes"]:
            for repeat in range(int(benchmark["repeats"])):
                record = generate(str(probe["prompt"]), int(benchmark["max_new_tokens"]))
                probes.append(
                    _probe_record(
                        str(probe["target_tag"]),
                        repeat,
                        *record,
                        str(probe["expected_smoke_output"]),
                    )
                )
        capacity = benchmark["capacity_probe"]
        capacity_values = generate(str(capacity["prompt"]), int(capacity["max_new_tokens"]))
        capacity_record = _probe_record(
            str(capacity["target_tag"]), 0, *capacity_values, None
        )
        final_gpu_mib = gpu_used_mib()
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
        sampler.stop()
        log_thread.join(timeout=5)

    offload_lines = [
        line
        for line in logs
        if "offload" in line.lower() or "cuda" in line.lower() or "buffer size" in line.lower()
    ]
    static_gpu_delta_mib = static_gpu_mib - baseline_gpu_mib
    if static_gpu_delta_mib < 6_000:
        raise BenchmarkError(
            f"llama.cpp did not place the Q8 model on CUDA: static GPU delta is only "
            f"{static_gpu_delta_mib} MiB"
        )
    return {
        "schema_version": 1,
        "status": "pass",
        "benchmark_id": config["benchmark_id"],
        "variant": "gguf-q8",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "artifact": {
            "repo_id": variant["repo_id"],
            "revision": variant["revision"],
            "format": variant["artifact_format"],
            "file": model.name,
            "bytes": model.stat().st_size,
            "sha256": variant["model_sha256"],
        },
        "backend": {
            "name": variant["backend"],
            "release": variant["llama_release"],
            "commit": variant["llama_commit"],
            "cuda": variant["cuda_version"],
            "n_gpu_layers": variant["n_gpu_layers"],
            "flash_attention": bool(variant["flash_attention"]),
            "command": command,
            "offload_evidence": offload_lines,
        },
        "load": {"seconds": load_seconds},
        "memory": {
            "process_peak_rss_bytes": sampler.peak_rss_bytes,
            "gpu_baseline_mib": baseline_gpu_mib,
            "gpu_static_after_load_mib": static_gpu_mib,
            "gpu_static_delta_mib": static_gpu_delta_mib,
            "gpu_final_mib": final_gpu_mib,
            "gpu_peak_mib": sampler.peak_gpu_mib,
            "gpu_peak_delta_mib": sampler.peak_gpu_mib - baseline_gpu_mib,
            "samples": sampler.samples,
        },
        "warmup": {
            "input_tokens": warmup[0],
            "new_tokens": warmup[1],
            "latency_seconds": warmup[2],
            "output": warmup[3],
        },
        "probes": probes,
        "summary": summarize_probes(probes),
        "capacity_probe": {
            "scope": capacity["scope"],
            "source_characters": capacity["source_characters"],
            **capacity_record,
        },
        "server_log_tail": logs[-80:],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/hymt2_teacher_benchmark.yaml"
    )
    parser.add_argument(
        "--lock", type=Path, default=ROOT / "configs/hymt2_teacher_benchmark.lock.json"
    )
    parser.add_argument(
        "--variant",
        required=True,
        choices=("original-bf16", "bnb-int8", "gguf-q8"),
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config = load_config(args.config)
    lock = load_lock(args.lock)
    runtime_base = resolve_runtime_base(config)
    reports = runtime_base / PurePosixPath(str(config["runtime"]["reports_subdir"]))
    output = args.output or reports / f"{args.variant}.json"
    try:
        if args.variant == "original-bf16":
            report = benchmark_original_bf16(config, lock, runtime_base)
        elif args.variant == "bnb-int8":
            report = benchmark_bnb(config, lock, runtime_base)
        else:
            report = benchmark_gguf(config, lock, runtime_base)
    except Exception as exc:
        failure = {
            "schema_version": 1,
            "status": "fail",
            "benchmark_id": config["benchmark_id"],
            "variant": args.variant,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "error": {"type": type(exc).__name__, "message": str(exc)},
        }
        atomic_write_json(output, failure)
        raise
    atomic_write_json(output, report)
    print(f"benchmark status: {report['status']}")
    print(f"report: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
