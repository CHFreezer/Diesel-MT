from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from benchmark_hymt2_teacher_variants import (  # noqa: E402
    BenchmarkError,
    _probe_record,
    load_config,
    load_lock,
    resolve_runtime_base,
    summarize_probes,
)


def test_repository_benchmark_config_locks_requested_variants() -> None:
    config = load_config(ROOT / "configs/hymt2_teacher_benchmark.yaml")
    lock = load_lock(ROOT / "configs/hymt2_teacher_benchmark.lock.json")
    original = config["variants"]["original-bf16"]
    assert original["repo_id"] == "tencent/Hy-MT2-7B"
    assert original["dtype"] == "bfloat16"
    assert original["device_map"] == "auto"
    assert original["runtime_subdir"] == "teacher/hymt2-7b-bf16"
    assert config["variants"]["bnb-int8"]["runtime_subdir"] == (
        original["runtime_subdir"]
    )
    assert config["benchmark"]["quality_reference_variant"] == "original-bf16"
    assert all(
        "expected_smoke_output" in probe and "fp8_reference" not in probe
        for probe in config["benchmark"]["probes"]
    )
    assert config["variants"]["bnb-int8"]["bitsandbytes_version"] == "0.49.2"
    assert config["variants"]["bnb-int8"]["bnb_cuda_version"] == "130"
    assert config["variants"]["bnb-int8"]["load_in_8bit"] is True
    gguf = config["variants"]["gguf-q8"]
    assert gguf["model_bytes"] == 7_981_928_896
    assert gguf["model_sha256"] == (
        "58b3ad55dd6f6fa08c695cddc34fb5f8f708a844f78ae10508071914b0ed67c0"
    )
    assert lock["artifacts"]["original-source"]["total_bytes"] == 16_075_624_007
    assert lock["artifacts"]["gguf-q8"]["total_bytes"] == 7_981_972_164


def test_frozen_teacher_selection_uses_original_bf16_quality_baseline() -> None:
    import yaml

    selection = yaml.safe_load(
        (ROOT / "configs/hymt2_teacher_selection.yaml").read_text(encoding="utf-8")
    )
    assert selection["status"] == "frozen"
    assert selection["teacher"]["artifact_repo"] == "tencent/Hy-MT2-7B-GGUF"
    assert selection["teacher"]["revision"] == "ab8472660ac61fac25f1af43fac2599d52a8a775"
    assert selection["teacher"]["sha256"] == (
        "58b3ad55dd6f6fa08c695cddc34fb5f8f708a844f78ae10508071914b0ed67c0"
    )
    assert selection["backend"]["name"] == "llama.cpp-cuda"
    assert selection["evidence"]["quality_reference"] == "original-bf16"
    assert selection["evidence"]["five_tag_reference_matches"] == 10
    assert selection["decision"]["use"] == "sequence-level-distillation-source"
    assert selection["runtime"]["retained_quality_reference_subdir"] == (
        "teacher/hymt2-7b-bf16/snapshot"
    )
    comparison = json.loads(
        (ROOT / "artifacts/model-training/reports/teacher/runtime-comparison.json").read_text(
            encoding="utf-8"
        )
    )
    selection_sha256 = hashlib.sha256(
        (ROOT / "configs/hymt2_teacher_selection.yaml").read_bytes()
    ).hexdigest()
    benchmark_config_sha256 = hashlib.sha256(
        (ROOT / "configs/hymt2_teacher_benchmark.yaml").read_bytes()
    ).hexdigest()
    assert comparison["status"] == "complete"
    assert comparison["common_protocol"]["quality_reference_variant"] == (
        "original-bf16"
    )
    assert comparison["quality_comparison"]["short_probes"] == {
        "bnb_int8_exact_matches": 10,
        "gguf_q8_exact_matches": 10,
        "runs": 10,
    }
    assert comparison["identities"]["teacher_selection_sha256"] == selection_sha256
    assert comparison["identities"]["current_benchmark_config_sha256"] == (
        benchmark_config_sha256
    )
    assert comparison["post_benchmark_storage_layout"][
        "current_shared_weight_subdir"
    ] == "teacher/hymt2-7b-bf16"


def test_benchmark_runtime_override_must_be_absolute(tmp_path: Path) -> None:
    config = load_config(ROOT / "configs/hymt2_teacher_benchmark.yaml")
    name = config["runtime"]["override_env"]
    old = os.environ.get(name)
    try:
        os.environ[name] = str(tmp_path)
        assert resolve_runtime_base(config) == tmp_path.resolve()
        os.environ[name] = "relative"
        with pytest.raises(BenchmarkError, match="must be absolute"):
            resolve_runtime_base(config)
    finally:
        if old is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = old


def test_probe_summary_uses_measured_latency_and_throughput() -> None:
    probes = [
        {
            "latency_seconds": 1.0,
            "tokens_per_second": 4.0,
            "output": "a",
        },
        {
            "latency_seconds": 3.0,
            "tokens_per_second": 8.0,
            "output": "b",
        },
    ]
    summary = summarize_probes(probes)
    assert summary["runs"] == 2
    assert summary["nonempty"] == 2
    assert summary["mean_latency_seconds"] == 2.0
    assert summary["mean_tokens_per_second"] == 6.0


def test_probe_record_rejects_empty_generation() -> None:
    with pytest.raises(BenchmarkError, match="invalid benchmark output"):
        _probe_record("eng_Latn", 0, 5, 1, 0.1, "", "reference")
