from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from hymt2_distillation import (  # noqa: E402
    CHINESE_CONVERSION_ROUTES,
    ROUTES,
    DistillationError,
    build_prompt,
    deterministic_route_sample,
    filter_output,
    load_prompt_config,
    metric_scores,
    prompt_routes,
    read_parallel_jsonl,
    validate_prompt_config,
)
from model_training_contract import directed_routes  # noqa: E402


CONFIG_PATH = ROOT / "configs" / "hymt2_teacher_prompt_decode.yaml"
ZH_CONVERSION_CONFIG_PATH = ROOT / "configs" / "hymt2_teacher_prompt_decode_zh_conversion.yaml"
CALIBRATION_EVIDENCE_PATH = (
    ROOT
    / "artifacts"
    / "model-training"
    / "reports"
    / "teacher"
    / "calibration.json"
)


@pytest.fixture
def prompt_config() -> dict[str, object]:
    return load_prompt_config(CONFIG_PATH)


def _sample(source: str, target: str, index: int, *, split: str = "dev") -> dict[str, str]:
    return {
        "sample_id": f"sample-{source}-{target}-{index}",
        "sample_group_id": f"group-{index}",
        "source_text": f"source {index}",
        "target_text": f"target {index}",
        "src_lang": source,
        "tgt_lang": target,
        "split": split,
    }


def test_prompt_contract_freezes_five_names_and_eighteen_routes(
    prompt_config: dict[str, object],
) -> None:
    assert tuple(ROUTES) == tuple(
        f"{source}->{target}"
        for source, target in directed_routes()
        if f"{source}->{target}" not in CHINESE_CONVERSION_ROUTES
    )
    assert len(prompt_config["route_limits"]) == 18
    assert prompt_config["prompt"]["language_names"]["zho_Hans"] == "Chinese"
    assert prompt_config["prompt"]["language_names"]["zho_Hant"] == "Traditional Chinese"
    assert prompt_config["prompt"]["system_prompt"] is None
    prompt = build_prompt(prompt_config, "天气很好", "zho_Hant")
    assert "Traditional Chinese" in prompt
    assert prompt.endswith("天气很好")


def test_chinese_conversion_prompt_addendum_keeps_existing_language_names() -> None:
    config = load_prompt_config(ZH_CONVERSION_CONFIG_PATH)
    assert prompt_routes(config) == CHINESE_CONVERSION_ROUTES
    assert config["prompt"]["language_names"]["zho_Hans"] == "Chinese"
    assert config["prompt"]["language_names"]["zho_Hant"] == "Traditional Chinese"
    assert len(config["route_limits"]) == 2


def test_chinese_conversion_source_copy_policy_is_route_specific() -> None:
    config = load_prompt_config(ZH_CONVERSION_CONFIG_PATH)
    unchanged = filter_output(
        source_text="今天",
        target_text="今天",
        target_language="zho_Hant",
        finish_reason="stop",
        config=config,
    )
    assert unchanged["accepted"] is True
    convertible_copy = filter_output(
        source_text="软件",
        target_text="软件",
        target_language="zho_Hant",
        finish_reason="stop",
        config=config,
    )
    assert convertible_copy["accepted"] is False
    assert "source_copy" in convertible_copy["rejection_reasons"]


def test_prompt_contract_rejects_hardware_or_test_drift(
    prompt_config: dict[str, object],
) -> None:
    changed = copy.deepcopy(prompt_config)
    changed["runtime"]["gpu_model"] = "specific-device"
    with pytest.raises(DistillationError, match="unknown fields: gpu_model"):
        validate_prompt_config(changed)

    changed = copy.deepcopy(prompt_config)
    changed["calibration_input"]["split"] = "test"
    with pytest.raises(DistillationError, match="must use dev"):
        validate_prompt_config(changed)


def test_deterministic_route_sample_is_order_independent_and_complete() -> None:
    records = [
        _sample(source, target, index)
        for source, target in directed_routes()
        for index in range(4)
    ]
    first = deterministic_route_sample(records, per_route=2, seed="fixed")
    second = deterministic_route_sample(list(reversed(records)), per_route=2, seed="fixed")
    assert [record["sample_id"] for record in first] == [record["sample_id"] for record in second]
    assert len(first) == 36
    assert {f"{record['src_lang']}->{record['tgt_lang']}" for record in first} == set(ROUTES)


@pytest.mark.parametrize(
    ("source", "target", "language", "finish_reason", "reason"),
    [
        ("hello", "", "zho_Hans", "stop", "empty_output"),
        ("hello", "hello", "zho_Hans", "stop", "source_copy"),
        ("hello", "Translation: 你好", "zho_Hans", "stop", "extra_explanation"),
        ("hello", "今天天气很好", "zho_Hant", "stop", "simplified_output_for_traditional_target"),
        ("hello", "今天天氣很好", "zho_Hans", "stop", "traditional_output_for_simplified_target"),
        ("hello {name}", "你好", "zho_Hans", "stop", "placeholder_mismatch"),
        ("hello", "你好", "zho_Hans", "length", "truncated"),
    ],
)
def test_output_filter_covers_failure_modes(
    prompt_config: dict[str, object],
    source: str,
    target: str,
    language: str,
    finish_reason: str,
    reason: str,
) -> None:
    result = filter_output(
        source_text=source,
        target_text=target,
        target_language=language,
        finish_reason=finish_reason,
        config=prompt_config,
    )
    assert result["accepted"] is False
    assert reason in result["rejection_reasons"]


def test_output_filter_accepts_shared_and_native_traditional_text(
    prompt_config: dict[str, object],
) -> None:
    result = filter_output(
        source_text="the weather is nice today",
        target_text="今天天氣很好",
        target_language="zho_Hant",
        finish_reason="stop",
        config=prompt_config,
    )
    assert result["accepted"] is True
    assert result["chinese_script_evidence"]["traditional"] > 0

    common_variant = filter_output(
        source_text="i had chicken for lunch and it was delicious",
        target_text="我午餐吃了雞肉，非常好吃。",
        target_language="zho_Hant",
        finish_reason="stop",
        config=prompt_config,
    )
    assert common_variant["accepted"] is True
    assert common_variant["chinese_script_evidence"] == {"simplified": 2, "traditional": 1}


def test_metrics_report_exact_sacrebleu_and_chrf(prompt_config: dict[str, object]) -> None:
    result = metric_scores(["今天天氣很好"], ["今天天氣很好"], prompt_config)
    assert result["sacrebleu"] == 100.0
    assert result["chrf"] == 100.0
    assert "tok:char" in result["sacrebleu_signature"]


def test_jsonl_reader_rejects_test_records(tmp_path: Path) -> None:
    path = tmp_path / "input.jsonl"
    record = _sample("eng_Latn", "jpn_Jpan", 1, split="test")
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    with pytest.raises(DistillationError, match="non-train"):
        read_parallel_jsonl(path, expected_split="train")


def test_td07_calibration_evidence_freezes_greedy_without_test_access() -> None:
    evidence = json.loads(CALIBRATION_EVIDENCE_PATH.read_text(encoding="utf-8"))
    assert evidence["status"] == "complete"
    assert evidence["test_accessed"] is False
    assert evidence["input"]["split"] == "dev"
    assert evidence["input"]["selected_records"] == 216
    assert set(evidence["profiles"]["greedy-v1"]["routes"]) == set(ROUTES)
    assert evidence["decision"]["selected_profile"] == "greedy-v1"
    assert evidence["decision"]["selected_route_gate_failures"] == []
    assert evidence["replay"]["greedy-v1"]["exact"] is True
    assert evidence["replay"]["official-sampling-v1"]["exact"] is True
