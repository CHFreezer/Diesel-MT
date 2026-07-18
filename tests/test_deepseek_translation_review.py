from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from deepseek_translation_review import (  # noqa: E402
    TranslationReviewError,
    _validate_decisions,
    estimate_cost,
    load_api_key,
    load_config,
    load_full_review_items,
    make_batches,
    run_batches,
)
from mvp_60m_data_pipeline import sha256_file, write_json, write_jsonl  # noqa: E402


def _config() -> dict[str, object]:
    return copy.deepcopy(load_config(ROOT / "configs/deepseek_translation_review.yaml"))


def _item(index: int, *, source: str = "A source", target: str = "一个译文") -> dict[str, object]:
    return {
        "id": f"item-{index}",
        "kind": "teacher",
        "route": "eng_Latn->zho_Hans",
        "source_language": "eng_Latn",
        "target_language": "zho_Hans",
        "source_text": source,
        "candidate_translation": target,
        "semantic_group_id": f"group-{index}",
        "source_id": "fixture",
        "source_record_id": str(index),
    }


def _pass(item: dict[str, object]) -> dict[str, object]:
    return {
        "id": item["id"],
        "verdict": "pass",
        "categories": [],
        "confidence": 0.99,
        "source_evidence": "",
        "target_evidence": "",
        "note": "",
    }


def test_auth_script_is_supported_without_tracking_secret(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    script = tmp_path / "auth.ps1"
    script.write_text(
        "$env:ANTHROPIC_BASE_URL='https://api.deepseek.com'\n"
        "$env:ANTHROPIC_AUTH_TOKEN='fixture-secret'\n",
        encoding="utf-8",
    )
    assert load_api_key(env_name="DEEPSEEK_API_KEY", auth_script=script) == "fixture-secret"


def test_environment_key_precedes_auth_script(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "environment-secret")
    script = tmp_path / "auth.ps1"
    script.write_text("$env:ANTHROPIC_AUTH_TOKEN='file-secret'\n", encoding="utf-8")
    assert load_api_key(env_name="DEEPSEEK_API_KEY", auth_script=script) == "environment-secret"


def test_batches_are_deterministic_and_cost_is_bounded() -> None:
    config = _config()
    config["api"]["batch_max_records"] = 2  # type: ignore[index]
    config["api"]["batch_max_estimated_input_tokens"] = 100_000  # type: ignore[index]
    items = [_item(index) for index in range(5)]
    first = make_batches(items, config)
    second = make_batches(items, config)
    assert [batch["identity"] for batch in first] == [batch["identity"] for batch in second]
    assert [len(batch["items"]) for batch in first] == [2, 2, 1]
    estimate = estimate_cost(first, config)
    assert estimate["records"] == 5
    assert estimate["batches"] == 3
    assert 0 < estimate["estimated_total_usd"] < 0.01

    changed = [dict(item) for item in items]
    changed[0]["candidate_translation"] = "changed candidate"
    changed_batches = make_batches(changed, config)
    assert changed_batches[0]["identity"] != first[0]["identity"]


def test_response_requires_exact_order_and_sanitizes_evidence() -> None:
    config = _config()
    items = [_item(1, source="135,000 years", target="35,000 年")]
    valid = {
        "items": [
            {
                "id": "item-1",
                "verdict": "reject",
                "categories": ["numeric_error"],
                "confidence": 0.98,
                "source_evidence": "135,000",
                "target_evidence": "35,000",
                "note": "A leading digit was lost.",
            }
        ]
    }
    assert _validate_decisions(valid, items, config)[0]["verdict"] == "reject"
    wrapped = copy.deepcopy(valid)
    wrapped["items"][0]["target_evidence"] = "“35,000”"
    assert _validate_decisions(wrapped, items, config)[0]["target_evidence"] == "35,000"
    invalid = copy.deepcopy(valid)
    invalid["items"][0]["target_evidence"] = "not present"
    sanitized = _validate_decisions(invalid, items, config)[0]
    assert sanitized["target_evidence"] == ""
    assert sanitized["discarded_ungrounded_evidence"] == ["target"]
    too_long = copy.deepcopy(valid)
    too_long["items"][0]["note"] = "x" * 300
    truncated = _validate_decisions(too_long, items, config)[0]
    assert len(truncated["note"]) == 240
    assert truncated["truncated_response_fields"] == ["note"]
    reordered = {"items": [_pass(_item(2))]}
    with pytest.raises(TranslationReviewError, match="IDs/order"):
        _validate_decisions(reordered, items, config)


def test_pass_cannot_hide_error_claims() -> None:
    config = _config()
    item = _item(1)
    decision = _pass(item)
    decision["note"] = "maybe wrong"
    with pytest.raises(TranslationReviewError, match="contains error claims"):
        _validate_decisions({"items": [decision]}, [item], config)


def test_full_inputs_are_hash_bound_and_include_reverse_pairs(tmp_path: Path) -> None:
    config = _config()
    td03 = tmp_path / "td03"
    td04 = tmp_path / "td04"
    td03.mkdir()
    td04.mkdir()
    source = {
        "semantic_group_id": "group-1",
        "language_tag": "eng_Latn",
        "source_id": "fixture-source",
        "source_record_id": "source-1",
    }
    write_jsonl(td03 / "source-bank.jsonl", [source])
    accepted = {
        "teacher_job_id": "teacher-job-1",
        "record_id": "teacher-1",
        "semantic_group_id": "group-1",
        "src_lang": "eng_Latn",
        "tgt_lang": "zho_Hans",
        "source_text": "hello",
        "target_text": "你好",
    }
    reverse = {
        "record_id": "reverse-1",
        "forward_job_id": "teacher-job-1",
        "semantic_group_id": "group-1",
        "src_lang": "zho_Hant",
        "tgt_lang": "eng_Latn",
        "source_text": "你好",
        "target_text": "hello",
    }
    write_jsonl(td04 / "accepted-teacher.jsonl", [accepted])
    write_jsonl(td04 / "reverse-pairs.jsonl", [reverse])
    manifest = {
        "status": "complete",
        "generation_identity": "fixture-generation",
        "accepted_teacher": {"sha256": sha256_file(td04 / "accepted-teacher.jsonl")},
        "reverse_pairs": {"sha256": sha256_file(td04 / "reverse-pairs.jsonl")},
    }
    write_json(td04 / "manifest.json", manifest)
    items, evidence = load_full_review_items(tmp_path, config)
    assert [item["kind"] for item in items] == ["teacher", "reverse"]
    assert items[0]["source_id"] == "fixture-source"
    assert evidence["records"] == 2
    (td04 / "accepted-teacher.jsonl").write_text("{}\n", encoding="utf-8")
    with pytest.raises(TranslationReviewError, match="hash drift"):
        load_full_review_items(tmp_path, config)


def test_response_files_resume_without_second_api_call(tmp_path: Path) -> None:
    config = _config()
    config["api"]["batch_max_records"] = 2  # type: ignore[index]
    config["api"]["batch_max_estimated_input_tokens"] = 100_000  # type: ignore[index]
    items = [_item(index) for index in range(3)]
    batches = make_batches(items, config)

    class FakeClient:
        def __init__(self) -> None:
            self.calls = 0

        def review_batch(self, batch_items: list[dict[str, object]]) -> dict[str, object]:
            self.calls += 1
            return {
                "api_response_id": f"response-{self.calls}",
                "model": "deepseek-v4-flash",
                "system_fingerprint": "fixture",
                "finish_reason": "stop",
                "latency_seconds": 0.01,
                "request_attempts": 1,
                "usage": {
                    "prompt_tokens": 10,
                    "prompt_cache_hit_tokens": 0,
                    "prompt_cache_miss_tokens": 10,
                    "completion_tokens": 5,
                    "reasoning_tokens": 0,
                    "total_tokens": 15,
                },
                "decisions": [_pass(item) for item in batch_items],
            }

    client = FakeClient()
    first = run_batches(
        batches,
        response_root=tmp_path / "responses",
        client=client,  # type: ignore[arg-type]
        config=config,
        concurrency=2,
    )
    assert len(first) == 2
    assert client.calls == 2
    second = run_batches(
        batches,
        response_root=tmp_path / "responses",
        client=client,  # type: ignore[arg-type]
        config=config,
        concurrency=2,
    )
    assert len(second) == 2
    assert client.calls == 2
