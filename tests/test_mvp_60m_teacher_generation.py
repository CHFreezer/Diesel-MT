from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from generate_mvp_60m_teacher import (  # noqa: E402
    build_jobs,
    build_reverse_pairs,
    generate_one,
    load_config,
)
from hymt2_distillation import DistillationError, load_prompt_config  # noqa: E402
from hymt2_decode_ab import (  # noqa: E402
    _comparison,
    load_decode_ab_config,
    select_decode_ab_items,
)


def test_job_builder_has_16_fixed_routes_and_four_quality_actual_hant_routes() -> None:
    config = load_config(ROOT / "configs/mvp_60m_teacher_generation.yaml")
    rows = []
    counts = {"eng_Latn": 12000, "zho_Hans": 12000, "jpn_Jpan": 12000, "kor_Hang": 12000, "zho_Hant": 3}
    for language, count in counts.items():
        for index in range(count):
            rows.append({"record_id": f"{language}-{index}", "semantic_group_id": f"g-{language}-{index}", "language_tag": language, "text": f"text {index}"})
    jobs = build_jobs(rows, config)
    route_counts = {}
    for job in jobs:
        route_counts[job["route"]] = route_counts.get(job["route"], 0) + 1
    assert len(route_counts) == 20
    assert sum(count == 12000 for route, count in route_counts.items() if not route.startswith("zho_Hant->")) == 16
    assert all(count == 3 for route, count in route_counts.items() if route.startswith("zho_Hant->"))


def test_reverse_pairs_are_bounded_and_preserve_shared_group() -> None:
    accepted = []
    for index in range(10):
        accepted.append({
            "job_id": f"f-{index}", "semantic_group_id": f"g-{index}",
            "src_lang": "eng_Latn", "tgt_lang": "zho_Hant",
            "source_text": f"There are {index} items.", "normalized_output": f"共有 {index} 個項目。",
        })
    reverse = build_reverse_pairs(accepted, {"zho_Hant->eng_Latn": 8})
    assert len(reverse) == 4
    assert all(row["counts_as_native_hant"] is False for row in reverse)
    assert all(row["semantic_group_id"].startswith("g-") for row in reverse)


def test_peg_native_output_parse_error_is_auditable_rejection() -> None:
    class BrokenPegNativeTeacher:
        def generate(self, **_: object) -> dict[str, object]:
            raise DistillationError(
                'teacher request failed after 3 attempts: HTTP 500: '
                '{"error":{"message":"The model produced output that does not match '
                'the expected peg-native format"}}'
            )

    cross = load_prompt_config(ROOT / "configs/hymt2_teacher_prompt_decode.yaml")
    conversion = load_prompt_config(ROOT / "configs/hymt2_teacher_prompt_decode_zh_conversion.yaml")
    job = {
        "job_id": "job-1",
        "job_rank": 0,
        "route": "eng_Latn->zho_Hans",
        "src_lang": "eng_Latn",
        "tgt_lang": "zho_Hans",
        "source_record_id": "source-1",
        "semantic_group_id": "group-1",
        "source_text": "A valid source sentence.",
    }
    record = generate_one(
        BrokenPegNativeTeacher(),
        job,
        cross_config=cross,
        conversion_config=conversion,
        generation_identity="generation-1",
    )
    assert record["accepted"] is False
    assert record["teacher_error_code"] == "peg_native_output_parse_error"
    assert record["rejection_reasons"][0] == "teacher_output_format_error"
    assert record["request_attempts"] == cross["runtime"]["request_attempts"]


def test_non_format_teacher_failure_remains_fatal() -> None:
    class UnavailableTeacher:
        def generate(self, **_: object) -> dict[str, object]:
            raise DistillationError("teacher request failed after 3 attempts: connection refused")

    cross = load_prompt_config(ROOT / "configs/hymt2_teacher_prompt_decode.yaml")
    conversion = load_prompt_config(ROOT / "configs/hymt2_teacher_prompt_decode_zh_conversion.yaml")
    job = {
        "job_id": "job-2",
        "job_rank": 0,
        "route": "eng_Latn->zho_Hans",
        "src_lang": "eng_Latn",
        "tgt_lang": "zho_Hans",
        "source_record_id": "source-2",
        "semantic_group_id": "group-2",
        "source_text": "Another valid source sentence.",
    }
    try:
        generate_one(
            UnavailableTeacher(),
            job,
            cross_config=cross,
            conversion_config=conversion,
            generation_identity="generation-1",
        )
    except DistillationError as error:
        assert "connection refused" in str(error)
    else:
        raise AssertionError("non-format teacher failures must remain fatal")


def test_generation_config_has_exact_audited_route_limits() -> None:
    config = load_config(ROOT / "configs/mvp_60m_teacher_generation.yaml")
    limits = config["generation"]["route_max_output_tokens"]
    assert len(limits) == 20
    assert limits["eng_Latn->kor_Hang"] == 128
    assert limits["zho_Hans->kor_Hang"] == 192
    assert limits["zho_Hans->jpn_Jpan"] == 128
    assert config["generation"]["parallel_slots"] == 64
    assert config["generation"]["server_context_size"] == 32768


def test_generate_one_uses_generation_specific_output_limit() -> None:
    class CapturingTeacher:
        def __init__(self) -> None:
            self.max_tokens = 0

        def generate(self, **kwargs: object) -> dict[str, object]:
            self.max_tokens = int(kwargs["max_tokens"])
            return {
                "raw_output": "有效译文",
                "finish_reason": "stop",
                "prompt_tokens": 10,
                "completion_tokens": 4,
                "latency_seconds": 0.1,
                "request_attempts": 1,
                "seed": 1,
            }

    cross = load_prompt_config(ROOT / "configs/hymt2_teacher_prompt_decode.yaml")
    conversion = load_prompt_config(ROOT / "configs/hymt2_teacher_prompt_decode_zh_conversion.yaml")
    teacher = CapturingTeacher()
    record = generate_one(
        teacher,
        {
            "job_id": "job-3",
            "job_rank": 0,
            "route": "eng_Latn->zho_Hans",
            "src_lang": "eng_Latn",
            "tgt_lang": "zho_Hans",
            "source_record_id": "source-3",
            "semantic_group_id": "group-3",
            "source_text": "A valid source sentence.",
        },
        cross_config=cross,
        conversion_config=conversion,
        generation_identity="generation-2",
        max_output_tokens=137,
    )
    assert teacher.max_tokens == 137
    assert record["accepted"] is True


def test_generate_one_can_use_official_sampling_profile() -> None:
    class CapturingTeacher:
        def __init__(self) -> None:
            self.profile: dict[str, object] = {}

        def generate(self, **kwargs: object) -> dict[str, object]:
            self.profile = dict(kwargs["profile"])  # type: ignore[arg-type]
            return {
                "raw_output": "Valid translation.",
                "finish_reason": "stop",
                "prompt_tokens": 10,
                "completion_tokens": 3,
                "latency_seconds": 0.1,
                "request_attempts": 1,
                "seed": 1,
            }

    cross = load_prompt_config(ROOT / "configs/hymt2_teacher_prompt_decode.yaml")
    conversion = load_prompt_config(ROOT / "configs/hymt2_teacher_prompt_decode_zh_conversion.yaml")
    teacher = CapturingTeacher()
    record = generate_one(
        teacher,
        {
            "job_id": "job-official",
            "job_rank": 0,
            "route": "jpn_Jpan->eng_Latn",
            "src_lang": "jpn_Jpan",
            "tgt_lang": "eng_Latn",
            "source_record_id": "source-official",
            "semantic_group_id": "group-official",
            "source_text": "今日は晴れです。",
        },
        cross_config=cross,
        conversion_config=conversion,
        generation_identity="decode-ab",
        profile_name="official-sampling-v1",
    )
    assert teacher.profile == {
        "temperature": 0.7,
        "top_p": 0.6,
        "top_k": 20,
        "repeat_penalty": 1.05,
        "seed": 20260715,
    }
    assert record["profile"] == "official-sampling-v1"


def test_decode_ab_selection_forces_known_cases_without_growing_budget(tmp_path: Path) -> None:
    config = load_decode_ab_config(ROOT / "configs/hymt2_decode_ab.yaml")
    config["ab"]["records"] = 3
    config["ab"]["forced_cases"] = [
        {"route": "jpn_Jpan->eng_Latn", "source_record_id": "forced"}
    ]
    review_config = {
        "inputs": {
            "td04_manifest": "td04/manifest.json",
            "accepted_teacher": "td04/accepted-teacher.jsonl",
            "reverse_pairs": "td04/reverse-pairs.jsonl",
            "source_bank": "td03/source-bank.jsonl",
        },
        "review": {"staged_review_seed": 7},
    }
    (tmp_path / "td04").mkdir()
    (tmp_path / "td03").mkdir()
    accepted = []
    sources = []
    for index, source_record_id in enumerate(("a", "b", "c", "forced")):
        group = f"g-{index}"
        accepted.append(
            {
                "record_id": f"teacher-{index}",
                "teacher_job_id": f"job-{index}",
                "semantic_group_id": group,
                "src_lang": "jpn_Jpan",
                "tgt_lang": "eng_Latn",
                "source_text": f"source {index}",
                "target_text": f"target {index}",
            }
        )
        sources.append(
            {
                "semantic_group_id": group,
                "language_tag": "jpn_Jpan",
                "source_id": "fixture",
                "source_record_id": source_record_id,
            }
        )
    import json
    import hashlib

    def write_lines(path: Path, rows: list[dict[str, object]]) -> str:
        data = b"".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode() + b"\n"
            for row in rows
        )
        path.write_bytes(data)
        return hashlib.sha256(data).hexdigest()

    accepted_sha = write_lines(tmp_path / "td04" / "accepted-teacher.jsonl", accepted)
    reverse_sha = write_lines(tmp_path / "td04" / "reverse-pairs.jsonl", [])
    write_lines(tmp_path / "td03" / "source-bank.jsonl", sources)
    (tmp_path / "td04" / "manifest.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "generation_identity": "fixture",
                "accepted_teacher": {"sha256": accepted_sha},
                "reverse_pairs": {"sha256": reverse_sha},
            }
        ),
        encoding="utf-8",
    )
    selected, evidence = select_decode_ab_items(tmp_path, review_config, config)
    assert len(selected) == 3
    assert evidence["forced_cases"] == 1
    assert any(item["source_record_id"] == "forced" for item in selected)


def test_decode_ab_ignores_reviewer_variance_for_identical_outputs() -> None:
    rows = [
        {
            "id": "same",
            "route": "eng_Latn->zho_Hans",
            "source_id": "fixture",
            "source_record_id": "1",
            "source_text": "Hello",
            "baseline_translation": "你好",
            "challenger_translation": "你好",
            "exact_match": True,
            "baseline_filter_accepted": True,
            "challenger_filter_accepted": True,
        }
    ]
    baseline = [{"review_id": "same", "verdict": "pass"}]
    challenger = [{"review_id": "same", "verdict": "reject"}]
    queue, _, summary = _comparison(
        rows,
        baseline,
        challenger,
        {
            "ab": {
                "forced_cases": [],
                "blind_seed": "fixture",
                "both_pass_manual_sample": 0,
            }
        },
    )
    assert queue == []
    assert summary["review_classification_counts"] == {"both_pass": 1}
    assert summary["reviewer_variance_on_exact_matches"] == 1
