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
