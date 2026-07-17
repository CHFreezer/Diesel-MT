from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from mvp_60m_data_pipeline import AbilityDataError  # noqa: E402
from publish_mvp_60m_mixed_corpus import (  # noqa: E402
    assemble_records,
    build_sampling_plan,
    load_config,
    validate_token_lengths,
)


def _row(index: int, route: str, provenance: str, group_prefix: str) -> dict[str, object]:
    source, target = route.split("->")
    return {
        "record_id": f"{provenance}-{index}",
        "semantic_group_id": f"{group_prefix}-{index}",
        "src_lang": source,
        "tgt_lang": target,
        "source_text": f"source text {index}",
        "target_text": f"target text {index}",
        "provenance": provenance,
    }


def test_two_stage_preview_is_exactly_80_20_without_raw_fill() -> None:
    config = load_config(ROOT / "configs/mvp_60m_mixed_corpus.yaml")
    teacher = [_row(index, "eng_Latn->jpn_Jpan", "teacher", "t") for index in range(5)]
    human = [_row(100 + index, "jpn_Jpan->eng_Latn", "human", "h") for index in range(3)]
    rows, audit = assemble_records(teacher, [], human)
    plan = build_sampling_plan(rows, config)
    assert audit["class_records"] == {"human": 3, "teacher": 5}
    assert plan["preview"]["class_counts"] == {"human": 20000, "teacher": 80000}
    assert plan["raw_duplicate_fill"] is False


def test_teacher_human_semantic_group_overlap_is_rejected() -> None:
    teacher = [_row(1, "eng_Latn->jpn_Jpan", "teacher", "shared")]
    human = [_row(1, "jpn_Jpan->eng_Latn", "human", "shared")]
    human[0]["semantic_group_id"] = teacher[0]["semantic_group_id"]
    with pytest.raises(AbilityDataError, match="semantic groups overlap"):
        assemble_records(teacher, [], human)


class _TinyTokenizer:
    src_lang = "eng_Latn"

    def __call__(self, texts, **_kwargs):
        return {"input_ids": [list(range(len(text.split()) + 2)) for text in texts]}


def test_zero_truncation_gate_rejects_overflow() -> None:
    rows = [{
        "sample_id": "x", "src_lang": "eng_Latn", "tgt_lang": "jpn_Jpan",
        "source_text": "one two", "target_text": "one two three four",
    }]
    with pytest.raises(AbilityDataError, match="zero-truncation gate failed"):
        validate_token_lengths(rows, _TinyTokenizer(), maximum_source=8, maximum_target=5)
