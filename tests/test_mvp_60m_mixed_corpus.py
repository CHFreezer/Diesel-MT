from __future__ import annotations

import sys
import copy
import json
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from model_data_source_contract import canonical_sha256  # noqa: E402
from model_training_contract import directed_routes  # noqa: E402
from mvp_60m_data_pipeline import (  # noqa: E402
    AbilityDataError,
    sha256_file,
    write_json,
    write_jsonl,
)
from publish_mvp_60m_mixed_corpus import (  # noqa: E402
    assemble_records,
    build_sampling_plan,
    load_config,
    publish,
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


def test_publish_verifies_input_hashes_and_writes_manifest_last(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    runtime = tmp_path / "runtime"
    (repository / "configs").mkdir(parents=True)
    (repository / "reports").mkdir()
    (runtime / "td03").mkdir(parents=True)
    (runtime / "td04").mkdir()

    generation_config = {"schema_version": 1, "identity": {"name": "fixture"}}
    (repository / "configs" / "generation.yaml").write_text(
        yaml.safe_dump(generation_config), encoding="utf-8"
    )
    generation_hash = canonical_sha256(generation_config)
    compact = repository / "reports" / "td03.json"
    compact.write_text("{}\n", encoding="utf-8")

    teacher = []
    for index, (source, target) in enumerate(directed_routes()):
        teacher.append(
            {
                **_row(index, f"{source}->{target}", "teacher", "teacher-group"),
                "generation_identity": "fixture-generation",
                "profile": "greedy-v1",
            }
        )
    human = [_row(100, "eng_Latn->jpn_Jpan", "human", "human-group")]
    teacher_count, teacher_hash = write_jsonl(runtime / "td04" / "accepted.jsonl", teacher)
    human_count, human_hash = write_jsonl(runtime / "td03" / "human.jsonl", human)
    reverse_count, reverse_hash = write_jsonl(runtime / "td04" / "reverse.jsonl", [])
    td03 = {
        "status": "complete",
        "source_bank": {"records": 5},
        "human_anchors": {"records": human_count, "sha256": human_hash},
    }
    td04 = {
        "status": "complete",
        "generation_config_sha256": generation_hash,
        "raw": {"records": teacher_count},
        "accepted_teacher": {"records": teacher_count, "sha256": teacher_hash},
        "reverse_pairs": {"records": reverse_count, "sha256": reverse_hash},
    }
    write_json(runtime / "td03" / "manifest.json", td03)
    write_json(runtime / "td04" / "manifest.json", td04)

    config = copy.deepcopy(load_config(ROOT / "configs/mvp_60m_mixed_corpus.yaml"))
    config["inputs"] = {
        "td03_runtime_manifest": "td03/manifest.json",
        "td03_compact_manifest": "reports/td03.json",
        "td03_compact_manifest_sha256": sha256_file(compact),
        "td04_runtime_manifest": "td04/manifest.json",
        "td04_generation_config": "configs/generation.yaml",
        "td04_generation_config_sha256": generation_hash,
        "human_anchors": "td03/human.jsonl",
        "accepted_teacher": "td04/accepted.jsonl",
        "reverse_pairs": "td04/reverse.jsonl",
    }
    config["quality"]["required_human_records"] = 1
    config["quality"]["required_fixed_non_hant_teacher_records_per_route"] = 1
    config_path = repository / "configs" / "mixed.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    report = repository / "reports" / "td05.json"

    manifest = publish(
        repository,
        runtime,
        config_path,
        tokenizer=_TinyTokenizer(),
        compact_report_path=report,
    )
    assert manifest["status"] == "complete"
    assert manifest["corpus"]["records"] == 21
    assert manifest["audit"]["class_records"] == {"human": 1, "teacher": 20}
    assert manifest["invariants"]["twenty_teacher_routes"] is True
    assert json.loads((runtime / "td05" / "manifest.json").read_text(encoding="utf-8")) == manifest
    assert report.is_file()
