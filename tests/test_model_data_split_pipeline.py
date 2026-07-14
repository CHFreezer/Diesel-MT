from __future__ import annotations

import copy
import hashlib
import json
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import model_data_split_pipeline as pipeline  # noqa: E402
from model_data_split_pipeline import (  # noqa: E402
    ContaminationError,
    InputError,
    LeakageError,
    SPLIT_PROFILE,
    audit_directed_samples,
    dry_run_plan,
    load_contamination_registry,
    prepare_finalized_samples,
    publish_finalized_data,
    scan_reference_records,
    sha256_bytes,
    split_for_component,
)
from model_training_contract import (  # noqa: E402
    canonical_json_bytes,
    config_sha256,
    load_model_data_config,
)


CONFIG_PATH = ROOT / "configs" / "mvp_model_data.yaml"
REGISTRY_PATH = ROOT / "configs" / "mvp_model_contamination.yaml"


@pytest.fixture
def config() -> dict[str, Any]:
    return load_model_data_config(CONFIG_PATH)


def _component_id(group_ids: list[str]) -> str:
    return f"component-sha256:{sha256_bytes(canonical_json_bytes(sorted(group_ids)))}"


def group_for_split(split: str, *, start: int = 0) -> str:
    for index in range(start, start + 100_000):
        group_id = f"group-{index:06d}"
        if split_for_component(_component_id([group_id])) == split:
            return group_id
    raise AssertionError(f"could not find deterministic group for {split}")


def group_samples(
    config: dict[str, Any],
    group_id: str,
    index: int,
    *,
    alignment_key: str | None = None,
    text_overrides: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    text_overrides = text_overrides or {}
    texts = {
        language: text_overrides.get(
            language,
            f"{language} {hashlib.sha256(f'{language}:{index}'.encode()).hexdigest()[:24]}",
        )
        for language in config["languages"]["model_tags"]
    }
    result = []
    for pair_index, pair in enumerate(config["directions"]["undirected_pairs"]):
        source, target = pair["tags"]
        result.append(
            {
                "sample_id": f"sample-{index:06d}-{pair_index:02d}",
                "sample_group_id": group_id,
                "source_id": "massive-1.1",
                "source_version": "1.1",
                "license": "CC-BY-4.0",
                "src_lang": source,
                "tgt_lang": target,
                "source_text": texts[source],
                "target_text": texts[target],
                "split": "train",
                "provenance": {
                    "kind": "human_parallel",
                    "source_record_id": f"train:{index}",
                    "alignment_key": alignment_key or f'{{"id":{index},"partition":"train"}}',
                },
            }
        )
    return result


def canonical_result_bytes(prepared: dict[str, Any]) -> bytes:
    return canonical_json_bytes(
        {
            "by_split": prepared["by_split"],
            "test_groups": prepared["test_groups"],
            "report": prepared["report"],
        }
    )


def clean_reference_scan(*, complete: bool = True) -> dict[str, Any]:
    return {
        "reference_sets": [],
        "blocking_hits": 0,
        "registry_sha256": "a" * 64,
        "registry_complete_for_m0": complete,
        "reference_identities": [],
    }


def write_td03_input(root: Path, samples: list[dict[str, Any]]) -> None:
    corpus = b"".join(canonical_json_bytes(sample) for sample in samples)
    corpus_path = root / "corpus" / "mvp" / "human_parallel.jsonl"
    corpus_path.parent.mkdir(parents=True, exist_ok=True)
    corpus_path.write_bytes(corpus)
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "identity": {"fixture": True},
        "identity_sha256": "b" * 64,
        "sample_schema_version": 1,
        "canonical_sample_scope": "nine undirected pairs; reverse route expansion is TD-04",
        "records": len(samples),
        "files": [
            {
                "path": "corpus/mvp/human_parallel.jsonl",
                "bytes": len(corpus),
                "sha256": sha256_bytes(corpus),
                "records": len(samples),
            }
        ],
    }
    (root / "corpus" / "mvp" / "manifest.json").write_bytes(canonical_json_bytes(manifest))


def test_split_profile_is_frozen_to_locked_massive_partition_counts() -> None:
    assert SPLIT_PROFILE["bucket_count"] == 16_521
    assert (
        SPLIT_PROFILE["train_buckets"],
        SPLIT_PROFILE["dev_buckets"],
        SPLIT_PROFILE["test_buckets"],
    ) == (11_514, 2_033, 2_974)
    assert config_sha256(SPLIT_PROFILE) == "935583630fcd06d39b9cf5c89bac92a76ef4e56d33ccfe1b3c11f030d9ecff0d"


def test_group_split_reverse_expansion_and_input_order_are_deterministic(
    config: dict[str, Any],
) -> None:
    group_ids = [
        group_for_split("train"),
        group_for_split("dev", start=100_000),
        group_for_split("test", start=200_000),
    ]
    samples = [
        sample
        for index, group_id in enumerate(group_ids, 1)
        for sample in group_samples(config, group_id, index)
    ]
    shuffled = list(samples)
    random.Random(20260715).shuffle(shuffled)
    first = prepare_finalized_samples(samples, config)
    second = prepare_finalized_samples(shuffled, config)
    assert canonical_result_bytes(first) == canonical_result_bytes(second)
    assert first["report"]["output"]["samples_by_split"] == {
        "train": 18,
        "dev": 18,
        "test": 18,
    }
    assert len(first["report"]["leakage_audit"]["route_counts"]) == 18
    assert set(first["report"]["leakage_audit"]["route_counts"].values()) == {3}
    assert len(first["test_groups"]) == 1


def test_exact_text_components_and_pair_dedup_are_deterministic(
    config: dict[str, Any],
) -> None:
    first = group_samples(config, "group-exact-a", 7)
    second = group_samples(config, "group-exact-b", 8)
    for left, right in zip(first, second, strict=True):
        right["source_text"] = left["source_text"]
        right["target_text"] = left["target_text"]
    prepared = prepare_finalized_samples(first + second, config)
    report = prepared["report"]
    assert report["component_binding"]["split_components"] == 1
    assert report["component_binding"]["exact_duplicate_text_keys"] == 5
    assert report["exact_deduplication"]["pair_exact_duplicates_removed"] == 9
    assert report["output"]["undirected_samples"] == 9
    assert report["output"]["directed_samples"] == 18


def test_near_duplicates_are_bound_before_split(config: dict[str, Any]) -> None:
    first = group_samples(
        config,
        "group-near-a",
        11,
        text_overrides={"eng_Latn": "please play the jazz music now"},
    )
    second = group_samples(
        config,
        "group-near-b",
        12,
        text_overrides={"eng_Latn": "please play the jazz music now."},
    )
    prepared = prepare_finalized_samples(first + second, config)
    assert prepared["report"]["component_binding"]["near_duplicate_links"] >= 1
    assert prepared["report"]["component_binding"]["split_components"] == 1
    group_splits = {
        sample["sample_group_id"]: sample["split"]
        for records in prepared["by_split"].values()
        for sample in records
    }
    assert len(set(group_splits.values())) == 1


def test_reverse_relation_split_mutation_is_rejected(config: dict[str, Any]) -> None:
    prepared = prepare_finalized_samples(group_samples(config, "group-reverse", 20), config)
    directed = [copy.deepcopy(sample) for records in prepared["by_split"].values() for sample in records]
    directed[0]["split"] = "test" if directed[0]["split"] != "test" else "train"
    with pytest.raises(LeakageError, match="group crosses|forward/reverse"):
        audit_directed_samples(directed, config)


def test_exact_and_near_cross_split_mutations_are_rejected(config: dict[str, Any]) -> None:
    exact_a = group_samples(
        config,
        "group-cross-exact-a",
        30,
        text_overrides={"eng_Latn": "shared exact English sentence"},
    )
    exact_b = group_samples(
        config,
        "group-cross-exact-b",
        31,
        text_overrides={"eng_Latn": "shared exact English sentence"},
    )
    exact = prepare_finalized_samples(exact_a + exact_b, config)
    exact_directed = [copy.deepcopy(sample) for records in exact["by_split"].values() for sample in records]
    original = next(sample["split"] for sample in exact_directed if sample["sample_group_id"] == "group-cross-exact-a")
    changed = "test" if original != "test" else "train"
    for sample in exact_directed:
        if sample["sample_group_id"] == "group-cross-exact-b":
            sample["split"] = changed
    with pytest.raises(LeakageError, match="exact normalized text crosses"):
        audit_directed_samples(exact_directed, config, check_near_duplicates=False)

    near_a = group_samples(
        config,
        "group-cross-near-a",
        32,
        text_overrides={"eng_Latn": "please start the morning news report"},
    )
    near_b = group_samples(
        config,
        "group-cross-near-b",
        33,
        text_overrides={"eng_Latn": "please start the morning news report."},
    )
    near = prepare_finalized_samples(near_a + near_b, config)
    near_directed = [copy.deepcopy(sample) for records in near["by_split"].values() for sample in records]
    original = next(sample["split"] for sample in near_directed if sample["sample_group_id"] == "group-cross-near-a")
    changed = "test" if original != "test" else "train"
    for sample in near_directed:
        if sample["sample_group_id"] == "group-cross-near-b":
            sample["split"] = changed
    with pytest.raises(LeakageError, match="near-duplicate"):
        audit_directed_samples(near_directed, config)


def test_wrong_alignment_group_and_derivation_are_rejected(config: dict[str, Any]) -> None:
    first = group_samples(config, "group-align-a", 40, alignment_key="same-alignment")
    second = group_samples(config, "group-align-b", 41, alignment_key="same-alignment")
    with pytest.raises(LeakageError, match="alignment key"):
        prepare_finalized_samples(first + second, config)

    first = group_samples(config, "group-derived-a", 42)
    second = group_samples(config, "group-derived-b", 43)
    with pytest.raises(LeakageError, match="not bound to its parent group"):
        prepare_finalized_samples(
            first + second,
            config,
            derived_links=[
                {
                    "child_sample_id": second[0]["sample_id"],
                    "parent_sample_id": first[0]["sample_id"],
                    "reason": "fixture derivation",
                }
            ],
        )


def test_reverse_or_unknown_input_orientation_is_rejected(config: dict[str, Any]) -> None:
    samples = group_samples(config, "group-bad-orientation", 50)
    samples[0]["src_lang"], samples[0]["tgt_lang"] = samples[0]["tgt_lang"], samples[0]["src_lang"]
    samples[0]["source_text"], samples[0]["target_text"] = samples[0]["target_text"], samples[0]["source_text"]
    with pytest.raises(InputError, match="canonical orientation"):
        prepare_finalized_samples(samples, config)


def test_reference_scan_distinguishes_report_only_and_blocking_hits(
    config: dict[str, Any],
) -> None:
    samples = group_samples(
        config,
        "group-reference",
        60,
        text_overrides={"eng_Latn": "please play my favorite jazz music"},
    )
    prepared = prepare_finalized_samples(samples, config)
    report = scan_reference_records(
        prepared["candidate_entries"],
        [
            {
                "reference_id": "tokenizer-holdout",
                "kind": "tokenizer_holdout",
                "policy": "report",
                "match": "exact",
                "records": [
                    {
                        "language": "eng_Latn",
                        "text": "please play my favorite jazz music",
                        "record_id": "tok-1",
                    }
                ],
            },
            {
                "reference_id": "formal-mt-eval",
                "kind": "mt_evaluation",
                "policy": "block",
                "match": "exact-and-near",
                "records": [
                    {
                        "language": "eng_Latn",
                        "text": "please play my favorite jazz music.",
                        "record_id": "eval-1",
                    }
                ],
            },
        ],
    )
    assert report["reference_sets"][0]["hits"] == 1
    assert report["reference_sets"][0]["policy"] == "report"
    assert report["reference_sets"][1]["hits"] == 1
    assert report["blocking_hits"] == 1

    with pytest.raises(InputError, match="must use policy=block"):
        scan_reference_records(
            prepared["candidate_entries"],
            [
                {
                    "reference_id": "unsafe-eval",
                    "kind": "mt_evaluation",
                    "policy": "report",
                    "match": "exact-and-near",
                    "records": [],
                }
            ],
        )

    with pytest.raises(InputError, match="must use match=exact-and-near"):
        scan_reference_records(
            prepared["candidate_entries"],
            [
                {
                    "reference_id": "unsafe-exact-only-eval",
                    "kind": "mt_evaluation",
                    "policy": "block",
                    "match": "exact",
                    "records": [],
                }
            ],
        )


def test_blocking_contamination_or_incomplete_registry_never_publishes_manifest(
    tmp_path: Path, config: dict[str, Any]
) -> None:
    prepared = prepare_finalized_samples(group_samples(config, "group-blocked", 70), config)
    identity = {"manifest_sha256": "c" * 64}
    blocked_scan = clean_reference_scan()
    blocked_scan["blocking_hits"] = 1
    blocked_scan["reference_sets"] = [
        {
            "reference_id": "eval",
            "kind": "mt_evaluation",
            "policy": "block",
            "records": 1,
            "hits": 1,
            "reported_hits": [],
        }
    ]
    with pytest.raises(ContaminationError, match="blocking external"):
        publish_finalized_data(
            prepared,
            identity,
            blocked_scan,
            tmp_path / "blocked",
            require_complete_references=True,
        )
    assert not (tmp_path / "blocked" / "corpus" / "mvp" / "finalized" / "manifest.json").exists()
    assert (tmp_path / "blocked" / "reports" / "td04-contamination-blocked.json").is_file()

    with pytest.raises(ContaminationError, match="incomplete"):
        publish_finalized_data(
            prepared,
            identity,
            clean_reference_scan(complete=False),
            tmp_path / "incomplete",
            require_complete_references=True,
        )
    assert not (tmp_path / "incomplete" / "corpus" / "mvp" / "finalized" / "manifest.json").exists()


def test_publication_is_byte_stable_and_manifest_is_last(
    tmp_path: Path, config: dict[str, Any]
) -> None:
    samples = group_samples(config, "group-publish-a", 80) + group_samples(
        config, "group-publish-b", 81
    )
    shuffled = list(samples)
    random.Random(9).shuffle(shuffled)
    first_prepared = prepare_finalized_samples(samples, config)
    second_prepared = prepare_finalized_samples(shuffled, config)
    identity = {
        "manifest_sha256": "d" * 64,
        "manifest_identity_sha256": "e" * 64,
        "corpus_sha256": "f" * 64,
        "records": len(samples),
    }
    first = tmp_path / "first"
    second = tmp_path / "second"
    first_result = publish_finalized_data(
        first_prepared,
        identity,
        clean_reference_scan(),
        first,
        require_complete_references=True,
    )
    second_result = publish_finalized_data(
        second_prepared,
        identity,
        clean_reference_scan(),
        second,
        require_complete_references=True,
    )
    paths = [
        "corpus/mvp/finalized/train.jsonl",
        "corpus/mvp/finalized/dev.jsonl",
        "corpus/mvp/finalized/test.jsonl",
        "corpus/mvp/finalized/test-groups.jsonl",
        "corpus/mvp/finalized/manifest.json",
        "reports/td04-dedup-leakage.json",
    ]
    assert all((first / path).read_bytes() == (second / path).read_bytes() for path in paths)
    assert first_result["manifest_sha256"] == second_result["manifest_sha256"]


def test_interrupted_publication_removes_completion_marker(
    tmp_path: Path, config: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    prepared = prepare_finalized_samples(group_samples(config, "group-interrupt", 90), config)
    real_atomic = pipeline.atomic_write_bytes

    def fail_dev(path: Path, data: bytes) -> None:
        if path.name == "dev.jsonl":
            raise OSError("injected TD-04 publish failure")
        real_atomic(path, data)

    monkeypatch.setattr(pipeline, "atomic_write_bytes", fail_dev)
    with pytest.raises(OSError, match="injected TD-04"):
        publish_finalized_data(
            prepared,
            {"manifest_sha256": "0" * 64},
            clean_reference_scan(),
            tmp_path,
            require_complete_references=True,
        )
    assert not (tmp_path / "corpus" / "mvp" / "finalized" / "manifest.json").exists()


def test_locked_registry_tracks_tokenizer_and_external_model_evaluation() -> None:
    registry = load_contamination_registry(REGISTRY_PATH)
    assert [reference["kind"] for reference in registry["reference_sets"]] == [
        "tokenizer_corpus",
        "tokenizer_holdout",
        "tokenizer_evaluation",
        "mt_evaluation",
    ]
    assert [reference["policy"] for reference in registry["reference_sets"]] == [
        "report",
        "report",
        "report",
        "block",
    ]
    assert [reference["match"] for reference in registry["reference_sets"]] == [
        "exact",
        "exact",
        "exact",
        "exact-and-near",
    ]
    assert registry["requirements"]["formal_mt_evaluation"]["status"] == "locked"
    assert pipeline.registry_is_complete(registry) is True
    for reference in registry["reference_sets"]:
        identity, specs = pipeline._manifest_file_specs(reference, ROOT)
        assert identity["manifest_sha256"] == reference["manifest"]["sha256"]
        assert len(specs) == (10 if reference["kind"] == "mt_evaluation" else 5)


def test_cli_dry_run_is_side_effect_free(
    tmp_path: Path, config: dict[str, Any]
) -> None:
    input_root = tmp_path / "input"
    out_root = tmp_path / "output"
    samples = group_samples(config, "group-cli", 100)
    write_td03_input(input_root, samples)
    result = subprocess.run(
        [
            sys.executable,
            "scripts/finalize_model_data.py",
            "--config",
            str(CONFIG_PATH),
            "--registry",
            str(REGISTRY_PATH),
            "--input-root",
            str(input_root),
            "--out",
            str(out_root),
            "--dry-run",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    plan = json.loads(result.stdout)
    assert plan["status"] == "dry-run"
    assert plan["input_records"] == 9
    assert plan["reference_registry_complete_for_m0"] is True
    assert not out_root.exists()
