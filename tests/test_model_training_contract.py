from __future__ import annotations

import copy
import hashlib
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import model_training_contract as contract  # noqa: E402
from model_training_contract import (  # noqa: E402
    ContractError,
    canonical_json_bytes,
    config_sha256,
    directed_routes,
    load_model_data_config,
    load_source_lock,
    load_student_config,
    product_directions,
    resolve_runtime_root,
    validate_model_data_config,
    validate_parallel_sample,
    validate_repo_relative_path,
    validate_route,
    validate_source_lock,
    validate_student_config,
)


DATA_CONFIG_PATH = ROOT / "configs" / "mvp_model_data.yaml"
STUDENT_CONFIG_PATH = ROOT / "configs" / "mvp_e8_d2_v48k.yaml"
SOURCE_LOCK_PATH = ROOT / "configs" / "mvp_model_data.lock.json"
TOKENIZER_MANIFEST = ROOT / "artifacts" / "tokenizers" / "mvp-tokenizer-v0" / "artifact_manifest.json"


@pytest.fixture
def data_config() -> dict[str, object]:
    return load_model_data_config(DATA_CONFIG_PATH)


@pytest.fixture
def student_config() -> dict[str, object]:
    return load_student_config(STUDENT_CONFIG_PATH)


def test_configs_and_source_lock_are_strict_and_hash_bound(
    data_config: dict[str, object], student_config: dict[str, object]
) -> None:
    lock = load_source_lock(SOURCE_LOCK_PATH, data_config)

    assert config_sha256(data_config) == "1c3fda336a5fae183ea48e813c442daabee5b754bfbd792bad15fabaeb2c52b7"
    assert config_sha256(student_config) == "ce2b50c258ffe2accee58697300e680f7871b51a6241eff2a986ea0bc2146252"
    assert lock["config_sha256"] == config_sha256(data_config)
    assert lock["source_order"] == ["massive-1.1"]
    assert lock["sources"][0]["archive"]["bytes"] == 40_251_390
    assert lock["sources"][0]["verification"]["selected_bytes"] == 51_782_238


def test_canonical_hash_ignores_mapping_order(data_config: dict[str, object]) -> None:
    reordered = dict(reversed(list(data_config.items())))
    assert canonical_json_bytes(reordered) == canonical_json_bytes(data_config)
    assert config_sha256(reordered) == config_sha256(data_config)
    assert canonical_json_bytes(data_config).endswith(b"\n")


def test_language_pair_and_direction_counts_are_frozen(data_config: dict[str, object]) -> None:
    assert tuple(data_config["languages"]["model_tags"]) == contract.LANGUAGE_TAGS
    assert len(data_config["directions"]["undirected_pairs"]) == 10
    assert len(directed_routes()) == 20
    assert len(set(directed_routes())) == 20
    assert len(product_directions()) == 12
    assert len(set(product_directions())) == 12
    assert not set(directed_routes()) & set(contract.EXCLUDED_ROUTES)


@pytest.mark.parametrize(
    ("source", "target", "message"),
    [
        ("eng_Latn", "eng_Latn", "same-language"),
        ("fra_Latn", "eng_Latn", "unknown language"),
    ],
)
def test_invalid_routes_fail(source: str, target: str, message: str) -> None:
    with pytest.raises(ContractError, match=message):
        validate_route(source, target)


def test_all_frozen_routes_validate() -> None:
    assert tuple(validate_route(*route) for route in directed_routes()) == directed_routes()


def test_parallel_sample_schema_and_human_provenance(data_config: dict[str, object]) -> None:
    sample = {
        "sample_id": "massive-1.1:train:0:eng_Latn--jpn_Jpan",
        "sample_group_id": "massive-1.1:train:0",
        "source_id": "massive-1.1",
        "source_version": "1.1",
        "license": "CC-BY-4.0",
        "src_lang": "eng_Latn",
        "tgt_lang": "jpn_Jpan",
        "source_text": "wake me up at five",
        "target_text": "5時に起こして",
        "split": "train",
        "provenance": {
            "kind": "human_parallel",
            "source_record_id": "train:0",
            "alignment_key": "partition,id",
        },
    }
    assert validate_parallel_sample(sample, data_config) == sample


def test_parallel_sample_rejects_unknown_and_missing_fields(data_config: dict[str, object]) -> None:
    sample = {
        "sample_id": "id",
        "sample_group_id": "group",
        "source_id": "source",
        "source_version": "1",
        "license": "CC-BY-4.0",
        "src_lang": "eng_Latn",
        "tgt_lang": "kor_Hang",
        "source_text": "hello",
        "target_text": "안녕하세요",
        "split": "train",
        "unexpected": True,
    }
    with pytest.raises(ContractError, match="unknown fields: unexpected"):
        validate_parallel_sample(sample, data_config)
    del sample["unexpected"]
    del sample["license"]
    with pytest.raises(ContractError, match="missing fields: license"):
        validate_parallel_sample(sample, data_config)


def test_teacher_provenance_requires_complete_hashed_identity(data_config: dict[str, object]) -> None:
    sample = {
        "sample_id": "id",
        "sample_group_id": "group",
        "source_id": "massive-1.1",
        "source_version": "1.1",
        "license": "CC-BY-4.0",
        "src_lang": "eng_Latn",
        "tgt_lang": "zho_Hant",
        "source_text": "hello",
        "target_text": "您好",
        "split": "train",
        "provenance": {
            "kind": "teacher_synthetic",
            "teacher_model": "tencent/Hy-MT2-7B",
            "teacher_revision": "locked-revision",
            "prompt_version": "v1",
            "decode_config_sha256": "0" * 64,
        },
    }
    with pytest.raises(ContractError, match="missing fields: generation_manifest_sha256"):
        validate_parallel_sample(sample, data_config)
    sample["provenance"]["generation_manifest_sha256"] = "not-a-hash"
    with pytest.raises(ContractError, match="lowercase SHA-256"):
        validate_parallel_sample(sample, data_config)


def test_data_config_rejects_unknown_missing_and_changed_pairs(data_config: dict[str, object]) -> None:
    unknown = copy.deepcopy(data_config)
    unknown["implicit_default"] = True
    with pytest.raises(ContractError, match="unknown fields: implicit_default"):
        validate_model_data_config(unknown)

    missing = copy.deepcopy(data_config)
    del missing["paths"]
    with pytest.raises(ContractError, match="missing fields: paths"):
        validate_model_data_config(missing)

    wrong_pair = copy.deepcopy(data_config)
    wrong_pair["directions"]["undirected_pairs"][0]["tags"] = ["eng_Latn", "kor_Hang"]
    with pytest.raises(ContractError, match="pair_id does not match"):
        validate_model_data_config(wrong_pair)


@pytest.mark.parametrize(
    "value",
    [
        "D:/Diesel-MT/data/model/raw",
        "../data/model/raw",
        "data/model/../secret",
        "data\\model\\raw",
        "artifacts/model-training/runtime",
    ],
)
def test_data_path_boundary_rejects_unsafe_or_wrong_roots(value: str) -> None:
    with pytest.raises(ContractError):
        validate_repo_relative_path(value, "data/model", "fixture path")


def test_runtime_root_uses_recordable_absolute_override(
    student_config: dict[str, object], tmp_path: Path
) -> None:
    default = resolve_runtime_root(student_config, ROOT, environ={})
    override = resolve_runtime_root(
        student_config,
        ROOT,
        environ={"DIESEL_MT_MODEL_RUNTIME": str(tmp_path)},
    )
    assert default == (ROOT / "artifacts/model-training/runtime").resolve()
    assert override == tmp_path.resolve()
    with pytest.raises(ContractError, match="absolute path"):
        resolve_runtime_root(
            student_config,
            ROOT,
            environ={"DIESEL_MT_MODEL_RUNTIME": "relative/runtime"},
        )


def test_student_identity_is_from_scratch_and_vocab_bound(student_config: dict[str, object]) -> None:
    assert student_config["identity"]["name"] == "mvp_e8_d2_v48k"
    assert student_config["identity"]["initialization"] == "from_scratch"
    assert student_config["model"]["vocab_size"] == 49_152
    assert student_config["model"]["encoder_layers"] == 8
    assert student_config["model"]["decoder_layers"] == 2
    assert student_config["model"]["tie_word_embeddings"] is True
    assert student_config["training_profile"]["status"] == "requires_td14_benchmark"
    assert student_config["training_profile"]["selection_mode"] == "benchmark_current_host"
    assert student_config["training_profile"]["hardware_identity_source"] == (
        "runtime_probe_and_run_manifest"
    )
    assert student_config["training_profile"]["device_preference_order"] == ["cuda", "cpu"]
    assert student_config["training_profile"]["precision_preference_order"] == [
        "bf16",
        "fp16",
        "fp32",
    ]
    assert all(
        value is None
        for value in student_config["training_profile"]["resource_budget"].values()
    )

    changed = copy.deepcopy(student_config)
    changed["model"]["vocab_size"] = 32_768
    with pytest.raises(ContractError, match="frozen MVP identity"):
        validate_student_config(changed)

    candidate = copy.deepcopy(student_config)
    candidate["training_profile"]["resource_budget"] = {
        "device_memory_budget_mib": 12_345,
        "device_memory_reserve_mib": 1_024,
        "max_device_memory_utilization": 0.85,
        "host_memory_budget_mib": 32_768,
        "dataloader_memory_budget_mib": 4_096,
        "oom_retry_limit": 2,
    }
    assert validate_student_config(candidate) == candidate
    assert config_sha256(candidate) != config_sha256(student_config)

    invalid_memory = copy.deepcopy(candidate)
    invalid_memory["training_profile"]["resource_budget"][
        "device_memory_budget_mib"
    ] = 0
    with pytest.raises(ContractError, match="integer >= 1"):
        validate_student_config(invalid_memory)


def test_student_config_rejects_unknown_fields(student_config: dict[str, object]) -> None:
    changed = copy.deepcopy(student_config)
    changed["model"]["pretrained_model"] = "third-party/model"
    with pytest.raises(ContractError, match="unknown fields: pretrained_model"):
        validate_student_config(changed)

    hardware_bound = copy.deepcopy(student_config)
    hardware_bound["training_profile"]["gpu_model"] = "specific-device"
    with pytest.raises(ContractError, match="unknown fields: gpu_model"):
        validate_student_config(hardware_bound)


def test_frozen_tokenizer_manifest_is_unchanged(student_config: dict[str, object]) -> None:
    digest = hashlib.sha256(TOKENIZER_MANIFEST.read_bytes()).hexdigest()
    assert digest == contract.TOKENIZER_MANIFEST_SHA256
    assert student_config["tokenizer"]["artifact_manifest_sha256"] == digest


def test_source_covers_ten_pairs_and_preserves_native_script_locales(
    data_config: dict[str, object]
) -> None:
    source = data_config["sources"][0]
    assert len(source["pair_coverage"]) == 10
    assert "zho_Hans--zho_Hant" in source["pair_coverage"]
    assert source["locale_to_model_tag"] == {
        "en-US": "eng_Latn",
        "zh-CN": "zho_Hans",
        "zh-TW": "zho_Hant",
        "ja-JP": "jpn_Jpan",
        "ko-KR": "kor_Hang",
    }
    assert "zh-TW" in source["native_script_evidence"]["zho_Hant"]
    assert source["translation_method"].startswith("professional human localization")


def test_source_lock_rejects_config_or_file_identity_drift(data_config: dict[str, object]) -> None:
    lock = load_source_lock(SOURCE_LOCK_PATH, data_config)
    changed_hash = copy.deepcopy(lock)
    changed_hash["config_sha256"] = "0" * 64
    with pytest.raises(ContractError, match="does not match"):
        validate_source_lock(changed_hash, data_config)

    changed_file = copy.deepcopy(lock)
    changed_file["sources"][0]["selected_files"][0]["sha256"] = "INVALID"
    with pytest.raises(ContractError, match="lowercase SHA-256"):
        validate_source_lock(changed_file, data_config)
