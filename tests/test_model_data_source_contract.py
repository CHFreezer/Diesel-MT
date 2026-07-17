from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from model_data_source_contract import (  # noqa: E402
    SourceContractError,
    canonical_sha256,
    load_mvp_60m_source_config,
    load_mvp_60m_source_lock,
    validate_mvp_60m_source_config,
    validate_mvp_60m_source_lock,
)


CONFIG_PATH = ROOT / "configs" / "mvp_60m_distillation_sources.yaml"
LOCK_PATH = ROOT / "configs" / "mvp_60m_distillation_sources.lock.json"


@pytest.fixture
def source_config() -> dict[str, object]:
    return load_mvp_60m_source_config(CONFIG_PATH)


def test_source_config_and_lock_are_hash_bound_and_bounded(
    source_config: dict[str, object],
) -> None:
    lock = load_mvp_60m_source_lock(LOCK_PATH, source_config)

    assert lock["config_sha256"] == canonical_sha256(source_config)
    assert lock["verification"] == {
        "verified_on": "2026-07-17",
        "method": (
            "all admitted sources: local full-file SHA-256 plus cached upstream archive "
            "and selected-member or sorted-concatenation SHA-256; HPLT zho_Hant remains "
            "explicitly rejected"
        ),
        "total_local_bytes": 1_832_030_713,
        "total_download_bytes": 1_914_686_972,
        "total_selected_bytes": 5_945_164_235,
    }
    assert len(lock["records"]) == 17


def test_first_wave_is_quality_first_hant_and_80_20_sampling_mix(
    source_config: dict[str, object],
) -> None:
    source_bank = source_config["source_bank"]
    hant = source_bank["native_hant_admission"]
    wave = source_config["teacher_generation"]["initial_wave"]
    mix = source_config["teacher_generation"]["training_mix"]

    assert source_bank["fixed_non_hant_unique_texts"] == 200_000
    assert source_bank["fixed_texts_per_non_hant_language"] == 50_000
    assert source_bank["actual_native_hant_unique_texts"] == 851
    assert hant["count_policy"] == "quality-gated-actual"
    assert hant["target_selected_texts"] is None
    assert hant["minimum_selected_texts"] is None
    assert hant["refill_to_quota"] == "prohibited"
    assert hant["lower_quality_backfill"] == "prohibited"
    assert hant["synthetic_counts_as_native"] == "prohibited"
    assert wave["fixed_quota_routes"] == 16
    assert wave["accepted_target_per_fixed_route"] == 10_000
    assert wave["fixed_route_accepted_target"] == 160_000
    assert wave["outgoing_hant_count_policy"] == "quality-gated-actual-without-refill"
    assert wave["reverse_pair_policy"]["counts_as_native_hant"] is False
    assert mix == {
        "teacher_sampling_weight": 0.80,
        "human_sampling_weight": 0.20,
        "raw_record_count_policy": (
            "quality-gated-actual-teacher-plus-quality-gated-human"
        ),
        "duplicate_fill": "prohibited",
    }


def test_mvp_pass_line_is_full_flores_dev_and_route_fail_closed(
    source_config: dict[str, object],
) -> None:
    pass_line = source_config["evaluation_boundary"]["mvp_pass_line"]

    assert pass_line["route_generations"] == 19_940
    assert pass_line["primary_metric"] == "sacrebleu-chrf-char6-word2-beta2"
    assert pass_line["minimum_macro_route_chrf"] == 25.0
    assert pass_line["minimum_route_chrf"] == 12.0
    assert pass_line["minimum_routes_at_or_above_chrf_20"] == 16
    assert pass_line["minimum_target_script_compliance_per_route"] == 0.99

    weakened = copy.deepcopy(source_config)
    weakened["evaluation_boundary"]["mvp_pass_line"]["minimum_route_chrf"] = 0.0
    with pytest.raises(SourceContractError, match="pass line"):
        validate_mvp_60m_source_config(weakened)


def test_chinese_tags_follow_flores_script_semantics_not_taiwan_locale(
    source_config: dict[str, object],
) -> None:
    boundary = source_config["coverage"]["chinese_script_boundary"]

    assert boundary == {
        "zho_Hans_semantics": "FLORES-200 zho_Hans",
        "zho_Hant_semantics": "FLORES-200 zho_Hant",
        "locale_policy": (
            "zho_Hant uses Taiwan standard Traditional Chinese as the primary output "
            "convention; formal standard written Traditional Chinese from Hong Kong or "
            "Macau is admissible supplementation and does not change the semantic identity"
        ),
        "admissible_hant": (
            "modern standard written Chinese in Traditional script from Taiwan, Hong "
            "Kong, or Macau with explicit script identity and source provenance"
        ),
        "excluded_hant": (
            "written Cantonese or Guangdong vernacular in any script, including yue_Hant, "
            "plus generic zh or cmn text without reliable language and script provenance"
        ),
        "cantonese_policy": (
            "Cantonese is a separate language capability outside the current five-tag and "
            "20-route scope and must never be mapped to zho_Hant"
        ),
        "conversion_policy": (
            "tool-converted Hans-to-Hant text is synthetic augmentation only and cannot "
            "satisfy native-Hant source or human-anchor admission"
        ),
    }

    localized = copy.deepcopy(source_config)
    localized["coverage"]["chinese_script_boundary"]["zho_Hant_semantics"] = (
        "Taiwan Mandarin"
    )
    with pytest.raises(SourceContractError, match="Chinese script capability"):
        validate_mvp_60m_source_config(localized)


def test_human_anchors_are_small_disjoint_and_not_a_foundation_pretrain(
    source_config: dict[str, object],
) -> None:
    anchors = source_config["human_anchors"]
    components = {component["source_id"]: component for component in anchors["components"]}

    assert anchors["count_policy"] == "per-component-quality-gated-actual-without-refill"
    assert anchors["total_independent_groups_ceiling"] == 12_500
    assert anchors["total_directed_records_ceiling"] == 50_000
    assert anchors["source_bank_overlap"] == "prohibited"
    assert components["alt-v20191206-en-ja-zh"]["directed_record_ceiling"] == 24_000
    assert components["massive-1.1-route-control"] == {
        "source_id": "massive-1.1-route-control",
        "independent_group_ceiling": 500,
        "routes_per_group": 20,
        "directed_record_ceiling": 10_000,
    }


def test_traditional_chinese_has_no_quota_and_candidate_domains_are_capped(
    source_config: dict[str, object],
) -> None:
    components = [
        component
        for component in source_config["source_bank"]["components"]
        if component["language_tag"] == "zho_Hant"
    ]

    assert sum(component["selected_texts"] for component in components) == 851
    hant = source_config["source_bank"]["native_hant_admission"]
    assert hant["domain_share_ceilings"] == {
        "technical": 0.15,
        "legal_and_government": 0.20,
    }
    assert {item["source_id"] for item in hant["candidate_sources"]} == {
        "hplt3-tokenizer-train",
        "massive-1.1-route-control",
        "taiwan-moj-law-api-20260710",
        "hkel-current-legislation",
        "mdn-translated-content-zh-tw",
        "tldr-pages-zh-tw",
        "ud-chinese-hk",
    }
    selected = {item["source_id"]: item["selected_texts"] for item in hant["candidate_sources"]}
    assert selected == {
        "hplt3-tokenizer-train": 0,
        "massive-1.1-route-control": 498,
        "taiwan-moj-law-api-20260710": 170,
        "hkel-current-legislation": 0,
        "mdn-translated-content-zh-tw": 121,
        "tldr-pages-zh-tw": 6,
        "ud-chinese-hk": 56,
    }

    changed = copy.deepcopy(source_config)
    changed["source_bank"]["native_hant_admission"]["target_selected_texts"] = 25_000
    with pytest.raises(SourceContractError, match="no quota fill"):
        validate_mvp_60m_source_config(changed)

    backfilled = copy.deepcopy(source_config)
    backfilled["source_bank"]["native_hant_admission"]["lower_quality_backfill"] = (
        "allowed"
    )
    with pytest.raises(SourceContractError, match="no quota fill"):
        validate_mvp_60m_source_config(backfilled)

    relabeled = copy.deepcopy(source_config)
    relabeled["teacher_generation"]["initial_wave"]["reverse_pair_policy"][
        "counts_as_native_hant"
    ] = True
    with pytest.raises(SourceContractError, match="reverse-pair"):
        validate_mvp_60m_source_config(relabeled)


def test_overflow_holdout_and_formal_test_boundaries_fail_closed(
    source_config: dict[str, object],
) -> None:
    truncating = copy.deepcopy(source_config)
    truncating["source_bank"]["requirements"]["overflow_policy"] = "truncate"
    with pytest.raises(SourceContractError, match="reject rather than truncate"):
        validate_mvp_60m_source_config(truncating)

    holdout = copy.deepcopy(source_config)
    holdout["source_bank"]["requirements"]["tokenizer_holdout_use"] = "allowed"
    with pytest.raises(SourceContractError, match="holdout"):
        validate_mvp_60m_source_config(holdout)

    test_access = copy.deepcopy(source_config)
    test_access["teacher_generation"]["weak_route_patch"]["formal_test_access"] = "allowed"
    with pytest.raises(SourceContractError, match="dev-only"):
        validate_mvp_60m_source_config(test_access)


def test_lock_rejects_config_local_file_and_archive_identity_drift(
    source_config: dict[str, object],
) -> None:
    lock = load_mvp_60m_source_lock(LOCK_PATH, source_config)

    changed_hash = copy.deepcopy(lock)
    changed_hash["config_sha256"] = "0" * 64
    with pytest.raises(SourceContractError, match="does not match"):
        validate_mvp_60m_source_lock(changed_hash, source_config)

    changed_local = copy.deepcopy(lock)
    changed_local["records"][0]["files"][0]["sha256"] = "not-a-hash"
    with pytest.raises(SourceContractError, match="lowercase SHA-256"):
        validate_mvp_60m_source_lock(changed_local, source_config)

    changed_archive = copy.deepcopy(lock)
    changed_archive["records"][3]["bytes"] += 1
    with pytest.raises(SourceContractError, match="total_download_bytes"):
        validate_mvp_60m_source_lock(changed_archive, source_config)
