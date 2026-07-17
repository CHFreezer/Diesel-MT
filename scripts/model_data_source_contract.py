"""TD-02 research contract for the ability-first 60M distillation MVP.

The contract locks verified dependencies and quality-first admission policy;
native Traditional Chinese counts remain an output of later source auditing.
Data materialization, teacher generation, training, and evaluation remain later
tasks. Historical M0/TD-16 artifacts are deliberately outside this module.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping

import yaml


MODEL_TAGS = ("eng_Latn", "zho_Hans", "zho_Hant", "jpn_Jpan", "kor_Hang")
DIRECTED_ROUTES = {
    f"{source}->{target}"
    for source in MODEL_TAGS
    for target in MODEL_TAGS
    if source != target
}
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


class SourceContractError(ValueError):
    """Raised when the TD-02 60M source identity or its byte lock drifts."""


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SourceContractError(f"{label} must be a mapping")
    return value


def _list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise SourceContractError(f"{label} must be a list")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    actual = set(value)
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    if missing or unknown:
        details = []
        if missing:
            details.append(f"missing fields: {', '.join(missing)}")
        if unknown:
            details.append(f"unknown fields: {', '.join(unknown)}")
        raise SourceContractError(f"{label} has {'; '.join(details)}")


def _string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SourceContractError(f"{label} must be a non-empty string")
    return value


def _positive_integer(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise SourceContractError(f"{label} must be a positive integer")
    return value


def _sha256(value: Any, label: str) -> str:
    text = _string(value, label)
    if not _SHA256_RE.fullmatch(text):
        raise SourceContractError(f"{label} must be a lowercase SHA-256")
    return text


def canonical_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8") + b"\n"
    return hashlib.sha256(payload).hexdigest()


def _load_mapping(path: Path) -> dict[str, Any]:
    if path.suffix.lower() == ".json":
        value = json.loads(path.read_text(encoding="utf-8"))
    else:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    return dict(_mapping(value, str(path)))


def validate_mvp_60m_source_config(config: Mapping[str, Any]) -> dict[str, Any]:
    _exact_keys(
        config,
        {
            "schema_version",
            "identity",
            "coverage",
            "evaluation_boundary",
            "source_bank",
            "human_anchors",
            "teacher_generation",
            "quality_gates",
            "resource_budget",
            "sources",
            "license_boundary",
        },
        "MVP 60M source config",
    )
    if config["schema_version"] != 4:
        raise SourceContractError("MVP 60M source schema_version must be 4")

    identity = _mapping(config["identity"], "identity")
    _exact_keys(identity, {"name", "purpose", "status"}, "identity")
    if identity["name"] != "mvp_60m_sequence_distillation_sources":
        raise SourceContractError("MVP 60M source identity changed")
    if identity["status"] != "research-in-progress":
        raise SourceContractError("MVP 60M source research status changed")

    coverage = _mapping(config["coverage"], "coverage")
    _exact_keys(
        coverage,
        {"model_tags", "directed_routes", "route_policy", "chinese_script_boundary"},
        "coverage",
    )
    if tuple(coverage["model_tags"]) != MODEL_TAGS or coverage["directed_routes"] != 20:
        raise SourceContractError("MVP 60M coverage must retain five tags and 20 routes")
    if "no pivot" not in _string(coverage["route_policy"], "coverage.route_policy"):
        raise SourceContractError("teacher routes must be direct, not pivoted")
    chinese_boundary = _mapping(
        coverage["chinese_script_boundary"], "coverage.chinese_script_boundary"
    )
    expected_chinese_boundary = {
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
    _exact_keys(
        chinese_boundary,
        set(expected_chinese_boundary),
        "coverage.chinese_script_boundary",
    )
    if dict(chinese_boundary) != expected_chinese_boundary:
        raise SourceContractError("Chinese script capability must remain aligned to FLORES")

    evaluation = _mapping(config["evaluation_boundary"], "evaluation_boundary")
    _exact_keys(
        evaluation,
        {
            "lock_record_id",
            "lock_path",
            "development_split",
            "formal_test_split",
            "policy",
            "mvp_pass_line",
        },
        "evaluation_boundary",
    )
    pass_line = _mapping(evaluation["mvp_pass_line"], "evaluation_boundary.mvp_pass_line")
    _exact_keys(
        pass_line,
        {
            "protocol",
            "source_sentences_per_tag",
            "route_generations",
            "primary_metric",
            "minimum_macro_route_chrf",
            "minimum_route_chrf",
            "minimum_routes_at_or_above_chrf_20",
            "minimum_target_script_compliance_per_route",
            "maximum_empty_output_rate_per_route",
            "maximum_source_copy_rate_per_route",
            "decision",
        },
        "evaluation_boundary.mvp_pass_line",
    )
    if dict(pass_line) != {
        "protocol": "full-five-tag-flores200-dev-20-direct-routes",
        "source_sentences_per_tag": 997,
        "route_generations": 19_940,
        "primary_metric": "sacrebleu-chrf-char6-word2-beta2",
        "minimum_macro_route_chrf": 25.0,
        "minimum_route_chrf": 12.0,
        "minimum_routes_at_or_above_chrf_20": 16,
        "minimum_target_script_compliance_per_route": 0.99,
        "maximum_empty_output_rate_per_route": 0.01,
        "maximum_source_copy_rate_per_route": 0.02,
        "decision": (
            "all thresholds must pass; otherwise only failing routes may trigger the "
            "single weak-route patch"
        ),
    }:
        raise SourceContractError("MVP dev pass line changed")
    expected_evaluation = {
        "lock_record_id": "flores200-evaluation-boundary",
        "lock_path": "configs/mvp_mt_evaluation.lock.json",
        "development_split": "FLORES-200 dev",
        "formal_test_split": "FLORES-200 devtest",
        "policy": (
            "dev may select the recipe and one weak-route patch; devtest is prohibited "
            "from TD-02 through training and remains one-shot"
        ),
        "mvp_pass_line": pass_line,
    }
    if dict(evaluation) != expected_evaluation:
        raise SourceContractError("FLORES dev/devtest isolation changed")

    source_bank = _mapping(config["source_bank"], "source_bank")
    _exact_keys(
        source_bank,
        {
            "selection_seed",
            "fixed_non_hant_unique_texts",
            "fixed_texts_per_non_hant_language",
            "native_hant_admission",
            "requirements",
            "components",
        },
        "source_bank",
    )
    _string(source_bank["selection_seed"], "source_bank.selection_seed")
    if source_bank["fixed_non_hant_unique_texts"] != 200_000:
        raise SourceContractError("the four non-Hant source banks must total 200000 texts")
    if source_bank["fixed_texts_per_non_hant_language"] != 50_000:
        raise SourceContractError("each non-Hant source bank must contain 50000 texts")

    hant_admission = _mapping(
        source_bank["native_hant_admission"], "source_bank.native_hant_admission"
    )
    _exact_keys(
        hant_admission,
        {
            "count_policy",
            "target_selected_texts",
            "minimum_selected_texts",
            "refill_to_quota",
            "lower_quality_backfill",
            "synthetic_counts_as_native",
            "same_route_duplicate_fill",
            "reuse_across_distinct_target_routes",
            "domain_share_ceilings",
            "candidate_sources",
        },
        "source_bank.native_hant_admission",
    )
    if not (
        hant_admission["count_policy"] == "quality-gated-actual"
        and hant_admission["target_selected_texts"] is None
        and hant_admission["minimum_selected_texts"] is None
        and hant_admission["refill_to_quota"] == "prohibited"
        and hant_admission["lower_quality_backfill"] == "prohibited"
        and hant_admission["synthetic_counts_as_native"] == "prohibited"
        and hant_admission["same_route_duplicate_fill"] == "prohibited"
        and hant_admission["reuse_across_distinct_target_routes"]
        == "allowed-with-shared-semantic-group"
    ):
        raise SourceContractError(
            "native Traditional Chinese must be quality-gated actual with no quota fill"
        )
    domain_ceilings = _mapping(
        hant_admission["domain_share_ceilings"],
        "source_bank.native_hant_admission.domain_share_ceilings",
    )
    if dict(domain_ceilings) != {"technical": 0.15, "legal_and_government": 0.20}:
        raise SourceContractError("native Hant technical/legal domain ceilings changed")
    candidate_fields = {"source_id", "domain", "lock_status"}
    candidates = _list(
        hant_admission["candidate_sources"],
        "source_bank.native_hant_admission.candidate_sources",
    )
    expected_candidates = {
        "hplt3-tokenizer-train": ("general", "requires-strict-requalification"),
        "massive-1.1-route-control": ("daily_and_dialogue", "verified-dependency"),
        "taiwan-moj-law-api-20260710": (
            "legal_and_government",
            "verified-dependency",
        ),
        "hkel-current-legislation": ("legal_and_government", "pending-byte-lock"),
        "mdn-translated-content-zh-tw": ("technical", "pending-byte-lock"),
        "tldr-pages-zh-tw": ("technical", "pending-byte-lock"),
        "ud-chinese-hk": ("daily_and_dialogue", "pending-byte-lock"),
    }
    actual_candidates: dict[str, tuple[str, str]] = {}
    for index, candidate_value in enumerate(candidates):
        candidate = _mapping(
            candidate_value,
            f"source_bank.native_hant_admission.candidate_sources[{index}]",
        )
        _exact_keys(
            candidate,
            candidate_fields,
            f"source_bank.native_hant_admission.candidate_sources[{index}]",
        )
        source_id = _string(candidate["source_id"], f"native Hant candidate {index}")
        if source_id in actual_candidates:
            raise SourceContractError(f"duplicate native Hant candidate: {source_id}")
        actual_candidates[source_id] = (
            _string(candidate["domain"], f"{source_id}.domain"),
            _string(candidate["lock_status"], f"{source_id}.lock_status"),
        )
    if actual_candidates != expected_candidates:
        raise SourceContractError("native Hant research candidates changed")

    requirements = _mapping(source_bank["requirements"], "source_bank.requirements")
    _exact_keys(
        requirements,
        {
            "minimum_characters",
            "maximum_characters",
            "minimum_student_tokens",
            "maximum_student_tokens",
            "tokenizer_path",
            "tokenizer_manifest_sha256",
            "allowed_split",
            "tokenizer_holdout_use",
            "exact_and_near_deduplication",
            "source_anchor_overlap",
            "overflow_policy",
        },
        "source_bank.requirements",
    )
    if not (
        requirements["minimum_characters"] == 20
        and requirements["maximum_characters"] == 256
        and requirements["minimum_student_tokens"] == 4
        and requirements["maximum_student_tokens"] == 256
    ):
        raise SourceContractError("source length bounds changed")
    if requirements["allowed_split"] != "train-only":
        raise SourceContractError("source bank must be train-only")
    if requirements["tokenizer_holdout_use"] != "prohibited":
        raise SourceContractError("tokenizer holdout must not enter teacher generation")
    if requirements["overflow_policy"] != "reject-never-truncate":
        raise SourceContractError("source overflow must reject rather than truncate")
    _sha256(requirements["tokenizer_manifest_sha256"], "tokenizer manifest hash")

    source_components = _list(source_bank["components"], "source_bank.components")
    component_fields = {"source_id", "language_tag", "selected_texts", "selection_unit"}
    totals_by_tag = {tag: 0 for tag in MODEL_TAGS}
    source_component_ids: set[str] = set()
    for index, component_value in enumerate(source_components):
        component = _mapping(component_value, f"source_bank.components[{index}]")
        _exact_keys(component, component_fields, f"source_bank.components[{index}]")
        source_id = _string(component["source_id"], f"source_bank.components[{index}].source_id")
        source_component_ids.add(source_id)
        tag = component["language_tag"]
        if tag not in totals_by_tag:
            raise SourceContractError(f"unknown source-bank language tag: {tag}")
        totals_by_tag[tag] += _positive_integer(
            component["selected_texts"], f"source_bank.components[{index}].selected_texts"
        )
        _string(component["selection_unit"], f"source_bank.components[{index}].selection_unit")
    expected_totals = {tag: 50_000 for tag in MODEL_TAGS if tag != "zho_Hant"}
    expected_totals["zho_Hant"] = 0
    if totals_by_tag != expected_totals:
        raise SourceContractError(
            "fixed source components must contain 50000 texts for each non-Hant tag "
            "and no native-Hant quota"
        )
    if sum(totals_by_tag.values()) != source_bank["fixed_non_hant_unique_texts"]:
        raise SourceContractError("fixed non-Hant source-bank total is inconsistent")

    anchors = _mapping(config["human_anchors"], "human_anchors")
    _exact_keys(
        anchors,
        {
            "selection_seed",
            "count_policy",
            "total_independent_groups_ceiling",
            "total_directed_records_ceiling",
            "source_bank_overlap",
            "components",
        },
        "human_anchors",
    )
    if anchors["source_bank_overlap"] != "prohibited":
        raise SourceContractError("human anchors must be disjoint from the teacher source bank")
    if anchors["count_policy"] != "per-component-quality-gated-actual-without-refill":
        raise SourceContractError("human anchors must use quality-gated actual counts")
    anchor_groups = 0
    anchor_records = 0
    anchor_source_ids: set[str] = set()
    anchor_fields = {
        "source_id",
        "independent_group_ceiling",
        "routes_per_group",
        "directed_record_ceiling",
    }
    for index, component_value in enumerate(_list(anchors["components"], "human_anchors.components")):
        component = _mapping(component_value, f"human_anchors.components[{index}]")
        _exact_keys(component, anchor_fields, f"human_anchors.components[{index}]")
        source_id = _string(component["source_id"], f"human_anchors.components[{index}].source_id")
        anchor_source_ids.add(source_id)
        groups = _positive_integer(
            component["independent_group_ceiling"],
            f"{source_id}.independent_group_ceiling",
        )
        routes = _positive_integer(component["routes_per_group"], f"{source_id}.routes_per_group")
        records = _positive_integer(
            component["directed_record_ceiling"],
            f"{source_id}.directed_record_ceiling",
        )
        if groups * routes != records:
            raise SourceContractError(f"{source_id} directed anchor count is inconsistent")
        anchor_groups += groups
        anchor_records += records
    if (
        anchor_groups != 22_750
        or anchor_records != 50_000
        or anchors["total_independent_groups_ceiling"] != anchor_groups
        or anchors["total_directed_records_ceiling"] != anchor_records
    ):
        raise SourceContractError(
            "human-anchor ceilings must remain 22750 groups / 50000 records"
        )

    teacher = _mapping(config["teacher_generation"], "teacher_generation")
    _exact_keys(
        teacher,
        {
            "teacher_lock_record_id",
            "teacher_selection_path",
            "teacher_selection_sha256",
            "prompt_decode_path",
            "prompt_decode_sha256",
            "initial_wave",
            "training_mix",
            "weak_route_patch",
        },
        "teacher_generation",
    )
    _sha256(teacher["teacher_selection_sha256"], "teacher selection hash")
    _sha256(teacher["prompt_decode_sha256"], "prompt/decode hash")

    wave = _mapping(teacher["initial_wave"], "teacher_generation.initial_wave")
    _exact_keys(
        wave,
        {
            "fixed_quota_routes",
            "fixed_route_definition",
            "accepted_target_per_fixed_route",
            "fixed_route_accepted_target",
            "candidate_scan_limit_per_fixed_route",
            "maximum_fixed_route_candidate_records",
            "outgoing_hant_routes",
            "outgoing_hant_count_policy",
            "accepted_teacher_records_policy",
            "reverse_pair_policy",
        },
        "teacher_generation.initial_wave",
    )
    expected_wave = {
        "fixed_quota_routes": 16,
        "fixed_route_definition": "source tag is not zho_Hant",
        "accepted_target_per_fixed_route": 10_000,
        "fixed_route_accepted_target": 160_000,
        "candidate_scan_limit_per_fixed_route": 12_000,
        "maximum_fixed_route_candidate_records": 192_000,
        "outgoing_hant_routes": 4,
        "outgoing_hant_count_policy": "quality-gated-actual-without-refill",
        "accepted_teacher_records_policy": (
            "160000-fixed-route-target-plus-quality-gated-outgoing-hant"
        ),
        "reverse_pair_policy": wave["reverse_pair_policy"],
    }
    if dict(wave) != expected_wave:
        raise SourceContractError(
            "initial teacher wave must retain 16 fixed routes and quality-gated Hant output routes"
        )
    reverse_policy = _mapping(
        wave["reverse_pair_policy"], "teacher_generation.initial_wave.reverse_pair_policy"
    )
    _exact_keys(
        reverse_policy,
        {
            "enabled",
            "source_pair",
            "reversed_target",
            "count_policy",
            "maximum_fraction_per_outgoing_hant_route",
            "counts_as_native_hant",
            "shared_semantic_group",
            "second_teacher_call",
        },
        "teacher_generation.initial_wave.reverse_pair_policy",
    )
    if dict(reverse_policy) != {
        "enabled": True,
        "source_pair": "accepted-non-hant-to-hant-teacher-pair",
        "reversed_target": "original-human-source-text",
        "count_policy": "quality-gated-actual",
        "maximum_fraction_per_outgoing_hant_route": 0.50,
        "counts_as_native_hant": False,
        "shared_semantic_group": "required",
        "second_teacher_call": "not-required",
    }:
        raise SourceContractError("one-hop Hant reverse-pair policy changed")

    mix = _mapping(teacher["training_mix"], "teacher_generation.training_mix")
    _exact_keys(
        mix,
        {
            "teacher_sampling_weight",
            "human_sampling_weight",
            "raw_record_count_policy",
            "duplicate_fill",
        },
        "teacher_generation.training_mix",
    )
    if dict(mix) != {
        "teacher_sampling_weight": 0.80,
        "human_sampling_weight": 0.20,
        "raw_record_count_policy": (
            "quality-gated-actual-teacher-plus-quality-gated-human"
        ),
        "duplicate_fill": "prohibited",
    }:
        raise SourceContractError(
            "initial training sampling must remain 80% teacher / 20% human without count fill"
        )

    patch = _mapping(teacher["weak_route_patch"], "teacher_generation.weak_route_patch")
    _exact_keys(
        patch,
        {
            "trigger",
            "maximum_increment_accepted_records_per_triggered_route",
            "refill_to_increment",
            "maximum_accepted_records_per_route",
            "maximum_teacher_records_all_routes",
            "maximum_patch_rounds_before_recipe_review",
            "formal_test_access",
        },
        "teacher_generation.weak_route_patch",
    )
    if not (
        patch["maximum_increment_accepted_records_per_triggered_route"] == 10_000
        and patch["refill_to_increment"] == "prohibited"
        and patch["maximum_accepted_records_per_route"] == 50_000
        and patch["maximum_teacher_records_all_routes"] == 1_000_000
        and patch["maximum_patch_rounds_before_recipe_review"] == 1
        and patch["formal_test_access"] == "prohibited"
    ):
        raise SourceContractError("weak-route expansion must remain dev-only and bounded")

    gates = _mapping(config["quality_gates"], "quality_gates")
    _exact_keys(
        gates,
        {
            "source_hard_reject",
            "teacher_hard_reject",
            "reverse_pair_hard_reject",
            "diagnostic_not_automatic_reject",
            "manual_audit",
        },
        "quality_gates",
    )
    required_source_gates = {
        "wrong-language-or-script",
        "written-cantonese-or-guangdong-vernacular",
        "character-or-token-overflow",
        "exact-or-near-duplicate",
        "flores-contamination",
    }
    if not required_source_gates <= set(_list(gates["source_hard_reject"], "source_hard_reject")):
        raise SourceContractError("source hard gates omit a required safety check")
    required_teacher_gates = {
        "empty-output",
        "prompt-echo-or-explanation",
        "wrong-target-script",
        "finish-reason-length",
    }
    if not required_teacher_gates <= set(_list(gates["teacher_hard_reject"], "teacher_hard_reject")):
        raise SourceContractError("teacher hard gates omit a required output check")
    if set(_list(gates["reverse_pair_hard_reject"], "reverse_pair_hard_reject")) != {
        "any-forward-teacher-hard-reject",
        "semantic-roundtrip-failure",
        "number-entity-or-placeholder-drift",
        "missing-shared-semantic-group",
    }:
        raise SourceContractError("reverse-pair hard gates changed")
    audit = _mapping(gates["manual_audit"], "quality_gates.manual_audit")
    if audit["accepted_samples_per_route"] != 20 or audit["rejected_samples_per_route_cap"] != 20:
        raise SourceContractError("manual audit must retain 20 accepted and up to 20 rejected samples per route")

    resource = _mapping(config["resource_budget"], "resource_budget")
    _exact_keys(
        resource,
        {
            "maximum_new_download_bytes",
            "maximum_selected_extract_bytes",
            "maximum_local_source_bytes",
            "generation_storage_policy",
        },
        "resource_budget",
    )
    for key in (
        "maximum_new_download_bytes",
        "maximum_selected_extract_bytes",
        "maximum_local_source_bytes",
    ):
        _positive_integer(resource[key], f"resource_budget.{key}")

    source_values = _list(config["sources"], "sources")
    source_fields = {
        "source_id",
        "roles",
        "version",
        "license",
        "homepage",
        "lock_record_ids",
        "language_tags",
        "available_records",
        "split_policy",
        "limitations",
    }
    sources_by_id: dict[str, Mapping[str, Any]] = {}
    referenced_lock_ids: set[str] = set()
    for index, source_value in enumerate(source_values):
        source = _mapping(source_value, f"sources[{index}]")
        _exact_keys(source, source_fields, f"sources[{index}]")
        source_id = _string(source["source_id"], f"sources[{index}].source_id")
        if source_id in sources_by_id:
            raise SourceContractError(f"duplicate source_id: {source_id}")
        sources_by_id[source_id] = source
        roles = set(_list(source["roles"], f"{source_id}.roles"))
        if not roles or not roles <= {"source-bank", "human-anchor"}:
            raise SourceContractError(f"{source_id} has invalid roles")
        tags = set(_list(source["language_tags"], f"{source_id}.language_tags"))
        if not tags or not tags <= set(MODEL_TAGS):
            raise SourceContractError(f"{source_id} has invalid language tags")
        _positive_integer(source["available_records"], f"{source_id}.available_records")
        if "test" in _string(source["split_policy"], f"{source_id}.split_policy"):
            raise SourceContractError(f"{source_id} must not admit source-native test data")
        if _string(source["version"], f"{source_id}.version").lower() == "latest":
            raise SourceContractError(f"{source_id} must not use a floating version")
        if not _string(source["homepage"], f"{source_id}.homepage").startswith("https://"):
            raise SourceContractError(f"{source_id}.homepage must use HTTPS")
        _string(source["license"], f"{source_id}.license")
        _list(source["limitations"], f"{source_id}.limitations")
        lock_ids = _list(source["lock_record_ids"], f"{source_id}.lock_record_ids")
        if not lock_ids:
            raise SourceContractError(f"{source_id} has no byte-lock record")
        for record_id in lock_ids:
            referenced_lock_ids.add(_string(record_id, f"{source_id}.lock_record_ids"))

    if not source_component_ids <= set(sources_by_id):
        raise SourceContractError("source bank references an unknown source")
    if not anchor_source_ids <= set(sources_by_id):
        raise SourceContractError("human anchors reference an unknown source")
    for source_id in source_component_ids:
        if "source-bank" not in sources_by_id[source_id]["roles"]:
            raise SourceContractError(f"{source_id} is not registered for source-bank use")
    for source_id in anchor_source_ids:
        if "human-anchor" not in sources_by_id[source_id]["roles"]:
            raise SourceContractError(f"{source_id} is not registered for human-anchor use")

    boundary = _mapping(config["license_boundary"], "license_boundary")
    _exact_keys(
        boundary,
        {"teacher_output_rule", "release_rule", "excluded_first_wave"},
        "license_boundary",
    )
    _string(boundary["teacher_output_rule"], "license_boundary.teacher_output_rule")
    _string(boundary["release_rule"], "license_boundary.release_rule")
    _list(boundary["excluded_first_wave"], "license_boundary.excluded_first_wave")

    return dict(config)


def load_mvp_60m_source_config(path: Path) -> dict[str, Any]:
    return validate_mvp_60m_source_config(_load_mapping(path))


def validate_mvp_60m_source_lock(
    lock: Mapping[str, Any], config: Mapping[str, Any]
) -> dict[str, Any]:
    validated_config = validate_mvp_60m_source_config(config)
    _exact_keys(
        lock,
        {"schema_version", "config_sha256", "record_order", "records", "verification"},
        "MVP 60M source lock",
    )
    if lock["schema_version"] != 4:
        raise SourceContractError("MVP 60M source lock schema_version must be 4")
    if _sha256(lock["config_sha256"], "lock config hash") != canonical_sha256(validated_config):
        raise SourceContractError("MVP 60M source lock does not match its config")

    sources = validated_config["sources"]
    expected_ids = {
        record_id
        for source in sources
        for record_id in source["lock_record_ids"]
    }
    expected_ids.add(validated_config["teacher_generation"]["teacher_lock_record_id"])
    expected_ids.add(validated_config["evaluation_boundary"]["lock_record_id"])
    record_order = _list(lock["record_order"], "lock.record_order")
    if len(record_order) != len(set(record_order)) or set(record_order) != expected_ids:
        raise SourceContractError("source lock record order is incomplete or duplicated")

    records = _list(lock["records"], "lock.records")
    if len(records) != len(record_order):
        raise SourceContractError("source lock record count differs from record_order")
    seen_ids: set[str] = set()
    local_bytes = 0
    download_bytes = 0
    selected_bytes = 0
    file_fields = {"path", "role", "bytes", "sha256"}
    for index, record_value in enumerate(records):
        record = _mapping(record_value, f"records[{index}]")
        kind = record.get("kind")
        if kind == "local-derived":
            _exact_keys(record, {"record_id", "kind", "files"}, f"records[{index}]")
            files = _list(record["files"], f"records[{index}].files")
        elif kind == "upstream-archive":
            _exact_keys(
                record,
                {"record_id", "kind", "download_uri", "bytes", "sha256", "selected_files"},
                f"records[{index}]",
            )
            if not _string(record["download_uri"], f"records[{index}].download_uri").startswith("https://"):
                raise SourceContractError("upstream archive URI must use HTTPS")
            download_bytes += _positive_integer(record["bytes"], f"records[{index}].bytes")
            _sha256(record["sha256"], f"records[{index}].sha256")
            files = _list(record["selected_files"], f"records[{index}].selected_files")
        else:
            raise SourceContractError(f"records[{index}] has unsupported kind: {kind}")

        record_id = _string(record["record_id"], f"records[{index}].record_id")
        if record_id in seen_ids or record_id not in expected_ids:
            raise SourceContractError(f"unknown or duplicate lock record: {record_id}")
        if record_id != record_order[index]:
            raise SourceContractError("lock record order differs from record_order")
        seen_ids.add(record_id)
        if not files:
            raise SourceContractError(f"{record_id} has no selected-file identity")
        paths: set[str] = set()
        for file_index, file_value in enumerate(files):
            file_record = _mapping(file_value, f"{record_id}.files[{file_index}]")
            _exact_keys(file_record, file_fields, f"{record_id}.files[{file_index}]")
            path = _string(file_record["path"], f"{record_id}.files[{file_index}].path")
            if path in paths:
                raise SourceContractError(f"{record_id} contains a duplicate selected path")
            paths.add(path)
            _string(file_record["role"], f"{record_id}.{path}.role")
            size = _positive_integer(file_record["bytes"], f"{record_id}.{path}.bytes")
            _sha256(file_record["sha256"], f"{record_id}.{path}.sha256")
            if kind == "local-derived":
                local_bytes += size
            else:
                selected_bytes += size

    if seen_ids != expected_ids:
        raise SourceContractError("source lock is incomplete")
    budget = validated_config["resource_budget"]
    if local_bytes > budget["maximum_local_source_bytes"]:
        raise SourceContractError("locked local sources exceed the local byte budget")
    if download_bytes > budget["maximum_new_download_bytes"]:
        raise SourceContractError("locked downloads exceed the download byte budget")
    if selected_bytes > budget["maximum_selected_extract_bytes"]:
        raise SourceContractError("locked selected files exceed the extraction byte budget")

    verification = _mapping(lock["verification"], "lock.verification")
    _exact_keys(
        verification,
        {
            "verified_on",
            "method",
            "total_local_bytes",
            "total_download_bytes",
            "total_selected_bytes",
        },
        "lock.verification",
    )
    _string(verification["verified_on"], "lock.verification.verified_on")
    _string(verification["method"], "lock.verification.method")
    if verification["total_local_bytes"] != local_bytes:
        raise SourceContractError("total_local_bytes is inconsistent")
    if verification["total_download_bytes"] != download_bytes:
        raise SourceContractError("total_download_bytes is inconsistent")
    if verification["total_selected_bytes"] != selected_bytes:
        raise SourceContractError("total_selected_bytes is inconsistent")
    return dict(lock)


def load_mvp_60m_source_lock(path: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    return validate_mvp_60m_source_lock(_load_mapping(path), config)
