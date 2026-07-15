"""Strict contracts shared by the Diesel-MT model-training workflow."""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

import yaml


LANGUAGE_TAGS = (
    "eng_Latn",
    "zho_Hans",
    "zho_Hant",
    "jpn_Jpan",
    "kor_Hang",
)
PRODUCT_LANGUAGES = ("Chinese", "English", "Japanese", "Korean")
MODEL_TO_PRODUCT = {
    "eng_Latn": "English",
    "zho_Hans": "Chinese",
    "zho_Hant": "Chinese",
    "jpn_Jpan": "Japanese",
    "kor_Hang": "Korean",
}
UNDIRECTED_PAIRS = (
    ("eng_Latn", "jpn_Jpan"),
    ("eng_Latn", "kor_Hang"),
    ("jpn_Jpan", "kor_Hang"),
    ("eng_Latn", "zho_Hans"),
    ("jpn_Jpan", "zho_Hans"),
    ("kor_Hang", "zho_Hans"),
    ("eng_Latn", "zho_Hant"),
    ("jpn_Jpan", "zho_Hant"),
    ("kor_Hang", "zho_Hant"),
    ("zho_Hans", "zho_Hant"),
)
EXCLUDED_ROUTES: tuple[tuple[str, str], ...] = ()
TOKENIZER_MANIFEST_SHA256 = (
    "eb79ae22f523f1d9c9fcf75b80f2b322e3c2882a8fddb7545b5933dd4053fa7f"
)

_PAIR_SETS = {frozenset(pair) for pair in UNDIRECTED_PAIRS}
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_ENV_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")


class ContractError(ValueError):
    """Raised when a model-training contract is incomplete or ambiguous."""


def canonical_json_bytes(value: Any) -> bytes:
    """Return the one canonical byte representation used for config identity."""

    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def config_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _load_mapping(path: Path) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ContractError(f"cannot load {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ContractError(f"{path} must contain a mapping")
    return value


def _exact_keys(
    value: Mapping[str, Any],
    required: set[str],
    context: str,
    *,
    optional: set[str] | None = None,
) -> None:
    optional = optional or set()
    missing = sorted(required - set(value))
    unknown = sorted(set(value) - required - optional)
    if missing:
        raise ContractError(f"{context} missing fields: {', '.join(missing)}")
    if unknown:
        raise ContractError(f"{context} unknown fields: {', '.join(unknown)}")


def _mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ContractError(f"{context} must be a mapping")
    return value


def _list(value: Any, context: str) -> list[Any]:
    if not isinstance(value, list):
        raise ContractError(f"{context} must be a list")
    return value


def _nonempty_string(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ContractError(f"{context} must be a non-empty string")
    return value


def _optional_integer(
    value: Any,
    context: str,
    *,
    minimum: int,
) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ContractError(f"{context} must be null or an integer >= {minimum}")
    return value


def _optional_utilization(value: Any, context: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 < value <= 1:
        raise ContractError(f"{context} must be null or a number in (0, 1]")
    return float(value)


def _sha256(value: Any, context: str) -> str:
    text = _nonempty_string(value, context)
    if not _SHA256_RE.fullmatch(text):
        raise ContractError(f"{context} must be a lowercase SHA-256")
    return text


def pair_id(tags: tuple[str, str] | list[str]) -> str:
    if len(tags) != 2:
        raise ContractError("pair must contain exactly two tags")
    return f"{tags[0]}--{tags[1]}"


def directed_routes() -> tuple[tuple[str, str], ...]:
    routes: list[tuple[str, str]] = []
    for left, right in UNDIRECTED_PAIRS:
        routes.extend(((left, right), (right, left)))
    return tuple(routes)


def product_directions() -> tuple[tuple[str, str], ...]:
    return tuple(
        (source, target)
        for source in PRODUCT_LANGUAGES
        for target in PRODUCT_LANGUAGES
        if source != target
    )


def validate_route(source: str, target: str) -> tuple[str, str]:
    unknown = sorted({source, target} - set(LANGUAGE_TAGS))
    if unknown:
        raise ContractError(f"route uses unknown language tags: {', '.join(unknown)}")
    if source == target:
        raise ContractError("same-language routes are forbidden")
    if frozenset((source, target)) not in _PAIR_SETS:
        raise ContractError(f"route is outside the 20-route allowlist: {source}->{target}")
    return source, target


def validate_repo_relative_path(value: Any, required_root: str, context: str) -> str:
    text = _nonempty_string(value, context)
    if "\\" in text or re.match(r"^[A-Za-z]:", text):
        raise ContractError(f"{context} must use a repository-relative POSIX path")
    path = PurePosixPath(text)
    if path.is_absolute() or ".." in path.parts or "." in path.parts:
        raise ContractError(f"{context} escapes its repository path boundary")
    normalized = path.as_posix()
    if normalized != text.rstrip("/"):
        raise ContractError(f"{context} is not a normalized POSIX path")
    root = PurePosixPath(required_root).as_posix().rstrip("/")
    if normalized != root and not normalized.startswith(root + "/"):
        raise ContractError(f"{context} must stay under {root}/")
    return normalized


def resolve_runtime_root(
    config: Mapping[str, Any],
    repository_root: Path,
    environ: Mapping[str, str] | None = None,
) -> Path:
    runtime = _mapping(config["runtime"], "student.runtime")
    env_name = _nonempty_string(runtime["hot_root_override_env"], "runtime env")
    environ = os.environ if environ is None else environ
    override = environ.get(env_name)
    if override:
        path = Path(override).expanduser()
        if not path.is_absolute():
            raise ContractError(f"{env_name} must resolve to an absolute path")
        return path.resolve()
    configured = validate_repo_relative_path(
        runtime["default_hot_root"], "artifacts/model-training/runtime", "runtime root"
    )
    return (repository_root / PurePosixPath(configured)).resolve()


def _validate_languages_and_directions(config: Mapping[str, Any]) -> None:
    languages = _mapping(config["languages"], "data.languages")
    _exact_keys(
        languages,
        {"product_languages", "model_tags", "model_to_product"},
        "data.languages",
    )
    if tuple(_list(languages["product_languages"], "product languages")) != PRODUCT_LANGUAGES:
        raise ContractError("product_languages must be the frozen four-language order")
    if tuple(_list(languages["model_tags"], "model tags")) != LANGUAGE_TAGS:
        raise ContractError("model_tags must be the frozen five-tag order")
    if dict(_mapping(languages["model_to_product"], "model_to_product")) != MODEL_TO_PRODUCT:
        raise ContractError("model_to_product must preserve both Chinese script tags")

    directions = _mapping(config["directions"], "data.directions")
    _exact_keys(directions, {"undirected_pairs", "excluded_routes", "counts"}, "data.directions")
    records = _list(directions["undirected_pairs"], "undirected_pairs")
    seen: set[frozenset[str]] = set()
    pair_ids: list[str] = []
    for index, record_value in enumerate(records):
        record = _mapping(record_value, f"undirected_pairs[{index}]")
        _exact_keys(record, {"pair_id", "tags"}, f"undirected_pairs[{index}]")
        tags = _list(record["tags"], f"undirected_pairs[{index}].tags")
        if len(tags) != 2 or any(tag not in LANGUAGE_TAGS for tag in tags):
            raise ContractError(f"undirected_pairs[{index}] has invalid tags")
        pair = frozenset(tags)
        if pair in seen:
            raise ContractError("undirected_pairs contains a duplicate pair")
        if pair not in _PAIR_SETS:
            raise ContractError("undirected_pairs contains a pair outside the allowlist")
        if record["pair_id"] != pair_id(tags):
            raise ContractError(f"undirected_pairs[{index}] pair_id does not match tag order")
        seen.add(pair)
        pair_ids.append(str(record["pair_id"]))
    if seen != _PAIR_SETS:
        raise ContractError("undirected_pairs must contain the frozen ten-pair set")

    excluded = [tuple(item) for item in _list(directions["excluded_routes"], "excluded_routes")]
    if tuple(excluded) != EXCLUDED_ROUTES:
        raise ContractError("excluded_routes must be empty for the complete 20-route contract")
    counts = _mapping(directions["counts"], "direction counts")
    _exact_keys(
        counts,
        {"product_languages", "model_tags", "undirected_pairs", "directed_routes", "product_directions"},
        "direction counts",
    )
    if dict(counts) != {
        "product_languages": 4,
        "model_tags": 5,
        "undirected_pairs": 10,
        "directed_routes": 20,
        "product_directions": 12,
    }:
        raise ContractError("direction counts do not match the frozen terminology")


def _validate_sample_schema(config: Mapping[str, Any]) -> None:
    schema = _mapping(config["sample_schema"], "data.sample_schema")
    _exact_keys(
        schema,
        {"version", "required_fields", "optional_fields", "split_values", "provenance_fields"},
        "data.sample_schema",
    )
    expected_required = [
        "sample_id",
        "sample_group_id",
        "source_id",
        "source_version",
        "license",
        "src_lang",
        "tgt_lang",
        "source_text",
        "target_text",
        "split",
    ]
    if schema["version"] != 1 or schema["required_fields"] != expected_required:
        raise ContractError("sample schema required fields or version changed")
    if schema["optional_fields"] != ["provenance"]:
        raise ContractError("sample schema optional fields must contain only provenance")
    if schema["split_values"] != ["train", "dev", "test"]:
        raise ContractError("sample split values must be train/dev/test")
    provenance = _mapping(schema["provenance_fields"], "provenance_fields")
    _exact_keys(
        provenance,
        {"human_parallel", "teacher_synthetic", "script_conversion"},
        "provenance_fields",
    )
    expected = {
        "human_parallel": ["kind", "source_record_id", "alignment_key"],
        "teacher_synthetic": [
            "kind",
            "teacher_model",
            "teacher_revision",
            "prompt_version",
            "decode_config_sha256",
            "generation_manifest_sha256",
        ],
        "script_conversion": [
            "kind",
            "tool",
            "tool_version",
            "source_sample_id",
            "generation_manifest_sha256",
        ],
    }
    if dict(provenance) != expected:
        raise ContractError("sample provenance field sets do not match the frozen schema")


def _validate_data_paths(config: Mapping[str, Any]) -> None:
    paths = _mapping(config["paths"], "data.paths")
    _exact_keys(paths, {"root", "raw", "cache", "interim", "corpus", "reports"}, "data.paths")
    for name, value in paths.items():
        validate_repo_relative_path(value, "data/model", f"data.paths.{name}")
    if paths["root"] != "data/model" or paths["corpus"] != "data/model/corpus/mvp":
        raise ContractError("data root and MVP corpus paths are frozen")


def _validate_source(source_value: Any, index: int) -> set[str]:
    source = _mapping(source_value, f"sources[{index}]")
    fields = {
        "source_id",
        "enabled",
        "name",
        "version",
        "source_type",
        "license",
        "license_url",
        "homepage",
        "download_uri",
        "format",
        "compression",
        "text_field",
        "alignment_key",
        "partition_field",
        "partition_map",
        "locale_to_model_tag",
        "pair_coverage",
        "domain",
        "translation_method",
        "native_script_evidence",
        "limitations",
        "citation",
    }
    _exact_keys(source, fields, f"sources[{index}]")
    source_id = _nonempty_string(source["source_id"], f"sources[{index}].source_id")
    version = _nonempty_string(source["version"], f"sources[{index}].version")
    if version.lower() == "latest" or "/latest" in str(source["download_uri"]).lower():
        raise ContractError(f"{source_id} must not use a floating version")
    if source["enabled"] is not True:
        return set()
    for field in ("name", "source_type", "license", "license_url", "homepage", "download_uri"):
        _nonempty_string(source[field], f"sources[{index}].{field}")
    if not str(source["download_uri"]).startswith("https://"):
        raise ContractError(f"{source_id} download_uri must use HTTPS")
    if source["format"] != "jsonl" or source["compression"] != "tar.gz":
        raise ContractError(f"{source_id} must use the locked JSONL tar.gz adapter")
    if source["text_field"] != "utt":
        raise ContractError(f"{source_id} must use raw utt text, not annotations")
    if source["alignment_key"] != ["partition", "id"]:
        raise ContractError(f"{source_id} alignment key must be partition + id")
    if source["partition_field"] != "partition" or source["partition_map"] != {
        "train": "train",
        "dev": "dev",
        "test": "test",
    }:
        raise ContractError(f"{source_id} partition contract changed")
    locales = _mapping(source["locale_to_model_tag"], f"{source_id}.locale_to_model_tag")
    if set(locales.values()) != set(LANGUAGE_TAGS) or len(locales) != len(LANGUAGE_TAGS):
        raise ContractError(f"{source_id} must map exactly one locale to each model tag")
    if locales.get("zh-CN") != "zho_Hans" or locales.get("zh-TW") != "zho_Hant":
        raise ContractError(f"{source_id} must preserve explicit Simplified/Traditional locales")
    coverage = set(_list(source["pair_coverage"], f"{source_id}.pair_coverage"))
    expected_ids = {pair_id(list(pair)) for pair in UNDIRECTED_PAIRS}
    if not coverage <= expected_ids:
        raise ContractError(f"{source_id} pair coverage contains an unknown pair")
    evidence = _mapping(source["native_script_evidence"], f"{source_id}.native_script_evidence")
    _exact_keys(evidence, {"zho_Hans", "zho_Hant"}, f"{source_id}.native_script_evidence")
    if "zh-CN" not in str(evidence["zho_Hans"]) or "zh-TW" not in str(evidence["zho_Hant"]):
        raise ContractError(f"{source_id} native script evidence must cite both locale labels")
    _list(source["limitations"], f"{source_id}.limitations")
    citation = _mapping(source["citation"], f"{source_id}.citation")
    _exact_keys(citation, {"paper_url", "bibkey"}, f"{source_id}.citation")
    return coverage


def validate_model_data_config(config: Mapping[str, Any]) -> dict[str, Any]:
    _exact_keys(
        config,
        {
            "schema_version",
            "identity",
            "reproducibility",
            "languages",
            "directions",
            "sample_schema",
            "paths",
            "budgets",
            "sources",
        },
        "model data config",
    )
    if config["schema_version"] != 2:
        raise ContractError("model data schema_version must be 2")
    identity = _mapping(config["identity"], "data.identity")
    _exact_keys(identity, {"name", "purpose", "status"}, "data.identity")
    if identity["name"] != "mvp_model_data" or identity["status"] != "locked-sources":
        raise ContractError("model data identity is not the active locked-source contract")
    reproducibility = _mapping(config["reproducibility"], "data.reproducibility")
    _exact_keys(
        reproducibility,
        {
            "canonical_serialization",
            "content_hash",
            "output_encoding",
            "output_newline",
            "final_trailing_newline",
            "config_hash_scope",
        },
        "data.reproducibility",
    )
    expected_reproducibility = {
        "canonical_serialization": "UTF-8 JSON, sorted keys, compact separators, LF terminator",
        "content_hash": "sha256",
        "output_encoding": "utf-8",
        "output_newline": "LF",
        "final_trailing_newline": True,
        "config_hash_scope": "entire parsed config",
    }
    if dict(reproducibility) != expected_reproducibility:
        raise ContractError("data reproducibility contract changed")
    _validate_languages_and_directions(config)
    _validate_sample_schema(config)
    _validate_data_paths(config)
    budgets = _mapping(config["budgets"], "data.budgets")
    _exact_keys(
        budgets,
        {
            "minimum_accepted_per_undirected_pair",
            "scan_limit_rows_per_locale",
            "download_max_bytes",
            "selected_extract_max_bytes",
            "source_rows_per_locale",
            "source_partition_rows_per_locale",
        },
        "data.budgets",
    )
    minimums = _mapping(
        budgets["minimum_accepted_per_undirected_pair"],
        "minimum accepted per pair",
    )
    _exact_keys(minimums, {"train", "dev", "test"}, "minimum accepted per pair")
    partitions = _mapping(
        budgets["source_partition_rows_per_locale"],
        "source partition rows",
    )
    _exact_keys(partitions, {"train", "dev", "test"}, "source partition rows")
    numeric_values = [
        *minimums.values(),
        budgets["scan_limit_rows_per_locale"],
        budgets["download_max_bytes"],
        budgets["selected_extract_max_bytes"],
        budgets["source_rows_per_locale"],
        *partitions.values(),
    ]
    if any(not isinstance(value, int) or value <= 0 for value in numeric_values):
        raise ContractError("all data budgets must be positive integers")
    if sum(partitions.values()) != budgets["source_rows_per_locale"]:
        raise ContractError("partition row counts must sum to source_rows_per_locale")
    if budgets["scan_limit_rows_per_locale"] != budgets["source_rows_per_locale"]:
        raise ContractError("scan limit must lock the complete selected locale")
    if any(minimums[name] > partitions[name] for name in minimums):
        raise ContractError("minimum accepted budget exceeds source partition rows")

    source_ids: set[str] = set()
    coverage: set[str] = set()
    sources = _list(config["sources"], "data.sources")
    if not sources:
        raise ContractError("at least one data source must be locked")
    for index, source in enumerate(sources):
        source_id = str(_mapping(source, f"sources[{index}]").get("source_id", ""))
        if source_id in source_ids:
            raise ContractError(f"duplicate source_id: {source_id}")
        source_ids.add(source_id)
        coverage.update(_validate_source(source, index))
    expected_ids = {pair_id(list(pair)) for pair in UNDIRECTED_PAIRS}
    if coverage != expected_ids:
        missing = sorted(expected_ids - coverage)
        raise ContractError(f"enabled sources do not cover all ten pairs: {', '.join(missing)}")
    return dict(config)


def load_model_data_config(path: Path) -> dict[str, Any]:
    return validate_model_data_config(_load_mapping(path))


def validate_student_config(config: Mapping[str, Any]) -> dict[str, Any]:
    _exact_keys(
        config,
        {
            "schema_version",
            "identity",
            "tokenizer",
            "model",
            "runtime",
            "publish",
            "reproducibility",
            "training_profile",
        },
        "student config",
    )
    if config["schema_version"] != 2:
        raise ContractError("student schema_version must be 2")
    identity = _mapping(config["identity"], "student.identity")
    _exact_keys(identity, {"name", "architecture", "initialization", "random_seed"}, "student.identity")
    if identity["name"] != "mvp_e8_d2_v48k":
        raise ContractError("student name must be mvp_e8_d2_v48k")
    if identity["architecture"] != "M2M100ForConditionalGeneration":
        raise ContractError("student architecture must use M2M100 semantics")
    if identity["initialization"] != "from_scratch":
        raise ContractError("student must be initialized from scratch")
    if not isinstance(identity["random_seed"], int):
        raise ContractError("student random_seed must be an integer")

    tokenizer = _mapping(config["tokenizer"], "student.tokenizer")
    _exact_keys(
        tokenizer,
        {"path", "artifact_manifest_sha256", "expected_vocab_size", "required_language_tokens"},
        "student.tokenizer",
    )
    validate_repo_relative_path(tokenizer["path"], "artifacts/tokenizers/mvp-tokenizer-v0", "tokenizer path")
    if _sha256(tokenizer["artifact_manifest_sha256"], "tokenizer manifest") != TOKENIZER_MANIFEST_SHA256:
        raise ContractError("student tokenizer manifest is not the frozen MVP artifact")
    if tokenizer["expected_vocab_size"] != 49_152:
        raise ContractError("student tokenizer vocab size must be 49,152")
    if tuple(tokenizer["required_language_tokens"]) != LANGUAGE_TAGS:
        raise ContractError("student language token order changed")

    model = _mapping(config["model"], "student.model")
    model_fields = {
        "vocab_size",
        "d_model",
        "encoder_ffn_dim",
        "decoder_ffn_dim",
        "encoder_layers",
        "decoder_layers",
        "encoder_attention_heads",
        "decoder_attention_heads",
        "max_position_embeddings",
        "activation_function",
        "dropout",
        "attention_dropout",
        "activation_dropout",
        "encoder_layerdrop",
        "decoder_layerdrop",
        "scale_embedding",
        "tie_word_embeddings",
        "use_cache",
        "bos_token_id",
        "pad_token_id",
        "eos_token_id",
        "decoder_start_token_id",
        "forced_eos_token_id",
    }
    _exact_keys(model, model_fields, "student.model")
    frozen_values = {
        "vocab_size": 49_152,
        "d_model": 512,
        "encoder_ffn_dim": 2_048,
        "decoder_ffn_dim": 2_048,
        "encoder_layers": 8,
        "decoder_layers": 2,
        "encoder_attention_heads": 8,
        "decoder_attention_heads": 8,
        "max_position_embeddings": 1_024,
        "activation_function": "relu",
        "dropout": 0.1,
        "attention_dropout": 0.1,
        "activation_dropout": 0.0,
        "encoder_layerdrop": 0.0,
        "decoder_layerdrop": 0.0,
        "scale_embedding": True,
        "tie_word_embeddings": True,
        "use_cache": True,
        "bos_token_id": 0,
        "pad_token_id": 1,
        "eos_token_id": 2,
        "decoder_start_token_id": 2,
        "forced_eos_token_id": None,
    }
    if dict(model) != frozen_values:
        raise ContractError("student model fields differ from the frozen MVP identity")

    runtime = _mapping(config["runtime"], "student.runtime")
    _exact_keys(
        runtime,
        {"default_hot_root", "hot_root_override_env", "checkpoints_subdir", "staging_subdir", "logs_subdir"},
        "student.runtime",
    )
    validate_repo_relative_path(runtime["default_hot_root"], "artifacts/model-training/runtime", "runtime root")
    if not _ENV_RE.fullmatch(str(runtime["hot_root_override_env"])):
        raise ContractError("runtime override environment variable is invalid")
    for field in ("checkpoints_subdir", "staging_subdir", "logs_subdir"):
        value = _nonempty_string(runtime[field], f"student.runtime.{field}")
        if PurePosixPath(value).is_absolute() or ".." in PurePosixPath(value).parts or "\\" in value:
            raise ContractError(f"student.runtime.{field} must be a safe relative subdirectory")

    publish = _mapping(config["publish"], "student.publish")
    _exact_keys(publish, {"root", "hf", "ct2_float32", "ct2_int8"}, "student.publish")
    for field, value in publish.items():
        validate_repo_relative_path(value, "artifacts/models/mvp_e8_d2_v48k", f"student.publish.{field}")

    reproducibility = _mapping(config["reproducibility"], "student.reproducibility")
    _exact_keys(
        reproducibility,
        {"canonical_serialization", "content_hash", "config_hash_scope"},
        "student.reproducibility",
    )
    if dict(reproducibility) != {
        "canonical_serialization": "UTF-8 JSON, sorted keys, compact separators, LF terminator",
        "content_hash": "sha256",
        "config_hash_scope": "entire parsed config",
    }:
        raise ContractError("student reproducibility contract changed")

    profile = _mapping(config["training_profile"], "student.training_profile")
    _exact_keys(
        profile,
        {
            "status",
            "selection_mode",
            "hardware_identity_source",
            "device_preference_order",
            "precision_preference_order",
            "resource_budget",
            "unfrozen_fields",
        },
        "student.training_profile",
    )
    if profile["status"] != "requires_td14_benchmark":
        raise ContractError("training profile must remain provisional until TD-14")
    if profile["selection_mode"] != "benchmark_current_host":
        raise ContractError("training profile must be selected by a current-host benchmark")
    if profile["hardware_identity_source"] != "runtime_probe_and_run_manifest":
        raise ContractError("hardware identity must come from runtime probing and the run manifest")

    device_preferences = _list(
        profile["device_preference_order"], "student.training_profile.device_preference_order"
    )
    if (
        not device_preferences
        or any(not isinstance(value, str) for value in device_preferences)
        or len(device_preferences) != len(set(device_preferences))
        or not set(device_preferences) <= {"cuda", "cpu"}
    ):
        raise ContractError("device preferences must be a unique non-empty cuda/cpu subset")
    precision_preferences = _list(
        profile["precision_preference_order"],
        "student.training_profile.precision_preference_order",
    )
    if (
        not precision_preferences
        or any(not isinstance(value, str) for value in precision_preferences)
        or len(precision_preferences) != len(set(precision_preferences))
        or not set(precision_preferences) <= {"bf16", "fp16", "fp32"}
    ):
        raise ContractError("precision preferences must be a unique non-empty supported subset")

    resource_budget = _mapping(
        profile["resource_budget"], "student.training_profile.resource_budget"
    )
    resource_budget_fields = {
        "device_memory_budget_mib",
        "device_memory_reserve_mib",
        "max_device_memory_utilization",
        "host_memory_budget_mib",
        "dataloader_memory_budget_mib",
        "oom_retry_limit",
    }
    _exact_keys(
        resource_budget,
        resource_budget_fields,
        "student.training_profile.resource_budget",
    )
    present_budget_fields = [value is not None for value in resource_budget.values()]
    if any(present_budget_fields) and not all(present_budget_fields):
        raise ContractError("resource budget candidates must fill every field or leave all fields null")
    _optional_integer(
        resource_budget["device_memory_budget_mib"],
        "student.training_profile.resource_budget.device_memory_budget_mib",
        minimum=1,
    )
    _optional_integer(
        resource_budget["device_memory_reserve_mib"],
        "student.training_profile.resource_budget.device_memory_reserve_mib",
        minimum=0,
    )
    _optional_utilization(
        resource_budget["max_device_memory_utilization"],
        "student.training_profile.resource_budget.max_device_memory_utilization",
    )
    host_memory_budget = _optional_integer(
        resource_budget["host_memory_budget_mib"],
        "student.training_profile.resource_budget.host_memory_budget_mib",
        minimum=1,
    )
    dataloader_memory_budget = _optional_integer(
        resource_budget["dataloader_memory_budget_mib"],
        "student.training_profile.resource_budget.dataloader_memory_budget_mib",
        minimum=1,
    )
    _optional_integer(
        resource_budget["oom_retry_limit"],
        "student.training_profile.resource_budget.oom_retry_limit",
        minimum=0,
    )
    if (
        host_memory_budget is not None
        and dataloader_memory_budget is not None
        and dataloader_memory_budget > host_memory_budget
    ):
        raise ContractError("dataloader memory budget must not exceed the host memory budget")
    expected_unfrozen = [
        "device",
        "precision",
        "resource_budget.device_memory_budget_mib",
        "resource_budget.device_memory_reserve_mib",
        "resource_budget.max_device_memory_utilization",
        "resource_budget.host_memory_budget_mib",
        "resource_budget.dataloader_memory_budget_mib",
        "resource_budget.oom_retry_limit",
        "micro_batch_size",
        "gradient_accumulation_steps",
        "gradient_checkpointing",
        "max_source_length",
        "max_target_length",
        "dataloader_workers",
        "optimizer",
        "scheduler",
        "warmup_steps",
        "max_optimizer_steps",
        "validation_frequency",
        "checkpoint_frequency",
    ]
    if profile["unfrozen_fields"] != expected_unfrozen:
        raise ContractError("TD-14 training-profile field list changed")
    return dict(config)


def load_student_config(path: Path) -> dict[str, Any]:
    return validate_student_config(_load_mapping(path))


def validate_parallel_sample(sample: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    schema = _mapping(config["sample_schema"], "data.sample_schema")
    required = set(schema["required_fields"])
    optional = set(schema["optional_fields"])
    _exact_keys(sample, required, "parallel sample", optional=optional)
    for field in (
        "sample_id",
        "sample_group_id",
        "source_id",
        "source_version",
        "license",
        "source_text",
        "target_text",
    ):
        _nonempty_string(sample[field], f"parallel sample.{field}")
    validate_route(str(sample["src_lang"]), str(sample["tgt_lang"]))
    if sample["split"] not in schema["split_values"]:
        raise ContractError("parallel sample split is not train/dev/test")
    provenance_value = sample.get("provenance")
    if provenance_value is not None:
        provenance = _mapping(provenance_value, "parallel sample.provenance")
        kind = _nonempty_string(provenance.get("kind"), "parallel sample.provenance.kind")
        field_sets = _mapping(schema["provenance_fields"], "provenance_fields")
        if kind not in field_sets:
            raise ContractError(f"unsupported provenance kind: {kind}")
        expected = set(field_sets[kind])
        _exact_keys(provenance, expected, f"parallel sample.provenance[{kind}]")
        for field in expected:
            _nonempty_string(provenance[field], f"parallel sample.provenance.{field}")
        for field in expected & {"decode_config_sha256", "generation_manifest_sha256"}:
            _sha256(provenance[field], f"parallel sample.provenance.{field}")
    return dict(sample)


def validate_source_lock(lock: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    _exact_keys(lock, {"schema_version", "config_sha256", "source_order", "sources"}, "source lock")
    if lock["schema_version"] != 2:
        raise ContractError("source lock schema_version must be 2")
    if _sha256(lock["config_sha256"], "source lock config hash") != config_sha256(config):
        raise ContractError("source lock does not match the model data config")
    enabled = [source for source in config["sources"] if source["enabled"]]
    expected_order = [source["source_id"] for source in enabled]
    if lock["source_order"] != expected_order:
        raise ContractError("source lock order does not match enabled config sources")
    records = _list(lock["sources"], "source lock sources")
    if len(records) != len(enabled):
        raise ContractError("source lock record count does not match enabled sources")
    by_id = {source["source_id"]: source for source in enabled}
    for index, record_value in enumerate(records):
        record = _mapping(record_value, f"source lock sources[{index}]")
        _exact_keys(
            record,
            {
                "source_id",
                "version",
                "license",
                "homepage",
                "download_uri",
                "archive",
                "selected_files",
                "verification",
            },
            f"source lock sources[{index}]",
        )
        source_id = str(record["source_id"])
        source = by_id.get(source_id)
        if source is None:
            raise ContractError(f"source lock contains unknown source: {source_id}")
        for field in ("version", "license", "homepage", "download_uri"):
            if record[field] != source[field]:
                raise ContractError(f"source lock {source_id}.{field} differs from config")
        archive = _mapping(record["archive"], f"{source_id}.archive")
        _exact_keys(archive, {"bytes", "sha256", "etag", "last_modified"}, f"{source_id}.archive")
        if not isinstance(archive["bytes"], int) or archive["bytes"] <= 0:
            raise ContractError(f"{source_id} archive bytes must be positive")
        _sha256(archive["sha256"], f"{source_id} archive hash")
        selected_files = _list(record["selected_files"], f"{source_id}.selected_files")
        expected_paths = {
            "1.1/LICENSE",
            "1.1/NOTICE.md",
            "1.1/data/en-US.jsonl",
            "1.1/data/zh-CN.jsonl",
            "1.1/data/zh-TW.jsonl",
            "1.1/data/ja-JP.jsonl",
            "1.1/data/ko-KR.jsonl",
        }
        paths: set[str] = set()
        selected_bytes = 0
        for file_index, file_value in enumerate(selected_files):
            file_record = _mapping(file_value, f"{source_id}.selected_files[{file_index}]")
            _exact_keys(file_record, {"path", "role", "bytes", "sha256"}, f"{source_id}.selected_files[{file_index}]")
            path = _nonempty_string(file_record["path"], "selected file path")
            if path in paths:
                raise ContractError(f"{source_id} selected_files contains duplicate path")
            paths.add(path)
            if not isinstance(file_record["bytes"], int) or file_record["bytes"] <= 0:
                raise ContractError(f"{source_id} selected file bytes must be positive")
            selected_bytes += file_record["bytes"]
            _sha256(file_record["sha256"], f"{source_id} selected file hash")
            _nonempty_string(file_record["role"], f"{source_id} selected file role")
        if paths != expected_paths:
            raise ContractError(f"{source_id} selected_files must lock five locales plus license/notice")
        verification = _mapping(record["verification"], f"{source_id}.verification")
        _exact_keys(
            verification,
            {
                "verified_on",
                "method",
                "alignment_key",
                "rows_per_locale",
                "partition_rows_per_locale",
                "selected_bytes",
            },
            f"{source_id}.verification",
        )
        if verification["alignment_key"] != ["partition", "id"]:
            raise ContractError(f"{source_id} verification alignment key changed")
        if verification["rows_per_locale"] != config["budgets"]["source_rows_per_locale"]:
            raise ContractError(f"{source_id} verified row count differs from config")
        if verification["partition_rows_per_locale"] != config["budgets"]["source_partition_rows_per_locale"]:
            raise ContractError(f"{source_id} verified partitions differ from config")
        if verification["selected_bytes"] != selected_bytes:
            raise ContractError(f"{source_id} selected byte total is inconsistent")
        if archive["bytes"] > config["budgets"]["download_max_bytes"]:
            raise ContractError(f"{source_id} archive exceeds the download budget")
        if selected_bytes > config["budgets"]["selected_extract_max_bytes"]:
            raise ContractError(f"{source_id} selected files exceed the extraction budget")
    return dict(lock)


def load_source_lock(path: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    return validate_source_lock(_load_mapping(path), config)
