"""Shared strict contracts and local llama.cpp runtime for Hy-MT2 distillation."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import threading
import time
import unicodedata
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence

import yaml
from opencc import OpenCC
from sacrebleu.metrics import BLEU, CHRF

from artifact_io import (
    atomic_write_bytes,
    atomic_write_json as _shared_atomic_write_json,
    canonical_json_bytes as _shared_canonical_json_bytes,
    sha256_bytes,
    sha256_file,
    write_jsonl as _shared_write_jsonl,
)
from model_data_pipeline import script_counts, wrong_script_dominates
from model_training_contract import LANGUAGE_TAGS, config_sha256, directed_routes, validate_route


class DistillationError(RuntimeError):
    """Raised when a TD-07/TD-08 contract or runtime invariant is violated."""


ALL_ROUTES = tuple(f"{source}->{target}" for source, target in directed_routes())
CHINESE_CONVERSION_ROUTES = (
    "zho_Hans->zho_Hant",
    "zho_Hant->zho_Hans",
)
# Backward-compatible name for immutable TD-07/TD-08 v1 artifacts.
ROUTES = tuple(route for route in ALL_ROUTES if route not in CHINESE_CONVERSION_ROUTES)
LEGACY_PROMPT_IDENTITY = {
    "name": "hymt2-teacher-prompt-decode-v1",
    "status": "frozen",
    "purpose": "td07-human-dev-calibration-and-td08-train-only-generation",
}
CHINESE_CONVERSION_PROMPT_IDENTITY = {
    "name": "hymt2-teacher-prompt-decode-zh-conversion-v2",
    "status": "frozen",
    "purpose": "td07-hans-hant-dev-calibration-and-td08-train-only-addendum",
}
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_PLACEHOLDER_RE = re.compile(
    r"\{\{[^{}]+\}\}|\$\{[^{}]+\}|\{[^{}]+\}|%(?:\d+\$)?[a-zA-Z]|<[^<>\s]+>"
)
_EXPLANATION_RE = re.compile(
    r"^(?:translation|translated text|here(?:'s| is) the translation|译文|翻译结果|翻譯結果)\s*[:：]",
    re.IGNORECASE,
)
_PROMPT_ECHO_MARKERS = (
    "translate the following text into",
    "only output the translated result",
    "将以下文本翻译",
    "將以下文本翻譯",
)
_REPEATED_CHARACTER_RE = re.compile(r"([^\s])\1{11,}")
_REPEATED_SEGMENT_RE = re.compile(r"(.{4,}?)\1{2,}")
_S2T = OpenCC("s2t")
_T2S = OpenCC("t2s")


def canonical_json_bytes(value: Any) -> bytes:
    return _shared_canonical_json_bytes(value, allow_nan=False)


def atomic_write_json(path: Path, value: Any) -> None:
    _shared_atomic_write_json(path, value, sort_keys=True, allow_nan=True)


def atomic_write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    _shared_write_jsonl(path, records, allow_nan=False)


def load_yaml(path: Path) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise DistillationError(f"cannot load YAML {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise DistillationError(f"{path} must contain a mapping")
    return value


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DistillationError(f"cannot load JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise DistillationError(f"{path} must contain an object")
    return value


def _exact_keys(value: Mapping[str, Any], required: set[str], context: str) -> None:
    missing = sorted(required - set(value))
    unknown = sorted(set(value) - required)
    if missing:
        raise DistillationError(f"{context} missing fields: {', '.join(missing)}")
    if unknown:
        raise DistillationError(f"{context} unknown fields: {', '.join(unknown)}")


def _mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise DistillationError(f"{context} must be a mapping")
    return value


def _positive_integer(value: Any, context: str, *, allow_zero: bool = False) -> int:
    minimum = 0 if allow_zero else 1
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise DistillationError(f"{context} must be an integer >= {minimum}")
    return value


def _rate(value: Any, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1:
        raise DistillationError(f"{context} must be in [0, 1]")
    return float(value)


def _repo_path(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise DistillationError(f"{context} must be a repository-relative POSIX path")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "." in path.parts:
        raise DistillationError(f"{context} must not escape the repository")
    return path.as_posix()


def validate_prompt_config(config: Mapping[str, Any]) -> dict[str, Any]:
    base_keys = {
            "schema_version",
            "identity",
            "teacher_selection",
            "calibration_input",
            "prompt",
            "decode_profiles",
            "selection",
            "filter",
            "calibration_gates",
            "route_limits",
            "runtime",
            "outputs",
    }

    identity = _mapping(config["identity"], "identity")
    _exact_keys(identity, {"name", "status", "purpose"}, "identity")
    if dict(identity) == LEGACY_PROMPT_IDENTITY:
        expected_keys = base_keys
        expected_routes = ROUTES
        expected_schema = 1
    elif dict(identity) == CHINESE_CONVERSION_PROMPT_IDENTITY:
        expected_keys = base_keys | {"routes"}
        expected_routes = CHINESE_CONVERSION_ROUTES
        expected_schema = 2
    else:
        raise DistillationError("prompt identity is not the frozen TD-07 identity")
    _exact_keys(config, expected_keys, "prompt config")
    if config["schema_version"] != expected_schema:
        raise DistillationError(f"prompt config schema_version must be {expected_schema}")
    if "routes" in config and tuple(config["routes"]) != expected_routes:
        raise DistillationError("prompt config route scope changed")

    selection = _mapping(config["teacher_selection"], "teacher_selection")
    _exact_keys(selection, {"path", "file_sha256", "selection_id"}, "teacher_selection")
    _repo_path(selection["path"], "teacher_selection.path")
    if not _SHA256_RE.fullmatch(str(selection["file_sha256"])):
        raise DistillationError("teacher_selection.file_sha256 is invalid")
    if selection["selection_id"] != "hymt2-7b-distillation-teacher-v1":
        raise DistillationError("teacher selection identity changed")

    calibration = _mapping(config["calibration_input"], "calibration_input")
    _exact_keys(
        calibration,
        {"path", "file_sha256", "split", "samples_per_route", "selection_seed", "test_access"},
        "calibration_input",
    )
    _repo_path(calibration["path"], "calibration_input.path")
    if not _SHA256_RE.fullmatch(str(calibration["file_sha256"])):
        raise DistillationError("calibration_input.file_sha256 is invalid")
    if calibration["split"] != "dev" or calibration["test_access"] != "prohibited":
        raise DistillationError("TD-07 must use dev and prohibit test")
    _positive_integer(calibration["samples_per_route"], "calibration samples_per_route")
    if not isinstance(calibration["selection_seed"], str) or not calibration["selection_seed"]:
        raise DistillationError("calibration selection_seed must be non-empty")

    prompt = _mapping(config["prompt"], "prompt")
    _exact_keys(prompt, {"version", "template", "system_prompt", "language_names"}, "prompt")
    if prompt["system_prompt"] is not None:
        raise DistillationError("Hy-MT2 frozen prompt must not use a system prompt")
    if "{target_language}" not in str(prompt["template"]) or "{source_text}" not in str(prompt["template"]):
        raise DistillationError("prompt template must contain target_language and source_text")
    names = _mapping(prompt["language_names"], "prompt.language_names")
    if dict(names) != {
        "eng_Latn": "English",
        "zho_Hans": "Chinese",
        "zho_Hant": "Traditional Chinese",
        "jpn_Jpan": "Japanese",
        "kor_Hang": "Korean",
    }:
        raise DistillationError("language-name mapping changed")

    profiles = _mapping(config["decode_profiles"], "decode_profiles")
    if set(profiles) != {"greedy-v1", "official-sampling-v1"}:
        raise DistillationError("decode profiles must contain the two frozen candidates")
    for name, raw_profile in profiles.items():
        profile = _mapping(raw_profile, f"decode_profiles.{name}")
        _exact_keys(profile, {"temperature", "top_p", "top_k", "repeat_penalty", "seed"}, f"decode_profiles.{name}")
        if not isinstance(profile["temperature"], (int, float)) or float(profile["temperature"]) < 0:
            raise DistillationError(f"{name}.temperature is invalid")
        _rate(profile["top_p"], f"{name}.top_p")
        _positive_integer(profile["top_k"], f"{name}.top_k", allow_zero=True)
        if not isinstance(profile["repeat_penalty"], (int, float)) or float(profile["repeat_penalty"]) <= 0:
            raise DistillationError(f"{name}.repeat_penalty is invalid")
        _positive_integer(profile["seed"], f"{name}.seed", allow_zero=True)

    choice = _mapping(config["selection"], "selection")
    _exact_keys(choice, {"selected_profile", "rule", "sampling_minimum_macro_chrf_delta", "tie_breaker"}, "selection")
    if choice["selected_profile"] not in profiles or choice["tie_breaker"] != "greedy-v1":
        raise DistillationError("selected/tie-break decode profile is invalid")
    if not isinstance(choice["sampling_minimum_macro_chrf_delta"], (int, float)):
        raise DistillationError("sampling macro chrF delta must be numeric")

    filters = _mapping(config["filter"], "filter")
    filter_fields = {
        "version",
        "unicode_normalization",
        "collapse_whitespace",
        "reject_empty",
        "reject_prompt_echo",
        "reject_explanation_prefix",
        "reject_exact_source_copy",
        "reject_wrong_script_dominance",
        "reject_chinese_script_counterevidence",
        "reject_finish_reason_length",
        "reject_abnormal_repetition",
        "preserve_placeholders",
        "minimum_output_source_character_ratio",
        "maximum_output_source_character_ratio",
        "length_ratio_floor_characters",
    }
    if expected_routes == CHINESE_CONVERSION_ROUTES:
        filter_fields.add("source_copy_policy")
    _exact_keys(filters, filter_fields, "filter")
    expected_filter_version = (
        "hymt2-output-filter-zh-conversion-v2"
        if expected_routes == CHINESE_CONVERSION_ROUTES
        else "hymt2-output-filter-v1"
    )
    if filters["unicode_normalization"] != "NFC" or filters["version"] != expected_filter_version:
        raise DistillationError("output filter identity changed")
    for field in filter_fields - {
        "version",
        "unicode_normalization",
        "minimum_output_source_character_ratio",
        "maximum_output_source_character_ratio",
        "length_ratio_floor_characters",
        "source_copy_policy",
    }:
        if filters[field] is not True:
            raise DistillationError(f"filter.{field} must be true")
    if expected_routes == CHINESE_CONVERSION_ROUTES and filters["source_copy_policy"] != (
        "allow-only-when-opencc-script-conversion-is-identity"
    ):
        raise DistillationError("Chinese conversion source-copy policy changed")
    minimum_ratio = float(filters["minimum_output_source_character_ratio"])
    maximum_ratio = float(filters["maximum_output_source_character_ratio"])
    if not 0 < minimum_ratio < maximum_ratio:
        raise DistillationError("output/source length-ratio bounds are invalid")
    _positive_integer(filters["length_ratio_floor_characters"], "filter length floor")

    gates = _mapping(config["calibration_gates"], "calibration_gates")
    _exact_keys(
        gates,
        {
            "minimum_route_chrf",
            "reference_interpretation",
            "minimum_route_accepted_rate",
            "minimum_route_script_compliance_rate",
            "maximum_route_source_copy_rate",
            "maximum_route_truncation_rate",
            "replay_samples_per_route",
            "require_exact_replay",
            "sacrebleu_tokenizer",
            "chrf_word_order",
        },
        "calibration_gates",
    )
    if not isinstance(gates["minimum_route_chrf"], (int, float)) or float(gates["minimum_route_chrf"]) < 0:
        raise DistillationError("minimum_route_chrf is invalid")
    if not isinstance(gates["reference_interpretation"], str) or "locale-specific" not in gates["reference_interpretation"]:
        raise DistillationError("calibration reference interpretation must preserve the MASSIVE localization boundary")
    for field in (
        "minimum_route_accepted_rate",
        "minimum_route_script_compliance_rate",
        "maximum_route_source_copy_rate",
        "maximum_route_truncation_rate",
    ):
        _rate(gates[field], f"calibration_gates.{field}")
    _positive_integer(gates["replay_samples_per_route"], "replay_samples_per_route")
    if gates["require_exact_replay"] is not True or gates["sacrebleu_tokenizer"] != "char":
        raise DistillationError("calibration must require exact replay and char SacreBLEU")
    _positive_integer(gates["chrf_word_order"], "chrf_word_order", allow_zero=True)

    limits = config["route_limits"]
    if not isinstance(limits, list) or len(limits) != len(expected_routes):
        raise DistillationError("route_limits must contain every configured route")
    found: set[str] = set()
    for index, raw_limit in enumerate(limits):
        limit = _mapping(raw_limit, f"route_limits[{index}]")
        _exact_keys(limit, {"route", "max_source_characters", "max_output_tokens", "stop"}, f"route_limits[{index}]")
        route = str(limit["route"])
        if route in found or route not in expected_routes:
            raise DistillationError(f"invalid or duplicate route limit: {route}")
        found.add(route)
        _positive_integer(limit["max_source_characters"], f"{route}.max_source_characters")
        _positive_integer(limit["max_output_tokens"], f"{route}.max_output_tokens")
        if not isinstance(limit["stop"], list) or any(not isinstance(item, str) for item in limit["stop"]):
            raise DistillationError(f"{route}.stop must be a string list")
    if found != set(expected_routes):
        raise DistillationError("route_limits coverage is incomplete")

    runtime = _mapping(config["runtime"], "runtime")
    _exact_keys(
        runtime,
        {
            "host",
            "port",
            "request_timeout_seconds",
            "request_attempts",
            "context_size",
            "n_gpu_layers",
            "flash_attention",
            "maximum_batch_size",
            "external_network",
        },
        "runtime",
    )
    if runtime["host"] != "127.0.0.1" or runtime["external_network"] != "prohibited":
        raise DistillationError("teacher runtime must be loopback-only with external network prohibited")
    for field in ("port", "request_timeout_seconds", "request_attempts", "context_size", "maximum_batch_size"):
        _positive_integer(runtime[field], f"runtime.{field}")
    if runtime["n_gpu_layers"] != "all" or runtime["flash_attention"] is not True or runtime["maximum_batch_size"] != 1:
        raise DistillationError("teacher runtime must preserve the frozen serial CUDA profile")

    outputs = _mapping(config["outputs"], "outputs")
    _exact_keys(outputs, {"records", "report"}, "outputs")
    for name, value in outputs.items():
        _repo_path(value, f"outputs.{name}")
    return dict(config)


def load_prompt_config(path: Path) -> dict[str, Any]:
    return validate_prompt_config(load_yaml(path))


def route_id(source: str, target: str) -> str:
    validate_route(source, target)
    return f"{source}->{target}"


def route_limits(config: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {str(record["route"]): record for record in config["route_limits"]}


def prompt_routes(config: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(config.get("routes", ROUTES))


def read_parallel_jsonl(
    path: Path,
    *,
    expected_split: str,
    expected_sha256: str | None = None,
) -> list[dict[str, Any]]:
    if expected_split not in {"train", "dev"}:
        raise DistillationError("teacher inputs may only be train or bounded dev")
    if expected_sha256 and sha256_file(path) != expected_sha256:
        raise DistillationError(f"input SHA-256 differs from the frozen identity: {path}")
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DistillationError(f"invalid JSONL at {path}:{line_number}") from exc
            if not isinstance(record, dict) or record.get("split") != expected_split:
                raise DistillationError(f"teacher input contains a non-{expected_split} record")
            source = str(record.get("src_lang", ""))
            target = str(record.get("tgt_lang", ""))
            route_id(source, target)
            for field in ("sample_id", "sample_group_id", "source_text", "target_text"):
                if not isinstance(record.get(field), str) or not record[field]:
                    raise DistillationError(f"teacher input record missing {field}")
            records.append(record)
    if not records:
        raise DistillationError(f"teacher input is empty: {path}")
    return records


def deterministic_route_sample(
    records: Sequence[Mapping[str, Any]],
    *,
    per_route: int,
    seed: str,
    routes: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    selected_routes = tuple(routes or ROUTES)
    by_route: dict[str, list[tuple[str, Mapping[str, Any]]]] = defaultdict(list)
    for record in records:
        route = route_id(str(record["src_lang"]), str(record["tgt_lang"]))
        identity = str(record["sample_id"])
        score = sha256_bytes(f"{seed}\0{route}\0{identity}".encode("utf-8"))
        by_route[route].append((score, record))
    missing = sorted(set(selected_routes) - set(by_route))
    if missing:
        raise DistillationError(f"input does not cover all routes: {', '.join(missing)}")
    selected: list[dict[str, Any]] = []
    for route in selected_routes:
        candidates = sorted(by_route[route], key=lambda item: (item[0], str(item[1]["sample_id"])))
        if len(candidates) < per_route:
            raise DistillationError(f"{route} has only {len(candidates)} records, needs {per_route}")
        selected.extend(dict(record) for _, record in candidates[:per_route])
    return selected


def build_prompt(config: Mapping[str, Any], source_text: str, target: str) -> str:
    names = config["prompt"]["language_names"]
    if target not in names:
        raise DistillationError(f"unknown target language: {target}")
    return str(config["prompt"]["template"]).format(
        target_language=names[target],
        source_text=source_text,
    )


def derived_seed(profile: Mapping[str, Any], sample_id: str) -> int:
    base = int(profile["seed"])
    digest = hashlib.sha256(f"{base}\0{sample_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF


def normalize_output(text: str, config: Mapping[str, Any]) -> str:
    value = unicodedata.normalize(str(config["filter"]["unicode_normalization"]), text)
    if config["filter"]["collapse_whitespace"]:
        value = re.sub(r"\s+", " ", value, flags=re.UNICODE)
    return value.strip()


def _conversion_changes(text: str, converted: str) -> int:
    return sum(left != right for left, right in zip(text, converted)) + abs(len(text) - len(converted))


def chinese_script_evidence(text: str) -> dict[str, int]:
    return {
        "simplified": _conversion_changes(text, _S2T.convert(text)),
        "traditional": _conversion_changes(text, _T2S.convert(text)),
    }


def _placeholder_counts(text: str) -> Counter[str]:
    return Counter(_PLACEHOLDER_RE.findall(text))


def filter_output(
    *,
    source_text: str,
    target_text: str,
    target_language: str,
    finish_reason: str,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    normalized = normalize_output(target_text, config)
    filters = config["filter"]
    reasons: list[str] = []
    source_compact = re.sub(r"\s+", "", source_text).casefold()
    target_compact = re.sub(r"\s+", "", normalized).casefold()
    if filters["reject_empty"] and not normalized:
        reasons.append("empty_output")
    lowered = normalized.casefold()
    if filters["reject_prompt_echo"] and any(marker in lowered for marker in _PROMPT_ECHO_MARKERS):
        reasons.append("prompt_echo")
    if filters["reject_explanation_prefix"] and _EXPLANATION_RE.search(normalized):
        reasons.append("extra_explanation")
    if filters["reject_exact_source_copy"] and source_compact and source_compact == target_compact:
        conversion_is_identity = False
        if filters.get("source_copy_policy") == "allow-only-when-opencc-script-conversion-is-identity":
            if target_language == "zho_Hant":
                conversion_is_identity = _S2T.convert(source_text) == source_text
            elif target_language == "zho_Hans":
                conversion_is_identity = _T2S.convert(source_text) == source_text
        if not conversion_is_identity:
            reasons.append("source_copy")
    if filters["reject_finish_reason_length"] and finish_reason == "length":
        reasons.append("truncated")
    if filters["reject_abnormal_repetition"] and (
        _REPEATED_CHARACTER_RE.search(normalized) or _REPEATED_SEGMENT_RE.search(normalized)
    ):
        reasons.append("abnormal_repetition")
    if filters["preserve_placeholders"] and _placeholder_counts(source_text) != _placeholder_counts(normalized):
        reasons.append("placeholder_mismatch")

    if normalized and filters["reject_wrong_script_dominance"] and wrong_script_dominates(normalized, target_language):
        reasons.append("wrong_script_dominance")
    evidence = chinese_script_evidence(normalized) if target_language in {"zho_Hans", "zho_Hant"} else {"simplified": 0, "traditional": 0}
    if filters["reject_chinese_script_counterevidence"]:
        if target_language == "zho_Hans" and evidence["traditional"] > 0 and evidence["simplified"] == 0:
            reasons.append("traditional_output_for_simplified_target")
        if target_language == "zho_Hant" and evidence["simplified"] > 0 and evidence["traditional"] == 0:
            reasons.append("simplified_output_for_traditional_target")

    source_characters = len(source_text.replace(" ", ""))
    target_characters = len(normalized.replace(" ", ""))
    denominator = max(source_characters, int(filters["length_ratio_floor_characters"]))
    ratio = target_characters / denominator
    if ratio < float(filters["minimum_output_source_character_ratio"]):
        reasons.append("output_too_short")
    if ratio > float(filters["maximum_output_source_character_ratio"]):
        reasons.append("output_too_long")
    unique_reasons = list(dict.fromkeys(reasons))
    return {
        "normalized_output": normalized,
        "accepted": not unique_reasons,
        "rejection_reasons": unique_reasons,
        "source_characters": source_characters,
        "target_characters": target_characters,
        "output_source_character_ratio": round(ratio, 6),
        "script_counts": script_counts(normalized),
        "chinese_script_evidence": evidence,
        "script_compliant": not any(
            reason
            in {
                "wrong_script_dominance",
                "traditional_output_for_simplified_target",
                "simplified_output_for_traditional_target",
            }
            for reason in unique_reasons
        ),
    }


def metric_scores(outputs: Sequence[str], references: Sequence[str], config: Mapping[str, Any]) -> dict[str, Any]:
    if not outputs or len(outputs) != len(references):
        raise DistillationError("metric inputs must be non-empty and aligned")
    gates = config["calibration_gates"]
    bleu = BLEU(tokenize=str(gates["sacrebleu_tokenizer"]), effective_order=True)
    chrf = CHRF(word_order=int(gates["chrf_word_order"]))
    bleu_score = bleu.corpus_score(list(outputs), [list(references)])
    chrf_score = chrf.corpus_score(list(outputs), [list(references)])
    return {
        "sacrebleu": round(float(bleu_score.score), 6),
        "chrf": round(float(chrf_score.score), 6),
        "sacrebleu_signature": str(bleu.get_signature()),
        "chrf_signature": str(chrf.get_signature()),
    }


def summarize_generation_records(records: Sequence[Mapping[str, Any]], config: Mapping[str, Any]) -> dict[str, Any]:
    if not records:
        raise DistillationError("cannot summarize empty generation records")
    by_route: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        by_route[str(record["route"])].append(record)
    summaries: dict[str, Any] = {}
    for route in prompt_routes(config):
        route_records = by_route.get(route, [])
        if not route_records:
            raise DistillationError(f"generation records missing route {route}")
        outputs = [str(record["normalized_output"]) for record in route_records]
        references = [str(record["reference_text"]) for record in route_records]
        reason_counts = Counter(
            reason for record in route_records for reason in record["rejection_reasons"]
        )
        count = len(route_records)
        metrics = metric_scores(outputs, references, config)
        summaries[route] = {
            "records": count,
            "accepted": sum(bool(record["accepted"]) for record in route_records),
            "accepted_rate": round(sum(bool(record["accepted"]) for record in route_records) / count, 6),
            "script_compliance_rate": round(sum(bool(record["script_compliant"]) for record in route_records) / count, 6),
            "source_copy_rate": round(reason_counts["source_copy"] / count, 6),
            "truncation_rate": round(reason_counts["truncated"] / count, 6),
            "rejection_reasons": dict(sorted(reason_counts.items())),
            "latency_seconds": round(sum(float(record["latency_seconds"]) for record in route_records), 6),
            "completion_tokens": sum(int(record["completion_tokens"]) for record in route_records),
            **metrics,
        }
    macro = {
        name: round(sum(float(summary[name]) for summary in summaries.values()) / len(summaries), 6)
        for name in ("chrf", "sacrebleu", "accepted_rate", "script_compliance_rate")
    }
    return {"routes": summaries, "macro": macro}


def route_gate_failures(summary: Mapping[str, Any], config: Mapping[str, Any]) -> list[str]:
    gates = config["calibration_gates"]
    failures: list[str] = []
    for route in prompt_routes(config):
        record = summary["routes"][route]
        checks = {
            "chrf": float(record["chrf"]) >= float(gates["minimum_route_chrf"]),
            "accepted_rate": float(record["accepted_rate"]) >= float(gates["minimum_route_accepted_rate"]),
            "script_compliance_rate": float(record["script_compliance_rate"])
            >= float(gates["minimum_route_script_compliance_rate"]),
            "source_copy_rate": float(record["source_copy_rate"])
            <= float(gates["maximum_route_source_copy_rate"]),
            "truncation_rate": float(record["truncation_rate"])
            <= float(gates["maximum_route_truncation_rate"]),
        }
        failures.extend(f"{route}:{name}" for name, passed in checks.items() if not passed)
    return failures


def resolve_runtime_paths(repository_root: Path, prompt_config: Mapping[str, Any]) -> dict[str, Path]:
    selection_path = repository_root / PurePosixPath(str(prompt_config["teacher_selection"]["path"]))
    if sha256_file(selection_path) != prompt_config["teacher_selection"]["file_sha256"]:
        raise DistillationError("teacher selection file differs from the prompt contract")
    selection = load_yaml(selection_path)
    if selection.get("status") != "frozen" or selection.get("selection_id") != prompt_config["teacher_selection"]["selection_id"]:
        raise DistillationError("teacher selection is not the frozen TD-06 identity")
    runtime = _mapping(selection["runtime"], "teacher selection runtime")
    override = os.environ.get(str(runtime["override_env"]))
    root = Path(override).expanduser().resolve() if override else (repository_root / PurePosixPath(str(runtime["default_root"]))).resolve()
    snapshot = root / PurePosixPath(str(runtime["artifact_subdir"]))
    backend = root / PurePosixPath(str(runtime["backend_subdir"]))
    model = snapshot / str(selection["teacher"]["filename"])
    servers = list(backend.rglob("llama-server.exe"))
    if len(servers) != 1:
        raise DistillationError(f"expected one llama-server.exe below {backend}, found {len(servers)}")
    if not model.is_file() or model.stat().st_size != int(selection["teacher"]["bytes"]):
        raise DistillationError("teacher GGUF file is missing or has the wrong size")
    if sha256_file(model) != selection["teacher"]["sha256"]:
        raise DistillationError("teacher GGUF SHA-256 differs from the frozen selection")
    return {"root": root, "snapshot": snapshot, "backend": backend, "model": model, "server": servers[0]}


def _http_json(url: str, payload: Mapping[str, Any] | None = None, timeout: float = 10) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="GET" if data is None else "POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            value = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", errors="replace").strip()
        detail = body or str(error.reason)
        raise DistillationError(f"HTTP {error.code}: {detail}") from error
    if not isinstance(value, dict):
        raise DistillationError(f"non-object response from {url}")
    return value


class LlamaCppTeacher:
    """One serial loopback-only llama.cpp server bound to the frozen GGUF teacher."""

    def __init__(self, repository_root: Path, config: Mapping[str, Any]) -> None:
        self.repository_root = repository_root
        self.config = config
        self.paths = resolve_runtime_paths(repository_root, config)
        runtime = config["runtime"]
        self.base_url = f"http://{runtime['host']}:{runtime['port']}"
        self.process: subprocess.Popen[str] | None = None
        self.logs: list[str] = []
        self.log_thread: threading.Thread | None = None
        self.command = [
            str(self.paths["server"]),
            "--model",
            str(self.paths["model"]),
            "--host",
            str(runtime["host"]),
            "--port",
            str(runtime["port"]),
            "--ctx-size",
            str(runtime["context_size"]),
            "--n-gpu-layers",
            str(runtime["n_gpu_layers"]),
            "--flash-attn",
            "on" if runtime["flash_attention"] else "off",
            "--parallel",
            str(runtime["maximum_batch_size"]),
            "--jinja",
            "--metrics",
        ]

    def __enter__(self) -> "LlamaCppTeacher":
        environment = dict(os.environ)
        environment.update(
            {
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "HF_DATASETS_OFFLINE": "1",
                "NO_PROXY": "127.0.0.1,localhost",
            }
        )
        creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        self.process = subprocess.Popen(
            self.command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            creationflags=creationflags,
            env=environment,
        )

        def collect_logs() -> None:
            assert self.process is not None and self.process.stdout is not None
            for line in self.process.stdout:
                self.logs.append(line.rstrip())
                if len(self.logs) > 2_000:
                    del self.logs[:1_000]

        self.log_thread = threading.Thread(target=collect_logs, daemon=True)
        self.log_thread.start()
        deadline = time.monotonic() + 180
        while True:
            if self.process.poll() is not None:
                raise DistillationError("llama-server exited before readiness: " + "\n".join(self.logs[-40:]))
            try:
                health = _http_json(self.base_url + "/health", timeout=1)
                if health.get("status") == "ok":
                    return self
            except (OSError, urllib.error.URLError, json.JSONDecodeError, DistillationError):
                pass
            if time.monotonic() >= deadline:
                raise DistillationError("timed out waiting for llama-server readiness")
            time.sleep(0.2)

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=10)
        if self.log_thread is not None:
            self.log_thread.join(timeout=5)

    def generate(
        self,
        *,
        prompt: str,
        profile: Mapping[str, Any],
        sample_id: str,
        max_tokens: int,
        stop: Sequence[str],
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": "hymt2-q8",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": int(max_tokens),
            "temperature": float(profile["temperature"]),
            "top_p": float(profile["top_p"]),
            "top_k": int(profile["top_k"]),
            "repeat_penalty": float(profile["repeat_penalty"]),
            "seed": derived_seed(profile, sample_id),
            "stream": False,
        }
        if stop:
            payload["stop"] = list(stop)
        attempts = int(self.config["runtime"]["request_attempts"])
        last_error: Exception | None = None
        for attempt in range(1, attempts + 1):
            started = time.perf_counter()
            try:
                response = _http_json(
                    self.base_url + "/v1/chat/completions",
                    payload,
                    timeout=float(self.config["runtime"]["request_timeout_seconds"]),
                )
                elapsed = time.perf_counter() - started
                choice = response["choices"][0]
                usage = response["usage"]
                return {
                    "raw_output": str(choice["message"]["content"]).strip(),
                    "finish_reason": str(choice.get("finish_reason", "")),
                    "prompt_tokens": int(usage["prompt_tokens"]),
                    "completion_tokens": int(usage["completion_tokens"]),
                    "latency_seconds": round(elapsed, 6),
                    "request_attempts": attempt,
                    "seed": payload["seed"],
                }
            except (OSError, urllib.error.URLError, KeyError, IndexError, TypeError, ValueError, DistillationError) as error:
                last_error = error
                if attempt < attempts:
                    time.sleep(min(2 ** (attempt - 1), 4))
        raise DistillationError(f"teacher request failed after {attempts} attempts: {last_error}")


def generation_record(
    sample: Mapping[str, Any],
    *,
    profile_name: str,
    config: Mapping[str, Any],
    response: Mapping[str, Any],
) -> dict[str, Any]:
    source = str(sample["src_lang"])
    target = str(sample["tgt_lang"])
    route = route_id(source, target)
    filtered = filter_output(
        source_text=str(sample["source_text"]),
        target_text=str(response["raw_output"]),
        target_language=target,
        finish_reason=str(response["finish_reason"]),
        config=config,
    )
    raw_output = str(response["raw_output"])
    return {
        "record_id": sha256_bytes(
            f"{profile_name}\0{sample['sample_id']}\0{sha256_bytes(raw_output.encode('utf-8'))}".encode("utf-8")
        ),
        "profile": profile_name,
        "route": route,
        "sample_id": sample["sample_id"],
        "sample_group_id": sample["sample_group_id"],
        "split": sample["split"],
        "src_lang": source,
        "tgt_lang": target,
        "source_text": sample["source_text"],
        "reference_text": sample["target_text"],
        "raw_output": raw_output,
        "raw_output_sha256": sha256_bytes(raw_output.encode("utf-8")),
        "normalized_output_sha256": sha256_bytes(str(filtered["normalized_output"]).encode("utf-8")),
        **response,
        **filtered,
    }


def run_profile(
    teacher: LlamaCppTeacher,
    samples: Sequence[Mapping[str, Any]],
    *,
    profile_name: str,
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    profile = config["decode_profiles"][profile_name]
    limits = route_limits(config)
    records: list[dict[str, Any]] = []
    for sample in samples:
        route = route_id(str(sample["src_lang"]), str(sample["tgt_lang"]))
        limit = limits[route]
        if len(str(sample["source_text"])) > int(limit["max_source_characters"]):
            raise DistillationError(f"{sample['sample_id']} exceeds frozen source limit for {route}")
        prompt = build_prompt(config, str(sample["source_text"]), str(sample["tgt_lang"]))
        response = teacher.generate(
            prompt=prompt,
            profile=profile,
            sample_id=str(sample["sample_id"]),
            max_tokens=int(limit["max_output_tokens"]),
            stop=limit["stop"],
        )
        records.append(
            generation_record(
                sample,
                profile_name=profile_name,
                config=config,
                response=response,
            )
        )
    return records
