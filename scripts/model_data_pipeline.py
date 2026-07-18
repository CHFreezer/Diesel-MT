#!/usr/bin/env python3
"""Deterministic locked-source pipeline for MVP parallel model data."""

from __future__ import annotations

import json
import os
import re
import tarfile
import time
import unicodedata
import urllib.error
import urllib.request
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from artifact_io import atomic_write_bytes, sha256_bytes, sha256_file
from model_training_contract import (
    ContractError,
    canonical_json_bytes,
    config_sha256,
    validate_model_data_config,
    validate_parallel_sample,
    validate_source_lock,
)


PIPELINE_VERSION = "td03-v1"
USER_AGENT = "Diesel-MT-model-data/td03-v1"
PARTITION_ORDER = {"train": 0, "dev": 1, "test": 2}
CLEANING_PROFILE: dict[str, Any] = {
    "unicode_normalization": "NFC",
    "collapse_unicode_whitespace": True,
    "min_characters": 1,
    "max_characters": 512,
    "length_ratio_floor_characters": 4,
    "max_length_ratio": 8.0,
    "wrong_script_min_letters": 4,
    "wrong_script_dominance": 0.85,
    "max_repeated_character_run": 16,
    "reject_html_tags_and_entities": True,
    "reject_unicode_replacement_character": True,
}

HTML_TAG_RE = re.compile(r"<\s*(?:/?[A-Za-z][^>]*|!--.*?)>", re.DOTALL)
HTML_ENTITY_RE = re.compile(r"&(?:#[0-9]{1,7}|#x[0-9A-Fa-f]{1,6}|[A-Za-z][A-Za-z0-9]{1,31});")
REPEATED_CHARACTER_RE = re.compile(
    rf"([^\s])\1{{{int(CLEANING_PROFILE['max_repeated_character_run']) - 1},}}"
)


class PipelineError(RuntimeError):
    """Base class for expected command-line pipeline failures."""

    exit_code = 2


class FetchError(PipelineError):
    """Locked source archive could not be obtained or verified."""


class SourceError(PipelineError):
    """Locked archive contents do not satisfy the source adapter contract."""


class BuildError(PipelineError):
    """Canonical samples or outputs could not be built safely."""


@dataclass(frozen=True)
class LocaleRecord:
    alignment_key: str
    split: str
    source_record_id: str
    text: str
    rejection_reason: str | None


def archive_cache_path(cache_root: Path, source_record: Mapping[str, Any]) -> Path:
    archive = source_record["archive"]
    filename = str(source_record["download_uri"]).rsplit("/", 1)[-1]
    return cache_root / str(source_record["source_id"]) / f"{archive['sha256'][:16]}-{filename}"


def _archive_is_valid(path: Path, archive: Mapping[str, Any]) -> bool:
    return (
        path.is_file()
        and path.stat().st_size == int(archive["bytes"])
        and sha256_file(path) == archive["sha256"]
    )


def download_archive(
    url: str,
    destination: Path,
    expected_bytes: int,
    expected_sha256: str,
    *,
    timeout: int = 120,
    retries: int = 4,
) -> None:
    """Download exactly one locked archive, resuming a validated byte prefix."""

    if retries <= 0:
        raise FetchError("download retries must be positive")
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    if destination.exists():
        if _archive_is_valid(destination, {"bytes": expected_bytes, "sha256": expected_sha256}):
            return
        destination.unlink()

    last_error: Exception | None = None
    for attempt in range(retries):
        if partial.exists() and partial.stat().st_size > expected_bytes:
            partial.unlink()
        start = partial.stat().st_size if partial.exists() else 0
        if start == expected_bytes:
            if sha256_file(partial) == expected_sha256:
                os.replace(partial, destination)
                return
            partial.unlink()
            start = 0

        headers = {
            "Accept-Encoding": "identity",
            "Range": f"bytes={start}-{expected_bytes - 1}",
            "User-Agent": USER_AGENT,
        }
        try:
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=timeout) as response:
                status = int(getattr(response, "status", 200))
                if start and status != 206:
                    raise FetchError(f"server ignored resume range for {url}")
                mode = "ab" if start else "wb"
                with partial.open(mode) as handle:
                    remaining = expected_bytes - start
                    while remaining:
                        chunk = response.read(min(1024 * 1024, remaining))
                        if not chunk:
                            break
                        handle.write(chunk)
                        remaining -= len(chunk)
                    handle.flush()
                    os.fsync(handle.fileno())
            actual = partial.stat().st_size if partial.exists() else 0
            if actual != expected_bytes:
                raise FetchError(f"short download for {url}: {actual}/{expected_bytes} bytes")
            if sha256_file(partial) != expected_sha256:
                partial.unlink()
                raise FetchError(f"downloaded archive SHA-256 differs from source lock: {url}")
            os.replace(partial, destination)
            return
        except (OSError, urllib.error.URLError, FetchError) as exc:
            last_error = exc
            if attempt + 1 < retries:
                time.sleep(min(2**attempt, 4))
    raise FetchError(f"download failed for {url}: {last_error}")


def ensure_locked_archive(
    source_record: Mapping[str, Any],
    cache_root: Path,
    *,
    offline: bool,
    use_cache: bool,
    timeout: int,
    retries: int,
) -> Path:
    path = archive_cache_path(cache_root, source_record)
    archive = source_record["archive"]
    if _archive_is_valid(path, archive):
        return path
    if offline or use_cache:
        raise FetchError(
            f"validated cache missing or corrupt for {source_record['source_id']}: {path}"
        )
    if path.exists():
        path.unlink()
    download_archive(
        str(source_record["download_uri"]),
        path,
        int(archive["bytes"]),
        str(archive["sha256"]),
        timeout=timeout,
        retries=retries,
    )
    if not _archive_is_valid(path, archive):
        raise FetchError(f"downloaded cache failed locked verification: {path}")
    return path


def read_locked_tar_members(
    archive_path: Path, source_record: Mapping[str, Any]
) -> dict[str, bytes]:
    """Read only exact locked regular files; never extract archive paths to disk."""

    selected = {str(record["path"]): record for record in source_record["selected_files"]}
    result: dict[str, bytes] = {}
    try:
        with tarfile.open(archive_path, mode="r:gz") as archive:
            member_counts = Counter(member.name for member in archive.getmembers())
            for path, locked in selected.items():
                if member_counts[path] != 1:
                    raise SourceError(
                        f"{source_record['source_id']} archive must contain one regular member: {path}"
                    )
                member = archive.getmember(path)
                if not member.isfile() or member.issym() or member.islnk():
                    raise SourceError(f"locked archive member is not a regular file: {path}")
                extracted = archive.extractfile(member)
                if extracted is None:
                    raise SourceError(f"cannot read locked archive member: {path}")
                data = extracted.read()
                if len(data) != int(locked["bytes"]):
                    raise SourceError(f"locked member byte size differs: {path}")
                if sha256_bytes(data) != locked["sha256"]:
                    raise SourceError(f"locked member SHA-256 differs: {path}")
                result[path] = data
    except (tarfile.TarError, OSError) as exc:
        raise SourceError(f"cannot read locked archive {archive_path}: {exc}") from exc
    return result


def _script_bucket(character: str) -> str | None:
    codepoint = ord(character)
    if (
        0x0041 <= codepoint <= 0x005A
        or 0x0061 <= codepoint <= 0x007A
        or 0x00C0 <= codepoint <= 0x024F
    ):
        return "latin"
    if 0x3040 <= codepoint <= 0x309F:
        return "hiragana"
    if 0x30A0 <= codepoint <= 0x30FF or 0x31F0 <= codepoint <= 0x31FF:
        return "katakana"
    if 0xAC00 <= codepoint <= 0xD7AF or 0x1100 <= codepoint <= 0x11FF:
        return "hangul"
    if (
        0x3400 <= codepoint <= 0x4DBF
        or 0x4E00 <= codepoint <= 0x9FFF
        or 0xF900 <= codepoint <= 0xFAFF
    ):
        return "han"
    return None


def script_counts(text: str) -> dict[str, int]:
    counts = Counter(bucket for character in text if (bucket := _script_bucket(character)))
    return {name: int(counts.get(name, 0)) for name in ("latin", "hiragana", "katakana", "hangul", "han")}


def wrong_script_dominates(text: str, language: str) -> bool:
    counts = script_counts(text)
    total = sum(counts.values())
    if total < int(CLEANING_PROFILE["wrong_script_min_letters"]):
        return False
    if language == "eng_Latn":
        expected = counts["latin"]
    elif language == "jpn_Jpan":
        expected = counts["hiragana"] + counts["katakana"] + counts["han"]
    elif language == "kor_Hang":
        expected = counts["hangul"]
    elif language in {"zho_Hans", "zho_Hant"}:
        expected = counts["han"]
    else:
        raise BuildError(f"unsupported cleaning language: {language}")
    return expected == 0 and max(counts.values()) / total >= float(
        CLEANING_PROFILE["wrong_script_dominance"]
    )


def normalize_text(value: Any, language: str) -> tuple[str, str | None]:
    """Normalize presentation only; never case-fold, transliterate, or convert scripts."""

    if not isinstance(value, str):
        return "", "non_string_text"
    text = unicodedata.normalize(str(CLEANING_PROFILE["unicode_normalization"]), value)
    for character in text:
        if unicodedata.category(character) == "Cc" and character not in "\t\n\r":
            return text, "control_character"
    text = re.sub(r"\s+", " ", text, flags=re.UNICODE).strip()
    if not text:
        return text, "empty_text"
    if "\ufffd" in text and CLEANING_PROFILE["reject_unicode_replacement_character"]:
        return text, "unicode_replacement_character"
    if CLEANING_PROFILE["reject_html_tags_and_entities"] and (
        HTML_TAG_RE.search(text) or HTML_ENTITY_RE.search(text)
    ):
        return text, "html_residue"
    if len(text) < int(CLEANING_PROFILE["min_characters"]):
        return text, "too_short"
    if len(text) > int(CLEANING_PROFILE["max_characters"]):
        return text, "too_long"
    if REPEATED_CHARACTER_RE.search(text):
        return text, "abnormal_repetition"
    if wrong_script_dominates(text, language):
        return text, "wrong_script_dominance"
    return text, None


def pair_rejection_reason(source_text: str, target_text: str) -> str | None:
    source_characters = len(source_text.replace(" ", ""))
    target_characters = len(target_text.replace(" ", ""))
    shorter = max(min(source_characters, target_characters), int(CLEANING_PROFILE["length_ratio_floor_characters"]))
    longer = max(source_characters, target_characters)
    if longer / shorter > float(CLEANING_PROFILE["max_length_ratio"]):
        return "length_ratio"
    return None


def _alignment_key(partition: str, record_id: Any) -> str:
    return json.dumps(
        {"id": record_id, "partition": partition},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def parse_massive_locale(
    data: bytes,
    language: str,
    source: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, LocaleRecord]:
    """Parse one locked MASSIVE JSONL locale into aligned normalized records."""

    try:
        text = data.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise SourceError(f"{source['source_id']} {language} is not valid UTF-8: {exc}") from exc
    partition_field = str(source["partition_field"])
    text_field = str(source["text_field"])
    partition_map = source["partition_map"]
    records: dict[str, LocaleRecord] = {}
    partition_counts: Counter[str] = Counter()
    scan_limit = int(config["budgets"]["scan_limit_rows_per_locale"])
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            raise SourceError(f"{source['source_id']} {language} contains blank JSONL line {line_number}")
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SourceError(
                f"{source['source_id']} {language} invalid JSONL line {line_number}: {exc}"
            ) from exc
        if not isinstance(row, dict):
            raise SourceError(f"{source['source_id']} {language} line {line_number} is not an object")
        if partition_field not in row or "id" not in row or text_field not in row:
            raise SourceError(
                f"{source['source_id']} {language} line {line_number} lacks partition, id, or text"
            )
        raw_partition = row[partition_field]
        if raw_partition not in partition_map:
            raise SourceError(
                f"{source['source_id']} {language} has unknown partition: {raw_partition!r}"
            )
        split = str(partition_map[raw_partition])
        key = _alignment_key(split, row["id"])
        if key in records:
            raise SourceError(f"{source['source_id']} {language} duplicate alignment key: {key}")
        normalized, rejection_reason = normalize_text(row[text_field], language)
        records[key] = LocaleRecord(
            alignment_key=key,
            split=split,
            source_record_id=f"{raw_partition}:{row['id']}",
            text=normalized,
            rejection_reason=rejection_reason,
        )
        partition_counts[split] += 1
        if len(records) > scan_limit:
            raise SourceError(f"{source['source_id']} {language} exceeds locked scan limit")

    expected_rows = int(config["budgets"]["source_rows_per_locale"])
    expected_partitions = dict(config["budgets"]["source_partition_rows_per_locale"])
    if len(records) != expected_rows:
        raise SourceError(
            f"{source['source_id']} {language} row count differs: {len(records)} != {expected_rows}"
        )
    if {name: partition_counts[name] for name in PARTITION_ORDER} != expected_partitions:
        raise SourceError(f"{source['source_id']} {language} partition counts differ from lock")
    return records


SOURCE_ADAPTERS: dict[
    str,
    Callable[[bytes, str, Mapping[str, Any], Mapping[str, Any]], dict[str, LocaleRecord]],
] = {"massive-1.1": parse_massive_locale}


def _checkpoint_identity(
    config: Mapping[str, Any], lock: Mapping[str, Any], source_record: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "pipeline_version": PIPELINE_VERSION,
        "cleaning_profile_sha256": config_sha256(CLEANING_PROFILE),
        "config_sha256": config_sha256(config),
        "lock_sha256": config_sha256(lock),
        "source_id": source_record["source_id"],
        "source_archive_sha256": source_record["archive"]["sha256"],
    }


def _checkpoint_paths(
    out_root: Path, identity_sha256: str, source_id: str, language: str
) -> tuple[Path, Path]:
    root = out_root / "interim" / "model-data" / identity_sha256 / source_id
    return root / f"{language}.jsonl", root / f"{language}.meta.json"


def _locale_records_bytes(records: Mapping[str, LocaleRecord]) -> bytes:
    payload = []
    for key in sorted(records, key=lambda item: (PARTITION_ORDER[records[item].split], item)):
        record = records[key]
        payload.append(
            canonical_json_bytes(
                {
                    "alignment_key": record.alignment_key,
                    "rejection_reason": record.rejection_reason,
                    "source_record_id": record.source_record_id,
                    "split": record.split,
                    "text": record.text,
                }
            )
        )
    return b"".join(payload)


def _write_locale_checkpoint(
    out_root: Path,
    identity: Mapping[str, Any],
    source_id: str,
    language: str,
    records: Mapping[str, LocaleRecord],
) -> None:
    identity_sha256 = config_sha256(identity)
    data_path, meta_path = _checkpoint_paths(out_root, identity_sha256, source_id, language)
    data = _locale_records_bytes(records)
    atomic_write_bytes(data_path, data)
    meta = {
        "schema_version": 1,
        "status": "complete",
        "identity": dict(identity),
        "identity_sha256": identity_sha256,
        "language": language,
        "records": len(records),
        "data_sha256": sha256_bytes(data),
    }
    atomic_write_bytes(meta_path, canonical_json_bytes(meta))


def _load_locale_checkpoint(
    out_root: Path,
    identity: Mapping[str, Any],
    source_id: str,
    language: str,
) -> dict[str, LocaleRecord] | None:
    identity_sha256 = config_sha256(identity)
    data_path, meta_path = _checkpoint_paths(out_root, identity_sha256, source_id, language)
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        data = data_path.read_bytes()
        if (
            meta.get("status") != "complete"
            or meta.get("identity") != identity
            or meta.get("identity_sha256") != identity_sha256
            or meta.get("language") != language
            or meta.get("data_sha256") != sha256_bytes(data)
        ):
            return None
        records: dict[str, LocaleRecord] = {}
        for line in data.decode("utf-8", errors="strict").splitlines():
            row = json.loads(line)
            if set(row) != {
                "alignment_key",
                "rejection_reason",
                "source_record_id",
                "split",
                "text",
            }:
                return None
            record = LocaleRecord(**row)
            if record.alignment_key in records or record.split not in PARTITION_ORDER:
                return None
            records[record.alignment_key] = record
        if len(records) != int(meta.get("records", -1)):
            return None
        return records
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError):
        return None


def _locked_data_path(source_record: Mapping[str, Any], language: str) -> str:
    matches = [
        str(record["path"])
        for record in source_record["selected_files"]
        if record["role"] == f"data:{language}"
    ]
    if len(matches) != 1:
        raise SourceError(
            f"{source_record['source_id']} lock must contain exactly one data file for {language}"
        )
    return matches[0]


def _source_locale_records(
    config: Mapping[str, Any],
    lock: Mapping[str, Any],
    source: Mapping[str, Any],
    source_record: Mapping[str, Any],
    selected_files: Mapping[str, bytes],
    out_root: Path,
    *,
    resume: bool,
) -> tuple[dict[str, dict[str, LocaleRecord]], dict[str, bool]]:
    adapter = SOURCE_ADAPTERS.get(str(source_record["source_id"]))
    if adapter is None:
        raise SourceError(f"no source adapter for {source_record['source_id']}")
    identity = _checkpoint_identity(config, lock, source_record)
    locale_records: dict[str, dict[str, LocaleRecord]] = {}
    resumed: dict[str, bool] = {}
    for language in config["languages"]["model_tags"]:
        records = (
            _load_locale_checkpoint(
                out_root, identity, str(source_record["source_id"]), str(language)
            )
            if resume
            else None
        )
        resumed[str(language)] = records is not None
        if records is None:
            locked_path = _locked_data_path(source_record, str(language))
            records = adapter(selected_files[locked_path], str(language), source, config)
            _write_locale_checkpoint(
                out_root,
                identity,
                str(source_record["source_id"]),
                str(language),
                records,
            )
        locale_records[str(language)] = records
    reference_language = str(config["languages"]["model_tags"][0])
    reference_keys = set(locale_records[reference_language])
    for language, records in locale_records.items():
        if set(records) != reference_keys:
            missing = len(reference_keys - set(records))
            extra = len(set(records) - reference_keys)
            raise SourceError(
                f"{source_record['source_id']} alignment differs for {language}: missing={missing}, extra={extra}"
            )
    return locale_records, resumed


def _group_id(
    source_record: Mapping[str, Any],
    alignment_key: str,
    records: Mapping[str, LocaleRecord],
) -> str:
    identity = {
        "source_id": source_record["source_id"],
        "source_version": source_record["version"],
        "alignment_key": alignment_key,
        "locale_content_sha256": {
            language: sha256_bytes(record.text.encode("utf-8"))
            for language, record in sorted(records.items())
        },
    }
    return f"group-sha256:{sha256_bytes(canonical_json_bytes(identity))}"


def _sample_id(sample_identity: Mapping[str, Any]) -> str:
    return f"sample-sha256:{sha256_bytes(canonical_json_bytes(sample_identity))}"


def _build_source_samples(
    config: Mapping[str, Any],
    source_record: Mapping[str, Any],
    locale_records: Mapping[str, Mapping[str, LocaleRecord]],
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    pairs = list(config["directions"]["undirected_pairs"])
    reference_language = str(config["languages"]["model_tags"][0])
    ordered_keys = sorted(
        locale_records[reference_language],
        key=lambda item: (
            PARTITION_ORDER[locale_records[reference_language][item].split],
            item,
        ),
    )
    samples: list[dict[str, Any]] = []
    sample_ids: set[str] = set()
    rejection_reasons: Counter[str] = Counter()
    rejection_by_pair: dict[str, Counter[str]] = {
        str(pair["pair_id"]): Counter() for pair in pairs
    }
    pair_counts: dict[str, Counter[str]] = {
        str(pair["pair_id"]): Counter() for pair in pairs
    }
    locale_stats: dict[str, Any] = {}
    for language, records in locale_records.items():
        locale_rejections = Counter(
            record.rejection_reason for record in records.values() if record.rejection_reason
        )
        locale_stats[language] = {
            "rows": len(records),
            "clean_text_rows": len(records) - sum(locale_rejections.values()),
            "rejected_text_rows": sum(locale_rejections.values()),
            "rejection_reasons": dict(sorted(locale_rejections.items())),
        }

    for alignment_key in ordered_keys:
        group_records = {
            language: records[alignment_key] for language, records in locale_records.items()
        }
        splits = {record.split for record in group_records.values()}
        source_record_ids = {record.source_record_id for record in group_records.values()}
        if len(splits) != 1 or len(source_record_ids) != 1:
            raise SourceError(
                f"{source_record['source_id']} aligned locales disagree on split or record id: {alignment_key}"
            )
        split = next(iter(splits))
        group_id = _group_id(source_record, alignment_key, group_records)
        for pair in pairs:
            pair_id = str(pair["pair_id"])
            source_language, target_language = [str(value) for value in pair["tags"]]
            source_locale = group_records[source_language]
            target_locale = group_records[target_language]
            pair_counts[pair_id]["scanned"] += 1
            pair_counts[pair_id][f"scanned_{split}"] += 1
            reason: str | None = None
            if source_locale.rejection_reason:
                reason = f"{source_language}:{source_locale.rejection_reason}"
            elif target_locale.rejection_reason:
                reason = f"{target_language}:{target_locale.rejection_reason}"
            else:
                reason = pair_rejection_reason(source_locale.text, target_locale.text)
            if reason:
                rejection_reasons[reason] += 1
                rejection_by_pair[pair_id][reason] += 1
                pair_counts[pair_id]["rejected"] += 1
                pair_counts[pair_id][f"rejected_{split}"] += 1
                continue

            sample_identity = {
                "sample_group_id": group_id,
                "src_lang": source_language,
                "tgt_lang": target_language,
                "source_text": source_locale.text,
                "target_text": target_locale.text,
            }
            sample = {
                "sample_id": _sample_id(sample_identity),
                "sample_group_id": group_id,
                "source_id": source_record["source_id"],
                "source_version": source_record["version"],
                "license": source_record["license"],
                "src_lang": source_language,
                "tgt_lang": target_language,
                "source_text": source_locale.text,
                "target_text": target_locale.text,
                "split": split,
                "provenance": {
                    "kind": "human_parallel",
                    "source_record_id": source_locale.source_record_id,
                    "alignment_key": alignment_key,
                },
            }
            try:
                validate_parallel_sample(sample, config)
            except ContractError as exc:
                raise BuildError(f"canonical sample violates TD-01 contract: {exc}") from exc
            if sample["sample_id"] in sample_ids:
                raise BuildError(f"duplicate stable sample id: {sample['sample_id']}")
            sample_ids.add(str(sample["sample_id"]))
            samples.append(sample)
            pair_counts[pair_id]["accepted"] += 1
            pair_counts[pair_id][f"accepted_{split}"] += 1

    pair_report = {
        pair_id: {name: int(value) for name, value in sorted(counts.items())}
        for pair_id, counts in pair_counts.items()
    }
    rejection_report = {
        "source_id": source_record["source_id"],
        "total_rejected_pairs": int(sum(rejection_reasons.values())),
        "by_reason": dict(sorted(rejection_reasons.items())),
        "by_pair": {
            pair_id: dict(sorted(counts.items()))
            for pair_id, counts in rejection_by_pair.items()
            if counts
        },
    }
    source_report = {
        "source_id": source_record["source_id"],
        "source_version": source_record["version"],
        "license": source_record["license"],
        "locales": locale_stats,
        "pairs": pair_report,
        "accepted_samples": len(samples),
    }
    return samples, source_report, rejection_report


def _output_record(relative_path: str, data: bytes, records: int | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": relative_path,
        "bytes": len(data),
        "sha256": sha256_bytes(data),
    }
    if records is not None:
        result["records"] = records
    return result


def _publish_outputs(
    out_root: Path,
    files: list[tuple[str, bytes, int | None]],
    manifest: Mapping[str, Any],
) -> Path:
    manifest_path = out_root / "corpus" / "mvp" / "manifest.json"
    # Once publication begins, remove the prior completion marker. Any
    # interruption can leave complete individual files, but never a false
    # complete corpus. The new manifest is always published last.
    manifest_path.unlink(missing_ok=True)
    try:
        for relative_path, data, _records in files:
            atomic_write_bytes(out_root / Path(relative_path), data)
        atomic_write_bytes(manifest_path, canonical_json_bytes(manifest))
    except BaseException:
        manifest_path.unlink(missing_ok=True)
        raise
    return manifest_path


def build_model_data(
    config: Mapping[str, Any],
    lock: Mapping[str, Any],
    out_root: Path,
    cache_root: Path,
    *,
    offline: bool,
    use_cache: bool,
    resume: bool = False,
    timeout: int = 120,
    retries: int = 4,
) -> dict[str, Any]:
    """Build canonical TD-03 samples and publish their manifest last."""

    validated_config = validate_model_data_config(config)
    validated_lock = validate_source_lock(lock, validated_config)
    sources_by_id = {
        str(source["source_id"]): source
        for source in validated_config["sources"]
        if source["enabled"]
    }
    lock_records = {
        str(record["source_id"]): record for record in validated_lock["sources"]
    }

    all_samples: list[dict[str, Any]] = []
    source_reports: list[dict[str, Any]] = []
    rejection_reports: list[dict[str, Any]] = []
    resume_report: dict[str, dict[str, bool]] = {}
    license_outputs: list[tuple[str, bytes, int | None]] = []
    for source_id in validated_lock["source_order"]:
        source = sources_by_id[str(source_id)]
        source_record = lock_records[str(source_id)]
        archive_path = ensure_locked_archive(
            source_record,
            cache_root,
            offline=offline,
            use_cache=use_cache,
            timeout=timeout,
            retries=retries,
        )
        selected_files = read_locked_tar_members(archive_path, source_record)
        locale_records, resumed = _source_locale_records(
            validated_config,
            validated_lock,
            source,
            source_record,
            selected_files,
            out_root,
            resume=resume,
        )
        samples, source_report, rejection_report = _build_source_samples(
            validated_config, source_record, locale_records
        )
        all_samples.extend(samples)
        source_reports.append(source_report)
        rejection_reports.append(rejection_report)
        resume_report[str(source_id)] = resumed
        for record in source_record["selected_files"]:
            if record["role"] in {"license", "notice"}:
                name = Path(str(record["path"])).name
                relative = f"corpus/mvp/sources/{source_id}/{name}"
                license_outputs.append((relative, selected_files[str(record["path"])], None))

    if len({sample["sample_id"] for sample in all_samples}) != len(all_samples):
        raise BuildError("stable sample IDs collide across sources")
    corpus_bytes = b"".join(canonical_json_bytes(sample) for sample in all_samples)
    cleaning_sha256 = config_sha256(CLEANING_PROFILE)
    route_counts: dict[str, int] = {}
    for source_report in source_reports:
        for pair_id, counts in source_report["pairs"].items():
            accepted = int(counts.get("accepted", 0))
            first, second = pair_id.split("--", 1)
            route_counts[f"{first}->{second}"] = route_counts.get(f"{first}->{second}", 0) + accepted
            route_counts[f"{second}->{first}"] = route_counts.get(f"{second}->{first}", 0) + accepted
    rejection_payload = {
        "schema_version": 1,
        "pipeline_version": PIPELINE_VERSION,
        "sources": rejection_reports,
        "total_rejected_pairs": sum(
            int(report["total_rejected_pairs"]) for report in rejection_reports
        ),
    }
    rejection_bytes = canonical_json_bytes(rejection_payload)
    corpus_record = _output_record(
        "corpus/mvp/human_parallel.jsonl", corpus_bytes, len(all_samples)
    )
    rejection_record = _output_record(
        "reports/data-rejections.json", rejection_bytes
    )
    build_payload = {
        "schema_version": 1,
        "pipeline_version": PIPELINE_VERSION,
        "config_sha256": config_sha256(validated_config),
        "lock_sha256": config_sha256(validated_lock),
        "cleaning_profile": CLEANING_PROFILE,
        "cleaning_profile_sha256": cleaning_sha256,
        "source_reports": source_reports,
        "totals": {
            "accepted_undirected_samples": len(all_samples),
            "rejected_pairs": rejection_payload["total_rejected_pairs"],
        },
        "directed_route_potential_counts": dict(sorted(route_counts.items())),
        "outputs": [corpus_record, rejection_record],
    }
    build_bytes = canonical_json_bytes(build_payload)
    files: list[tuple[str, bytes, int | None]] = [
        ("corpus/mvp/human_parallel.jsonl", corpus_bytes, len(all_samples)),
        *license_outputs,
        ("reports/data-rejections.json", rejection_bytes, None),
        ("reports/data-build.json", build_bytes, None),
    ]
    file_records = [
        _output_record(relative, data, records) for relative, data, records in files
    ]
    identity = {
        "pipeline_version": PIPELINE_VERSION,
        "config_sha256": config_sha256(validated_config),
        "lock_sha256": config_sha256(validated_lock),
        "cleaning_profile_sha256": cleaning_sha256,
        "source_archives": {
            source_id: lock_records[source_id]["archive"]["sha256"]
            for source_id in validated_lock["source_order"]
        },
    }
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "identity": identity,
        "identity_sha256": config_sha256(identity),
        "sample_schema_version": validated_config["sample_schema"]["version"],
        "canonical_sample_scope": "ten undirected pairs; reverse route expansion is TD-04",
        "records": len(all_samples),
        "files": file_records,
    }
    manifest_path = _publish_outputs(out_root, files, manifest)
    return {
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "identity_sha256": manifest["identity_sha256"],
        "records": len(all_samples),
        "rejected_pairs": rejection_payload["total_rejected_pairs"],
        "resume_checkpoints_used": resume_report,
    }


def dry_run_plan(
    config: Mapping[str, Any],
    lock: Mapping[str, Any],
    out_root: Path,
    cache_root: Path,
    *,
    offline: bool,
    use_cache: bool,
    resume: bool,
) -> dict[str, Any]:
    """Return a side-effect-free execution plan after strict contract validation."""

    validated_config = validate_model_data_config(config)
    validated_lock = validate_source_lock(lock, validated_config)
    sources = []
    for source_record in validated_lock["sources"]:
        path = archive_cache_path(cache_root, source_record)
        if not path.exists():
            status = "missing"
        elif _archive_is_valid(path, source_record["archive"]):
            status = "valid"
        else:
            status = "invalid"
        sources.append(
            {
                "source_id": source_record["source_id"],
                "download_uri": source_record["download_uri"],
                "archive_sha256": source_record["archive"]["sha256"],
                "archive_bytes": source_record["archive"]["bytes"],
                "cache_status": status,
            }
        )
    return {
        "status": "dry-run",
        "pipeline_version": PIPELINE_VERSION,
        "config_sha256": config_sha256(validated_config),
        "lock_sha256": config_sha256(validated_lock),
        "cleaning_profile_sha256": config_sha256(CLEANING_PROFILE),
        "network_allowed": not (offline or use_cache),
        "resume": resume,
        "sources": sources,
        "maximum_canonical_samples": int(
            validated_config["budgets"]["source_rows_per_locale"]
        )
        * len(validated_config["directions"]["undirected_pairs"]),
        "outputs": [
            "corpus/mvp/human_parallel.jsonl",
            "corpus/mvp/sources/<source_id>/LICENSE|NOTICE",
            "reports/data-rejections.json",
            "reports/data-build.json",
            "corpus/mvp/manifest.json (published last)",
        ],
        "operations": [
            "validate config-bound source lock",
            "validate cache or resume locked archive download",
            "verify exact regular tar members by byte size and SHA-256",
            "parse aligned locales and conservatively normalize text",
            "reuse identity-bound locale checkpoints when requested",
            "build stable group/sample identities for ten undirected pairs",
            "atomically publish corpus and reports, then completion manifest",
        ],
        "runtime_roots": {
            "out": str(out_root),
            "cache": str(cache_root),
        },
    }
