"""Deterministic HPLT 3.0 tokenizer-corpus acquisition pipeline."""

from __future__ import annotations

import contextlib
import ctypes
import datetime as dt
import hashlib
import heapq
import html
import io
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import unicodedata
import urllib.error
import urllib.request
from collections import Counter, deque
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterator

import yaml
import zstandard


LANGUAGES = ("eng_Latn", "zho_Hans", "jpn_Jpan", "kor_Hang")
REQUIRED_SOURCE_FIELDS = {
    "source_id",
    "source_type",
    "license",
    "homepage",
    "download_uri",
    "version_or_snapshot",
    "languages",
    "expected_files",
    "checksum_or_size",
    "enabled",
    "notes",
}
LOCK_SCHEMA_VERSION = 1
USER_AGENT = "Diesel-MT-tokenizer-dataset-fetch/1.0"
RAM_CHECKPOINT_SCHEMA_VERSION = 1
MIB = 1024 * 1024
GIB = 1024 * MIB
DEFAULT_BATCH_CHARACTERS = 2_000_000
MINHASH_SENTINEL = (1 << 128) - 1

# Worker configuration is installed once by ProcessPoolExecutor.initializer.
# The sequential path uses the same initializer and worker function so both
# execution modes exercise identical content logic.
_WORKER_LANGUAGE: str | None = None
_WORKER_CLEANING: dict[str, Any] | None = None
_WORKER_DEDUP: dict[str, Any] | None = None
_WORKER_SEED = 0
_WORKER_SCRIPT_THRESHOLD = 0.0


class PipelineError(RuntimeError):
    exit_code = 5


class ConfigError(PipelineError):
    exit_code = 2


class LockError(PipelineError):
    exit_code = 3


class FetchError(PipelineError):
    exit_code = 4


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temporary)
        raise


def load_config(path: Path) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError(f"cannot load config {path}: {exc}") from exc
    validate_config(value)
    return value


def validate_config(config: Any) -> None:
    if not isinstance(config, dict) or config.get("schema_version") != 1:
        raise ConfigError("config.schema_version must be 1")
    for key in ("dataset", "reproducibility", "profiles", "cleaning", "deduplication", "quality", "sources"):
        if key not in config:
            raise ConfigError(f"config missing required field: {key}")
    dataset = config["dataset"]
    for key in ("name", "version_or_snapshot", "homepage", "license", "terms_url", "base_uri", "quality_wds", "max_shards_per_language"):
        if key not in dataset or dataset[key] in (None, ""):
            raise ConfigError(f"dataset missing required field: {key}")
    if not (str(dataset["homepage"]).startswith("https://") and str(dataset["base_uri"]).startswith("https://")):
        raise ConfigError("dataset homepage and base_uri must use HTTPS")
    minimum_wds = int(dataset["quality_wds"].get("minimum", -1))
    maximum_wds = int(dataset["quality_wds"].get("maximum", -1))
    if minimum_wds < 0 or maximum_wds < minimum_wds or int(dataset["max_shards_per_language"]) <= 0:
        raise ConfigError("dataset WDS range and max_shards_per_language must be positive and ordered")
    for key in ("contract", "canonical_json", "content_hash", "output_encoding", "output_newline", "final_trailing_newline"):
        if key not in config["reproducibility"]:
            raise ConfigError(f"reproducibility missing required field: {key}")
    outputs: set[str] = set()
    source_ids: set[str] = set()
    for index, source in enumerate(config["sources"]):
        missing = REQUIRED_SOURCE_FIELDS - set(source)
        if missing:
            raise ConfigError(f"sources[{index}] missing fields: {', '.join(sorted(missing))}")
        if not str(source["source_id"]).strip() or source["source_id"] in source_ids:
            raise ConfigError(f"sources[{index}] has empty or duplicate source_id")
        source_ids.add(source["source_id"])
        if source["enabled"]:
            for key in ("license", "homepage", "download_uri", "version_or_snapshot"):
                if not str(source[key]).strip():
                    raise ConfigError(f"enabled source {source['source_id']} has empty {key}")
            mapping = source["languages"]
            if not isinstance(mapping, dict) or not mapping.get("source") or not mapping.get("output"):
                raise ConfigError(f"enabled source {source['source_id']} has invalid language mapping")
            if mapping["output"] in outputs:
                raise ConfigError(f"duplicate enabled output language: {mapping['output']}")
            if not str(source["download_uri"]).startswith("https://"):
                raise ConfigError(f"enabled source {source['source_id']} download_uri must use HTTPS")
            outputs.add(mapping["output"])
    if outputs != set(LANGUAGES):
        raise ConfigError(f"enabled sources must map exactly to {', '.join(LANGUAGES)}")
    for name in ("smoke", "mvp"):
        profile = config["profiles"].get(name)
        if not isinstance(profile, dict):
            raise ConfigError(f"missing profile: {name}")
        for key in ("enabled_languages", "character_budget_per_language", "locked_prefix_bytes_per_shard", "random_seed", "concurrency", "corpus_subdir"):
            if key not in profile:
                raise ConfigError(f"profile {name} missing {key}")
        if set(profile["enabled_languages"]) != set(LANGUAGES):
            raise ConfigError(f"profile {name} must enable exactly four project languages")
        if (
            int(profile["character_budget_per_language"]) <= 0
            or int(profile["locked_prefix_bytes_per_shard"]) <= 0
            or int(profile["concurrency"]) <= 0
        ):
            raise ConfigError(f"profile {name} budgets must be positive")
        output_path = Path(str(profile["corpus_subdir"]))
        if output_path.is_absolute() or ".." in output_path.parts:
            raise ConfigError(f"profile {name} corpus_subdir must be a relative path without parent traversal")
        per_language = profile.get("locked_prefix_bytes_by_language", {})
        if set(per_language) - set(LANGUAGES) or any(int(value) <= 0 for value in per_language.values()):
            raise ConfigError(f"profile {name} has invalid per-language locked prefix bytes")
    cleaning = config["cleaning"]
    for key in (
        "algorithm_version",
        "min_characters",
        "max_characters",
        "max_replacement_character_ratio",
        "max_control_character_ratio",
        "reject_html_pattern",
        "reject_template_pattern",
        "reject_keyword_stuffing_pattern",
        "max_keyword_stuffing_matches",
        "reject_repeated_spam_pattern",
        "max_repeated_spam_matches",
        "split_on_lines",
        "collapse_horizontal_whitespace",
    ):
        if key not in cleaning:
            raise ConfigError(f"cleaning missing required field: {key}")
    if int(cleaning["min_characters"]) <= 0 or int(cleaning["max_characters"]) < int(cleaning["min_characters"]):
        raise ConfigError("cleaning character limits must be positive and ordered")
    if int(cleaning["max_keyword_stuffing_matches"]) < 0:
        raise ConfigError("cleaning.max_keyword_stuffing_matches cannot be negative")
    if int(cleaning["max_repeated_spam_matches"]) < 0:
        raise ConfigError("cleaning.max_repeated_spam_matches cannot be negative")
    for pattern_name in (
        "reject_html_pattern",
        "reject_template_pattern",
        "reject_keyword_stuffing_pattern",
        "reject_repeated_spam_pattern",
    ):
        try:
            re.compile(str(cleaning[pattern_name]))
        except re.error as exc:
            raise ConfigError(f"cleaning.{pattern_name} is not a valid regular expression: {exc}") from exc
    dedup = config["deduplication"]
    if set(dedup.get("approximate_languages", [])) != {"eng_Latn", "zho_Hans"}:
        raise ConfigError("deduplication.approximate_languages must be English and Simplified Chinese")
    minhash = dedup.get("minhash", {})
    for key in ("token_unit", "ngram_size", "hash", "permutations", "bands", "threshold", "seed", "max_ngrams_per_text", "tie_break"):
        if key not in minhash:
            raise ConfigError(f"deduplication.minhash missing required field: {key}")
    if (
        int(minhash["ngram_size"]) <= 0
        or int(minhash["permutations"]) <= 0
        or int(minhash["bands"]) <= 0
        or int(minhash["permutations"]) % int(minhash["bands"]) != 0
        or not 0.0 <= float(minhash["threshold"]) <= 1.0
        or int(minhash["max_ngrams_per_text"]) <= 0
    ):
        raise ConfigError("deduplication.minhash parameters are invalid")
    thresholds = config["quality"].get("language_min_script_ratio", {})
    if set(thresholds) != set(LANGUAGES) or any(not 0.0 <= float(value) <= 1.0 for value in thresholds.values()):
        raise ConfigError("quality language script-ratio thresholds must cover exactly four languages")
    if int(config["quality"].get("review_sample_count", 0)) <= 0:
        raise ConfigError("quality.review_sample_count must be positive")


def enabled_sources(config: dict[str, Any]) -> list[dict[str, Any]]:
    return sorted((s for s in config["sources"] if s["enabled"]), key=lambda s: s["languages"]["output"])


def request_bytes(url: str, *, timeout: int, retries: int = 3, headers: dict[str, str] | None = None) -> bytes:
    merged = {"User-Agent": USER_AGENT, "Accept-Encoding": "identity"}
    merged.update(headers or {})
    last: Exception | None = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=merged), timeout=timeout) as response:
                return response.read()
        except (OSError, urllib.error.URLError) as exc:
            last = exc
            if attempt + 1 < retries:
                time.sleep(2**attempt)
    raise FetchError(f"network request failed for {url}: {last}")


def remote_size(url: str, *, timeout: int = 60) -> int:
    request = urllib.request.Request(url, method="HEAD", headers={"User-Agent": USER_AGENT, "Accept-Encoding": "identity"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            size = response.headers.get("Content-Length")
    except (OSError, urllib.error.URLError) as exc:
        raise FetchError(f"HEAD failed for {url}: {exc}") from exc
    if not size or not size.isdigit():
        raise FetchError(f"remote Content-Length missing for {url}")
    return int(size)


def parse_map(data: bytes, minimum_wds: int, maximum_wds: int) -> list[tuple[int, int, str]]:
    try:
        lines = data.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise LockError(f"map is not UTF-8: {exc}") from exc
    parsed: list[tuple[int, int, str]] = []
    pattern = re.compile(r"/(\d+)_(\d+)\.jsonl\.zst$")
    for line in lines:
        line = line.strip()
        match = pattern.search(line)
        if not match:
            continue
        wds, shard = map(int, match.groups())
        if minimum_wds <= wds <= maximum_wds:
            parsed.append((wds, shard, line))
    parsed.sort(key=lambda item: (-item[0], item[1], item[2]))
    if not parsed:
        raise LockError("map contains no shard in configured WDS range")
    return parsed


def parse_md5_list(data: bytes) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in data.decode("ascii").splitlines():
        parts = line.strip().split()
        if len(parts) == 2 and re.fullmatch(r"[0-9a-fA-F]{32}", parts[0]):
            result[parts[1].replace("\\", "/")] = parts[0].lower()
    return result


def cache_path_for(cache_root: Path, source_id: str, url: str) -> Path:
    name = url.rsplit("/", 1)[-1]
    return cache_root / "hplt3" / source_id / f"{name}.prefix"


def download_locked_prefix(url: str, path: Path, target_bytes: int, *, timeout: int = 120, retries: int = 4) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".part")
    if path.exists():
        current = path.stat().st_size
        if current == target_bytes:
            return
        if current < target_bytes:
            with contextlib.suppress(FileNotFoundError):
                partial.unlink()
            os.replace(path, partial)
        else:
            path.unlink()
    last: Exception | None = None
    for attempt in range(retries):
        start = partial.stat().st_size if partial.exists() else 0
        if start > target_bytes:
            partial.unlink()
            start = 0
        if start == target_bytes:
            os.replace(partial, path)
            return
        headers = {
            "User-Agent": USER_AGENT,
            "Accept-Encoding": "identity",
            "Range": f"bytes={start}-{target_bytes - 1}",
        }
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=timeout) as response:
                if start and response.status != 206:
                    raise FetchError(f"server ignored resume range for {url}")
                mode = "ab" if start else "wb"
                with partial.open(mode) as handle:
                    remaining = target_bytes - start
                    while remaining:
                        chunk = response.read(min(1024 * 1024, remaining))
                        if not chunk:
                            break
                        handle.write(chunk)
                        remaining -= len(chunk)
                    handle.flush()
                    os.fsync(handle.fileno())
            if partial.stat().st_size == target_bytes:
                os.replace(partial, path)
                return
            raise FetchError(f"short download for {url}: {partial.stat().st_size}/{target_bytes} bytes")
        except (OSError, urllib.error.URLError, FetchError) as exc:
            last = exc
            if attempt + 1 < retries:
                time.sleep(2**attempt)
    raise FetchError(f"download failed for {url}: {last}")


def resolve_lock(config: dict[str, Any], config_path: Path, lock_path: Path, cache_root: Path, profile_name: str) -> dict[str, Any]:
    profile = config["profiles"][profile_name]
    dataset = config["dataset"]
    selected_count = int(dataset["max_shards_per_language"])
    source_locks: list[dict[str, Any]] = []
    for source in enabled_sources(config):
        map_uri = source["download_uri"]
        map_data = request_bytes(map_uri, timeout=60)
        metadata_cache = cache_root / "hplt3" / source["source_id"] / "metadata"
        atomic_write_bytes(metadata_cache / f"{source['languages']['source']}.map", map_data)
        parsed = parse_map(map_data, int(dataset["quality_wds"]["minimum"]), int(dataset["quality_wds"]["maximum"]))
        source_lang = source["languages"]["source"]
        md5_uri = f"{dataset['base_uri']}/{source_lang}.md5"
        md5_data = request_bytes(md5_uri, timeout=60)
        atomic_write_bytes(metadata_cache / f"{source_lang}.md5", md5_data)
        md5s = parse_md5_list(md5_data)
        shards: list[dict[str, Any]] = []
        for order, (wds, shard_number, url) in enumerate(parsed[:selected_count]):
            full_size = remote_size(url)
            per_language = profile.get("locked_prefix_bytes_by_language", {})
            requested_bytes = int(per_language.get(source["languages"]["output"], profile["locked_prefix_bytes_per_shard"]))
            locked_bytes = min(full_size, requested_bytes)
            cache_path = cache_path_for(cache_root, source["source_id"], url)
            download_locked_prefix(url, cache_path, locked_bytes)
            relative_name = f"{source_lang}/{url.rsplit('/', 1)[-1]}"
            upstream_md5 = md5s.get(relative_name)
            if not upstream_md5:
                raise LockError(f"official MD5 missing for {relative_name}")
            shards.append(
                {
                    "logical_order": order,
                    "wds": wds,
                    "shard_number": shard_number,
                    "url": url,
                    "remote_size": full_size,
                    "upstream_md5": upstream_md5,
                    "locked_bytes": locked_bytes,
                    "sha256": sha256_file(cache_path),
                    "sha256_scope": "bytes=0-(locked_bytes-1)",
                }
            )
        source_locks.append(
            {
                "source_id": source["source_id"],
                "source_language": source_lang,
                "output_language": source["languages"]["output"],
                "map_uri": map_uri,
                "map_sha256": sha256_bytes(map_data),
                "md5_uri": md5_uri,
                "shards": shards,
            }
        )
    lock = {
        "schema_version": LOCK_SCHEMA_VERSION,
        "dataset_name": dataset["name"],
        "version_or_snapshot": dataset["version_or_snapshot"],
        "resolved_profile": profile_name,
        "quality_wds": dataset["quality_wds"],
        "config_sha256": sha256_file(config_path),
        "sources": sorted(source_locks, key=lambda item: item["output_language"]),
    }
    atomic_write_bytes(lock_path, canonical_json_bytes(lock))
    return lock


def load_lock(path: Path, config: dict[str, Any], profile_name: str, config_path: Path | None = None) -> dict[str, Any]:
    try:
        lock = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LockError(f"cannot load source lock {path}: {exc}") from exc
    if lock.get("schema_version") != LOCK_SCHEMA_VERSION:
        raise LockError("source lock schema_version must be 1")
    if lock.get("dataset_name") != config["dataset"]["name"] or lock.get("version_or_snapshot") != config["dataset"]["version_or_snapshot"]:
        raise LockError("source lock dataset identity differs from config")
    if lock.get("quality_wds") != config["dataset"]["quality_wds"]:
        raise LockError("source lock WDS range differs from config")
    if config_path is not None and lock.get("config_sha256") != sha256_file(config_path):
        raise LockError("source lock config SHA-256 differs from the selected config file")
    if len(lock.get("sources", [])) != 4:
        raise LockError("source lock must contain exactly four enabled sources")
    if lock["sources"] != sorted(lock["sources"], key=lambda item: item.get("output_language", "")):
        raise LockError("source lock sources must be sorted by output_language")
    profile = config["profiles"][profile_name]
    registry = {source["source_id"]: source for source in enabled_sources(config)}
    outputs: set[str] = set()
    for source in lock["sources"]:
        for key in ("source_id", "source_language", "output_language", "map_uri", "map_sha256", "shards"):
            if not source.get(key):
                raise LockError(f"source lock entry missing {key}")
        configured = registry.get(source["source_id"])
        if configured is None:
            raise LockError(f"source lock contains unknown or disabled source: {source['source_id']}")
        if source["source_language"] != configured["languages"]["source"] or source["output_language"] != configured["languages"]["output"]:
            raise LockError(f"source lock language mapping differs from config: {source['source_id']}")
        if source["output_language"] in outputs:
            raise LockError(f"source lock has duplicate output language: {source['output_language']}")
        outputs.add(source["output_language"])
        if not re.fullmatch(r"[0-9a-f]{64}", source["map_sha256"]):
            raise LockError(f"invalid map SHA-256: {source['source_id']}")
        if not source["shards"]:
            raise LockError(f"source lock has no shards: {source['source_id']}")
        logical_orders = [int(shard.get("logical_order", -1)) for shard in source["shards"]]
        if logical_orders != list(range(len(source["shards"]))):
            raise LockError(f"source lock shard order is not contiguous: {source['source_id']}")
        for shard in source["shards"]:
            for key in ("logical_order", "wds", "shard_number", "url", "remote_size", "upstream_md5", "locked_bytes", "sha256", "sha256_scope"):
                if key not in shard or shard[key] in (None, ""):
                    raise LockError(f"locked shard missing {key}: {source['source_id']}")
            if not (int(config["dataset"]["quality_wds"]["minimum"]) <= int(shard["wds"]) <= int(config["dataset"]["quality_wds"]["maximum"])):
                raise LockError(f"locked shard WDS is outside config range: {source['source_id']}")
            if int(shard["remote_size"]) <= 0 or int(shard["locked_bytes"]) <= 0 or int(shard["locked_bytes"]) > int(shard["remote_size"]):
                raise LockError(f"locked shard has invalid sizes: {source['source_id']}")
            if not re.fullmatch(r"[0-9a-f]{32}", str(shard["upstream_md5"])):
                raise LockError(f"invalid upstream MD5: {source['source_id']}")
            required_bytes = int(profile.get("locked_prefix_bytes_by_language", {}).get(source["output_language"], profile["locked_prefix_bytes_per_shard"]))
            if int(shard["locked_bytes"]) < min(int(shard["remote_size"]), required_bytes):
                raise LockError(f"lock resolved for too small a prefix for profile {profile_name}: {source['source_id']}")
            if not re.fullmatch(r"[0-9a-f]{64}", shard["sha256"]):
                raise LockError(f"invalid shard SHA-256: {source['source_id']}")
            if shard["sha256_scope"] != "bytes=0-(locked_bytes-1)":
                raise LockError(f"invalid SHA-256 scope: {source['source_id']}")
    if outputs != set(LANGUAGES):
        raise LockError("source lock must map exactly the four project languages")
    return lock


def ensure_cache(lock: dict[str, Any], cache_root: Path, *, offline: bool, use_cache: bool) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for source in lock["sources"]:
        for shard in sorted(source["shards"], key=lambda item: item["logical_order"]):
            path = cache_path_for(cache_root, source["source_id"], shard["url"])
            valid = path.exists() and path.stat().st_size == int(shard["locked_bytes"]) and sha256_file(path) == shard["sha256"]
            if not valid:
                if path.exists():
                    path.unlink()
                if offline or use_cache:
                    raise FetchError(f"validated cache missing or corrupt for {source['source_id']} shard {shard['url']}")
                download_locked_prefix(shard["url"], path, int(shard["locked_bytes"]))
                if sha256_file(path) != shard["sha256"]:
                    path.unlink(missing_ok=True)
                    raise FetchError(f"downloaded prefix SHA-256 differs from source lock: {shard['url']}")
            result[f"{source['source_id']}:{shard['logical_order']}"] = path
    return result


def iter_jsonl_zst_prefix(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("rb") as raw:
        reader = zstandard.ZstdDecompressor().stream_reader(raw)
        text = io.TextIOWrapper(reader, encoding="utf-8", errors="strict", newline="")
        try:
            for line in text:
                if line.strip():
                    try:
                        value = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(value, dict):
                        yield value
        except (zstandard.ZstdError, UnicodeDecodeError):
            # A lock may intentionally end before the remote zstd frame. Complete
            # JSONL records yielded before the boundary are the locked input.
            return
        finally:
            with contextlib.suppress(Exception):
                text.detach()


def split_and_clean(text: Any, rules: dict[str, Any]) -> Iterator[tuple[str | None, str | None]]:
    if not isinstance(text, str):
        yield None, "missing_text"
        return
    units = text.splitlines() if rules["split_on_lines"] else [text]
    if not units:
        units = [""]
    html_re = re.compile(rules["reject_html_pattern"])
    template_re = re.compile(rules["reject_template_pattern"])
    keyword_stuffing_re = re.compile(rules["reject_keyword_stuffing_pattern"])
    repeated_spam_re = re.compile(rules["reject_repeated_spam_pattern"])
    for raw in units:
        value = raw.replace("\u00a0", " ")
        if rules["collapse_horizontal_whitespace"]:
            value = re.sub(r"[\t\v\f \u2000-\u200b\u202f\u205f\u3000]+", " ", value)
        value = html.unescape(value).strip()
        if not value:
            yield None, "empty"
            continue
        if len(value) < int(rules["min_characters"]):
            yield None, "too_short"
            continue
        if len(value) > int(rules["max_characters"]):
            yield None, "too_long"
            continue
        replacement_ratio = value.count("\ufffd") / len(value)
        if replacement_ratio > float(rules["max_replacement_character_ratio"]):
            yield None, "replacement_characters"
            continue
        control_count = sum(unicodedata.category(char) in {"Cc", "Cs"} for char in value)
        if control_count / len(value) > float(rules["max_control_character_ratio"]):
            yield None, "control_characters"
            continue
        if html_re.search(value):
            yield None, "html_residue"
            continue
        if template_re.search(value):
            yield None, "template_residue"
            continue
        if sum(1 for _ in keyword_stuffing_re.finditer(value)) > int(rules["max_keyword_stuffing_matches"]):
            yield None, "keyword_stuffing"
            continue
        if sum(1 for _ in repeated_spam_re.finditer(value)) > int(rules["max_repeated_spam_matches"]):
            yield None, "keyword_stuffing"
            continue
        yield value, None


def content_id(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def one_permutation_minhash(text: str, parameters: dict[str, Any]) -> tuple[int, ...]:
    n = int(parameters["ngram_size"])
    permutations = int(parameters["permutations"])
    seed = int(parameters["seed"])
    max_ngrams = int(parameters["max_ngrams_per_text"])
    compact = re.sub(r"\s+", " ", text)
    count = max(0, len(compact) - n + 1)
    if count <= max_ngrams:
        positions = range(count)
    else:
        # Bound CPU for very long web segments while retaining deterministic
        # coverage across the entire text instead of truncating to a prefix.
        positions = (index * count // max_ngrams for index in range(max_ngrams))
    tokens = {compact[index : index + n] for index in positions}
    key = seed.to_bytes(16, "little", signed=False)
    bins = [-1] * permutations
    limit = (1 << 128) // permutations
    for token in tokens:
        value = int.from_bytes(hashlib.blake2b(token.encode("utf-8"), digest_size=16, key=key).digest(), "big")
        bucket = min(permutations - 1, value // limit)
        remainder = value % limit
        if bins[bucket] < 0 or remainder < bins[bucket]:
            bins[bucket] = remainder
    return tuple(bins)


def minhash_similarity(left: tuple[int, ...], right: tuple[int, ...]) -> float:
    comparable = [(a, b) for a, b in zip(left, right) if a >= 0 or b >= 0]
    if not comparable:
        return 1.0
    return sum(a == b and a >= 0 for a, b in comparable) / len(comparable)


def signature_buckets(signature: tuple[int, ...], bands: int) -> list[str]:
    rows = len(signature) // bands
    result: list[str] = []
    for band in range(bands):
        section = signature[band * rows : (band + 1) * rows]
        payload = ",".join(map(str, section)).encode("ascii")
        result.append(hashlib.sha256(payload).hexdigest()[:24])
    return result


def script_ratio(language: str, text: str) -> float:
    meaningful = 0
    matches = 0
    if language == "eng_Latn":
        for char in text:
            if char.isalpha():
                meaningful += 1
                matches += "A" <= char <= "Z" or "a" <= char <= "z"
    elif language == "zho_Hans":
        for char in text:
            if char.isalpha():
                meaningful += 1
                matches += "\u3400" <= char <= "\u9fff"
    elif language == "jpn_Jpan":
        for char in text:
            if char.isalpha():
                meaningful += 1
                matches += ("\u3040" <= char <= "\u30ff") or ("\u3400" <= char <= "\u9fff")
    else:
        for char in text:
            if char.isalpha():
                meaningful += 1
                matches += "\uac00" <= char <= "\ud7af"
    return matches / meaningful if meaningful else 0.0


def signature_bucket_keys(signature: tuple[int, ...], bands: int) -> tuple[int, ...]:
    """Return compact keys equivalent to the configured 24-hex-digit buckets."""
    rows = len(signature) // bands
    result: list[int] = []
    for band in range(bands):
        section = signature[band * rows : (band + 1) * rows]
        payload = ",".join(map(str, section)).encode("ascii")
        result.append(int.from_bytes(hashlib.sha256(payload).digest()[:12], "big"))
    return tuple(result)


def pack_minhash(signature: tuple[int, ...]) -> bytes:
    return b"".join((value if value >= 0 else MINHASH_SENTINEL).to_bytes(16, "big") for value in signature)


def packed_minhash_similarity(left: bytes, right: bytes) -> float:
    if len(left) != len(right) or len(left) % 16:
        raise ValueError("packed MinHash signatures have incompatible lengths")
    comparable = 0
    equal = 0
    sentinel = MINHASH_SENTINEL.to_bytes(16, "big")
    for offset in range(0, len(left), 16):
        a = left[offset : offset + 16]
        b = right[offset : offset + 16]
        if a != sentinel or b != sentinel:
            comparable += 1
            equal += a == b and a != sentinel
    return equal / comparable if comparable else 1.0


def _initialize_worker(language: str, cleaning: dict[str, Any], dedup: dict[str, Any], seed: int, script_threshold: float) -> None:
    global _WORKER_LANGUAGE, _WORKER_CLEANING, _WORKER_DEDUP, _WORKER_SEED, _WORKER_SCRIPT_THRESHOLD
    _WORKER_LANGUAGE = language
    _WORKER_CLEANING = cleaning
    _WORKER_DEDUP = dedup
    _WORKER_SEED = seed
    _WORKER_SCRIPT_THRESHOLD = script_threshold


def _fingerprint_cleaned(text: str) -> tuple[bytes, bytes, bytes, int, bool, bytes | None, tuple[int, ...]]:
    assert _WORKER_LANGUAGE is not None and _WORKER_DEDUP is not None
    encoded = text.encode("utf-8")
    cid = hashlib.sha256(encoded).digest()
    priority = hashlib.sha256(str(_WORKER_SEED).encode("ascii") + b":" + cid.hex().encode("ascii")).digest()
    signature_bytes: bytes | None = None
    bucket_keys: tuple[int, ...] = ()
    if _WORKER_LANGUAGE in _WORKER_DEDUP["approximate_languages"]:
        parameters = _WORKER_DEDUP["minhash"]
        signature = one_permutation_minhash(text, parameters)
        signature_bytes = pack_minhash(signature)
        bucket_keys = signature_bucket_keys(signature, int(parameters["bands"]))
    passes_script_check = script_ratio(_WORKER_LANGUAGE, text) >= _WORKER_SCRIPT_THRESHOLD
    return priority, cid, encoded, len(text), passes_script_check, signature_bytes, bucket_keys


def _process_document_batch(
    batch: list[tuple[Any, str]],
) -> list[tuple[str, int, int, int, dict[str, int], list[tuple[bytes, bytes, bytes, int, bool, bytes | None, tuple[int, ...]]]]]:
    """Clean and fingerprint a batch; document boundaries remain explicit."""
    assert _WORKER_CLEANING is not None
    results = []
    for raw_text, source_url in batch:
        input_units = 0
        cleaned_units = 0
        cleaned_characters = 0
        filters: Counter[str] = Counter()
        fingerprints = []
        for cleaned, reason in split_and_clean(raw_text, _WORKER_CLEANING):
            input_units += 1
            if reason:
                filters[reason] += 1
                continue
            assert cleaned is not None
            cleaned_units += 1
            cleaned_characters += len(cleaned)
            fingerprints.append(_fingerprint_cleaned(cleaned))
        results.append((source_url, input_units, cleaned_units, cleaned_characters, dict(filters), fingerprints))
    return results


def _iter_locked_documents(source: dict[str, Any], cached: dict[str, Path]) -> Iterator[dict[str, Any]]:
    for shard in sorted(source["shards"], key=lambda item: item["logical_order"]):
        path = cached[f"{source['source_id']}:{shard['logical_order']}"]
        yield from iter_jsonl_zst_prefix(path)


def _iter_document_batches(documents: Iterator[dict[str, Any]], target_characters: int) -> Iterator[list[tuple[Any, str]]]:
    batch: list[tuple[Any, str]] = []
    characters = 0
    for document in documents:
        raw_text = document.get("text")
        source_url = str(document.get("u") or "")
        batch.append((raw_text, source_url))
        characters += len(raw_text) if isinstance(raw_text, str) else 1
        if characters >= target_characters or len(batch) >= 512:
            yield batch
            batch = []
            characters = 0
    if batch:
        yield batch


def _ordered_document_results(
    documents: Iterator[dict[str, Any]],
    language: str,
    cleaning: dict[str, Any],
    dedup: dict[str, Any],
    seed: int,
    script_threshold: float,
    workers: int,
    batch_characters: int,
) -> Iterator[tuple[str, int, int, int, dict[str, int], list[tuple[bytes, bytes, bytes, int, bool, bytes | None, tuple[int, ...]]]]]:
    batches = _iter_document_batches(documents, batch_characters)
    if workers == 1:
        _initialize_worker(language, cleaning, dedup, seed, script_threshold)
        for batch in batches:
            yield from _process_document_batch(batch)
        return

    executor = ProcessPoolExecutor(
        max_workers=workers,
        initializer=_initialize_worker,
        initargs=(language, cleaning, dedup, seed, script_threshold),
    )
    pending: deque[Future[Any]] = deque()
    exhausted = False

    def submit_one() -> None:
        nonlocal exhausted
        if exhausted:
            return
        try:
            batch = next(batches)
        except StopIteration:
            exhausted = True
            return
        pending.append(executor.submit(_process_document_batch, batch))

    try:
        for _ in range(workers):
            submit_one()
        while pending:
            result = pending.popleft().result()
            submit_one()
            yield from result
    finally:
        for future in pending:
            future.cancel()
        executor.shutdown(wait=True, cancel_futures=True)


class _MemoryStatusEx(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


class _ProcessMemoryCounters(ctypes.Structure):
    _fields_ = [
        ("cb", ctypes.c_ulong),
        ("PageFaultCount", ctypes.c_ulong),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    ]


def memory_snapshot() -> tuple[int, int, int]:
    """Return total physical, available physical, and current-process RSS."""
    total = available = rss = 0
    if os.name == "nt":
        status = _MemoryStatusEx()
        status.dwLength = ctypes.sizeof(status)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            total = int(status.ullTotalPhys)
            available = int(status.ullAvailPhys)
        counters = _ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        get_current_process = ctypes.windll.kernel32.GetCurrentProcess
        get_current_process.restype = ctypes.c_void_p
        process = get_current_process()
        get_process_memory = ctypes.windll.psapi.GetProcessMemoryInfo
        get_process_memory.argtypes = [ctypes.c_void_p, ctypes.POINTER(_ProcessMemoryCounters), ctypes.c_ulong]
        if get_process_memory(process, ctypes.byref(counters), counters.cb):
            rss = int(counters.WorkingSetSize)
    else:
        with contextlib.suppress(OSError, ValueError):
            pages = os.sysconf("SC_PHYS_PAGES")
            available_pages = os.sysconf("SC_AVPHYS_PAGES")
            page_size = os.sysconf("SC_PAGE_SIZE")
            total = int(pages * page_size)
            available = int(available_pages * page_size)
        with contextlib.suppress(OSError, ValueError):
            fields = Path("/proc/self/statm").read_text(encoding="ascii").split()
            rss = int(fields[1]) * int(os.sysconf("SC_PAGE_SIZE"))
    return total, available, rss


def _memory_limits(max_memory_gib: float | None, min_available_memory_gib: float | None) -> tuple[int, int]:
    total, _, _ = memory_snapshot()
    if total <= 0:
        total = 8 * GIB
    if max_memory_gib is not None and max_memory_gib <= 0:
        raise PipelineError("max_memory_gib must be positive")
    if min_available_memory_gib is not None and min_available_memory_gib < 0:
        raise PipelineError("min_available_memory_gib cannot be negative")
    max_rss = int(max_memory_gib * GIB) if max_memory_gib is not None else min(48 * GIB, max(2 * GIB, int(total * 0.40)))
    min_available = (
        int(min_available_memory_gib * GIB)
        if min_available_memory_gib is not None
        else min(32 * GIB, max(512 * MIB, int(total * 0.25)))
    )
    return max_rss, min_available


def _sample_resources(tracker: dict[str, int], language: str, max_rss: int, min_available: int) -> None:
    _, available, rss = memory_snapshot()
    tracker["peak_main_rss_bytes"] = max(tracker.get("peak_main_rss_bytes", 0), rss)
    if available:
        previous = tracker.get("minimum_available_memory_bytes")
        tracker["minimum_available_memory_bytes"] = available if previous is None else min(previous, available)
    if rss > max_rss:
        raise PipelineError(
            f"RAM-first safety stop for {language}: main RSS {rss / GIB:.2f} GiB exceeds {max_rss / GIB:.2f} GiB"
        )
    if available and available < min_available:
        raise PipelineError(
            f"RAM-first safety stop for {language}: system available memory {available / GIB:.2f} GiB is below {min_available / GIB:.2f} GiB"
        )


def percentile_from_counts(counts: dict[int, int] | Counter[int], fraction: float) -> int:
    total = sum(counts.values())
    if total <= 0:
        return 0
    wanted = min(total - 1, int((total - 1) * fraction))
    seen = 0
    for value, count in sorted(counts.items()):
        seen += count
        if seen > wanted:
            return int(value)
    raise AssertionError("unreachable percentile state")


def git_identity(repo_root: Path) -> tuple[str, bool]:
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo_root, text=True, capture_output=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=repo_root, text=True, capture_output=True, check=True).stdout.strip())
        return commit, dirty
    except (OSError, subprocess.CalledProcessError):
        return "unavailable", True


def relevant_hashes(repo_root: Path) -> dict[str, str]:
    paths = [
        repo_root / "scripts" / "fetch_tokenizer_datasets.py",
        repo_root / "scripts" / "tokenizer_dataset_pipeline.py",
        repo_root / "requirements.lock",
    ]
    return {path.relative_to(repo_root).as_posix(): sha256_file(path) for path in paths if path.exists()}


def _build_provenance(
    config: dict[str, Any], config_path: Path, lock_path: Path, profile_name: str, seed: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    repo_root = Path(__file__).resolve().parents[1]
    commit, dirty = git_identity(repo_root)
    dependency_hash = sha256_file(repo_root / "requirements.lock")
    code_hashes = relevant_hashes(repo_root)
    provenance = {
        "algorithm_version": {
            "cleaning": config["cleaning"]["algorithm_version"],
            "deduplication": config["deduplication"]["algorithm_version"],
            "pipeline": "ordered-ram-first-v1",
        },
        "config_sha256": sha256_file(config_path),
        "source_lock_sha256": sha256_file(lock_path),
        "git_commit": commit,
        "git_dirty": dirty,
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "dependency_lock_sha256": dependency_hash,
        "source_code_sha256": code_hashes,
        "profile": profile_name,
        "seed": seed,
    }
    checkpoint_identity = {
        "algorithm_version": provenance["algorithm_version"],
        "config_sha256": provenance["config_sha256"],
        "source_lock_sha256": provenance["source_lock_sha256"],
        "dependency_lock_sha256": dependency_hash,
        "source_code_sha256": code_hashes,
        "profile": profile_name,
        "seed": seed,
    }
    return provenance, checkpoint_identity


def _checkpoint_path(checkpoint_dir: Path, language: str) -> Path:
    return checkpoint_dir / f"{language}.json"


def _load_language_checkpoint(
    checkpoint_dir: Path,
    staging_dir: Path,
    corpus_dir: Path,
    language: str,
    identity: dict[str, Any],
) -> dict[str, Any] | None:
    path = _checkpoint_path(checkpoint_dir, language)
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if (
        state.get("schema_version") != RAM_CHECKPOINT_SCHEMA_VERSION
        or state.get("identity") != identity
        or state.get("language") != language
    ):
        return None
    output = state.get("output")
    if not isinstance(output, dict) or not re.fullmatch(r"[0-9a-f]{64}", str(output.get("sha256", ""))):
        return None
    candidates = (corpus_dir / f"{language}.txt", staging_dir / f"{language}.txt")
    for artifact in candidates:
        if artifact.is_file() and artifact.stat().st_size == int(output.get("bytes", -1)) and sha256_file(artifact) == output["sha256"]:
            state["_artifact_path"] = artifact
            return state
    return None


def _write_language_checkpoint(checkpoint_dir: Path, state: dict[str, Any]) -> None:
    serializable = {key: value for key, value in state.items() if not key.startswith("_")}
    atomic_write_bytes(_checkpoint_path(checkpoint_dir, state["language"]), canonical_json_bytes(serializable))


def _transfer_staged_file(staged_path: Path, final_path: Path, expected_sha256: str, expected_bytes: int) -> str:
    """Copy one staged corpus with one sequential stream, then publish atomically."""
    final_path.parent.mkdir(parents=True, exist_ok=True)
    if staged_path.resolve() == final_path.resolve():
        return str(final_path)
    if staged_path.drive.lower() == final_path.drive.lower():
        os.replace(staged_path, final_path)
    else:
        fd, temporary = tempfile.mkstemp(prefix=f".{final_path.name}.", suffix=".transfer.tmp", dir=final_path.parent)
        digest = hashlib.sha256()
        copied = 0
        try:
            with staged_path.open("rb") as source, os.fdopen(fd, "wb") as destination:
                while chunk := source.read(8 * MIB):
                    destination.write(chunk)
                    digest.update(chunk)
                    copied += len(chunk)
                destination.flush()
                os.fsync(destination.fileno())
            if copied != expected_bytes or digest.hexdigest() != expected_sha256:
                raise PipelineError(f"staged transfer verification failed for {final_path.name}")
            os.replace(temporary, final_path)
        except BaseException:
            with contextlib.suppress(FileNotFoundError):
                os.unlink(temporary)
            raise
        staged_path.unlink()
    if final_path.stat().st_size != expected_bytes:
        raise PipelineError(f"published corpus verification failed for {final_path.name}")
    return str(final_path)


def _build_language_ram_first(
    source: dict[str, Any],
    cached: dict[str, Path],
    config: dict[str, Any],
    budget: int,
    seed: int,
    workers: int,
    staging_dir: Path,
    checkpoint_dir: Path,
    checkpoint_identity: dict[str, Any],
    resource_tracker: dict[str, int],
    max_rss: int,
    min_available: int,
) -> dict[str, Any]:
    language = source["output_language"]
    source_id = source["source_id"]
    approximate = language in config["deduplication"]["approximate_languages"]
    parameters = config["deduplication"]["minhash"]
    band_count = int(parameters["bands"])
    threshold = float(parameters["threshold"])
    exact_seen: set[bytes] = set()
    lsh_buckets: list[dict[int, bytes]] | None = [dict() for _ in range(band_count)] if approximate else None
    # Tuple order intentionally starts with the exact selection keys so the
    # in-place sort has no key-array allocation: priority, cid, text, chars, QA.
    candidates: list[tuple[bytes, bytes, bytes, int, bool, str]] = []
    stats: dict[str, Any] = {
        "documents": 0,
        "documents_with_source_url": 0,
        "input_units": 0,
        "cleaned_units": 0,
        "cleaned_characters": 0,
        "candidate_samples": 0,
        "candidate_characters": 0,
        "filters": Counter(),
        "exact_duplicates": 0,
        "approximate_duplicates": 0,
    }
    candidate_target = budget + budget // 20
    batch_characters = min(DEFAULT_BATCH_CHARACTERS, max(1_000, budget // 4))
    next_progress = 100_000
    documents = _iter_locked_documents(source, cached)
    results = _ordered_document_results(
        documents,
        language,
        config["cleaning"],
        config["deduplication"],
        seed,
        float(config["quality"]["language_min_script_ratio"][language]),
        workers,
        batch_characters,
    )
    try:
        for source_url, input_units, cleaned_units, cleaned_characters, filters, fingerprints in results:
            stats["documents"] += 1
            stats["documents_with_source_url"] += bool(source_url)
            stats["input_units"] += input_units
            stats["cleaned_units"] += cleaned_units
            stats["cleaned_characters"] += cleaned_characters
            stats["filters"].update(filters)
            for priority, cid, text_bytes, characters, script_pass, signature, bucket_keys in fingerprints:
                if cid in exact_seen:
                    stats["exact_duplicates"] += 1
                    continue
                if lsh_buckets is not None:
                    assert signature is not None
                    possible: list[bytes] = []
                    for band, bucket in enumerate(bucket_keys):
                        other = lsh_buckets[band].get(bucket)
                        if other is not None and other not in possible:
                            possible.append(other)
                    if any(packed_minhash_similarity(signature, other) >= threshold for other in possible):
                        stats["approximate_duplicates"] += 1
                        continue
                exact_seen.add(cid)
                if lsh_buckets is not None:
                    assert signature is not None
                    for band, bucket in enumerate(bucket_keys):
                        lsh_buckets[band].setdefault(bucket, signature)
                candidates.append((priority, cid, text_bytes, characters, script_pass, source_url))
                stats["candidate_characters"] += characters
                if len(candidates) >= next_progress:
                    _sample_resources(resource_tracker, language, max_rss, min_available)
                    print(
                        f"[{language}] candidates={len(candidates):,} characters={stats['candidate_characters']:,} "
                        f"main_rss={resource_tracker['peak_main_rss_bytes'] / GIB:.2f}GiB",
                        file=sys.stderr,
                        flush=True,
                    )
                    next_progress += 100_000
            if stats["candidate_characters"] >= candidate_target:
                break
    finally:
        results.close()
        documents.close()

    stats["candidate_samples"] = len(candidates)
    if budget >= 1_000_000 and stats["candidate_characters"] < budget:
        raise PipelineError(
            f"locked input exhausted before {language} reached its {budget:,}-character budget "
            f"({stats['candidate_characters']:,} candidates)"
        )
    _sample_resources(resource_tracker, language, max_rss, min_available)
    print(f"[{language}] sorting {len(candidates):,} RAM candidates", file=sys.stderr, flush=True)
    candidates.sort()

    staging_dir.mkdir(parents=True, exist_ok=True)
    staged_path = staging_dir / f"{language}.txt"
    for stale in staging_dir.glob(f".{language}.*.tmp"):
        stale.unlink(missing_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{language}.", suffix=".tmp", dir=staging_dir)
    selected_characters = 0
    selected_lines = 0
    length_counts: Counter[int] = Counter()
    review_heap: list[tuple[int, bytes, bool, bytes]] = []
    review_count = int(config["quality"]["review_sample_count"])
    review_seed = int(config["quality"]["review_seed"])
    output_digest = hashlib.sha256()
    try:
        with os.fdopen(fd, "wb") as handle:
            for _priority, cid, text_bytes, characters, script_pass, source_url in candidates:
                if selected_characters + characters > budget:
                    continue
                line = text_bytes + b"\n"
                handle.write(line)
                output_digest.update(line)
                selected_characters += characters
                selected_lines += 1
                length_counts[characters] += 1
                rank = int.from_bytes(hashlib.sha256(str(review_seed).encode("ascii") + b":" + cid.hex().encode("ascii")).digest(), "big")
                source_url_digest = hashlib.sha256(source_url.encode("utf-8")).digest()
                item = (-rank, cid, script_pass, source_url_digest)
                if len(review_heap) < review_count:
                    heapq.heappush(review_heap, item)
                elif rank < -review_heap[0][0]:
                    heapq.heapreplace(review_heap, item)
                if selected_characters == budget:
                    break
            handle.flush()
            os.fsync(handle.fileno())
        minimum_characters = 1 if budget < 1_000_000 else budget - int(config["cleaning"]["max_characters"])
        if selected_lines == 0 or selected_characters < minimum_characters:
            raise PipelineError(
                f"deterministic selection for {language} retained only {selected_characters:,}/{budget:,} characters"
            )
        os.replace(temporary, staged_path)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temporary)
        raise

    ordered_review = sorted(review_heap, key=lambda item: (-item[0], item[1]))
    review = [
        {
            "content_id": cid.hex(),
            "source_url_sha256": source_url_digest.hex(),
            "status": "pass" if script_pass else "review",
        }
        for _negative_rank, cid, script_pass, source_url_digest in ordered_review
    ]
    stats.update(
        {
            "sampled_out": len(candidates) - selected_lines,
            "final_lines": selected_lines,
            "final_characters": selected_characters,
            "length_counts": {str(length): count for length, count in sorted(length_counts.items())},
            "filters": dict(sorted(stats["filters"].items())),
        }
    )
    output = {
        "sha256": output_digest.hexdigest(),
        "bytes": staged_path.stat().st_size,
        "lines": selected_lines,
        "characters": selected_characters,
    }
    state = {
        "schema_version": RAM_CHECKPOINT_SCHEMA_VERSION,
        "identity": checkpoint_identity,
        "language": language,
        "output": output,
        "stats": stats,
        "source_sample_counts": {source_id: selected_lines},
        "review": review,
        "_artifact_path": staged_path,
    }
    _write_language_checkpoint(checkpoint_dir, state)
    _sample_resources(resource_tracker, language, max_rss, min_available)
    print(
        f"[{language}] staged {selected_lines:,} lines / {selected_characters:,} chars at {staged_path}",
        file=sys.stderr,
        flush=True,
    )
    return state


def build_corpus(
    config: dict[str, Any],
    config_path: Path,
    lock: dict[str, Any],
    lock_path: Path,
    out_root: Path,
    cache_root: Path,
    profile_name: str,
    seed: int,
    *,
    offline: bool,
    use_cache: bool,
    resume: bool = False,
    staging_root: Path | None = None,
    max_memory_gib: float | None = None,
    min_available_memory_gib: float | None = None,
) -> dict[str, Any]:
    started = time.time()
    profile = config["profiles"][profile_name]
    budget = int(profile["character_budget_per_language"])
    workers = int(profile["concurrency"])
    if workers <= 0:
        raise PipelineError("profile concurrency must be positive")
    max_rss, min_available = _memory_limits(max_memory_gib, min_available_memory_gib)
    resource_tracker: dict[str, int] = {"peak_main_rss_bytes": 0}
    _sample_resources(resource_tracker, "startup", max_rss, min_available)
    provenance, checkpoint_identity = _build_provenance(config, config_path, lock_path, profile_name, seed)

    # Validation is a sequential read of each locked prefix. No candidate or
    # index data is written during this phase.
    cached = ensure_cache(lock, cache_root, offline=offline, use_cache=use_cache)
    interim = out_root / "interim" / profile_name / "ram-first"
    checkpoint_dir = interim / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    corpus_dir = out_root / profile["corpus_subdir"]
    corpus_dir.mkdir(parents=True, exist_ok=True)
    for language in LANGUAGES:
        for stale in corpus_dir.glob(f".{language}.*.transfer.tmp"):
            stale.unlink(missing_ok=True)
    namespace = hashlib.sha256(str(out_root.resolve()).encode("utf-8")).hexdigest()[:16]
    staging_dir = (
        staging_root / namespace / profile_name
        if staging_root is not None
        else interim / "staging"
    )
    manifest_path = corpus_dir / "manifest.jsonl"
    run_path = corpus_dir / "run.json"
    report_path = out_root / "reports" / f"tokenizer_corpus_{profile_name}.md"
    # A stale manifest must never make an interrupted rebuild look complete.
    manifest_path.unlink(missing_ok=True)
    run_path.unlink(missing_ok=True)
    report_path.unlink(missing_ok=True)

    states: dict[str, dict[str, Any]] = {}
    transfers: dict[str, Future[str]] = {}
    sources = sorted(lock["sources"], key=lambda item: item["output_language"])
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="tokenizer-corpus-transfer") as transfer_pool:
        for source in sources:
            language = source["output_language"]
            state = (
                _load_language_checkpoint(checkpoint_dir, staging_dir, corpus_dir, language, checkpoint_identity)
                if resume
                else None
            )
            if state is None:
                state = _build_language_ram_first(
                    source,
                    cached,
                    config,
                    budget,
                    seed,
                    workers,
                    staging_dir,
                    checkpoint_dir,
                    checkpoint_identity,
                    resource_tracker,
                    max_rss,
                    min_available,
                )
            else:
                print(f"[{language}] resume checkpoint verified", file=sys.stderr, flush=True)
            states[language] = state
            artifact = Path(state["_artifact_path"])
            final_path = corpus_dir / f"{language}.txt"
            if artifact.resolve() != final_path.resolve():
                output = state["output"]
                transfers[language] = transfer_pool.submit(
                    _transfer_staged_file,
                    artifact,
                    final_path,
                    output["sha256"],
                    int(output["bytes"]),
                )
            else:
                state["_artifact_path"] = final_path

        # The deterministic manifest is the completion marker and is not
        # published until every background transfer has verified and renamed.
        for language in LANGUAGES:
            if language in transfers:
                final_path = Path(transfers[language].result())
                states[language]["_artifact_path"] = final_path
                print(f"[{language}] background transfer verified at {final_path}", file=sys.stderr, flush=True)

    output_metadata = {language: states[language]["output"] for language in LANGUAGES}
    stats = {language: states[language]["stats"] for language in LANGUAGES}
    source_proportions = {language: states[language]["source_sample_counts"] for language in LANGUAGES}
    review = {language: states[language]["review"] for language in LANGUAGES}

    source_config = {source["source_id"]: source for source in enabled_sources(config)}
    manifest_records: list[dict[str, Any]] = []
    for language in LANGUAGES:
        language_sources = []
        for source in lock["sources"]:
            if source["output_language"] == language:
                registry = source_config[source["source_id"]]
                language_sources.append(
                    {
                        "source_id": source["source_id"],
                        "source_language": source["source_language"],
                        "output_language": language,
                        "version_or_snapshot": registry["version_or_snapshot"],
                        "license": registry["license"],
                        "homepage": registry["homepage"],
                        "map_sha256": source["map_sha256"],
                        "shard_prefix_sha256": [item["sha256"] for item in source["shards"]],
                    }
                )
        item = output_metadata[language]
        language_stats = stats[language]
        cleaned_units = max(1, int(language_stats["cleaned_units"]))
        manifest_records.append(
            {
                "language": language,
                "file": f"{language}.txt",
                "sha256": item["sha256"],
                "bytes": item["bytes"],
                "samples": item["lines"],
                "characters": item["characters"],
                "documents": language_stats["documents"],
                "documents_with_source_url": language_stats["documents_with_source_url"],
                "input_units": language_stats["input_units"],
                "cleaned_units": language_stats["cleaned_units"],
                "cleaned_characters": language_stats["cleaned_characters"],
                "candidate_samples": language_stats["candidate_samples"],
                "candidate_characters": language_stats["candidate_characters"],
                "filters": dict(sorted(language_stats["filters"].items())),
                "exact_duplicates": language_stats["exact_duplicates"],
                "exact_duplicate_rate": round(language_stats["exact_duplicates"] / cleaned_units, 9),
                "approximate_duplicates": language_stats["approximate_duplicates"],
                "approximate_duplicate_rate": round(language_stats["approximate_duplicates"] / cleaned_units, 9),
                "sampled_out": language_stats["sampled_out"],
                "source_sample_counts": source_proportions[language],
                "sources": language_sources,
                "provenance": provenance,
            }
        )
    atomic_write_bytes(manifest_path, b"".join(canonical_json_bytes(record) for record in manifest_records))

    lines = [
        f"# Tokenizer corpus quality report: {profile_name}",
        "",
        "Deterministic rebuild command:",
        "",
        f"`python scripts/fetch_tokenizer_datasets.py --profile {profile_name} --use-cache --offline`",
        "",
        "## Provenance",
        "",
        f"- Config SHA-256: `{provenance['config_sha256']}`",
        f"- Source lock SHA-256: `{provenance['source_lock_sha256']}`",
        f"- Dependency lock SHA-256: `{provenance['dependency_lock_sha256']}`",
        f"- Pipeline: `{provenance['algorithm_version']['pipeline']}`; seed: `{seed}`",
        "",
        "## Summary",
        "",
        "| Language | Lines | Characters | p50 | p95 | Exact dedup | Approx dedup | Sampled out | SHA-256 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for language in LANGUAGES:
        language_stats = stats[language]
        length_counts = {int(key): int(value) for key, value in language_stats["length_counts"].items()}
        lines.append(
            f"| {language} | {language_stats['final_lines']} | {language_stats['final_characters']} | "
            f"{percentile_from_counts(length_counts, .5)} | {percentile_from_counts(length_counts, .95)} | "
            f"{language_stats['exact_duplicates']} | {language_stats['approximate_duplicates']} | "
            f"{language_stats['sampled_out']} | `{output_metadata[language]['sha256']}` |"
        )
    lines.extend(["", "## Filtering and sources", ""])
    for language in LANGUAGES:
        language_stats = stats[language]
        input_units = max(1, int(language_stats["input_units"]))
        cleaned_units = max(1, int(language_stats["cleaned_units"]))
        filter_text = ", ".join(
            f"`{key}`={value} ({value / input_units:.4%})" for key, value in sorted(language_stats["filters"].items())
        ) or "none"
        source_total = max(1, sum(source_proportions[language].values()))
        source_text = ", ".join(
            f"`{key}`={value} ({value / source_total:.4%})" for key, value in source_proportions[language].items()
        )
        lines.extend(
            [
                f"### {language}",
                "",
                f"Documents: {language_stats['documents']}; input units: {language_stats['input_units']}; "
                f"documents with source URL: {language_stats['documents_with_source_url']}; "
                f"cleaned units: {language_stats['cleaned_units']}; cleaned characters: {language_stats['cleaned_characters']}; "
                f"candidate units: {language_stats['candidate_samples']}; candidate characters: {language_stats['candidate_characters']}.",
                "",
                f"Exact dedup rate: {language_stats['exact_duplicates'] / cleaned_units:.6%}; "
                f"approximate dedup rate: {language_stats['approximate_duplicates'] / cleaned_units:.6%}.",
                "",
                f"Filter reason counts and input rates: {filter_text}.",
                "",
                f"Final source proportions: {source_text}.",
                "",
            ]
        )
    lines.extend(
        [
            "## Fixed manual-review sample",
            "",
            "The report records stable content IDs and automated script checks only; source text is intentionally omitted.",
            "",
        ]
    )
    for language in LANGUAGES:
        review_items = review[language]
        review_required = sum(item["status"] == "review" for item in review_items)
        lines.extend(
            [
                f"### {language}",
                "",
                f"Automated result: {'pass' if review_required == 0 else 'review required'}; "
                f"{len(review_items)} stable samples, {review_required} below the configured script-ratio threshold. "
                "Retained lines passed the configured HTML, replacement-character, control-character and length filters.",
                "",
                ", ".join(f"`{item['content_id'][:16]}` ({item['status']})" for item in review_items) + ".",
                "",
            ]
        )
    atomic_write_bytes(report_path, ("\n".join(lines).rstrip() + "\n").encode("utf-8"))

    _sample_resources(resource_tracker, "finalize", max_rss, min_available)
    run_record = {
        "started_at_utc": dt.datetime.fromtimestamp(started, dt.timezone.utc).isoformat(),
        "finished_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "duration_seconds": round(time.time() - started, 3),
        "out_root": str(out_root.resolve()),
        "cache_root": str(cache_root.resolve()),
        "staging_root": str(staging_root.resolve()) if staging_root is not None else str(staging_dir.resolve()),
        "profile": profile_name,
        "concurrency": workers,
        "offline": offline,
        "use_cache": use_cache,
        "resume": resume,
        "max_main_rss_bytes": max_rss,
        "minimum_available_memory_limit_bytes": min_available,
        "peak_main_rss_bytes": resource_tracker["peak_main_rss_bytes"],
        "minimum_available_memory_bytes": resource_tracker.get("minimum_available_memory_bytes"),
        "manifest_sha256": sha256_file(manifest_path),
        "report_sha256": sha256_file(report_path),
    }
    atomic_write_bytes(run_path, json.dumps(run_record, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8") + b"\n")
    return {
        "corpus_dir": str(corpus_dir),
        "report": str(report_path),
        "manifest_sha256": run_record["manifest_sha256"],
        "outputs": output_metadata,
        "duration_seconds": run_record["duration_seconds"],
        "peak_main_rss_bytes": run_record["peak_main_rss_bytes"],
    }


def dry_run_plan(
    config: dict[str, Any],
    lock: dict[str, Any],
    out_root: Path,
    cache_root: Path,
    profile_name: str,
    seed: int,
    offline: bool,
    use_cache: bool,
    *,
    staging_root: Path | None = None,
    max_memory_gib: float | None = None,
    min_available_memory_gib: float | None = None,
) -> dict[str, Any]:
    profile = config["profiles"][profile_name]
    return {
        "action": "build",
        "profile": profile_name,
        "sources": [
            {
                "source_id": source["source_id"],
                "mapping": f"{source['source_language']} -> {source['output_language']}",
                "shards": [
                    {
                        "wds": shard["wds"],
                        "url": shard["url"],
                        "locked_bytes": shard["locked_bytes"],
                    }
                    for shard in source["shards"]
                ],
            }
            for source in lock["sources"]
        ],
        "quality_wds": config["dataset"]["quality_wds"],
        "character_budget_per_language": profile["character_budget_per_language"],
        "seed": seed,
        "concurrency": profile["concurrency"],
        "cache": str(cache_root),
        "staging": str(staging_root) if staging_root is not None else str(out_root / "interim" / profile_name / "ram-first" / "staging"),
        "output": str(out_root / profile["corpus_subdir"]),
        "ram_first": {
            "single_language_resident": True,
            "candidate_database": False,
            "max_memory_gib": max_memory_gib,
            "min_available_memory_gib": min_available_memory_gib,
            "background_transfers": 1,
        },
        "network_allowed": not (offline or use_cache),
        "operations": ["validate locked cache or download locked prefixes", "single-stream JSONL read", "ordered parallel cleaning and fingerprints", "RAM exact and configured MinHash deduplication", "RAM seeded balanced sampling", "stage completed language", "one-at-a-time verified background transfer", "atomic manifest, run record, and quality report output"],
    }
