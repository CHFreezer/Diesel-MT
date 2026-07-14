#!/usr/bin/env python3
"""TD-04 deterministic group split, deduplication, and leakage protection."""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Iterator, Mapping, Sequence

import yaml

from model_training_contract import (
    ContractError,
    canonical_json_bytes,
    config_sha256,
    directed_routes,
    pair_id,
    validate_model_data_config,
    validate_parallel_sample,
)


PIPELINE_VERSION = "td04-v1"
SPLIT_PROFILE: dict[str, Any] = {
    "version": "massive-locked-count-hash-v1",
    "hash": "sha256",
    "bucket_count": 16521,
    "train_buckets": 11514,
    "dev_buckets": 2033,
    "test_buckets": 2974,
}
DEDUP_PROFILE: dict[str, Any] = {
    "version": "unicode-char-trigram-minhash-v1",
    "exact_text": "NFC output text, case-sensitive",
    "near_normalization": "NFC output text, Unicode casefold, collapsed whitespace",
    "ngram_size": 3,
    "minimum_ngrams": 5,
    "jaccard_threshold": 0.82,
    "minhash_permutations": 24,
    "minhash_bands": 6,
    "exhaustive_text_limit": 2000,
    "max_reported_hits": 200,
}
REFERENCE_KINDS = {
    "tokenizer_corpus",
    "tokenizer_holdout",
    "tokenizer_evaluation",
    "mt_evaluation",
    "same_source_version",
}
REFERENCE_POLICIES = {"report", "block"}
REFERENCE_FORMATS = {
    "tokenizer-manifest-jsonl",
    "tokenizer-evaluation-manifest-json",
}
SPLIT_ORDER = {"train": 0, "dev": 1, "test": 2}
MASK64 = (1 << 64) - 1


class SplitPipelineError(RuntimeError):
    """Base class for expected TD-04 failures."""

    exit_code = 2


class InputError(SplitPipelineError):
    """TD-03 input or contamination registry is invalid."""


class LeakageError(SplitPipelineError):
    """A group, reverse route, derivation, or near duplicate crosses splits."""


class ContaminationError(SplitPipelineError):
    """A blocking external reference overlap prevents publication."""


@dataclass(frozen=True)
class TextEntry:
    language: str
    text: str
    normalized: str
    groups: tuple[str, ...]
    splits: tuple[str, ...] = ()


class UnionFind:
    """Deterministic union-find whose representative is always lexical minimum."""

    def __init__(self, values: Iterable[str]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> str:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return left_root
        root, child = sorted((left_root, right_root))
        self.parent[child] = root
        return root


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _exact_keys(
    value: Mapping[str, Any],
    required: set[str],
    context: str,
    *,
    optional: set[str] | None = None,
) -> None:
    optional = optional or set()
    missing = required - set(value)
    unknown = set(value) - required - optional
    if missing:
        raise InputError(f"{context} missing fields: {', '.join(sorted(missing))}")
    if unknown:
        raise InputError(f"{context} unknown fields: {', '.join(sorted(unknown))}")


def _locked_relative_path(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise InputError(f"{context} must be a non-empty POSIX repository-relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or path.parts[0] != "data":
        raise InputError(f"{context} must remain below data/")
    return path.as_posix()


def validate_contamination_registry(registry: Mapping[str, Any]) -> dict[str, Any]:
    _exact_keys(
        registry,
        {"schema_version", "identity", "reference_sets", "requirements", "derived_sample_links"},
        "contamination registry",
    )
    if registry["schema_version"] != 1:
        raise InputError("contamination registry schema_version must be 1")
    identity = registry["identity"]
    if not isinstance(identity, dict):
        raise InputError("contamination registry identity must be an object")
    _exact_keys(identity, {"name", "status"}, "contamination registry identity")
    if identity != {
        "name": "mvp_model_contamination_registry",
        "status": "td04-locked-runtime-inputs",
    }:
        raise InputError("contamination registry identity changed")
    reference_sets = registry["reference_sets"]
    if not isinstance(reference_sets, list):
        raise InputError("contamination registry reference_sets must be a list")
    reference_ids: set[str] = set()
    for index, reference in enumerate(reference_sets):
        if not isinstance(reference, dict):
            raise InputError(f"reference_sets[{index}] must be an object")
        _exact_keys(
            reference,
            {"reference_id", "kind", "policy", "manifest"},
            f"reference_sets[{index}]",
        )
        reference_id = reference["reference_id"]
        if not isinstance(reference_id, str) or not reference_id or reference_id in reference_ids:
            raise InputError("reference IDs must be unique non-empty strings")
        reference_ids.add(reference_id)
        if reference["kind"] not in REFERENCE_KINDS:
            raise InputError(f"unsupported reference kind: {reference['kind']}")
        if reference["policy"] not in REFERENCE_POLICIES:
            raise InputError(f"unsupported reference policy: {reference['policy']}")
        if reference["kind"] in {"mt_evaluation", "same_source_version"} and reference["policy"] != "block":
            raise InputError(f"{reference['kind']} references must use policy=block")
        manifest = reference["manifest"]
        if not isinstance(manifest, dict):
            raise InputError(f"{reference_id}.manifest must be an object")
        _exact_keys(manifest, {"path", "format", "bytes", "sha256"}, f"{reference_id}.manifest")
        _locked_relative_path(manifest["path"], f"{reference_id}.manifest.path")
        if manifest["format"] not in REFERENCE_FORMATS:
            raise InputError(f"unsupported reference manifest format: {manifest['format']}")
        if not isinstance(manifest["bytes"], int) or manifest["bytes"] <= 0:
            raise InputError(f"{reference_id}.manifest.bytes must be positive")
        if not re.fullmatch(r"[0-9a-f]{64}", str(manifest["sha256"])):
            raise InputError(f"{reference_id}.manifest.sha256 is invalid")
    requirements = registry["requirements"]
    if not isinstance(requirements, dict) or set(requirements) != {
        "formal_mt_evaluation",
        "same_source_versions",
    }:
        raise InputError("contamination registry requirements are incomplete")
    for name, requirement in requirements.items():
        if not isinstance(requirement, dict):
            raise InputError(f"requirements.{name} must be an object")
        _exact_keys(requirement, {"status", "required_before_m0", "note"}, f"requirements.{name}")
        if requirement["required_before_m0"] is not True or not isinstance(requirement["note"], str):
            raise InputError(f"requirements.{name} must be explicit before M0")
    links = registry["derived_sample_links"]
    if not isinstance(links, list):
        raise InputError("derived_sample_links must be a list")
    for index, link in enumerate(links):
        if not isinstance(link, dict):
            raise InputError(f"derived_sample_links[{index}] must be an object")
        _exact_keys(link, {"child_sample_id", "parent_sample_id", "reason"}, f"derived_sample_links[{index}]")
        if not all(isinstance(link[field], str) and link[field] for field in link):
            raise InputError("derived sample links require non-empty string fields")
    return dict(registry)


def load_contamination_registry(path: Path) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise InputError(f"cannot load contamination registry {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise InputError("contamination registry root must be an object")
    return validate_contamination_registry(value)


def registry_is_complete(registry: Mapping[str, Any]) -> bool:
    kinds = {reference["kind"] for reference in registry["reference_sets"]}
    formal = registry["requirements"]["formal_mt_evaluation"]["status"]
    versions = registry["requirements"]["same_source_versions"]["status"]
    return "mt_evaluation" in kinds and formal == "locked" and versions in {
        "none_identified",
        "locked",
    }


def _near_normalize(text: str) -> str:
    return " ".join(text.casefold().split())


def _ngrams(normalized: str) -> frozenset[str]:
    size = int(DEDUP_PROFILE["ngram_size"])
    padded = f"^{normalized}$"
    if len(padded) <= size:
        return frozenset({padded})
    return frozenset(padded[index : index + size] for index in range(len(padded) - size + 1))


def _jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    union = len(left | right)
    return len(left & right) / union if union else 1.0


def _minhash_signature(grams: frozenset[str]) -> tuple[int, ...]:
    permutations = int(DEDUP_PROFILE["minhash_permutations"])
    minima = [MASK64] * permutations
    for gram in grams:
        digest = hashlib.blake2b(gram.encode("utf-8"), digest_size=16, person=b"DieselMT-TD04").digest()
        first = int.from_bytes(digest[:8], "big")
        second = int.from_bytes(digest[8:], "big") | 1
        for index in range(permutations):
            value = (first + index * second) & MASK64
            if value < minima[index]:
                minima[index] = value
    return tuple(minima)


def _signature_bands(signature: tuple[int, ...]) -> Iterator[tuple[int, tuple[int, ...]]]:
    bands = int(DEDUP_PROFILE["minhash_bands"])
    width = len(signature) // bands
    for band in range(bands):
        start = band * width
        yield band, signature[start : start + width]


def _candidate_pairs(entries: Sequence[TextEntry]) -> Iterator[tuple[int, int]]:
    if len(entries) <= int(DEDUP_PROFILE["exhaustive_text_limit"]):
        for left in range(len(entries)):
            for right in range(left + 1, len(entries)):
                yield left, right
        return
    buckets: dict[tuple[int, tuple[int, ...]], list[int]] = defaultdict(list)
    signatures = [_minhash_signature(_ngrams(entry.normalized)) for entry in entries]
    pairs: set[tuple[int, int]] = set()
    for index, signature in enumerate(signatures):
        candidates: set[int] = set()
        for band_key in _signature_bands(signature):
            candidates.update(buckets[band_key])
        for candidate in candidates:
            pairs.add((candidate, index))
        for band_key in _signature_bands(signature):
            buckets[band_key].append(index)
    yield from sorted(pairs)


def find_near_duplicates(entries: Sequence[TextEntry]) -> Iterator[tuple[TextEntry, TextEntry, float]]:
    grams = [_ngrams(entry.normalized) for entry in entries]
    minimum = int(DEDUP_PROFILE["minimum_ngrams"])
    threshold = float(DEDUP_PROFILE["jaccard_threshold"])
    for left_index, right_index in _candidate_pairs(entries):
        if min(len(grams[left_index]), len(grams[right_index])) < minimum:
            continue
        score = _jaccard(grams[left_index], grams[right_index])
        if score >= threshold:
            yield entries[left_index], entries[right_index], score


def split_for_component(component_id: str, profile: Mapping[str, Any] = SPLIT_PROFILE) -> str:
    if (
        int(profile["train_buckets"])
        + int(profile["dev_buckets"])
        + int(profile["test_buckets"])
        != int(profile["bucket_count"])
    ):
        raise InputError("split profile bucket counts are inconsistent")
    payload = canonical_json_bytes(
        {"split_profile_version": profile["version"], "component_id": component_id}
    )
    bucket = int(sha256_bytes(payload), 16) % int(profile["bucket_count"])
    train_end = int(profile["train_buckets"])
    dev_end = train_end + int(profile["dev_buckets"])
    if bucket < train_end:
        return "train"
    if bucket < dev_end:
        return "dev"
    return "test"


def _stable_sample_id(sample: Mapping[str, Any]) -> str:
    identity = {
        "sample_group_id": sample["sample_group_id"],
        "src_lang": sample["src_lang"],
        "tgt_lang": sample["tgt_lang"],
        "source_text": sample["source_text"],
        "target_text": sample["target_text"],
    }
    return f"sample-sha256:{sha256_bytes(canonical_json_bytes(identity))}"


def _validate_undirected_inputs(
    samples: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    derived_links: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    canonical_pairs = {
        frozenset(str(tag) for tag in pair["tags"]): tuple(str(tag) for tag in pair["tags"])
        for pair in config["directions"]["undirected_pairs"]
    }
    ordered: list[dict[str, Any]] = []
    by_id: dict[str, dict[str, Any]] = {}
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    alignment_groups: dict[tuple[str, str, str], str] = {}
    for value in samples:
        try:
            sample = validate_parallel_sample(value, config)
        except ContractError as exc:
            raise InputError(f"input sample violates TD-01 contract: {exc}") from exc
        sample_id = str(sample["sample_id"])
        if sample_id in by_id:
            raise InputError(f"duplicate input sample ID: {sample_id}")
        expected = canonical_pairs.get(frozenset({str(sample["src_lang"]), str(sample["tgt_lang"])}))
        if expected is None or (sample["src_lang"], sample["tgt_lang"]) != expected:
            raise InputError(
                "TD-04 input must contain only the configured canonical orientation before reverse expansion"
            )
        by_id[sample_id] = sample
        groups[str(sample["sample_group_id"])].append(sample)
        provenance = sample.get("provenance")
        if isinstance(provenance, dict) and provenance.get("kind") == "human_parallel":
            identity = (
                str(sample["source_id"]),
                str(sample["source_version"]),
                str(provenance["alignment_key"]),
            )
            previous = alignment_groups.setdefault(identity, str(sample["sample_group_id"]))
            if previous != sample["sample_group_id"]:
                raise LeakageError("the same source alignment key was assigned to different groups")
        ordered.append(sample)
    for group_id, group_samples in groups.items():
        if len({sample["split"] for sample in group_samples}) != 1:
            raise LeakageError(f"input group already crosses splits: {group_id}")
    links = list(derived_links)
    for sample in ordered:
        provenance = sample.get("provenance")
        if isinstance(provenance, dict) and provenance.get("kind") == "script_conversion":
            links.append(
                {
                    "child_sample_id": sample["sample_id"],
                    "parent_sample_id": provenance["source_sample_id"],
                    "reason": "script_conversion provenance",
                }
            )
    for link in links:
        child = by_id.get(str(link["child_sample_id"]))
        parent = by_id.get(str(link["parent_sample_id"]))
        if child is None or parent is None:
            raise LeakageError("derived sample link refers to a missing parent or child")
        if child["sample_group_id"] != parent["sample_group_id"]:
            raise LeakageError(
                f"derived sample is not bound to its parent group: {link['child_sample_id']}"
            )
    return sorted(ordered, key=lambda sample: str(sample["sample_id"])), groups


def _text_entries(
    samples: Sequence[Mapping[str, Any]],
    group_splits: Mapping[str, str] | None = None,
) -> dict[str, list[TextEntry]]:
    groups_by_text: dict[tuple[str, str], set[str]] = defaultdict(set)
    originals: dict[tuple[str, str], str] = {}
    for sample in samples:
        for language_field, text_field in (
            ("src_lang", "source_text"),
            ("tgt_lang", "target_text"),
        ):
            language = str(sample[language_field])
            text = str(sample[text_field])
            normalized = _near_normalize(text)
            key = (language, normalized)
            originals.setdefault(key, text)
            groups_by_text[key].add(str(sample["sample_group_id"]))
    result: dict[str, list[TextEntry]] = defaultdict(list)
    for (language, normalized), group_ids in sorted(groups_by_text.items()):
        splits = (
            tuple(sorted({group_splits[group_id] for group_id in group_ids}, key=SPLIT_ORDER.get))
            if group_splits is not None
            else ()
        )
        result[language].append(
            TextEntry(
                language=language,
                text=originals[(language, normalized)],
                normalized=normalized,
                groups=tuple(sorted(group_ids)),
                splits=splits,
            )
        )
    return result


def _assign_split_components(
    samples: Sequence[Mapping[str, Any]], groups: Mapping[str, Sequence[Mapping[str, Any]]]
) -> tuple[dict[str, str], dict[str, str], dict[str, Any]]:
    union = UnionFind(groups)
    exact_groups: dict[tuple[str, str], set[str]] = defaultdict(set)
    for sample in samples:
        exact_groups[(str(sample["src_lang"]), str(sample["source_text"]))].add(
            str(sample["sample_group_id"])
        )
        exact_groups[(str(sample["tgt_lang"]), str(sample["target_text"]))].add(
            str(sample["sample_group_id"])
        )
    exact_duplicate_text_keys = 0
    exact_group_links = 0
    for group_ids in exact_groups.values():
        if len(group_ids) > 1:
            exact_duplicate_text_keys += 1
            ordered = sorted(group_ids)
            for group_id in ordered[1:]:
                union.union(ordered[0], group_id)
                exact_group_links += 1

    near_hit_count = 0
    reported_near_hits: list[dict[str, Any]] = []
    for language, entries in _text_entries(samples).items():
        for entry in entries:
            if len(entry.groups) > 1:
                for group_id in entry.groups[1:]:
                    union.union(entry.groups[0], group_id)
                near_hit_count += 1
                if len(reported_near_hits) < int(DEDUP_PROFILE["max_reported_hits"]):
                    reported_near_hits.append(
                        {
                            "language": language,
                            "left_text_sha256": sha256_bytes(entry.text.encode("utf-8")),
                            "right_text_sha256": sha256_bytes(entry.normalized.encode("utf-8")),
                            "jaccard": 1.0,
                        }
                    )
        for left, right, score in find_near_duplicates(entries):
            if left.normalized == right.normalized:
                continue
            near_hit_count += 1
            for left_group in left.groups:
                for right_group in right.groups:
                    union.union(left_group, right_group)
            if len(reported_near_hits) < int(DEDUP_PROFILE["max_reported_hits"]):
                reported_near_hits.append(
                    {
                        "language": language,
                        "left_text_sha256": sha256_bytes(left.text.encode("utf-8")),
                        "right_text_sha256": sha256_bytes(right.text.encode("utf-8")),
                        "jaccard": round(score, 6),
                    }
                )

    components: dict[str, list[str]] = defaultdict(list)
    for group_id in groups:
        components[union.find(group_id)].append(group_id)
    group_splits: dict[str, str] = {}
    group_components: dict[str, str] = {}
    component_counts: Counter[str] = Counter()
    for group_ids in sorted((sorted(values) for values in components.values())):
        component_id = f"component-sha256:{sha256_bytes(canonical_json_bytes(group_ids))}"
        split = split_for_component(component_id)
        component_counts[split] += 1
        for group_id in group_ids:
            group_splits[group_id] = split
            group_components[group_id] = component_id
    report = {
        "exact_duplicate_text_keys": exact_duplicate_text_keys,
        "exact_group_links": exact_group_links,
        "near_duplicate_links": near_hit_count,
        "reported_near_duplicate_links": reported_near_hits,
        "split_components": len(components),
        "component_counts_by_split": {
            split: int(component_counts[split]) for split in SPLIT_ORDER
        },
    }
    return group_splits, group_components, report


def _deduplicate_pairs(
    samples: Sequence[Mapping[str, Any]], group_splits: Mapping[str, str]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    pair_winners: dict[tuple[Any, ...], str] = {}
    removed: list[str] = []
    source_counts: Counter[tuple[str, str, str]] = Counter()
    target_counts: Counter[tuple[str, str, str]] = Counter()
    for value in sorted(samples, key=lambda sample: str(sample["sample_id"])):
        sample = dict(value)
        sample["split"] = group_splits[str(sample["sample_group_id"])]
        language_text = tuple(
            sorted(
                (
                    (str(sample["src_lang"]), str(sample["source_text"])),
                    (str(sample["tgt_lang"]), str(sample["target_text"])),
                )
            )
        )
        key = (pair_id([str(sample["src_lang"]), str(sample["tgt_lang"])]), language_text)
        if key in pair_winners:
            removed.append(str(sample["sample_id"]))
            continue
        pair_winners[key] = str(sample["sample_id"])
        source_counts[(str(sample["src_lang"]), str(sample["source_text"]), str(sample["tgt_lang"]))] += 1
        target_counts[(str(sample["tgt_lang"]), str(sample["target_text"]), str(sample["src_lang"]))] += 1
        kept.append(sample)
    report = {
        "pair_exact_duplicates_removed": len(removed),
        "removed_sample_ids": removed[: int(DEDUP_PROFILE["max_reported_hits"])],
        "source_exact_collision_keys": sum(1 for count in source_counts.values() if count > 1),
        "source_exact_duplicate_occurrences": sum(count - 1 for count in source_counts.values() if count > 1),
        "target_exact_collision_keys": sum(1 for count in target_counts.values() if count > 1),
        "target_exact_duplicate_occurrences": sum(count - 1 for count in target_counts.values() if count > 1),
    }
    return kept, report


def _expand_reverse_routes(
    samples: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    route_order = {route: index for index, route in enumerate(directed_routes())}
    for value in samples:
        forward = dict(value)
        reverse = dict(value)
        reverse["src_lang"], reverse["tgt_lang"] = forward["tgt_lang"], forward["src_lang"]
        reverse["source_text"], reverse["target_text"] = forward["target_text"], forward["source_text"]
        reverse["sample_id"] = _stable_sample_id(reverse)
        try:
            validate_parallel_sample(forward, config)
            validate_parallel_sample(reverse, config)
        except ContractError as exc:
            raise InputError(f"direction expansion violates TD-01 contract: {exc}") from exc
        result.extend((forward, reverse))
    if len({sample["sample_id"] for sample in result}) != len(result):
        raise LeakageError("forward/reverse expansion produced duplicate sample IDs")
    return sorted(
        result,
        key=lambda sample: (
            SPLIT_ORDER[str(sample["split"])],
            route_order[(str(sample["src_lang"]), str(sample["tgt_lang"]))],
            str(sample["sample_group_id"]),
            str(sample["sample_id"]),
        ),
    )


def audit_directed_samples(
    samples: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    *,
    check_near_duplicates: bool = True,
) -> dict[str, Any]:
    group_splits: dict[str, set[str]] = defaultdict(set)
    relations: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    exact_text_splits: dict[tuple[str, str], set[str]] = defaultdict(set)
    for sample in samples:
        try:
            validate_parallel_sample(sample, config)
        except ContractError as exc:
            raise LeakageError(f"directed output violates sample contract: {exc}") from exc
        group_id = str(sample["sample_group_id"])
        split = str(sample["split"])
        group_splits[group_id].add(split)
        endpoints = tuple(
            sorted(
                (
                    (str(sample["src_lang"]), str(sample["source_text"])),
                    (str(sample["tgt_lang"]), str(sample["target_text"])),
                )
            )
        )
        relations[(group_id, endpoints)].append(sample)
        exact_text_splits[(str(sample["src_lang"]), str(sample["source_text"]))].add(split)
        exact_text_splits[(str(sample["tgt_lang"]), str(sample["target_text"]))].add(split)
    crossing_groups = [group_id for group_id, splits in group_splits.items() if len(splits) > 1]
    if crossing_groups:
        raise LeakageError(f"group crosses train/dev/test: {crossing_groups[0]}")
    for relation, records in relations.items():
        directions = {(record["src_lang"], record["tgt_lang"]) for record in records}
        if len(records) != 2 or len(directions) != 2:
            raise LeakageError(f"forward/reverse relation is incomplete or duplicated: {relation[0]}")
        if len({record["split"] for record in records}) != 1:
            raise LeakageError(f"forward/reverse relation crosses splits: {relation[0]}")
    exact_crossings = [key for key, splits in exact_text_splits.items() if len(splits) > 1]
    if exact_crossings:
        raise LeakageError(f"exact normalized text crosses splits: {exact_crossings[0][0]}")
    near_crossings = 0
    if check_near_duplicates:
        for entries in _text_entries(samples, {group: next(iter(splits)) for group, splits in group_splits.items()}).values():
            for entry in entries:
                if len(entry.splits) > 1:
                    raise LeakageError(
                        f"near-normalized text crosses splits for language {entry.language}"
                    )
            for left, right, _score in find_near_duplicates(entries):
                if left.normalized == right.normalized:
                    continue
                if set(left.splits) != set(right.splits):
                    near_crossings += 1
                    raise LeakageError(
                        f"near-duplicate text crosses splits for language {left.language}"
                    )
    route_counts = Counter((str(sample["src_lang"]), str(sample["tgt_lang"])) for sample in samples)
    return {
        "groups": len(group_splits),
        "relations": len(relations),
        "near_crossings": near_crossings,
        "route_counts": {
            f"{source}->{target}": int(route_counts[(source, target)])
            for source, target in directed_routes()
        },
    }


def prepare_finalized_samples(
    samples: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    *,
    derived_links: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    validated_config = validate_model_data_config(config)
    ordered, groups = _validate_undirected_inputs(samples, validated_config, derived_links)
    group_splits, group_components, component_report = _assign_split_components(ordered, groups)
    deduplicated, exact_report = _deduplicate_pairs(ordered, group_splits)
    directed = _expand_reverse_routes(deduplicated, validated_config)
    audit = audit_directed_samples(directed, validated_config)
    by_split = {
        split: [sample for sample in directed if sample["split"] == split]
        for split in SPLIT_ORDER
    }
    test_groups = [
        {
            "sample_group_id": group_id,
            "split_component_id": group_components[group_id],
        }
        for group_id in sorted(group_splits)
        if group_splits[group_id] == "test"
    ]
    report = {
        "schema_version": 1,
        "status": "prepared",
        "pipeline_version": PIPELINE_VERSION,
        "split_profile": SPLIT_PROFILE,
        "split_profile_sha256": config_sha256(SPLIT_PROFILE),
        "dedup_profile": DEDUP_PROFILE,
        "dedup_profile_sha256": config_sha256(DEDUP_PROFILE),
        "input": {
            "undirected_samples": len(ordered),
            "sample_groups": len(groups),
        },
        "component_binding": component_report,
        "exact_deduplication": exact_report,
        "output": {
            "undirected_samples": len(deduplicated),
            "directed_samples": len(directed),
            "samples_by_split": {split: len(records) for split, records in by_split.items()},
            "test_groups": len(test_groups),
            "test_group_ids_sha256": sha256_bytes(
                b"".join(canonical_json_bytes(record) for record in test_groups)
            ),
        },
        "leakage_audit": audit,
    }
    return {
        "by_split": by_split,
        "test_groups": test_groups,
        "report": report,
        "candidate_entries": _text_entries(directed, group_splits),
    }


def _reference_candidate_index(
    entries_by_language: Mapping[str, Sequence[TextEntry]],
) -> tuple[
    dict[tuple[str, str], list[TextEntry]],
    dict[tuple[str, int, tuple[int, ...]], list[tuple[TextEntry, frozenset[str]]]],
]:
    exact: dict[tuple[str, str], list[TextEntry]] = defaultdict(list)
    bands: dict[
        tuple[str, int, tuple[int, ...]], list[tuple[TextEntry, frozenset[str]]]
    ] = defaultdict(list)
    for language, entries in entries_by_language.items():
        for entry in entries:
            exact[(language, entry.text)].append(entry)
            grams = _ngrams(entry.normalized)
            if len(grams) < int(DEDUP_PROFILE["minimum_ngrams"]):
                continue
            for band, values in _signature_bands(_minhash_signature(grams)):
                bands[(language, band, values)].append((entry, grams))
    return exact, bands


def scan_reference_records(
    entries_by_language: Mapping[str, Sequence[TextEntry]],
    reference_sets: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    exact_index, band_index = _reference_candidate_index(entries_by_language)
    set_reports: list[dict[str, Any]] = []
    total_blocking_hits = 0
    for reference_set in reference_sets:
        records = reference_set.get("records")
        if not isinstance(records, list):
            raise InputError(
                f"inline reference set {reference_set.get('reference_id')} records must be a list"
            )
        report = _scan_one_reference_set(
            entries_by_language,
            exact_index,
            band_index,
            str(reference_set["reference_id"]),
            str(reference_set["kind"]),
            str(reference_set["policy"]),
            iter(records),
        )
        if report["policy"] == "block":
            total_blocking_hits += int(report["hits"])
        set_reports.append(report)
    return {
        "reference_sets": set_reports,
        "blocking_hits": total_blocking_hits,
    }


def _scan_one_reference_set(
    entries_by_language: Mapping[str, Sequence[TextEntry]],
    exact_index: Mapping[tuple[str, str], Sequence[TextEntry]],
    band_index: Mapping[
        tuple[str, int, tuple[int, ...]], Sequence[tuple[TextEntry, frozenset[str]]]
    ],
    reference_id: str,
    kind: str,
    policy: str,
    records: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    if kind not in REFERENCE_KINDS or policy not in REFERENCE_POLICIES:
        raise InputError(f"invalid reference set: {reference_id}")
    if kind in {"mt_evaluation", "same_source_version"} and policy != "block":
        raise InputError(f"{kind} reference set must use policy=block")
    hits = 0
    record_count = 0
    reported: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        record_count += 1
        if not isinstance(record, dict) or not isinstance(record.get("language"), str) or not isinstance(record.get("text"), str):
            raise InputError(f"invalid reference record in {reference_id}")
        language = str(record["language"])
        text = str(record["text"])
        if language not in entries_by_language:
            raise InputError(
                f"reference set {reference_id} contains an unknown or uncovered language: {language}"
            )
        record_id = str(record.get("record_id", index))
        match = "exact" if exact_index.get((language, text)) else None
        score = 1.0 if match else 0.0
        matched_entries = list(exact_index.get((language, text), []))
        if match is None:
            normalized = _near_normalize(text)
            grams = _ngrams(normalized)
            candidates: dict[tuple[str, str], tuple[TextEntry, frozenset[str]]] = {}
            if len(grams) >= int(DEDUP_PROFILE["minimum_ngrams"]):
                language_entries = entries_by_language.get(language, ())
                if len(language_entries) <= int(DEDUP_PROFILE["exhaustive_text_limit"]):
                    for entry in language_entries:
                        candidates[(entry.language, entry.normalized)] = (
                            entry,
                            _ngrams(entry.normalized),
                        )
                else:
                    signature = _minhash_signature(grams)
                    for band, values in _signature_bands(signature):
                        for entry, candidate_grams in band_index.get((language, band, values), []):
                            candidates[(entry.language, entry.normalized)] = (
                                entry,
                                candidate_grams,
                            )
            near_matches = []
            for entry, candidate_grams in candidates.values():
                similarity = _jaccard(grams, candidate_grams)
                if similarity >= float(DEDUP_PROFILE["jaccard_threshold"]):
                    near_matches.append((similarity, entry))
            if near_matches:
                score, best = max(
                    near_matches, key=lambda item: (item[0], item[1].normalized)
                )
                matched_entries = [best]
                match = "near"
        if match is not None:
            hits += 1
            if len(reported) < int(DEDUP_PROFILE["max_reported_hits"]):
                reported.append(
                    {
                        "record_id": record_id,
                        "language": language,
                        "match": match,
                        "jaccard": round(score, 6),
                        "reference_text_sha256": sha256_bytes(text.encode("utf-8")),
                        "candidate_splits": sorted(
                            {split for entry in matched_entries for split in entry.splits},
                            key=SPLIT_ORDER.get,
                        ),
                    }
                )
    return {
        "reference_id": reference_id,
        "kind": kind,
        "policy": policy,
        "records": record_count,
        "hits": hits,
        "reported_hits": reported,
    }


def _manifest_file_specs(
    reference: Mapping[str, Any], repo_root: Path
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = reference["manifest"]
    path = repo_root / Path(str(manifest["path"]))
    if not path.is_file() or path.stat().st_size != int(manifest["bytes"]):
        raise InputError(f"reference manifest missing or wrong size: {manifest['path']}")
    if sha256_file(path) != manifest["sha256"]:
        raise InputError(f"reference manifest SHA-256 differs: {manifest['path']}")
    specs: list[dict[str, Any]] = []
    if manifest["format"] == "tokenizer-manifest-jsonl":
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            specs.append(
                {
                    "language": row["language"],
                    "path": path.parent / row["file"],
                    "relative_path": f"{PurePosixPath(manifest['path']).parent.as_posix()}/{row['file']}",
                    "format": "text-lines",
                    "bytes": int(row["bytes"]),
                    "sha256": row["sha256"],
                }
            )
    else:
        value = json.loads(path.read_text(encoding="utf-8"))
        for language, row in sorted(value["files"].items()):
            specs.append(
                {
                    "language": language,
                    "path": path.parent / row["path"],
                    "relative_path": f"{PurePosixPath(manifest['path']).parent.as_posix()}/{row['path']}",
                    "format": "jsonl-text-field",
                    "bytes": int(row["bytes"]),
                    "sha256": row["sha256"],
                }
            )
    identity = {
        "reference_id": reference["reference_id"],
        "kind": reference["kind"],
        "policy": reference["policy"],
        "manifest_sha256": manifest["sha256"],
        "files": [
            {
                "language": spec["language"],
                "relative_path": spec["relative_path"],
                "bytes": spec["bytes"],
                "sha256": spec["sha256"],
            }
            for spec in specs
        ],
    }
    return identity, specs


def _iter_verified_reference_file(spec: Mapping[str, Any]) -> Iterator[dict[str, str]]:
    path = Path(spec["path"])
    digest = hashlib.sha256()
    byte_count = 0
    try:
        with path.open("rb") as handle:
            for line_number, raw_line in enumerate(handle, 1):
                digest.update(raw_line)
                byte_count += len(raw_line)
                decoded = raw_line.decode("utf-8", errors="strict").rstrip("\r\n")
                if not decoded:
                    continue
                if spec["format"] == "text-lines":
                    text = decoded
                else:
                    row = json.loads(decoded)
                    text = row.get("text")
                    if not isinstance(text, str):
                        raise InputError(f"reference JSONL row lacks text: {path}:{line_number}")
                yield {
                    "language": str(spec["language"]),
                    "text": text,
                    "record_id": f"{spec['relative_path']}:{line_number}",
                }
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise InputError(f"cannot scan reference file {path}: {exc}") from exc
    if byte_count != int(spec["bytes"]) or digest.hexdigest() != spec["sha256"]:
        raise InputError(f"reference file byte identity differs: {spec['relative_path']}")


def scan_registry_references(
    entries_by_language: Mapping[str, Sequence[TextEntry]],
    registry: Mapping[str, Any],
    repo_root: Path,
) -> dict[str, Any]:
    validated = validate_contamination_registry(registry)
    identities: list[dict[str, Any]] = []
    exact_index, band_index = _reference_candidate_index(entries_by_language)
    set_reports: list[dict[str, Any]] = []
    total_blocking_hits = 0
    for reference in validated["reference_sets"]:
        identity, specs = _manifest_file_specs(reference, repo_root)
        identities.append(identity)
        def records() -> Iterator[dict[str, str]]:
            for spec in specs:
                yield from _iter_verified_reference_file(spec)

        set_report = _scan_one_reference_set(
            entries_by_language,
            exact_index,
            band_index,
            str(reference["reference_id"]),
            str(reference["kind"]),
            str(reference["policy"]),
            records(),
        )
        set_reports.append(set_report)
        if set_report["policy"] == "block":
            total_blocking_hits += int(set_report["hits"])
    report = {
        "reference_sets": set_reports,
        "blocking_hits": total_blocking_hits,
    }
    report["registry_sha256"] = config_sha256(validated)
    report["registry_complete_for_m0"] = registry_is_complete(validated)
    report["reference_identities"] = identities
    return report


def load_td03_samples(
    input_root: Path, config: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest_path = input_root / "corpus" / "mvp" / "manifest.json"
    try:
        manifest_bytes = manifest_path.read_bytes()
        manifest = json.loads(manifest_bytes.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise InputError(f"cannot load TD-03 manifest: {exc}") from exc
    if manifest.get("status") != "complete" or manifest.get("canonical_sample_scope") != (
        "nine undirected pairs; reverse route expansion is TD-04"
    ):
        raise InputError("TD-03 manifest is incomplete or has the wrong sample scope")
    records = [record for record in manifest.get("files", []) if record.get("path") == "corpus/mvp/human_parallel.jsonl"]
    if len(records) != 1:
        raise InputError("TD-03 manifest must contain one human_parallel.jsonl")
    record = records[0]
    corpus_path = input_root / Path(record["path"])
    if (
        not corpus_path.is_file()
        or corpus_path.stat().st_size != int(record["bytes"])
        or sha256_file(corpus_path) != record["sha256"]
    ):
        raise InputError("TD-03 human_parallel.jsonl differs from its manifest")
    try:
        samples = [json.loads(line) for line in corpus_path.read_text(encoding="utf-8").splitlines()]
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise InputError(f"cannot parse TD-03 corpus: {exc}") from exc
    if len(samples) != int(record["records"]):
        raise InputError("TD-03 corpus record count differs from its manifest")
    identity = {
        "manifest_sha256": sha256_bytes(manifest_bytes),
        "manifest_identity_sha256": manifest["identity_sha256"],
        "corpus_sha256": record["sha256"],
        "records": len(samples),
    }
    return samples, identity


def _output_record(relative_path: str, data: bytes, records: int | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": relative_path,
        "bytes": len(data),
        "sha256": sha256_bytes(data),
    }
    if records is not None:
        result["records"] = records
    return result


def publish_finalized_data(
    prepared: Mapping[str, Any],
    input_identity: Mapping[str, Any],
    reference_scan: Mapping[str, Any],
    out_root: Path,
    *,
    require_complete_references: bool,
) -> dict[str, Any]:
    report = dict(prepared["report"])
    report["reference_scan"] = dict(reference_scan)
    report["status"] = "blocked" if int(reference_scan["blocking_hits"]) else "complete"
    blocked_path = out_root / "reports" / "td04-contamination-blocked.json"
    manifest_path = out_root / "corpus" / "mvp" / "finalized" / "manifest.json"
    if require_complete_references and not reference_scan.get("registry_complete_for_m0", False):
        report["status"] = "blocked"
        report["blocking_reason"] = "reference registry is incomplete for M0"
        atomic_write_bytes(blocked_path, canonical_json_bytes(report))
        manifest_path.unlink(missing_ok=True)
        raise ContaminationError("reference registry is incomplete for formal M0 publication")
    if int(reference_scan["blocking_hits"]):
        report["blocking_reason"] = "blocking external contamination hits"
        atomic_write_bytes(blocked_path, canonical_json_bytes(report))
        manifest_path.unlink(missing_ok=True)
        raise ContaminationError(
            f"blocking external contamination hits: {reference_scan['blocking_hits']}"
        )

    split_bytes = {
        split: b"".join(canonical_json_bytes(sample) for sample in prepared["by_split"][split])
        for split in SPLIT_ORDER
    }
    test_groups_bytes = b"".join(
        canonical_json_bytes(record) for record in prepared["test_groups"]
    )
    report_bytes = canonical_json_bytes(report)
    files: list[tuple[str, bytes, int | None]] = [
        (f"corpus/mvp/finalized/{split}.jsonl", split_bytes[split], len(prepared["by_split"][split]))
        for split in SPLIT_ORDER
    ]
    files.extend(
        [
            ("corpus/mvp/finalized/test-groups.jsonl", test_groups_bytes, len(prepared["test_groups"])),
            ("reports/td04-dedup-leakage.json", report_bytes, None),
        ]
    )
    file_records = [_output_record(path, data, records) for path, data, records in files]
    identity = {
        "pipeline_version": PIPELINE_VERSION,
        "input": dict(input_identity),
        "split_profile_sha256": config_sha256(SPLIT_PROFILE),
        "dedup_profile_sha256": config_sha256(DEDUP_PROFILE),
        "reference_registry_sha256": reference_scan.get("registry_sha256"),
        "reference_identities": reference_scan.get("reference_identities", []),
    }
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "identity": identity,
        "identity_sha256": config_sha256(identity),
        "scope": "18 directed routes after group split and deduplication",
        "records": sum(len(records) for records in prepared["by_split"].values()),
        "files": file_records,
    }
    manifest_path.unlink(missing_ok=True)
    blocked_path.unlink(missing_ok=True)
    try:
        for relative_path, data, _records in files:
            atomic_write_bytes(out_root / Path(relative_path), data)
        atomic_write_bytes(manifest_path, canonical_json_bytes(manifest))
    except BaseException:
        manifest_path.unlink(missing_ok=True)
        raise
    return {
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "identity_sha256": manifest["identity_sha256"],
        "records": manifest["records"],
        "samples_by_split": report["output"]["samples_by_split"],
    }


def dry_run_plan(
    config: Mapping[str, Any],
    registry: Mapping[str, Any],
    input_root: Path,
    out_root: Path,
) -> dict[str, Any]:
    validated_config = validate_model_data_config(config)
    validated_registry = validate_contamination_registry(registry)
    samples, input_identity = load_td03_samples(input_root, validated_config)
    return {
        "status": "dry-run",
        "pipeline_version": PIPELINE_VERSION,
        "input": input_identity,
        "input_records": len(samples),
        "split_profile": SPLIT_PROFILE,
        "split_profile_sha256": config_sha256(SPLIT_PROFILE),
        "dedup_profile_sha256": config_sha256(DEDUP_PROFILE),
        "reference_registry_sha256": config_sha256(validated_registry),
        "reference_registry_complete_for_m0": registry_is_complete(validated_registry),
        "reference_sets": [
            {
                "reference_id": reference["reference_id"],
                "kind": reference["kind"],
                "policy": reference["policy"],
                "manifest_sha256": reference["manifest"]["sha256"],
            }
            for reference in validated_registry["reference_sets"]
        ],
        "outputs": [
            "corpus/mvp/finalized/train.jsonl",
            "corpus/mvp/finalized/dev.jsonl",
            "corpus/mvp/finalized/test.jsonl",
            "corpus/mvp/finalized/test-groups.jsonl",
            "reports/td04-dedup-leakage.json",
            "corpus/mvp/finalized/manifest.json (published last)",
        ],
        "runtime_roots": {"input": str(input_root), "out": str(out_root)},
    }
