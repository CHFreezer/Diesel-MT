#!/usr/bin/env python3
"""TD-05 M0 dataset coverage, quality, review, and reproducibility acceptance."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from model_data_pipeline import (
    LocaleRecord,
    atomic_write_bytes,
    pair_rejection_reason,
    script_counts,
    wrong_script_dominates,
)
from model_data_split_pipeline import sha256_file
from model_training_contract import (
    LANGUAGE_TAGS,
    UNDIRECTED_PAIRS,
    canonical_json_bytes,
    config_sha256,
    directed_routes,
    pair_id,
    validate_model_data_config,
    validate_parallel_sample,
)


PIPELINE_VERSION = "td05-m0-acceptance-v1"
REVIEW_SEED = "diesel-mt-m0-manual-review-v1"
SPLIT_ORDER = ("train", "dev", "test")
REPRO_PATHS = (
    "corpus/mvp/human_parallel.jsonl",
    "corpus/mvp/manifest.json",
    "reports/td03-build.json",
    "reports/td03-rejections.json",
    "corpus/mvp/finalized/train.jsonl",
    "corpus/mvp/finalized/dev.jsonl",
    "corpus/mvp/finalized/test.jsonl",
    "corpus/mvp/finalized/test-groups.jsonl",
    "corpus/mvp/finalized/manifest.json",
    "reports/td04-dedup-leakage.json",
)


class M0AcceptanceError(RuntimeError):
    """M0 cannot be published because an acceptance gate failed."""


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise M0AcceptanceError(f"cannot load JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise M0AcceptanceError(f"JSON root must be an object: {path}")
    return value


def _quantiles(values: Sequence[float | int]) -> dict[str, float | int]:
    if not values:
        raise M0AcceptanceError("cannot summarize an empty numeric series")
    ordered = sorted(values)

    def nearest(probability: float) -> float | int:
        index = max(0, math.ceil(probability * len(ordered)) - 1)
        return ordered[index]

    return {
        "min": ordered[0],
        "p50": nearest(0.50),
        "p95": nearest(0.95),
        "p99": nearest(0.99),
        "max": ordered[-1],
    }


def _character_length(text: str) -> int:
    return len(text.replace(" ", ""))


def _length_ratio(source: str, target: str) -> float:
    left = _character_length(source)
    right = _character_length(target)
    return max(left, right) / max(min(left, right), 4)


def _mixed_script(text: str) -> bool:
    counts = script_counts(text)
    return sum(1 for value in counts.values() if value > 0) >= 2


def load_sampling_config(path: Path) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise M0AcceptanceError(f"cannot load direction sampling config {path}: {exc}") from exc
    if not isinstance(value, dict) or set(value) != {
        "schema_version",
        "identity",
        "strategy",
        "routes",
    }:
        raise M0AcceptanceError("direction sampling config fields are incomplete")
    if value["schema_version"] != 1 or value["identity"] != {
        "name": "mvp_direction_sampling",
        "status": "m0-locked",
    }:
        raise M0AcceptanceError("direction sampling identity changed")
    strategy = value["strategy"]
    if not isinstance(strategy, dict) or strategy != {
        "unit": "directed_sample",
        "epoch": "one deterministic pass over the frozen train split",
        "shuffle": "seeded sampler owned by the later training task",
        "default_weight": 1.0,
        "maximum_repeats_per_epoch": 1,
        "low_resource_oversampling": "prohibited",
    }:
        raise M0AcceptanceError("direction sampling strategy changed")
    records = value["routes"]
    if not isinstance(records, list) or len(records) != 18:
        raise M0AcceptanceError("direction sampling must contain exactly 18 routes")
    expected = {f"{source}->{target}" for source, target in directed_routes()}
    actual: set[str] = set()
    for record in records:
        if not isinstance(record, dict) or set(record) != {
            "route",
            "weight",
            "maximum_repeats_per_epoch",
        }:
            raise M0AcceptanceError("invalid direction sampling route record")
        if record["weight"] != 1.0 or record["maximum_repeats_per_epoch"] != 1:
            raise M0AcceptanceError("M0 direction sampling cannot repeat or reweight routes")
        actual.add(str(record["route"]))
    if actual != expected:
        raise M0AcceptanceError("direction sampling route coverage differs from the 18-route contract")
    return value


def _load_finalized_records(
    root: Path, config: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest_path = root / "corpus" / "mvp" / "finalized" / "manifest.json"
    manifest = _read_json(manifest_path)
    if manifest.get("status") != "complete" or manifest.get("records", 0) <= 0:
        raise M0AcceptanceError("TD-04 finalized manifest is incomplete")
    records: list[dict[str, Any]] = []
    for split in SPLIT_ORDER:
        relative = f"corpus/mvp/finalized/{split}.jsonl"
        specs = [record for record in manifest["files"] if record["path"] == relative]
        if len(specs) != 1:
            raise M0AcceptanceError(f"finalized manifest lacks {relative}")
        spec = specs[0]
        path = root / relative
        if (
            not path.is_file()
            or path.stat().st_size != int(spec["bytes"])
            or sha256_file(path) != spec["sha256"]
        ):
            raise M0AcceptanceError(f"finalized corpus identity differs: {relative}")
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            try:
                sample = json.loads(line)
                validate_parallel_sample(sample, config)
            except (json.JSONDecodeError, ValueError) as exc:
                raise M0AcceptanceError(f"invalid finalized sample {relative}:{line_number}: {exc}") from exc
            if sample["split"] != split:
                raise M0AcceptanceError(f"split field differs from file: {relative}:{line_number}")
            records.append(sample)
    if len(records) != int(manifest["records"]):
        raise M0AcceptanceError("finalized record count differs from manifest")
    return records, manifest


def analyze_dataset(
    root: Path,
    config: Mapping[str, Any],
    sampling: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[tuple[str, str], list[dict[str, Any]]]]:
    validated = validate_model_data_config(config)
    records, _final_manifest = _load_finalized_records(root, validated)
    canonical_pairs = {frozenset(pair): pair for pair in UNDIRECTED_PAIRS}
    route_counts: Counter[tuple[str, str, str]] = Counter()
    language_counts: Counter[str] = Counter()
    canonical: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    stats: dict[tuple[str, str], dict[str, Any]] = defaultdict(
        lambda: {
            "source_lengths": [],
            "target_lengths": [],
            "ratios": [],
            "script_compliant": 0,
            "mixed_script": 0,
            "source_ids": Counter(),
            "provenance": Counter(),
        }
    )
    teacher_fields = 0
    source_identities: set[tuple[str, str, str]] = set()
    provenance_kinds: set[str] = set()
    for sample in records:
        source = str(sample["src_lang"])
        target = str(sample["tgt_lang"])
        split = str(sample["split"])
        route_counts[(split, source, target)] += 1
        source_identities.add(
            (
                str(sample["source_id"]),
                str(sample["source_version"]),
                str(sample["license"]),
            )
        )
        provenance_kinds.add(str(sample.get("provenance", {}).get("kind", "unspecified")))
        language_counts[source] += 1
        language_counts[target] += 1
        pair = canonical_pairs[frozenset((source, target))]
        if (source, target) != pair:
            continue
        pair_name = pair_id(pair)
        canonical[(pair_name, split)].append(sample)
        bucket = stats[(pair_name, split)]
        bucket["source_lengths"].append(_character_length(str(sample["source_text"])))
        bucket["target_lengths"].append(_character_length(str(sample["target_text"])))
        bucket["ratios"].append(_length_ratio(str(sample["source_text"]), str(sample["target_text"])))
        compliant = not wrong_script_dominates(str(sample["source_text"]), source) and not wrong_script_dominates(
            str(sample["target_text"]), target
        )
        bucket["script_compliant"] += int(compliant)
        bucket["mixed_script"] += int(
            _mixed_script(str(sample["source_text"])) or _mixed_script(str(sample["target_text"]))
        )
        bucket["source_ids"][str(sample["source_id"])] += 1
        kind = str(sample.get("provenance", {}).get("kind", "unspecified"))
        bucket["provenance"][kind] += 1
        serialized_sample = json.dumps(sample, ensure_ascii=False)
        teacher_fields += sum(
            field in serialized_sample
            for field in ("teacher_model", "teacher_revision", "decode_config_sha256")
        )
    expected_routes = set(directed_routes())
    missing_route_splits = [
        f"{split}:{source}->{target}"
        for split in SPLIT_ORDER
        for source, target in directed_routes()
        if route_counts[(split, source, target)] <= 0
    ]
    if missing_route_splits:
        raise M0AcceptanceError(f"empty route/split: {missing_route_splits[0]}")
    if set(language_counts) != set(LANGUAGE_TAGS):
        raise M0AcceptanceError("finalized corpus does not cover all five model tags")
    if source_identities != {("massive-1.1", "1.1", "CC-BY-4.0")}:
        raise M0AcceptanceError("finalized corpus contains an unknown source/version/license identity")
    if provenance_kinds != {"human_parallel"} or teacher_fields:
        raise M0AcceptanceError("M0 must contain only human_parallel provenance and no teacher fields")
    pair_report: dict[str, Any] = {}
    td03_build = _read_json(root / "reports" / "td03-build.json")
    td03_pairs = td03_build["source_reports"][0]["pairs"]
    minimums = validated["budgets"]["minimum_accepted_per_undirected_pair"]
    for pair in UNDIRECTED_PAIRS:
        name = pair_id(pair)
        split_report: dict[str, Any] = {}
        for split in SPLIT_ORDER:
            bucket = stats[(name, split)]
            count = len(canonical[(name, split)])
            if count < int(minimums[split]):
                raise M0AcceptanceError(f"{name}/{split} fell below the frozen minimum")
            if bucket["script_compliant"] != count:
                raise M0AcceptanceError(f"{name}/{split} contains script-noncompliant output")
            split_report[split] = {
                "records": count,
                "source_distribution": dict(sorted(bucket["source_ids"].items())),
                "provenance_distribution": dict(sorted(bucket["provenance"].items())),
                "length_characters": {
                    "source": _quantiles(bucket["source_lengths"]),
                    "target": _quantiles(bucket["target_lengths"]),
                },
                "length_ratio": {
                    key: round(float(value), 6)
                    for key, value in _quantiles(bucket["ratios"]).items()
                },
                "script_compliant": count,
                "script_compliance_rate": 1.0,
                "mixed_script_records": int(bucket["mixed_script"]),
            }
        raw = td03_pairs[name]
        final_total = sum(split_report[split]["records"] for split in SPLIT_ORDER)
        pair_report[name] = {
            "source_id": "massive-1.1",
            "source_version": "1.1",
            "license": "CC-BY-4.0",
            "source_type": "human_parallel",
            "raw_scanned": int(raw["scanned"]),
            "td03_accepted": int(raw["accepted"]),
            "td03_rejected": int(raw["rejected"]),
            "td04_exact_duplicates_removed": int(raw["accepted"]) - final_total,
            "final_undirected": final_total,
            "final_directed": final_total * 2,
            "splits": split_report,
        }
    sampling_by_route = {record["route"]: record for record in sampling["routes"]}
    direction_report: dict[str, Any] = {}
    for source, target in directed_routes():
        route = f"{source}->{target}"
        by_split = {split: int(route_counts[(split, source, target)]) for split in SPLIT_ORDER}
        direction_report[route] = {
            "records_by_split": by_split,
            "records_total": sum(by_split.values()),
            "train_weight": sampling_by_route[route]["weight"],
            "maximum_repeats_per_epoch": sampling_by_route[route]["maximum_repeats_per_epoch"],
            "effective_train_exposure_per_epoch": by_split["train"],
        }
    contamination = _read_json(root / "reports" / "td04-dedup-leakage.json")["reference_scan"]
    formal = [record for record in contamination["reference_sets"] if record["kind"] == "mt_evaluation"]
    if len(formal) != 1 or formal[0]["hits"] != 0 or contamination["blocking_hits"] != 0:
        raise M0AcceptanceError("formal MT evaluation contamination gate failed")
    tokenizer_overlap = sum(
        int(record["hits"])
        for record in contamination["reference_sets"]
        if str(record["kind"]).startswith("tokenizer_")
    )
    return (
        {
            "finalized_manifest_sha256": sha256_file(
                root / "corpus" / "mvp" / "finalized" / "manifest.json"
            ),
            "directed_records": len(records),
            "undirected_records": len(records) // 2,
            "languages": {language: int(language_counts[language]) for language in LANGUAGE_TAGS},
            "routes": direction_report,
            "pairs": pair_report,
            "provenance_totals": {
                "human_parallel": sum(
                    count
                    for bucket in stats.values()
                    for kind, count in bucket["provenance"].items()
                    if kind == "human_parallel"
                ),
                "teacher_synthetic": 0,
                "script_conversion": 0,
            },
            "teacher_fields_found": teacher_fields,
            "contamination": {
                "formal_mt_evaluation": formal[0],
                "tokenizer_reference_hits_report_only": tokenizer_overlap,
                "registry_sha256": contamination["registry_sha256"],
            },
            "contract": {
                "model_tags": 5,
                "undirected_pairs": 9,
                "directed_routes": len(expected_routes),
                "zho_Hans_zho_Hant_routes": 0,
            },
        },
        canonical,
    )


def _stable_review_key(record: Mapping[str, Any]) -> str:
    identity = {
        "seed": REVIEW_SEED,
        "category": record["category"],
        "pair_id": record["pair_id"],
        "split": record["split"],
        "identity": record.get("sample_id", record.get("alignment_key")),
        "reason": record.get("expected_rejection_reason"),
    }
    return hashlib.sha256(canonical_json_bytes(identity)).hexdigest()


def _review_record(sample: Mapping[str, Any], pair_name: str) -> dict[str, Any]:
    record = {
        "category": "accepted",
        "expected_rejection_reason": None,
        "pair_id": pair_name,
        "sample_id": sample["sample_id"],
        "source_lang": sample["src_lang"],
        "source_text": sample["source_text"],
        "split": sample["split"],
        "target_lang": sample["tgt_lang"],
        "target_text": sample["target_text"],
    }
    record["review_id"] = _stable_review_key(record)
    return record


def _select_accepted(records: Sequence[dict[str, Any]], pair_name: str, quota: int) -> list[dict[str, Any]]:
    candidates = [_review_record(record, pair_name) for record in records]
    selected: list[dict[str, Any]] = []

    def add(record: dict[str, Any], tag: str) -> None:
        if record["review_id"] not in {item["review_id"] for item in selected}:
            selected.append({**record, "selection_tag": tag})

    add(min(candidates, key=lambda row: _character_length(row["source_text"]) + _character_length(row["target_text"])), "short-boundary")
    add(max(candidates, key=lambda row: _character_length(row["source_text"]) + _character_length(row["target_text"])), "long-boundary")
    add(max(candidates, key=lambda row: _length_ratio(row["source_text"], row["target_text"])), "ratio-boundary")
    if "zho_Hant" in pair_name:
        mixed = [
            row
            for row in candidates
            if _mixed_script(row["source_text"]) or _mixed_script(row["target_text"])
        ]
        if mixed:
            add(min(mixed, key=lambda row: row["review_id"]), "traditional-mixed-script")
    for record in sorted(candidates, key=lambda row: row["review_id"]):
        if len(selected) >= quota:
            break
        add(record, "stable-random")
    return selected[:quota]


def _load_locale_checkpoints(root: Path) -> dict[str, dict[str, LocaleRecord]]:
    candidates = list((root / "interim" / "td03").glob("*/massive-1.1"))
    valid = [path for path in candidates if all((path / f"{language}.jsonl").is_file() for language in LANGUAGE_TAGS)]
    if len(valid) != 1:
        raise M0AcceptanceError("expected one complete TD-03 checkpoint identity for manual rejection review")
    result: dict[str, dict[str, LocaleRecord]] = {}
    for language in LANGUAGE_TAGS:
        records: dict[str, LocaleRecord] = {}
        path = valid[0] / f"{language}.jsonl"
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            records[row["alignment_key"]] = LocaleRecord(
                alignment_key=row["alignment_key"],
                split=row["split"],
                source_record_id=row["source_record_id"],
                text=row["text"],
                rejection_reason=row["rejection_reason"],
            )
        result[language] = records
    return result


def _rejected_review_candidates(
    root: Path, config: Mapping[str, Any]
) -> dict[str, list[dict[str, Any]]]:
    locales = _load_locale_checkpoints(root)
    reference = locales[LANGUAGE_TAGS[0]]
    result: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for alignment_key in sorted(reference):
        for source, target in UNDIRECTED_PAIRS:
            left = locales[source][alignment_key]
            right = locales[target][alignment_key]
            reason = (
                f"{source}:{left.rejection_reason}"
                if left.rejection_reason
                else f"{target}:{right.rejection_reason}"
                if right.rejection_reason
                else pair_rejection_reason(left.text, right.text)
            )
            if not reason:
                continue
            name = pair_id((source, target))
            record = {
                "alignment_key": alignment_key,
                "category": "rejected",
                "expected_rejection_reason": reason,
                "pair_id": name,
                "source_lang": source,
                "source_text": left.text,
                "split": left.split,
                "target_lang": target,
                "target_text": right.text,
            }
            record["review_id"] = _stable_review_key(record)
            result[name].append(record)
    return result


def _select_rejected(records: Sequence[dict[str, Any]], quota: int = 20) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    by_reason: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_reason[str(record["expected_rejection_reason"])].append(record)
    for reason in sorted(by_reason):
        selected.append({**min(by_reason[reason], key=lambda row: row["review_id"]), "selection_tag": "reason-stratum"})
    selected_ids = {record["review_id"] for record in selected}
    for record in sorted(records, key=lambda row: row["review_id"]):
        if len(selected) >= min(quota, len(records)):
            break
        if record["review_id"] not in selected_ids:
            selected.append({**record, "selection_tag": "stable-random"})
            selected_ids.add(record["review_id"])
    return selected[: min(quota, len(records))]


def build_review_queue(
    root: Path,
    config: Mapping[str, Any],
    canonical: Mapping[tuple[str, str], Sequence[dict[str, Any]]],
) -> tuple[bytes, dict[str, Any]]:
    rejected = _rejected_review_candidates(root, config)
    queue: list[dict[str, Any]] = []
    counts: dict[str, Any] = {}
    for pair in UNDIRECTED_PAIRS:
        name = pair_id(pair)
        pair_counts = {"accepted_train": 0, "accepted_dev": 0, "accepted_test": 0, "rejected": 0}
        for split, quota in (("train", 20), ("dev", 10), ("test", 10)):
            selected = _select_accepted(canonical[(name, split)], name, quota)
            queue.extend(selected)
            pair_counts[f"accepted_{split}"] = len(selected)
        rejected_selected = _select_rejected(rejected[name])
        queue.extend(rejected_selected)
        pair_counts["rejected"] = len(rejected_selected)
        pair_counts["rejected_available"] = len(rejected[name])
        counts[name] = pair_counts
    queue = sorted(
        queue,
        key=lambda row: (
            [pair_id(pair) for pair in UNDIRECTED_PAIRS].index(row["pair_id"]),
            0 if row["category"] == "accepted" else 1,
            SPLIT_ORDER.index(row["split"]),
            row["review_id"],
        ),
    )
    data = b"".join(canonical_json_bytes(record) for record in queue)
    summary = {
        "records": len(queue),
        "accepted": sum(1 for record in queue if record["category"] == "accepted"),
        "rejected": sum(1 for record in queue if record["category"] == "rejected"),
        "traditional_mixed_script": sum(
            1 for record in queue if record["selection_tag"] == "traditional-mixed-script"
        ),
        "counts_by_pair": counts,
        "selection_seed": REVIEW_SEED,
        "sha256": hashlib.sha256(data).hexdigest(),
        "bytes": len(data),
    }
    return data, summary


def compare_builds(first: Path, second: Path) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    identical = True
    for relative in REPRO_PATHS:
        left = first / relative
        right = second / relative
        if not left.is_file() or not right.is_file():
            raise M0AcceptanceError(f"reproducibility input missing: {relative}")
        left_identity = {"bytes": left.stat().st_size, "sha256": sha256_file(left)}
        right_identity = {"bytes": right.stat().st_size, "sha256": sha256_file(right)}
        same = left_identity == right_identity
        identical &= same
        files.append({"path": relative, "first": left_identity, "second": right_identity, "identical": same})
    if not identical:
        differing = next(record["path"] for record in files if not record["identical"])
        raise M0AcceptanceError(f"independent builds differ: {differing}")
    return {
        "status": "pass",
        "identical": True,
        "first_conditions": "cold network source acquisition; fresh output; serial deterministic adapter",
        "second_conditions": "validated hot cache; network prohibited; offline fresh build followed by resume verification",
        "worker_note": "TD-03/TD-04 model-data code is deliberately serial and has no worker-count-dependent path",
        "files": files,
    }


def load_review_attestation(
    path: Path,
    queue: Mapping[str, Any],
    review_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise M0AcceptanceError(f"cannot load manual review attestation {path}: {exc}") from exc
    if not isinstance(value, dict) or set(value) != {
        "schema_version",
        "status",
        "queue",
        "reviewer",
        "method",
        "decisions",
        "notes",
    }:
        raise M0AcceptanceError("manual review attestation fields are incomplete")
    if value["schema_version"] != 1 or value["status"] != "complete":
        raise M0AcceptanceError("manual review is not complete")
    if value["queue"] != {
        "bytes": queue["bytes"],
        "records": queue["records"],
        "sha256": queue["sha256"],
    }:
        raise M0AcceptanceError("manual review attestation does not bind the current queue")
    decisions = value["decisions"]
    if not isinstance(decisions, dict) or set(decisions) != {
        "accepted_reviewed",
        "accepted_quality_flags",
        "rejected_reviewed",
        "rejected_rule_mismatches",
        "systemic_blocker",
    }:
        raise M0AcceptanceError("manual review decisions are incomplete")
    if (
        decisions["accepted_reviewed"] != queue["accepted"]
        or decisions["rejected_reviewed"] != queue["rejected"]
        or decisions["systemic_blocker"] is not False
    ):
        raise M0AcceptanceError("manual review counts are incomplete or a systemic blocker remains")
    accepted_ids = {
        str(record["review_id"])
        for record in review_records
        if record["category"] == "accepted"
    }
    flags = decisions["accepted_quality_flags"]
    if not isinstance(flags, list) or any(
        not isinstance(flag, dict)
        or set(flag) != {"review_id", "issue"}
        or flag["review_id"] not in accepted_ids
        or not isinstance(flag["issue"], str)
        or not flag["issue"]
        for flag in flags
    ):
        raise M0AcceptanceError("manual review accepted quality flags are invalid")
    if len({flag["review_id"] for flag in flags}) != len(flags):
        raise M0AcceptanceError("manual review accepted quality flags are duplicated")
    if decisions["rejected_rule_mismatches"] != []:
        raise M0AcceptanceError("manual review found a rejected-row rule mismatch")
    if not isinstance(value["reviewer"], str) or not value["reviewer"]:
        raise M0AcceptanceError("manual review reviewer must be named")
    if not isinstance(value["notes"], list) or not value["notes"]:
        raise M0AcceptanceError("manual review must record limitations/observations")
    return value


def prepare_review(
    root: Path,
    config: Mapping[str, Any],
    sampling: Mapping[str, Any],
) -> dict[str, Any]:
    _analysis, canonical = analyze_dataset(root, config, sampling)
    queue_bytes, queue = build_review_queue(root, config, canonical)
    path = root / "reports" / "td05-manual-review.jsonl"
    atomic_write_bytes(path, queue_bytes)
    return {**queue, "path": path.as_posix(), "status": "review-required"}


def accept_m0(
    root: Path,
    rebuild_root: Path,
    config: Mapping[str, Any],
    sampling: Mapping[str, Any],
    attestation_path: Path,
) -> dict[str, Any]:
    manifest_path = root / "corpus" / "mvp" / "m0-manifest.json"
    manifest_path.unlink(missing_ok=True)
    analysis, canonical = analyze_dataset(root, config, sampling)
    queue_bytes, queue = build_review_queue(root, config, canonical)
    queue_path = root / "reports" / "td05-manual-review.jsonl"
    atomic_write_bytes(queue_path, queue_bytes)
    review_records = [json.loads(line) for line in queue_bytes.decode("utf-8").splitlines()]
    attestation = load_review_attestation(attestation_path, queue, review_records)
    reproducibility = compare_builds(root, rebuild_root)
    report = {
        "schema_version": 1,
        "status": "complete",
        "pipeline_version": PIPELINE_VERSION,
        "dataset": analysis,
        "sampling": {
            "config_sha256": config_sha256(sampling),
            "strategy": sampling["strategy"],
        },
        "manual_review": {
            **queue,
            "path": "reports/td05-manual-review.jsonl",
            "attestation_sha256": sha256_file(attestation_path),
            "reviewer": attestation["reviewer"],
            "method": attestation["method"],
            "notes": attestation["notes"],
            "accepted_quality_flags": attestation["decisions"]["accepted_quality_flags"],
            "accepted_quality_flag_count": len(
                attestation["decisions"]["accepted_quality_flags"]
            ),
            "rejected_rule_mismatches": [],
            "systemic_blocker": False,
            "status": "pass-with-known-source-quality-warnings",
        },
        "reproducibility": reproducibility,
        "release_gates": {
            "five_model_tags": "pass",
            "nine_undirected_pairs": "pass",
            "eighteen_directed_routes": "pass",
            "independent_zho_hans_dev_test": "pass",
            "independent_zho_hant_dev_test": "pass",
            "source_and_license_traceability": "pass",
            "script_compliance": "pass",
            "formal_evaluation_contamination": "pass",
            "manual_review": "pass-with-known-source-quality-warnings",
            "byte_reproducibility": "pass",
            "teacher_targets_absent": "pass",
        },
        "consumers": ["TD-07", "TD-09", "TD-12", "TD-13", "TD-14", "TD-15", "TD-16"],
    }
    report_bytes = canonical_json_bytes(report)
    report_path = root / "reports" / "td05-m0-acceptance.json"
    atomic_write_bytes(report_path, report_bytes)
    final_manifest_path = root / "corpus" / "mvp" / "finalized" / "manifest.json"
    m0_manifest = {
        "schema_version": 1,
        "status": "complete",
        "release": "m0-model-training-data",
        "pipeline_version": PIPELINE_VERSION,
        "finalized_manifest": {
            "path": "corpus/mvp/finalized/manifest.json",
            "bytes": final_manifest_path.stat().st_size,
            "sha256": sha256_file(final_manifest_path),
        },
        "acceptance_report": {
            "path": "reports/td05-m0-acceptance.json",
            "bytes": len(report_bytes),
            "sha256": hashlib.sha256(report_bytes).hexdigest(),
        },
        "manual_review": {
            "path": "reports/td05-manual-review.jsonl",
            "bytes": len(queue_bytes),
            "records": queue["records"],
            "sha256": queue["sha256"],
        },
        "consumers": report["consumers"],
    }
    atomic_write_bytes(manifest_path, canonical_json_bytes(m0_manifest))
    return {
        "status": "complete",
        "records": analysis["directed_records"],
        "manifest": manifest_path.as_posix(),
        "manifest_sha256": sha256_file(manifest_path),
        "report": report_path.as_posix(),
        "report_sha256": sha256_file(report_path),
        "review_records": queue["records"],
    }
