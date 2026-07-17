"""Publish the TD-05 quality-actual 80/20 teacher/human training corpus."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterable, Mapping, Sequence

import yaml

from model_data_source_contract import canonical_sha256
from model_training_contract import directed_routes, load_student_config
from mvp_60m_data_pipeline import (
    AbilityDataError,
    MODEL_TAGS,
    canonical_json_bytes,
    normalized_identity,
    sha256_bytes,
    sha256_file,
    write_json,
    write_jsonl,
)
from mvp_student import load_frozen_tokenizer


ROUTES = tuple(f"{source}->{target}" for source, target in directed_routes())


def load_config(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise AbilityDataError("TD-05 config schema differs")
    if value.get("identity", {}).get("status") != "frozen":
        raise AbilityDataError("TD-05 config is not frozen")
    quality = value.get("quality", {})
    sampling = value.get("sampling", {})
    if quality.get("formal_test_access") != "prohibited":
        raise AbilityDataError("formal test access must remain prohibited")
    if not math.isclose(float(sampling.get("teacher_weight", -1)), 0.80):
        raise AbilityDataError("teacher sampling weight must remain 0.80")
    if not math.isclose(float(sampling.get("human_weight", -1)), 0.20):
        raise AbilityDataError("human sampling weight must remain 0.20")
    if sampling.get("raw_duplicate_fill") != "prohibited":
        raise AbilityDataError("raw duplicate fill must remain prohibited")
    return value


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AbilityDataError(f"JSON object expected: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise AbilityDataError(f"JSON object expected: {path}:{line_number}")
            rows.append(value)
    return rows


def _training_row(row: Mapping[str, Any], mixture_class: str) -> dict[str, Any]:
    sample_id = str(row["record_id"])
    group_id = str(row["semantic_group_id"])
    source = str(row["src_lang"])
    target = str(row["tgt_lang"])
    source_text = str(row["source_text"]).strip()
    target_text = str(row["target_text"]).strip()
    if source not in MODEL_TAGS or target not in MODEL_TAGS or source == target:
        raise AbilityDataError(f"unsupported TD-05 route: {source}->{target}")
    if not sample_id or not group_id or not source_text or not target_text:
        raise AbilityDataError("TD-05 record contains an empty required field")
    result = {
        "sample_id": sample_id,
        "sample_group_id": group_id,
        "split": "train",
        "src_lang": source,
        "tgt_lang": target,
        "source_text": source_text,
        "target_text": target_text,
        "mixture_class": mixture_class,
        "provenance": str(row["provenance"]),
        "input_record_id": sample_id,
    }
    for key in (
        "source_id",
        "source_record_id",
        "teacher_job_id",
        "forward_job_id",
        "generation_identity",
        "profile",
        "counts_as_native_hant",
    ):
        if key in row:
            result[key] = row[key]
    return result


def assemble_records(
    teacher: Sequence[Mapping[str, Any]],
    reverse: Sequence[Mapping[str, Any]],
    human: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = [*(_training_row(row, "teacher") for row in teacher)]
    rows.extend(_training_row(row, "teacher") for row in reverse)
    rows.extend(_training_row(row, "human") for row in human)
    seen_ids: set[str] = set()
    seen_pairs: set[tuple[str, str, str, str]] = set()
    class_groups: dict[str, set[str]] = defaultdict(set)
    class_routes: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        sample_id = str(row["sample_id"])
        if sample_id in seen_ids:
            raise AbilityDataError(f"duplicate TD-05 sample_id: {sample_id}")
        seen_ids.add(sample_id)
        pair = (
            str(row["src_lang"]),
            str(row["tgt_lang"]),
            normalized_identity(str(row["source_text"])),
            normalized_identity(str(row["target_text"])),
        )
        if pair in seen_pairs:
            raise AbilityDataError(f"duplicate directed text pair: {sample_id}")
        seen_pairs.add(pair)
        mixture_class = str(row["mixture_class"])
        class_groups[mixture_class].add(str(row["sample_group_id"]))
        class_routes[mixture_class][f"{row['src_lang']}->{row['tgt_lang']}"] += 1
    overlap = class_groups["teacher"] & class_groups["human"]
    if overlap:
        raise AbilityDataError(f"teacher/human semantic groups overlap: {len(overlap)}")
    rows.sort(key=lambda row: (str(row["mixture_class"]), str(row["sample_id"])))
    audit = {
        "records": len(rows),
        "unique_sample_ids": len(seen_ids),
        "unique_directed_text_pairs": len(seen_pairs),
        "teacher_human_group_overlap": len(overlap),
        "class_records": dict(sorted(Counter(str(row["mixture_class"]) for row in rows).items())),
        "class_groups": {key: len(value) for key, value in sorted(class_groups.items())},
        "class_route_counts": {
            key: dict(sorted(value.items())) for key, value in sorted(class_routes.items())
        },
    }
    return rows, audit


def _smooth_counts(weights: Mapping[str, float], exposures: int) -> Counter[str]:
    order = tuple(sorted(weights))
    scores = {key: 0.0 for key in order}
    total = sum(float(weights[key]) for key in order)
    counts: Counter[str] = Counter()
    for _ in range(exposures):
        for key in order:
            scores[key] += float(weights[key])
        chosen = max(order, key=lambda key: (scores[key], -order.index(key)))
        scores[chosen] -= total
        counts[chosen] += 1
    return counts


def build_sampling_plan(
    records: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> dict[str, Any]:
    sampling = config["sampling"]
    preview = int(sampling["preview_exposures"])
    class_weights = {
        "teacher": float(sampling["teacher_weight"]),
        "human": float(sampling["human_weight"]),
    }
    class_counts = _smooth_counts(class_weights, preview)
    routes_by_class: dict[str, set[str]] = defaultdict(set)
    for row in records:
        routes_by_class[str(row["mixture_class"])].add(
            f"{row['src_lang']}->{row['tgt_lang']}"
        )
    route_preview: dict[str, dict[str, int]] = {}
    for mixture_class, count in sorted(class_counts.items()):
        routes = sorted(routes_by_class[mixture_class])
        route_preview[mixture_class] = dict(
            sorted(_smooth_counts({route: 1.0 for route in routes}, count).items())
        )
    return {
        "schema_version": 1,
        "algorithm": sampling["algorithm"],
        "seed": int(sampling["seed"]),
        "class_weights": class_weights,
        "within_class_route_policy": sampling["within_class_route_policy"],
        "within_route_record_policy": sampling["within_route_record_policy"],
        "raw_duplicate_fill": False,
        "preview": {
            "exposures": preview,
            "class_counts": dict(sorted(class_counts.items())),
            "route_counts_by_class": route_preview,
        },
    }


def _token_lengths(
    rows: Sequence[Mapping[str, Any]], tokenizer: Any
) -> tuple[list[int], list[int]]:
    from mvp_60m_data_pipeline import tokenizer_lengths

    source = [0] * len(rows)
    target = [0] * len(rows)
    original = getattr(tokenizer, "src_lang", None)
    try:
        for language in MODEL_TAGS:
            source_indexes = [i for i, row in enumerate(rows) if row["src_lang"] == language]
            if source_indexes:
                tokenizer.src_lang = language
                values = tokenizer_lengths(tokenizer, [str(rows[i]["source_text"]) for i in source_indexes])
                for index, value in zip(source_indexes, values, strict=True):
                    source[index] = value
            target_indexes = [i for i, row in enumerate(rows) if row["tgt_lang"] == language]
            if target_indexes:
                tokenizer.src_lang = language
                values = tokenizer_lengths(tokenizer, [str(rows[i]["target_text"]) for i in target_indexes])
                for index, value in zip(target_indexes, values, strict=True):
                    target[index] = value
    finally:
        tokenizer.src_lang = original
    return source, target


def validate_token_lengths(
    rows: Sequence[Mapping[str, Any]],
    tokenizer: Any,
    *,
    maximum_source: int,
    maximum_target: int,
) -> dict[str, Any]:
    source, target = _token_lengths(rows, tokenizer)
    source_overflow = [rows[i]["sample_id"] for i, length in enumerate(source) if length > maximum_source]
    target_overflow = [rows[i]["sample_id"] for i, length in enumerate(target) if length > maximum_target]
    if source_overflow or target_overflow:
        raise AbilityDataError(
            f"zero-truncation gate failed: source={len(source_overflow)}, target={len(target_overflow)}"
        )
    return {
        "source_max_tokens": max(source, default=0),
        "target_max_tokens": max(target, default=0),
        "source_total_tokens": sum(source),
        "target_total_tokens": sum(target),
        "source_overflow": 0,
        "target_overflow": 0,
    }


def _verify_inputs(
    repository_root: Path, runtime_root: Path, config: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    inputs = config["inputs"]
    compact = repository_root / PurePosixPath(inputs["td03_compact_manifest"])
    if sha256_file(compact) != inputs["td03_compact_manifest_sha256"]:
        raise AbilityDataError("TD-03 compact manifest hash drift")
    td03 = read_json(runtime_root / PurePosixPath(inputs["td03_runtime_manifest"]))
    td04 = read_json(runtime_root / PurePosixPath(inputs["td04_runtime_manifest"]))
    if td03.get("status") != "complete" or td04.get("status") != "complete":
        raise AbilityDataError("TD-03 and TD-04 must both be complete")
    generation_config = repository_root / PurePosixPath(inputs["td04_generation_config"])
    generation = yaml.safe_load(generation_config.read_text(encoding="utf-8"))
    if canonical_sha256(generation) != inputs["td04_generation_config_sha256"]:
        raise AbilityDataError("TD-04 generation config hash drift")
    if td04.get("generation_config_sha256") != inputs["td04_generation_config_sha256"]:
        raise AbilityDataError("TD-04 manifest belongs to another generation config")
    for key, manifest, section in (
        ("human_anchors", td03, "human_anchors"),
        ("accepted_teacher", td04, "accepted_teacher"),
        ("reverse_pairs", td04, "reverse_pairs"),
    ):
        path = runtime_root / PurePosixPath(inputs[key])
        if sha256_file(path) != manifest[section]["sha256"]:
            raise AbilityDataError(f"{key} hash drift")
    return td03, td04


def publish(
    repository_root: Path,
    runtime_root: Path,
    config_path: Path,
    *,
    tokenizer: Any | None = None,
    compact_report_path: Path | None = None,
) -> dict[str, Any]:
    config = load_config(config_path)
    td03, td04 = _verify_inputs(repository_root, runtime_root, config)
    inputs = config["inputs"]
    human = read_jsonl(runtime_root / PurePosixPath(inputs["human_anchors"]))
    teacher = read_jsonl(runtime_root / PurePosixPath(inputs["accepted_teacher"]))
    reverse = read_jsonl(runtime_root / PurePosixPath(inputs["reverse_pairs"]))
    rows, audit = assemble_records(teacher, reverse, human)

    quality = config["quality"]
    if len(human) != int(quality["required_human_records"]):
        raise AbilityDataError("human anchor count drift")
    teacher_routes = audit["class_route_counts"].get("teacher", {})
    if set(teacher_routes) != set(ROUTES) or len(teacher_routes) != int(quality["required_teacher_routes"]):
        raise AbilityDataError("teacher records do not cover exactly 20 routes")
    fixed = int(quality["required_fixed_non_hant_teacher_records_per_route"])
    for route, count in teacher_routes.items():
        if not route.startswith("zho_Hant->") and int(count) < fixed:
            raise AbilityDataError(f"{route} has fewer than {fixed} teacher records")

    if tokenizer is None:
        student_path = repository_root / PurePosixPath(config["tokenizer"]["student_config"])
        student = load_student_config(student_path)
        tokenizer, token_info = load_frozen_tokenizer(student, repository_root)
    else:
        token_info = {"artifact_manifest_sha256": config["tokenizer"]["artifact_manifest_sha256"]}
    if token_info["artifact_manifest_sha256"] != config["tokenizer"]["artifact_manifest_sha256"]:
        raise AbilityDataError("tokenizer artifact identity drift")
    token_audit = validate_token_lengths(
        rows,
        tokenizer,
        maximum_source=int(config["tokenizer"]["maximum_source_tokens"]),
        maximum_target=int(config["tokenizer"]["maximum_target_tokens"]),
    )
    plan = build_sampling_plan(rows, config)
    output_root = runtime_root / "td05"
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "manifest.json"
    manifest_path.unlink(missing_ok=True)
    records, corpus_hash = write_jsonl(output_root / "training-corpus.jsonl", rows)
    write_json(output_root / "sampling-plan.json", plan)
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "task": "TD-05",
        "config_sha256": canonical_sha256(config),
        "formal_test_accessed": False,
        "inputs": {
            "td03_manifest_sha256": sha256_file(runtime_root / PurePosixPath(inputs["td03_runtime_manifest"])),
            "td04_manifest_sha256": sha256_file(runtime_root / PurePosixPath(inputs["td04_runtime_manifest"])),
        },
        "corpus": {"path": "training-corpus.jsonl", "records": records, "sha256": corpus_hash},
        "sampling_plan": {
            "path": "sampling-plan.json",
            "sha256": sha256_file(output_root / "sampling-plan.json"),
            "teacher_weight": 0.80,
            "human_weight": 0.20,
        },
        "audit": audit,
        "token_audit": token_audit,
        "invariants": {
            "twenty_teacher_routes": True,
            "zero_truncation": True,
            "raw_duplicate_fill": False,
            "teacher_human_group_overlap": 0,
            "manifest_published_last": True,
        },
    }
    write_json(manifest_path, manifest)
    report_path = compact_report_path
    if report_path is None:
        report_path = repository_root / PurePosixPath(config["outputs"]["compact_report"])
    compact = {
        **manifest,
        "runtime_manifest_sha256": sha256_file(manifest_path),
        "td03": {"source_records": td03["source_bank"]["records"], "human_records": td03["human_anchors"]["records"]},
        "td04": {"raw_records": td04["raw"]["records"], "accepted_records": td04["accepted_teacher"]["records"], "reverse_records": td04["reverse_pairs"]["records"]},
    }
    write_json(report_path, compact)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/mvp_60m_mixed_corpus.yaml"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repository_root = args.repository_root.resolve()
    config_path = args.config.resolve() if args.config.is_absolute() else (repository_root / args.config).resolve()
    publish(repository_root, args.runtime_root.resolve(), config_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
