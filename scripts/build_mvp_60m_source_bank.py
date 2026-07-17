"""Build the TD-03 source bank and human anchors from the frozen TD-02 lock."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from model_data_source_contract import canonical_sha256, load_mvp_60m_source_config, load_mvp_60m_source_lock
from mvp_60m_data_pipeline import (
    AbilityDataError,
    ParallelGroup,
    anchor_rows,
    filter_parallel_groups,
    near_identity,
    normalized_identity,
    parse_alt_groups,
    parse_massive_groups,
    parse_moj_parallel_groups,
    read_parallel_lines,
    select_group_role,
    select_unpc_hans,
    sha256_file,
    source_rows,
    write_json,
    write_jsonl,
    canonical_json_bytes,
    sha256_bytes,
)
from tokenizer_utils import reload_tokenizer


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--alt-en-ja", type=Path, required=True)
    parser.add_argument("--alt-en-zh", type=Path, required=True)
    parser.add_argument("--kftt-en", type=Path, required=True)
    parser.add_argument("--kftt-ja", type=Path, required=True)
    parser.add_argument("--korean-en", type=Path, required=True)
    parser.add_argument("--korean-ko", type=Path, required=True)
    parser.add_argument("--unpc-zh", type=Path, required=True)
    parser.add_argument("--massive", type=Path, required=True)
    parser.add_argument("--moj-zh", type=Path, nargs=2, required=True)
    parser.add_argument("--moj-en", type=Path, nargs=2, required=True)
    return parser.parse_args()


def main() -> int:
    args = arguments()
    root = args.repository_root.resolve()
    runtime = args.runtime_root.resolve()
    config = load_mvp_60m_source_config(root / "configs/mvp_60m_distillation_sources.yaml")
    lock = load_mvp_60m_source_lock(root / "configs/mvp_60m_distillation_sources.lock.json", config)
    tokenizer = reload_tokenizer(root / "artifacts/tokenizers/mvp-tokenizer-v0")
    flores: list[str] = []
    for language in ("eng_Latn", "zho_Hans", "zho_Hant", "jpn_Jpan", "kor_Hang"):
        flores.extend((root / "data/model/raw/flores200-original/dev" / f"{language}.dev").read_text(encoding="utf-8").splitlines())
    contamination_exact = {normalized_identity(text) for text in flores}
    contamination_near = {near_identity(text) for text in flores}

    hant_path = runtime / "td02/native-hant-selected.jsonl"
    hant_rows = [json.loads(line) for line in hant_path.read_text(encoding="utf-8").splitlines()]
    if len(hant_rows) != config["source_bank"]["actual_native_hant_unique_texts"]:
        raise AbilityDataError("TD-02 native-Hant count differs from source contract")
    if sha256_file(hant_path) != config["source_bank"]["selected_native_hant_sha256"]:
        raise AbilityDataError("TD-02 native-Hant selection hash differs")
    used_exact = {str(row["normalized_sha256"]) for row in hant_rows}
    used_near = {near_identity(str(row["text"])) for row in hant_rows}
    used_exact.update(contamination_exact)
    used_near.update(contamination_near)
    used_groups: set[tuple[str, str]] = set()
    forbidden_groups: set[tuple[str, str]] = set()
    for row in hant_rows:
        source_id = str(row["source_id"])
        record_id = str(row["source_record_id"])
        if source_id == "massive-1.1-route-control":
            forbidden_groups.add((source_id, record_id))
        elif source_id == "taiwan-moj-law-api-20260710":
            parts = record_id.split(":")
            url = ":".join(parts[1:-2])
            code = url.partition("pcode=")[2].lower()
            article = parts[-2]
            forbidden_groups.add((source_id, f"{code}:{article}"))

    raw_groups = {
        "alt": parse_alt_groups(args.alt_en_ja.resolve(), args.alt_en_zh.resolve()),
        "kftt": read_parallel_lines(
            "kftt-1.0-en-ja", {"eng_Latn": args.kftt_en.resolve(), "jpn_Jpan": args.kftt_ja.resolve()}
        ),
        "korean": read_parallel_lines(
            "korean-parallel-news-a1fb53d", {"eng_Latn": args.korean_en.resolve(), "kor_Hang": args.korean_ko.resolve()}
        ),
        "massive": parse_massive_groups(args.massive.resolve()),
        "moj": parse_moj_parallel_groups(
            [path.resolve() for path in args.moj_zh], [path.resolve() for path in args.moj_en]
        ),
    }
    groups: dict[str, list[ParallelGroup]] = {}
    rejection_reports: dict[str, dict[str, int]] = {}
    for name, values in raw_groups.items():
        groups[name], rejection_reports[name] = filter_parallel_groups(
            values,
            tokenizer=tokenizer,
            contamination_exact=contamination_exact,
            contamination_near=contamination_near,
        )

    anchor_specs = [
        ("alt", 4000, ("eng_Latn", "zho_Hans", "jpn_Jpan")),
        ("kftt", 3000, ("eng_Latn", "jpn_Jpan")),
        ("korean", 3000, ("eng_Latn", "kor_Hang")),
        ("moj", 2000, ("eng_Latn", "zho_Hant")),
        ("massive", 500, ("eng_Latn", "zho_Hans", "zho_Hant", "jpn_Jpan", "kor_Hang")),
    ]
    anchors: list[dict[str, object]] = []
    anchor_group_counts: dict[str, int] = {}
    for name, ceiling, languages in anchor_specs:
        selected = select_group_role(
            groups[name], count=ceiling, seed=f"td03-anchor-{name}", languages=languages,
            used_groups=used_groups, used_exact=used_exact, used_near=used_near,
            forbidden_groups=forbidden_groups,
        )
        anchor_group_counts[name] = len(selected)
        anchors.extend(anchor_rows(selected, languages))

    source_specs = [
        ("alt", "eng_Latn", 1000),
        ("kftt", "eng_Latn", 34000),
        ("korean", "eng_Latn", 15000),
        ("alt", "zho_Hans", 5000),
        ("kftt", "jpn_Jpan", 49000),
        ("alt", "jpn_Jpan", 1000),
        ("korean", "kor_Hang", 50000),
    ]
    sources: list[dict[str, object]] = []
    source_counts: Counter[str] = Counter()
    for name, language, count in source_specs:
        selected = select_group_role(
            groups[name], count=count, seed=f"td03-source-{name}-{language}", languages=(language,),
            used_groups=used_groups, used_exact=used_exact, used_near=used_near,
        )
        if len(selected) != count:
            raise AbilityDataError(f"{name}/{language} selected {len(selected)}, expected {count}")
        sources.extend(source_rows(selected, language))
        source_counts[language] += len(selected)
    unpc = select_unpc_hans(
        args.unpc_zh.resolve(), tokenizer=tokenizer, count=45000, seed="td03-source-unpc-zho-hans",
        used_exact=used_exact, used_near=used_near,
        contamination_exact=contamination_exact, contamination_near=contamination_near,
    )
    for row in unpc:
        row["record_id"] = f"src-{row['normalized_sha256'][:24]}"
        row["semantic_group_id"] = f"grp-unpc-{int(row['source_record_id']):08d}"
    sources.extend(unpc)
    source_counts["zho_Hans"] += len(unpc)
    for row in hant_rows:
        row["record_id"] = f"src-{row['normalized_sha256'][:24]}"
        source_id = str(row["source_id"])
        source_group_id = str(row["source_record_id"])
        if source_id == "taiwan-moj-law-api-20260710":
            parts = source_group_id.split(":")
            url = ":".join(parts[1:-2])
            code = url.partition("pcode=")[2].lower()
            source_group_id = f"{code}:{parts[-2]}"
        row["semantic_group_id"] = (
            f"grp-{sha256_bytes(canonical_json_bytes([source_id, source_group_id]))[:24]}"
        )
    sources.extend(hant_rows)
    source_counts["zho_Hant"] += len(hant_rows)

    expected = {"eng_Latn": 50000, "zho_Hans": 50000, "zho_Hant": 851, "jpn_Jpan": 50000, "kor_Hang": 50000}
    if dict(source_counts) != expected:
        raise AbilityDataError(f"source counts differ: {dict(source_counts)}")
    source_group_ids = {str(row["semantic_group_id"]) for row in sources}
    anchor_group_ids = {str(row["semantic_group_id"]) for row in anchors}
    group_overlap = source_group_ids & anchor_group_ids
    source_identities = {normalized_identity(str(row["text"])) for row in sources}
    source_near = {near_identity(str(row["text"])) for row in sources}
    anchor_texts = {
        str(row[field])
        for row in anchors
        for field in ("source_text", "target_text")
    }
    exact_overlap = source_identities & {normalized_identity(text) for text in anchor_texts}
    near_overlap = source_near & {near_identity(text) for text in anchor_texts}
    if group_overlap or exact_overlap or near_overlap:
        raise AbilityDataError(
            f"TD-03 isolation failed: groups={len(group_overlap)}, exact={len(exact_overlap)}, near={len(near_overlap)}"
        )
    td03 = runtime / "td03"
    source_records, source_hash = write_jsonl(td03 / "source-bank.jsonl", sources)
    anchor_records, anchor_hash = write_jsonl(td03 / "human-anchors.jsonl", anchors)
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "task": "TD-03",
        "source_config_sha256": canonical_sha256(config),
        "source_lock_sha256": sha256_file(root / "configs/mvp_60m_distillation_sources.lock.json"),
        "formal_test_accessed": False,
        "source_bank": {"path": "source-bank.jsonl", "records": source_records, "sha256": source_hash, "counts": expected},
        "human_anchors": {
            "path": "human-anchors.jsonl", "records": anchor_records, "sha256": anchor_hash,
            "independent_group_counts": anchor_group_counts,
            "route_counts": dict(sorted(Counter(f"{row['src_lang']}->{row['tgt_lang']}" for row in anchors).items())),
        },
        "candidate_groups": {name: {"raw": len(raw_groups[name]), "accepted": len(groups[name]), "rejections": rejection_reports[name]} for name in raw_groups},
        "invariants": {
            "zero_truncation": True,
            "source_anchor_group_overlap": len(group_overlap),
            "exact_or_near_overlap": len(exact_overlap | near_overlap),
            "flores_dev_contamination": 0,
        },
    }
    write_json(td03 / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
