#!/usr/bin/env python3
"""Verify and freeze the Diesel-MT MVP tokenizer artifact.

The freeze record is intentionally stored outside the immutable tokenizer
directory.  Its root identity is the SHA-256 of ``artifact_manifest.json``;
that manifest in turn pins every file in the published artifact.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in os.sys.path:
    os.sys.path.insert(0, str(SCRIPTS_DIR))

from artifact_io import sha256_file  # noqa: E402
from tokenizer_utils import (  # noqa: E402
    PROJECT_LANGUAGES,
    atomic_write_json,
    backend_pipeline,
    build_language_mapping,
    forced_bos_token_id,
    reload_tokenizer,
    verify_tokenizer,
)


SCHEMA_VERSION = 1
EXPECTED_VOCAB_SIZE = 49_152
LANGUAGE_TEXT = {
    "eng_Latn": "A tiny multilingual forward pass checks the frozen tokenizer.",
    "zho_Hans": "这是冻结分词器的微型前向测试。",
    "zho_Hant": "這是凍結分詞器的微型前向測試。",
    "jpn_Jpan": "これは凍結したトークナイザーの小さな前向きテストです。",
    "kor_Hang": "동결된 토크나이저의 작은 순방향 테스트입니다.",
}


class FreezeError(RuntimeError):
    """Raised when a freeze invariant is not satisfied."""


def portable_display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return resolved.name


def read_object(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise FreezeError(f"cannot read JSON object {path}: {error}") from error
    if not isinstance(value, dict):
        raise FreezeError(f"expected a JSON object in {path}")
    return value


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def verify_artifact_files(artifact_dir: Path) -> tuple[dict, str]:
    manifest_path = artifact_dir / "artifact_manifest.json"
    manifest = read_object(manifest_path)
    if manifest.get("schema_version") != 1 or not isinstance(manifest.get("files"), list):
        raise FreezeError("artifact manifest has an unsupported schema")
    expected_paths: set[str] = set()
    for record in manifest["files"]:
        if not isinstance(record, Mapping):
            raise FreezeError("artifact manifest contains a non-object file record")
        relative = str(record.get("path", ""))
        if not relative or relative in expected_paths:
            raise FreezeError(f"invalid or duplicate artifact path: {relative!r}")
        expected_paths.add(relative)
        path = artifact_dir / relative
        if not path.is_file():
            raise FreezeError(f"artifact file is missing: {path}")
        if path.stat().st_size != int(record.get("bytes", -1)):
            raise FreezeError(f"artifact byte count mismatch: {relative}")
        actual_sha256 = sha256_file(path)
        if actual_sha256 != record.get("sha256"):
            raise FreezeError(f"artifact SHA-256 mismatch: {relative}")
    actual_paths = {
        path.relative_to(artifact_dir).as_posix()
        for path in artifact_dir.rglob("*")
        if path.is_file() and path.name != "artifact_manifest.json"
    }
    if actual_paths != expected_paths:
        raise FreezeError(
            "artifact file set differs from manifest: "
            f"missing={sorted(expected_paths - actual_paths)}, extra={sorted(actual_paths - expected_paths)}"
        )
    return manifest, sha256_file(manifest_path)


def verify_save_reload(artifact_dir: Path) -> tuple[object, dict[str, int]]:
    tokenizer = reload_tokenizer(artifact_dir)
    verify_tokenizer(tokenizer, expected_vocab_size=EXPECTED_VOCAB_SIZE)
    mapping = build_language_mapping(tokenizer)
    saved_mapping = read_object(artifact_dir / "language_map.json").get("mapping")
    if saved_mapping != mapping:
        raise FreezeError(f"language_map.json differs from the loaded tokenizer: {saved_mapping}")
    for language in PROJECT_LANGUAGES:
        tokenizer.src_lang = language
        token_id = mapping[language]
        encoded_language = tokenizer.encode(language, add_special_tokens=False)
        encoded_text = tokenizer(LANGUAGE_TEXT[language], add_special_tokens=True)["input_ids"]
        if encoded_language != [token_id] or encoded_text[0] != token_id:
            raise FreezeError(f"language token contract failed for {language}")
        if encoded_text[-1] != tokenizer.eos_token_id:
            raise FreezeError(f"EOS contract failed for {language}")
    original_vocab = tokenizer.get_vocab()
    original_backend = backend_pipeline(tokenizer)
    with tempfile.TemporaryDirectory(prefix="diesel-mt-tokenizer-reload-") as directory:
        tokenizer.save_pretrained(directory)
        reloaded = reload_tokenizer(Path(directory))
        if reloaded.get_vocab() != original_vocab:
            raise FreezeError("vocabulary changed across explicit save/reload")
        if backend_pipeline(reloaded) != original_backend:
            raise FreezeError("backend pipeline changed across explicit save/reload")
    tokenizer.src_lang = "eng_Latn"
    return tokenizer, mapping


def micro_m2m100_forward(tokenizer: object, mapping: Mapping[str, int]) -> dict:
    import torch
    from transformers import M2M100Config, M2M100ForConditionalGeneration

    torch.manual_seed(20260713)
    vocabulary_size = len(tokenizer)
    config = M2M100Config(
        vocab_size=vocabulary_size,
        d_model=32,
        encoder_layers=1,
        decoder_layers=1,
        encoder_ffn_dim=64,
        decoder_ffn_dim=64,
        encoder_attention_heads=4,
        decoder_attention_heads=4,
        dropout=0.0,
        attention_dropout=0.0,
        activation_dropout=0.0,
        use_cache=False,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        decoder_start_token_id=mapping["eng_Latn"],
    )
    model = M2M100ForConditionalGeneration(config)
    model.eval()
    dimensions = {
        "config_vocab_size": model.config.vocab_size,
        "shared_embedding_rows": model.model.shared.num_embeddings,
        "encoder_embedding_rows": model.model.encoder.embed_tokens.num_embeddings,
        "decoder_embedding_rows": model.model.decoder.embed_tokens.num_embeddings,
        "lm_head_rows": model.lm_head.out_features,
    }
    if set(dimensions.values()) != {vocabulary_size}:
        raise FreezeError(f"micro M2M100 vocabulary dimensions disagree: {dimensions}")
    forwards: list[dict] = []
    languages = list(PROJECT_LANGUAGES)
    with torch.no_grad():
        for index, source_language in enumerate(languages):
            target_language = languages[(index + 1) % len(languages)]
            tokenizer.src_lang = source_language
            inputs = tokenizer(LANGUAGE_TEXT[source_language], return_tensors="pt")
            tokenizer.src_lang = target_language
            labels = tokenizer(
                LANGUAGE_TEXT[target_language], return_tensors="pt"
            )["input_ids"]
            target_id = forced_bos_token_id(tokenizer, target_language)
            model.config.decoder_start_token_id = target_id
            output = model(**inputs, labels=labels)
            loss = float(output.loss.item())
            if not math.isfinite(loss):
                raise FreezeError(
                    f"micro M2M100 produced a non-finite loss for {source_language}->{target_language}"
                )
            if output.logits.shape[-1] != vocabulary_size:
                raise FreezeError("micro M2M100 logits have the wrong vocabulary dimension")
            forwards.append(
                {
                    "source_language": source_language,
                    "target_language": target_language,
                    "forced_bos_token_id": target_id,
                    "input_tokens": int(inputs["input_ids"].shape[1]),
                    "label_tokens": int(labels.shape[1]),
                    "loss": loss,
                    "logits_shape": list(output.logits.shape),
                }
            )
    tokenizer.src_lang = "eng_Latn"
    return {
        "model": "M2M100ForConditionalGeneration",
        "config": {
            "d_model": 32,
            "encoder_layers": 1,
            "decoder_layers": 1,
            "attention_heads": 4,
            "ffn_dim": 64,
        },
        "dimensions": dimensions,
        "forwards": forwards,
    }


def coverage_summary(metrics: Mapping) -> dict:
    corpus_metrics = metrics.get("corpus_metrics")
    coverage = metrics.get("character_coverage")
    if not isinstance(corpus_metrics, Mapping) or set(corpus_metrics) != set(PROJECT_LANGUAGES):
        raise FreezeError("coverage report does not contain exactly the five project languages")
    if not isinstance(coverage, Mapping) or set(coverage) != set(PROJECT_LANGUAGES):
        raise FreezeError("character coverage does not contain exactly the five project languages")
    result: dict[str, dict] = {}
    for language in PROJECT_LANGUAGES:
        item = corpus_metrics[language]
        character = coverage[language]
        summary = {
            "tokens_per_non_whitespace_character": item["tokens_per_non_whitespace_character"],
            "token_length_p95": item["token_length_p95"],
            "token_length_p99": item["token_length_p99"],
            "source_character_loss_rate": item["source_character_loss_rate"],
            "frequency_weighted_coverage": character["frequency_weighted_coverage"],
            "unique_character_coverage": character["unique_character_coverage"],
            "roundtrip_exact_rate": item["roundtrip_exact_rate"],
        }
        numeric = [float(value) for value in summary.values()]
        if not all(math.isfinite(value) for value in numeric):
            raise FreezeError(f"non-finite coverage metric for {language}")
        result[language] = summary
    return result


def render_markdown(record: Mapping) -> str:
    lines = [
        "# mvp-tokenizer-v0 freeze acceptance",
        "",
        f"- Status: **{record['status']}**",
        f"- Vocabulary: {record['vocab_size']:,}",
        f"- Artifact manifest SHA-256: `{record['artifact_manifest_sha256']}`",
        f"- Tokenizer SHA-256: `{record['tokenizer_sha256']}`",
        f"- Training corpus manifest SHA-256: `{record['training_corpus_manifest_sha256']}`",
        f"- Holdout manifest SHA-256: `{record['holdout_manifest_sha256']}`",
        f"- Fixed evaluation manifest SHA-256: `{record['evaluation_manifest_sha256']}`",
        "",
        "## Coverage",
        "",
        "| Language | Tokens/char | P95 | P99 | Source loss | Frequency coverage | Unique coverage | Roundtrip |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for language in PROJECT_LANGUAGES:
        item = record["coverage"][language]
        lines.append(
            f"| {language} | {item['tokens_per_non_whitespace_character']:.4f} | "
            f"{item['token_length_p95']} | {item['token_length_p99']} | "
            f"{item['source_character_loss_rate']:.6%} | "
            f"{item['frequency_weighted_coverage']:.6%} | "
            f"{item['unique_character_coverage']:.6%} | "
            f"{item['roundtrip_exact_rate']:.6%} |"
        )
    parity = record["chinese_sequence_parity"]
    lines.extend(
        [
            "",
            "## Simplified/Traditional parity",
            "",
            f"- Traditional/Simplified tokens-per-character ratio: {parity['traditional_to_simplified_fertility_ratio']:.4f}",
            f"- Traditional/Simplified P95 ratio: {parity['traditional_to_simplified_p95_ratio']:.4f}",
            "",
            "## Integrity and runtime checks",
            "",
            f"- Artifact files verified: {record['artifact_files_verified']}",
            f"- Save/reload vocabulary and backend equality: {record['save_reload_verified']}",
            f"- Five language tokens verified: {record['language_tokens_verified']}",
            f"- Micro M2M100 forwards with finite loss: {len(record['micro_m2m100']['forwards'])}/5",
            f"- M2M100 embedding/lm_head rows: {record['vocab_size']:,}",
            "",
            "Synthetic rare-Unicode probes remain diagnostic and are not training data. "
            "The frozen decision uses the independently generated holdout metrics above.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts/tokenizers/mvp-tokenizer-v0"))
    parser.add_argument(
        "--metrics",
        type=Path,
        default=Path("artifacts/tokenizers/reports/mvp-tokenizer-v0/coverage-mvp-48k.json"),
    )
    parser.add_argument(
        "--holdout-manifest",
        type=Path,
        default=Path("data/tokenizer/holdout/mvp/manifest.jsonl"),
    )
    parser.add_argument(
        "--evaluation-manifest",
        type=Path,
        default=Path("data/tokenizer/evaluation/mvp-v0/manifest.json"),
    )
    parser.add_argument(
        "--comparison",
        type=Path,
        default=Path("artifacts/tokenizers/reports/mvp-tokenizer-v0/comparison.json"),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("artifacts/tokenizers/reports/mvp-tokenizer-v0/freeze_acceptance.json"),
    )
    parser.add_argument(
        "--output-markdown",
        type=Path,
        default=Path("artifacts/tokenizers/reports/mvp-tokenizer-v0/freeze_acceptance.md"),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    artifact_dir = args.artifact_dir.resolve()
    manifest, manifest_sha256 = verify_artifact_files(artifact_dir)
    tokenizer, mapping = verify_save_reload(artifact_dir)
    metrics = read_object(args.metrics)
    if int(metrics.get("vocab_size", -1)) != EXPECTED_VOCAB_SIZE:
        raise FreezeError(f"coverage report has wrong vocabulary size: {metrics.get('vocab_size')}")
    training_metadata = read_object(artifact_dir / "training_meta.json")
    if float(training_metadata.get("sample_fraction", -1)) != 1.0:
        raise FreezeError("frozen tokenizer was not trained on the complete balanced corpus")
    training_languages = training_metadata.get("languages")
    if not isinstance(training_languages, Mapping) or set(training_languages) != set(PROJECT_LANGUAGES):
        raise FreezeError("training metadata does not contain exactly five languages")
    micro = micro_m2m100_forward(tokenizer, mapping)
    record = {
        "schema_version": SCHEMA_VERSION,
        "status": "frozen",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "artifact": portable_display_path(artifact_dir),
        "vocab_size": len(tokenizer),
        "artifact_manifest_sha256": manifest_sha256,
        "artifact_files_verified": len(manifest["files"]),
        "tokenizer_sha256": sha256_file(artifact_dir / "tokenizer.json"),
        "training_corpus_manifest_sha256": sha256_file(artifact_dir / "corpus_manifest.jsonl"),
        "holdout_manifest_sha256": sha256_file(args.holdout_manifest),
        "evaluation_manifest_sha256": sha256_file(args.evaluation_manifest),
        "coverage_report_sha256": sha256_file(args.metrics),
        "fallback_comparison_sha256": sha256_file(args.comparison),
        "training_checkpoint_config_fingerprint": training_metadata[
            "checkpoint_config_fingerprint"
        ],
        "training_input_order_sha256": training_metadata["snapshot"][
            "input_order_sha256"
        ],
        "save_reload_verified": True,
        "language_tokens_verified": True,
        "language_token_ids": mapping,
        "coverage": coverage_summary(metrics),
        "chinese_sequence_parity": metrics["chinese_sequence_parity"],
        "micro_m2m100": micro,
    }
    atomic_write_json(args.output_json, record)
    atomic_write_text(args.output_markdown, render_markdown(record))
    print(
        json.dumps(
            {
                "status": record["status"],
                "artifact_manifest_sha256": manifest_sha256,
                "output_json": str(args.output_json),
                "output_markdown": str(args.output_markdown),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
