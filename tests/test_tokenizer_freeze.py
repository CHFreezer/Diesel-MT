from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from freeze_tokenizer_artifact import (  # noqa: E402
    coverage_summary,
    micro_m2m100_forward,
    verify_artifact_files,
)
from tokenizer_utils import (  # noqa: E402
    PROJECT_LANGUAGES,
    build_language_mapping,
    create_seed_tokenizer,
)


def test_artifact_manifest_verifies_every_published_file(tmp_path: Path) -> None:
    payloads = {"tokenizer.json": b"tokenizer\n", "language_map.json": b"mapping\n"}
    records = []
    for name, payload in payloads.items():
        (tmp_path / name).write_bytes(payload)
        records.append(
            {
                "path": name,
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    manifest = {"schema_version": 1, "files": sorted(records, key=lambda row: row["path"])}
    manifest_path = tmp_path / "artifact_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    actual, manifest_sha256 = verify_artifact_files(tmp_path)

    assert actual == manifest
    assert manifest_sha256 == hashlib.sha256(manifest_path.read_bytes()).hexdigest()


def test_coverage_summary_requires_and_preserves_five_languages() -> None:
    metrics = {
        "corpus_metrics": {
            language: {
                "tokens_per_non_whitespace_character": 0.5,
                "token_length_p95": 100,
                "token_length_p99": 200,
                "source_character_loss_rate": 0.0,
                "roundtrip_exact_rate": 1.0,
            }
            for language in PROJECT_LANGUAGES
        },
        "character_coverage": {
            language: {
                "frequency_weighted_coverage": 1.0,
                "unique_character_coverage": 1.0,
            }
            for language in PROJECT_LANGUAGES
        },
    }

    summary = coverage_summary(metrics)

    assert tuple(summary) == PROJECT_LANGUAGES
    assert all(row["token_length_p99"] == 200 for row in summary.values())


def test_micro_m2m100_forward_covers_every_language_direction() -> None:
    tokenizer = create_seed_tokenizer()
    mapping = build_language_mapping(tokenizer)

    result = micro_m2m100_forward(tokenizer, mapping)

    assert set(result["dimensions"].values()) == {len(tokenizer)}
    assert len(result["forwards"]) == len(PROJECT_LANGUAGES)
    assert all(row["loss"] > 0 for row in result["forwards"])
