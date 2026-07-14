from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_tokenizers import (  # noqa: E402
    LONG_CHARACTER_THRESHOLD,
    _evaluate_text,
    build_evaluation_sample_set,
    evaluate_tokenizer,
    load_evaluation_samples,
    render_candidate_report,
)
from tokenizer_utils import TRAINING_LANGUAGES, create_seed_tokenizer  # noqa: E402


def write_evaluation_corpus(root: Path, *, lines_per_language: int = 40) -> None:
    root.mkdir(parents=True)
    manifest = []
    for language in TRAINING_LANGUAGES:
        lines = []
        for index in range(lines_per_language):
            suffix = " long" * 120 if index % 5 == 0 else " short"
            lines.append(f"{language} deterministic evaluation row {index:04d}{suffix}")
        payload = ("\n".join(lines) + "\n").encode("utf-8")
        path = root / f"{language}.txt"
        path.write_bytes(payload)
        manifest.append(
            {
                "language": language,
                "file": path.name,
                "bytes": len(payload),
                "samples": len(lines),
                "characters": sum(len(line) for line in lines),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "split": "holdout",
            }
        )
    (root / "manifest.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in manifest),
        encoding="utf-8",
    )


def test_fixed_sample_set_is_reproducible_and_has_long_quota(tmp_path: Path) -> None:
    corpus_dir = tmp_path / "corpus"
    sample_dir = tmp_path / "samples"
    write_evaluation_corpus(corpus_dir)
    first = build_evaluation_sample_set(
        corpus_dir,
        sample_dir,
        seed=123,
        sample_size=12,
        long_quota=3,
    )
    first_payloads = {
        language: (sample_dir / f"{language}.jsonl").read_bytes()
        for language in TRAINING_LANGUAGES
    }
    second = build_evaluation_sample_set(
        corpus_dir,
        sample_dir,
        seed=123,
        sample_size=12,
        long_quota=3,
        rebuild=True,
    )
    assert first == second
    samples = load_evaluation_samples(sample_dir, second)
    for language, rows in samples.items():
        corpus_rows = [row for row in rows if row["source"] == "corpus"]
        assert len(corpus_rows) == 12
        assert sum(len(row["text"]) > LONG_CHARACTER_THRESHOLD for row in corpus_rows) >= 3
        assert len({row["text_sha256"] for row in corpus_rows}) == 12
        assert (sample_dir / f"{language}.jsonl").read_bytes() == first_payloads[language]


def test_fused_unknown_counts_every_source_character() -> None:
    tokenizer = create_seed_tokenizer()
    row = {
        "sample_id": "fused-unknown",
        "text": "𠀀𪚥",
    }
    item, unknown_positions, ids = _evaluate_text(tokenizer, row)
    assert item["unk_tokens"] == 1
    assert item["unk_source_characters"] == 2
    assert item["unmapped_unk_tokens"] == 0
    assert unknown_positions == {0, 1}
    assert ids == [tokenizer.unk_token_id]


def test_evaluation_separates_corpus_and_stress_probes(tmp_path: Path) -> None:
    tokenizer = create_seed_tokenizer()
    samples = {
        language: [
            {
                "sample_id": f"{language}-corpus",
                "language": language,
                "source": "corpus",
                "category": "corpus_random",
                "text": "Shared 漢字 test",
            },
            {
                "sample_id": f"{language}-empty",
                "language": language,
                "source": "synthetic_probe",
                "category": "empty",
                "text": "",
            },
        ]
        for language in TRAINING_LANGUAGES
    }
    metrics = evaluate_tokenizer(
        tokenizer,
        samples,
        label="seed",
        artifact_path=tmp_path,
        training_metadata={"sample_fraction": 0.1},
        sample_manifest={
            "sample_dir": str(tmp_path),
            "corpus_manifest_sha256": "fixture",
            "seed": 123,
            "corpus_samples_per_language": 1,
            "long_quota_per_language": 0,
        },
    )
    assert metrics["total_corpus_metrics"]["samples"] == len(TRAINING_LANGUAGES)
    assert metrics["all_sample_metrics"]["eng_Latn"]["samples"] == 2
    assert metrics["category_metrics"]["eng_Latn"]["empty"]["samples"] == 1
    assert metrics["total_corpus_metrics"]["source_character_loss_rate"] > 0
    report = render_candidate_report(metrics)
    assert "Sampled-corpus result" in report
    assert "Synthetic stress probes" in report
