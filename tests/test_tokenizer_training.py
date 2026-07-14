from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import tokenizer_utils  # noqa: E402
from tokenizer_utils import (  # noqa: E402
    PROJECT_LANGUAGES,
    LanguageAllowlist,
    build_language_mapping,
    create_seed_tokenizer,
    forced_bos_token_id,
    reload_tokenizer,
    verify_tokenizer,
)
from train_tokenizer import (  # noqa: E402
    BalancedBatchIterator,
    NativeTrainingHeartbeat,
    ProcessTreeMemorySnapshot,
    TrainingConfig,
    character_is_covered,
    evaluate_watchdog,
    load_balanced_corpus,
    train_candidate,
)


LANGUAGE_TEXT = {
    "eng_Latn": "The quick brown fox translates a deterministic sentence",
    "zho_Hans": "这是用于确定性分词器训练的中文测试句子",
    "jpn_Jpan": "これは決定的なトークナイザー訓練用の日本語文です",
    "kor_Hang": "이것은 결정적 토크나이저 학습을 위한 한국어 문장입니다",
}


def write_corpus_fixture(root: Path, *, repeats: int = 240) -> frozenset[str]:
    root.mkdir(parents=True)
    manifest_lines = []
    alphabet = {"▁"}
    for language, base in LANGUAGE_TEXT.items():
        lines = [f"{base} {index:04d} alpha beta gamma delta" for index in range(repeats)]
        payload = ("\n".join(lines) + "\n").encode("utf-8")
        path = root / f"{language}.txt"
        path.write_bytes(payload)
        characters = sum(len(line) for line in lines)
        manifest_lines.append(
            json.dumps(
                {
                    "language": language,
                    "file": path.name,
                    "bytes": len(payload),
                    "samples": len(lines),
                    "characters": characters,
                    "sha256": hashlib.sha256(payload).hexdigest(),
                },
                sort_keys=True,
            )
        )
        alphabet.update(character for character in "".join(lines) if not character.isspace())
    (root / "manifest.jsonl").write_text(
        "\n".join(manifest_lines) + "\n", encoding="utf-8"
    )
    return frozenset(alphabet)


def test_seed_tokenizer_language_contract() -> None:
    tokenizer = create_seed_tokenizer()
    verify_tokenizer(tokenizer, expected_vocab_size=10)
    assert build_language_mapping(tokenizer) == {
        "eng_Latn": 5,
        "zho_Hans": 6,
        "zho_Hant": 7,
        "jpn_Jpan": 8,
        "kor_Hang": 9,
    }
    assert forced_bos_token_id(tokenizer, "jpn_Jpan") == 8
    with pytest.raises(ValueError, match="not supported"):
        LanguageAllowlist().check("fra_Latn")


def test_verify_tokenizer_restores_src_lang_on_success_and_failure(monkeypatch) -> None:
    tokenizer = create_seed_tokenizer()
    tokenizer.src_lang = "jpn_Jpan"

    verify_tokenizer(tokenizer)
    assert tokenizer.src_lang == "jpn_Jpan"

    original_verify_backend_pipeline = tokenizer_utils.verify_backend_pipeline

    def fail_during_language_validation(tokenizer, *, expected_src_lang=None):
        if expected_src_lang == "zho_Hans":
            raise tokenizer_utils.TokenizerValidationError("injected validation failure")
        return original_verify_backend_pipeline(
            tokenizer, expected_src_lang=expected_src_lang
        )

    monkeypatch.setattr(
        tokenizer_utils, "verify_backend_pipeline", fail_during_language_validation
    )
    with pytest.raises(
        tokenizer_utils.TokenizerValidationError, match="injected validation failure"
    ):
        verify_tokenizer(tokenizer)
    assert tokenizer.src_lang == "jpn_Jpan"


def test_loading_is_deterministic_and_character_balanced(tmp_path: Path) -> None:
    corpus_dir = tmp_path / "corpus"
    write_corpus_fixture(corpus_dir)
    first = load_balanced_corpus(
        corpus_dir,
        seed=1234,
        sample_fraction=0.5,
        progress_interval_s=60,
    )
    second = load_balanced_corpus(
        corpus_dir,
        seed=1234,
        sample_fraction=0.5,
        progress_interval_s=60,
    )
    assert first.lines_by_language == second.lines_by_language
    character_totals = [
        item.training_characters for item in first.language_stats.values()
    ]
    longest_line = max(
        len(line) for lines in first.lines_by_language.values() for line in lines
    )
    assert max(character_totals) - min(character_totals) <= longest_line
    assert first.manifest_sha256 == second.manifest_sha256


def test_balanced_iterator_can_release_single_candidate_corpus(tmp_path: Path) -> None:
    corpus_dir = tmp_path / "corpus"
    write_corpus_fixture(corpus_dir, repeats=24)
    corpus = load_balanced_corpus(
        corpus_dir,
        seed=20260713,
        sample_fraction=1.0,
        progress_interval_s=60,
    )
    expected_lines = corpus.total_lines
    expected_characters = corpus.total_characters
    iterator = BalancedBatchIterator(
        corpus,
        batch_size=16,
        label="release-test",
        progress_interval_s=60,
        release_consumed_lines=True,
    )
    yielded = [line for batch in iterator for line in batch]
    assert len(yielded) == expected_lines
    assert corpus.lines_by_language == {}
    assert corpus.total_lines == expected_lines
    assert corpus.total_characters == expected_characters
    assert iterator.order_sha256


def test_train_save_reload_and_repeatability(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("RAYON_NUM_THREADS", raising=False)
    monkeypatch.delenv("TOKENIZERS_PARALLELISM", raising=False)
    corpus_dir = tmp_path / "corpus"
    alphabet = write_corpus_fixture(corpus_dir)
    corpus = load_balanced_corpus(
        corpus_dir,
        seed=20260713,
        sample_fraction=1.0,
        progress_interval_s=60,
    )
    config = TrainingConfig(
        vocab_size=256,
        min_frequency=2,
        limit_alphabet=None,
        num_threads=2,
        batch_size=32,
        heartbeat_interval_s=0,
        native_progress=False,
    )
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    staging = tmp_path / "staging"
    train_candidate(
        corpus,
        config,
        output_dir=first_dir,
        staging_dir=staging,
        initial_alphabet=alphabet,
    )
    train_candidate(
        corpus,
        config,
        output_dir=second_dir,
        staging_dir=staging,
        initial_alphabet=alphabet,
    )
    first_json = json.loads((first_dir / "tokenizer.json").read_text(encoding="utf-8"))
    second_json = json.loads((second_dir / "tokenizer.json").read_text(encoding="utf-8"))
    assert first_json == second_json
    tokenizer = reload_tokenizer(first_dir)
    assert len(tokenizer) == 256
    assert all(character_is_covered(tokenizer, character) for character in alphabet)
    assert tokenizer("测试", add_special_tokens=True)["input_ids"][0] == 5
    metadata = json.loads((first_dir / "training_meta.json").read_text(encoding="utf-8"))
    assert metadata["random_seed"] == 20260713
    assert metadata["input_order_sha256"]
    assert metadata["missing_corpus_characters"] == 0
    artifact_manifest = json.loads(
        (first_dir / "artifact_manifest.json").read_text(encoding="utf-8")
    )
    assert {item["path"] for item in artifact_manifest["files"]} >= {
        "tokenizer.json",
        "tokenizer_config.json",
        "language_map.json",
        "alphabet_audit.json",
        "training_meta.json",
        "corpus_manifest.jsonl",
    }


def test_native_heartbeat_is_newline_visible(capfd) -> None:
    with NativeTrainingHeartbeat("pytest-heartbeat", 0.1):
        time.sleep(0.7)
    captured = capfd.readouterr()
    assert "HEARTBEAT pytest-heartbeat: still running" in captured.out


def test_watchdog_uses_process_and_system_memory_thresholds() -> None:
    def tree(rss_gib: float) -> ProcessTreeMemorySnapshot:
        value = int(rss_gib * 1024**3)
        return ProcessTreeMemorySnapshot(
            process_ids=(123,),
            rss_bytes=value,
            private_bytes=value,
            root_rss_bytes=value,
            root_peak_rss_bytes=value,
            root_private_bytes=value,
        )

    process_stop = evaluate_watchdog(
        tree(80.1),
        available_bytes=40 * 1024**3,
        max_memory_gib=80,
        min_available_memory_gib=4,
        available_memory_warning_gib=16,
    )
    assert "process-tree RSS" in process_stop.stop_reason

    system_stop = evaluate_watchdog(
        tree(60),
        available_bytes=3 * 1024**3,
        max_memory_gib=80,
        min_available_memory_gib=4,
        available_memory_warning_gib=16,
    )
    assert "available RAM" in system_stop.stop_reason

    warning = evaluate_watchdog(
        tree(60),
        available_bytes=15 * 1024**3,
        max_memory_gib=80,
        min_available_memory_gib=4,
        available_memory_warning_gib=16,
    )
    assert warning.stop_reason is None
    assert warning.available_memory_warning is True


def test_cli_supervisor_emits_heartbeat(tmp_path: Path) -> None:
    corpus_dir = tmp_path / "corpus"
    alphabet = write_corpus_fixture(corpus_dir)
    memory_log = tmp_path / "memory.jsonl"
    # A small test-only alphabet is injected through the library tests; the CLI
    # uses the production alphabet, so only exercise its supervisor with --help
    # plus a deliberately slow worker import in a failing small-vocab run.
    environment = os.environ.copy()
    environment.pop("RAYON_NUM_THREADS", None)
    environment.pop("TOKENIZERS_PARALLELISM", None)
    result = subprocess.run(
        [
            sys.executable,
            "-u",
            str(ROOT / "scripts" / "train_tokenizer.py"),
            "--corpus-dir",
            str(corpus_dir),
            "--vocab-size",
            "256",
            "--output-dir",
            str(tmp_path / "output"),
            "--staging-dir",
            str(tmp_path / "staging"),
            "--heartbeat-interval",
            "0.1",
            "--load-progress-interval",
            "60",
            "--min-available-memory-gib",
            "0",
            "--available-memory-warning-gib",
            "0",
            "--watchdog-interval",
            "0.1",
            "--memory-log",
            str(memory_log),
            "--no-native-progress",
        ],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=30,
    )
    assert result.returncode != 0
    assert "SUPERVISOR heartbeat" in result.stdout
    assert "initial_alphabet" in result.stdout
    memory_records = [
        json.loads(line) for line in memory_log.read_text(encoding="utf-8").splitlines()
    ]
    assert any(record["event"] == "sample" for record in memory_records)
    summary = next(record for record in memory_records if record["event"] == "watchdog_summary")
    assert summary["samples"] > 0
    assert summary["sampled_peak_process_tree_rss_bytes"] > 0
