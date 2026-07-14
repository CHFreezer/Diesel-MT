#!/usr/bin/env python3
"""Train Diesel-MT tokenizers through a resumable Rust feed checkpoint.

Python owns the existing corpus validation, seeded sampling, character-budget
balancing and multilingual round-robin order.  A canonical snapshot on the D:
SSD is fed once through the official Rust Tokenizer preprocessing path.  The
resulting BPE trainer is immutable cache state: interrupted merge training can
restart from it without loading or passing the corpus again.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from tokenizer_checkpoint import (  # noqa: E402
    SNAPSHOT_SCHEMA_VERSION,
    canonical_sha256,
    ensure_helper,
    feed_checkpoint,
    inspect_checkpoint,
    inspect_length_prefixed_snapshot,
    nllb_from_trained_backend,
    prepare_backend_and_trainer,
    sha256_file,
    train_from_checkpoint,
    write_length_prefixed_snapshot,
)
from tokenizer_utils import (  # noqa: E402
    PROJECT_LANGUAGES,
    TRAINING_LANGUAGES,
    atomic_write_json,
    backend_pipeline,
    build_language_mapping,
    create_seed_tokenizer,
    reload_tokenizer,
    save_language_mapping,
    verify_tokenizer,
)
from train_tokenizer import (  # noqa: E402
    BALANCING_ALGORITHM,
    CORPUS_MANIFEST_NAME,
    MUST_COVER_ALPHABET,
    SAMPLING_ALGORITHM,
    VOCAB_CANDIDATE_NAMES,
    BalancedBatchIterator,
    build_alphabet_audit,
    configure_tokenizer_threads,
    effective_alphabet_limit,
    load_balanced_corpus,
    progress,
    publish_artifact,
    supervise_worker,
    write_artifact_manifest,
)


STATE_SCHEMA_VERSION = 1
CORPUS_STATE_SCHEMA_VERSION = 1
FEED_TEMPLATE_VOCAB_SIZE = 32768
FEED_TEMPLATE_MIN_FREQUENCY = 1


@dataclass(frozen=True)
class StatePaths:
    root: Path
    manifest: Path
    corpus_state: Path
    corpus_manifest: Path
    snapshot: Path
    seed_backend: Path
    trainer_config: Path
    checkpoint: Path
    native_target: Path
    staging: Path


def state_paths(root: Path) -> StatePaths:
    root = root.resolve()
    return StatePaths(
        root=root,
        manifest=root / "state-manifest.json",
        corpus_state=root / "corpus-state.json",
        corpus_manifest=root / CORPUS_MANIFEST_NAME,
        snapshot=root / "balanced-corpus.snapshot",
        seed_backend=root / "seed-backend.json",
        trainer_config=root / "trainer-config.json",
        checkpoint=root / "bpe-feed-state.bin",
        native_target=root / "native-target",
        staging=root / "staging",
    )


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def initial_alphabet_sha256() -> str:
    encoded = "".join(sorted(MUST_COVER_ALPHABET, key=ord)).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def default_state_dir() -> Path:
    if os.name == "nt":
        return Path(r"D:\Diesel-MT-tokenizer-stage\checkpoints\mvp-tokenizer-full")
    return Path(tempfile.gettempdir()) / "diesel-mt-tokenizer-checkpoints" / "mvp-full"


def state_config(args: argparse.Namespace, seed_tokenizer: object) -> dict:
    source_manifest = args.corpus_dir.resolve() / CORPUS_MANIFEST_NAME
    if not source_manifest.is_file():
        raise FileNotFoundError(f"corpus manifest not found: {source_manifest}")
    import tokenizers

    payload = {
        "schema_version": STATE_SCHEMA_VERSION,
        "snapshot_schema_version": SNAPSHOT_SCHEMA_VERSION,
        "corpus_manifest_sha256": sha256_file(source_manifest),
        "seed": args.seed,
        "sample_fraction": args.sample_fraction,
        "sampling_algorithm": SAMPLING_ALGORITHM,
        "balancing_algorithm": BALANCING_ALGORITHM,
        "initial_alphabet_size": len(MUST_COVER_ALPHABET),
        "initial_alphabet_sha256": initial_alphabet_sha256(),
        "seed_backend_pipeline": backend_pipeline(seed_tokenizer),
        "tokenizers_version": tokenizers.__version__,
    }
    payload["fingerprint"] = canonical_sha256(payload)
    return payload


def validate_existing_state(state: Mapping, expected_config: Mapping) -> None:
    actual = state.get("config")
    if actual != expected_config:
        actual_fingerprint = actual.get("fingerprint") if isinstance(actual, dict) else None
        raise RuntimeError(
            "checkpoint state belongs to a different corpus/pipeline configuration: "
            f"{actual_fingerprint!r} != {expected_config['fingerprint']!r}"
        )


def validate_checkpoint_summary(state: Mapping, actual: Mapping) -> None:
    expected = state.get("checkpoint")
    if not isinstance(expected, dict):
        raise RuntimeError("state manifest has no checkpoint summary")
    for key in (
        "schema_version",
        "tokenizers_engine_version",
        "tokenizer_sha256",
        "trainer_config_sha256",
        "snapshot_sha256",
        "input_order_sha256",
        "input_records",
        "input_utf8_bytes",
    ):
        if actual.get(key) != expected.get(key):
            raise RuntimeError(
                f"checkpoint summary mismatch for {key}: "
                f"{actual.get(key)!r} != {expected.get(key)!r}"
            )


def write_corpus_state(corpus: object, paths: StatePaths) -> dict:
    total_lines = corpus.total_lines
    total_characters = corpus.total_characters
    payload = {
        "schema_version": CORPUS_STATE_SCHEMA_VERSION,
        "seed": corpus.seed,
        "sample_fraction": corpus.sample_fraction,
        "manifest_sha256": corpus.manifest_sha256,
        "total_lines": total_lines,
        "total_characters": total_characters,
        "character_counts": {
            character: int(count)
            for character, count in sorted(corpus.character_counts.items(), key=lambda item: ord(item[0]))
        },
        "language_stats": {
            language: asdict(corpus.language_stats[language])
            for language in TRAINING_LANGUAGES
        },
        "load_elapsed_s": corpus.load_elapsed_s,
    }
    atomic_write_json(paths.corpus_state, payload)
    shutil.copy2(corpus.manifest_path, paths.corpus_manifest)
    return payload


def snapshot_payload(summary: object) -> dict:
    return {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "path": str(summary.path),
        "records": summary.records,
        "utf8_bytes": summary.utf8_bytes,
        "input_order_sha256": summary.input_order_sha256,
        "snapshot_sha256": summary.snapshot_sha256,
        "bytes": summary.path.stat().st_size,
    }


def finish_feed_checkpoint(
    args: argparse.Namespace,
    paths: StatePaths,
    state: dict,
) -> dict:
    helper = ensure_helper(target_dir=paths.native_target)
    if paths.checkpoint.is_file():
        checkpoint_summary = inspect_checkpoint(
            helper,
            checkpoint=paths.checkpoint,
            num_threads=args.num_threads,
        )
    else:
        progress(f"FEED starting Rust checkpoint build from {paths.snapshot}")
        checkpoint_summary = feed_checkpoint(
            helper,
            seed_backend=paths.seed_backend,
            trainer_config=paths.trainer_config,
            snapshot=paths.snapshot,
            output=paths.checkpoint,
            num_threads=args.num_threads,
        )
    snapshot = state["snapshot"]
    comparisons = {
        "snapshot_sha256": snapshot["snapshot_sha256"],
        "input_order_sha256": snapshot["input_order_sha256"],
        "input_records": snapshot["records"],
        "input_utf8_bytes": snapshot["utf8_bytes"],
    }
    for key, expected in comparisons.items():
        if checkpoint_summary.get(key) != expected:
            raise RuntimeError(
                f"Rust checkpoint disagrees with Python snapshot for {key}: "
                f"{checkpoint_summary.get(key)!r} != {expected!r}"
            )
    state = {
        **state,
        "status": "feed_complete",
        "checkpoint": checkpoint_summary,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    atomic_write_json(paths.manifest, state)
    progress(
        f"FEED checkpoint ready: records={checkpoint_summary['input_records']:,}, "
        f"checkpoint={paths.checkpoint}"
    )
    return state


def prepare_state(args: argparse.Namespace) -> dict:
    paths = state_paths(args.state_dir)
    paths.root.mkdir(parents=True, exist_ok=True)
    paths.staging.mkdir(parents=True, exist_ok=True)
    seed_tokenizer = create_seed_tokenizer()
    config = state_config(args, seed_tokenizer)
    state = read_json(paths.manifest) if paths.manifest.is_file() else None
    if state is not None:
        validate_existing_state(state, config)
        status = state.get("status")
        if status == "feed_complete":
            helper = ensure_helper(target_dir=paths.native_target)
            actual = inspect_checkpoint(
                helper,
                checkpoint=paths.checkpoint,
                num_threads=args.num_threads,
            )
            validate_checkpoint_summary(state, actual)
            progress(f"FEED cache hit: {paths.checkpoint}")
            return state
        if status == "snapshot_ready":
            return finish_feed_checkpoint(args, paths, state)
        if status == "building_snapshot" and paths.snapshot.is_file():
            progress("SNAPSHOT recovering metadata from completed snapshot")
            summary = inspect_length_prefixed_snapshot(paths.snapshot)
            state = {**state, "status": "snapshot_ready", "snapshot": snapshot_payload(summary)}
            atomic_write_json(paths.manifest, state)
            return finish_feed_checkpoint(args, paths, state)
        if status != "building_snapshot":
            raise RuntimeError(f"unsupported checkpoint state status: {status!r}")

    if state is None:
        occupied = [
            path
            for path in (
                paths.corpus_state,
                paths.corpus_manifest,
                paths.snapshot,
                paths.seed_backend,
                paths.trainer_config,
                paths.checkpoint,
            )
            if path.exists()
        ]
        if occupied:
            raise RuntimeError(
                "state directory contains files without a state manifest; refusing to "
                f"guess ownership: {occupied}"
            )
        feed_limit_alphabet = len(MUST_COVER_ALPHABET) + 2048
        prepare_backend_and_trainer(
            seed_tokenizer,
            vocab_size=FEED_TEMPLATE_VOCAB_SIZE,
            min_frequency=FEED_TEMPLATE_MIN_FREQUENCY,
            limit_alphabet=feed_limit_alphabet,
            initial_alphabet=sorted(MUST_COVER_ALPHABET, key=ord),
            seed_backend_path=paths.seed_backend,
            trainer_config_path=paths.trainer_config,
            show_progress=not args.no_native_progress,
        )
        state = {
            "schema_version": STATE_SCHEMA_VERSION,
            "status": "building_snapshot",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "config": config,
            "seed_backend_sha256": sha256_file(paths.seed_backend),
            "trainer_config_sha256": sha256_file(paths.trainer_config),
        }
        atomic_write_json(paths.manifest, state)
    else:
        for path in (paths.seed_backend, paths.trainer_config):
            if not path.is_file():
                raise RuntimeError(f"incomplete checkpoint state is missing {path}")

    corpus = load_balanced_corpus(
        args.corpus_dir,
        seed=args.seed,
        sample_fraction=args.sample_fraction,
        progress_interval_s=args.load_progress_interval,
        max_memory_gib=args.max_memory_gib,
        min_available_memory_gib=args.min_available_memory_gib,
    )
    corpus_state = write_corpus_state(corpus, paths)
    iterator = BalancedBatchIterator(
        corpus,
        batch_size=args.batch_size,
        label="checkpoint-feed",
        progress_interval_s=args.heartbeat_interval,
        max_memory_gib=args.max_memory_gib,
        min_available_memory_gib=args.min_available_memory_gib,
        release_consumed_lines=True,
    )
    progress(f"SNAPSHOT writing canonical Python corpus entry to {paths.snapshot}")
    summary = write_length_prefixed_snapshot(
        iterator,
        paths.snapshot,
        expected_records=corpus_state["total_lines"],
    )
    if iterator.order_sha256 != summary.input_order_sha256:
        raise RuntimeError(
            "BalancedBatchIterator order hash disagrees with the snapshot writer: "
            f"{iterator.order_sha256} != {summary.input_order_sha256}"
        )
    state = {
        **state,
        "status": "snapshot_ready",
        "corpus_state_sha256": sha256_file(paths.corpus_state),
        "copied_corpus_manifest_sha256": sha256_file(paths.corpus_manifest),
        "snapshot": snapshot_payload(summary),
    }
    atomic_write_json(paths.manifest, state)
    progress(
        f"SNAPSHOT complete: records={summary.records:,}, "
        f"utf8_bytes={summary.utf8_bytes:,}, sha256={summary.snapshot_sha256}"
    )
    return finish_feed_checkpoint(args, paths, state)


def load_complete_state(args: argparse.Namespace) -> tuple[StatePaths, dict, dict]:
    paths = state_paths(args.state_dir)
    if not paths.manifest.is_file():
        raise RuntimeError(f"checkpoint state manifest not found: {paths.manifest}")
    state = read_json(paths.manifest)
    if state.get("status") != "feed_complete":
        raise RuntimeError(
            f"checkpoint state is not feed_complete: {state.get('status')!r}"
        )
    helper = ensure_helper(target_dir=paths.native_target)
    actual = inspect_checkpoint(
        helper,
        checkpoint=paths.checkpoint,
        num_threads=args.num_threads,
    )
    validate_checkpoint_summary(state, actual)
    corpus_state = read_json(paths.corpus_state)
    if sha256_file(paths.corpus_state) != state["corpus_state_sha256"]:
        raise RuntimeError("corpus-state.json fingerprint does not match state manifest")
    if state["config"]["initial_alphabet_sha256"] != initial_alphabet_sha256():
        raise RuntimeError("current must-cover alphabet differs from checkpoint state")
    return paths, state, corpus_state


def candidate_output_path(output_root: Path, vocab_size: int, multiple: bool) -> Path:
    if not multiple:
        return output_root
    suffix = VOCAB_CANDIDATE_NAMES.get(vocab_size, f"vocab-{vocab_size}")
    return output_root / suffix


def train_candidates(args: argparse.Namespace) -> None:
    if args.output_dir is None:
        raise ValueError("--output-dir is required for phase train/all")
    paths, state, corpus_state = load_complete_state(args)
    helper = ensure_helper(target_dir=paths.native_target)
    output_root = args.output_dir.resolve()
    staging_root = (
        args.staging_dir.resolve() if args.staging_dir is not None else paths.staging
    )
    staging_root.mkdir(parents=True, exist_ok=True)
    character_counts = Counter(
        {str(character): int(count) for character, count in corpus_state["character_counts"].items()}
    )
    vocab_sizes = list(dict.fromkeys(args.vocab_size))
    multiple = len(vocab_sizes) > 1

    import tokenizers
    import transformers

    for vocab_size in vocab_sizes:
        alphabet_limit = effective_alphabet_limit(
            vocab_size,
            args.limit_alphabet,
            len(MUST_COVER_ALPHABET),
        )
        label = f"vocab-{vocab_size}"
        output_dir = candidate_output_path(output_root, vocab_size, multiple)
        working = Path(tempfile.mkdtemp(prefix=f"checkpoint-{label}-", dir=staging_root))
        artifact = working / "artifact"
        artifact.mkdir()
        backend_path = working / "trained-backend.json"
        started = time.perf_counter()
        try:
            progress(f"TRAIN {label} resuming from {paths.checkpoint}")
            native_summary = train_from_checkpoint(
                helper,
                seed_backend=paths.seed_backend,
                checkpoint=paths.checkpoint,
                output_backend=backend_path,
                vocab_size=vocab_size,
                min_frequency=args.min_frequency,
                limit_alphabet=alphabet_limit,
                num_threads=args.num_threads,
            )
            tokenizer = nllb_from_trained_backend(backend_path)
            verify_tokenizer(tokenizer, expected_vocab_size=vocab_size)
            alphabet_audit = build_alphabet_audit(
                tokenizer,
                character_counts,
                MUST_COVER_ALPHABET,
            )
            if alphabet_audit["missing_initial_alphabet"]:
                missing = alphabet_audit["missing_initial_alphabet"]
                raise RuntimeError(
                    f"{len(missing)} must-cover characters encode as <unk>: {missing[:20]!r}"
                )
            tokenizer.src_lang = "eng_Latn"
            tokenizer.save_pretrained(str(artifact))
            shutil.copy2(paths.corpus_manifest, artifact / "corpus_manifest.jsonl")
            language_mapping = build_language_mapping(tokenizer)
            save_language_mapping(language_mapping, artifact / "language_map.json")
            atomic_write_json(artifact / "alphabet_audit.json", alphabet_audit)
            metadata = {
                "schema_version": 2,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "training_mode": "rust-feed-checkpoint-v1",
                "vocab_size": vocab_size,
                "min_frequency": args.min_frequency,
                "limit_alphabet": alphabet_limit,
                "num_threads": args.num_threads,
                "batch_size": args.batch_size,
                "tokenizers_version": tokenizers.__version__,
                "transformers_version": transformers.__version__,
                "checkpoint_state": str(paths.root),
                "checkpoint": state["checkpoint"],
                "native_train": native_summary,
                "total_training_lines": corpus_state["total_lines"],
                "total_training_characters": corpus_state["total_characters"],
                "corpus_unique_characters": len(character_counts),
                "languages": corpus_state["language_stats"],
                "language_token_ids": language_mapping,
                "backend": backend_pipeline(tokenizer),
                "elapsed_s": time.perf_counter() - started,
            }
            atomic_write_json(artifact / "training_meta.json", metadata)
            artifact_manifest = write_artifact_manifest(artifact)
            publish_artifact(artifact, output_dir, artifact_manifest)
        finally:
            shutil.rmtree(working, ignore_errors=True)
        final_tokenizer = reload_tokenizer(output_dir)
        verify_tokenizer(final_tokenizer, expected_vocab_size=vocab_size)
        progress(f"DONE {label}: artifact={output_dir}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("prepare", "train", "all"), default="all")
    parser.add_argument(
        "--corpus-dir",
        type=Path,
        default=Path("data/tokenizer/corpus/mvp"),
    )
    parser.add_argument("--state-dir", type=Path, default=default_state_dir())
    parser.add_argument("--vocab-size", action="append", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--staging-dir", type=Path, default=None)
    parser.add_argument("--sample-fraction", type=float, default=1.0)
    parser.add_argument("--min-frequency", type=int, default=2)
    parser.add_argument("--limit-alphabet", type=int, default=None)
    parser.add_argument("--num-threads", type=int, default=min(16, os.cpu_count() or 16))
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--heartbeat-interval", type=float, default=10.0)
    parser.add_argument("--load-progress-interval", type=float, default=10.0)
    parser.add_argument("--max-memory-gib", type=float, default=80.0)
    parser.add_argument("--min-available-memory-gib", type=float, default=4.0)
    parser.add_argument("--available-memory-warning-gib", type=float, default=16.0)
    parser.add_argument("--watchdog-interval", type=float, default=1.0)
    parser.add_argument("--memory-log", type=Path, default=None)
    parser.add_argument("--no-native-progress", action="store_true")
    parser.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.memory_log is None:
        args.memory_log = args.state_dir / "logs" / f"{args.phase}-memory.jsonl"
    return args


def validate_args(args: argparse.Namespace) -> None:
    if not 0 < args.sample_fraction <= 1:
        raise ValueError("sample_fraction must be in (0, 1]")
    if any(size <= 0 for size in args.vocab_size):
        raise ValueError("all vocabulary sizes must be positive")
    if args.min_frequency < 1:
        raise ValueError("min_frequency must be at least 1")
    if args.batch_size < len(TRAINING_LANGUAGES):
        raise ValueError(f"batch_size must be at least {len(TRAINING_LANGUAGES)}")
    if args.phase in {"train", "all"} and args.output_dir is None:
        raise ValueError("--output-dir is required for phase train/all")


def run_worker(args: argparse.Namespace) -> int:
    validate_args(args)
    configure_tokenizer_threads(args.num_threads)
    progress("=== Diesel-MT checkpointed tokenizer training ===")
    progress(
        f"CONFIG phase={args.phase}, state={args.state_dir.resolve()}, "
        f"vocab_sizes={list(dict.fromkeys(args.vocab_size))}, threads={args.num_threads}"
    )
    if args.phase in {"prepare", "all"}:
        prepare_state(args)
    if args.phase in {"train", "all"}:
        train_candidates(args)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    raw_args = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(raw_args)
    if args._worker:
        return run_worker(args)
    return supervise_worker(raw_args, args, worker_script=Path(__file__).resolve())


if __name__ == "__main__":
    raise SystemExit(main())
