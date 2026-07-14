"""Canonical corpus snapshots and resumable Rust BPE trainer checkpoints.

Python remains the source of truth for sampling, balancing and multilingual
ordering.  It writes the exact strings produced by ``BalancedBatchIterator``
as length-prefixed UTF-8 records.  The pinned Rust helper then reuses the
official ``Tokenizer::train`` preprocessing path for ``feed`` and persists the
resulting trainer before the memory-intensive BPE merge stage.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from tokenizer_utils import PROJECT_LANGUAGES, atomic_write_json


ROOT = Path(__file__).resolve().parents[1]
HELPER_MANIFEST = ROOT / "tools" / "tokenizer-checkpoint" / "Cargo.toml"
CHECKPOINT_SCHEMA_VERSION = 2
SNAPSHOT_SCHEMA_VERSION = 1
SNAPSHOT_MAGIC = b"DMTTOKSNAPV1\0\0\0\0"


@dataclass(frozen=True)
class SnapshotSummary:
    path: Path
    records: int
    utf8_bytes: int
    input_order_sha256: str
    snapshot_sha256: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def write_length_prefixed_snapshot(
    batches: Iterable[str | Sequence[str]],
    output: Path,
    *,
    expected_records: int,
) -> SnapshotSummary:
    """Atomically persist the exact strings yielded by the Python corpus entry."""
    if expected_records <= 0:
        raise ValueError("expected_records must be positive")
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to replace existing snapshot: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    if temporary.exists():
        temporary.unlink()
    records = 0
    utf8_bytes = 0
    order_digest = hashlib.sha256()
    try:
        with temporary.open("xb") as handle:
            handle.write(SNAPSHOT_MAGIC)
            handle.write(expected_records.to_bytes(8, "big"))
            for batch in batches:
                values: Sequence[str] = (batch,) if isinstance(batch, str) else batch
                for text in values:
                    if not isinstance(text, str):
                        raise TypeError(
                            f"snapshot input must contain str, got {type(text).__name__}"
                        )
                    encoded = text.encode("utf-8")
                    length = len(encoded).to_bytes(8, "big")
                    handle.write(length)
                    handle.write(encoded)
                    order_digest.update(length)
                    order_digest.update(encoded)
                    records += 1
                    utf8_bytes += len(encoded)
            if records != expected_records:
                raise ValueError(
                    f"snapshot wrote {records:,} records, expected {expected_records:,}"
                )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return SnapshotSummary(
        path=output,
        records=records,
        utf8_bytes=utf8_bytes,
        input_order_sha256=order_digest.hexdigest(),
        snapshot_sha256=sha256_file(output),
    )


def inspect_length_prefixed_snapshot(path: Path) -> SnapshotSummary:
    """Validate an existing snapshot and reconstruct its content fingerprints."""
    path = path.resolve()
    order_digest = hashlib.sha256()
    records = 0
    utf8_bytes = 0
    with path.open("rb") as handle:
        if handle.read(len(SNAPSHOT_MAGIC)) != SNAPSHOT_MAGIC:
            raise ValueError(f"unsupported snapshot format: {path}")
        count_bytes = handle.read(8)
        if len(count_bytes) != 8:
            raise ValueError(f"truncated snapshot header: {path}")
        expected_records = int.from_bytes(count_bytes, "big")
        while True:
            length = handle.read(8)
            if not length:
                break
            if len(length) != 8:
                raise ValueError(f"truncated record length after record {records:,}")
            size = int.from_bytes(length, "big")
            encoded = handle.read(size)
            if len(encoded) != size:
                raise ValueError(f"truncated snapshot record {records + 1:,}")
            encoded.decode("utf-8")
            order_digest.update(length)
            order_digest.update(encoded)
            records += 1
            utf8_bytes += size
    if records != expected_records:
        raise ValueError(
            f"snapshot contains {records:,} records, header declares {expected_records:,}"
        )
    return SnapshotSummary(
        path=path,
        records=records,
        utf8_bytes=utf8_bytes,
        input_order_sha256=order_digest.hexdigest(),
        snapshot_sha256=sha256_file(path),
    )


def canonical_sha256(payload: Mapping | Sequence) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def prepare_backend_and_trainer(
    seed_tokenizer: object,
    *,
    vocab_size: int,
    min_frequency: int,
    limit_alphabet: int,
    initial_alphabet: Sequence[str],
    seed_backend_path: Path,
    trainer_config_path: Path,
    show_progress: bool,
) -> tuple[dict, dict]:
    """Create the same empty backend/trainer state used by Transformers."""
    backend = json.loads(seed_tokenizer.backend_tokenizer.to_str())
    added_tokens = backend.pop("added_tokens")
    backend["post_processor"] = None
    model = backend.get("model") or {}
    if model.get("type") != "BPE":
        raise ValueError(f"checkpoint path only supports BPE, got {model.get('type')!r}")
    model["vocab"] = {}
    model["merges"] = []

    special_tokens: list[dict] = []
    for original in added_tokens:
        record = dict(original)
        record.pop("id", None)
        if record.get("special", False):
            special_tokens.append(record)
    trainer_state = {
        "type": "BPE",
        "vocab_size": vocab_size,
        "min_frequency": min_frequency,
        "show_progress": show_progress,
        "special_tokens": special_tokens,
        "limit_alphabet": limit_alphabet,
        "initial_alphabet": list(initial_alphabet),
        "continuing_subword_prefix": model.get("continuing_subword_prefix"),
        "end_of_word_suffix": model.get("end_of_word_suffix"),
        "max_token_length": None,
    }
    atomic_write_json(seed_backend_path, backend)
    atomic_write_json(trainer_config_path, trainer_state)
    return backend, trainer_state


def helper_binary_path(target_dir: Path | None = None) -> Path:
    target = (
        target_dir.resolve()
        if target_dir is not None
        else HELPER_MANIFEST.parent / "target"
    )
    suffix = ".exe" if os.name == "nt" else ""
    return target / "release" / f"diesel-mt-tokenizer-checkpoint{suffix}"


def ensure_helper(
    *,
    target_dir: Path | None = None,
    cargo: str = "cargo",
) -> Path:
    """Build the pinned helper when its release binary is absent."""
    binary = helper_binary_path(target_dir)
    if binary.is_file():
        return binary
    environment = os.environ.copy()
    if target_dir is not None:
        environment["CARGO_TARGET_DIR"] = str(target_dir.resolve())
    subprocess.run(
        [
            cargo,
            "build",
            "--locked",
            "--release",
            "--manifest-path",
            str(HELPER_MANIFEST),
        ],
        cwd=ROOT,
        env=environment,
        check=True,
    )
    if not binary.is_file():
        raise RuntimeError(f"cargo reported success but helper is absent: {binary}")
    return binary


def run_helper(
    binary: Path,
    arguments: Sequence[str | Path],
    *,
    num_threads: int,
) -> dict:
    environment = os.environ.copy()
    environment["RAYON_NUM_THREADS"] = str(num_threads)
    environment["TOKENIZERS_PARALLELISM"] = "true"
    result = subprocess.run(
        [str(binary), *(str(argument) for argument in arguments)],
        cwd=ROOT,
        env=environment,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=True,
        stdout=subprocess.PIPE,
    )
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError("checkpoint helper returned no JSON summary")
    return json.loads(lines[-1])


def feed_checkpoint(
    binary: Path,
    *,
    seed_backend: Path,
    trainer_config: Path,
    snapshot: Path,
    output: Path,
    num_threads: int,
) -> dict:
    return run_helper(
        binary,
        [
            "feed",
            "--tokenizer",
            seed_backend,
            "--trainer-config",
            trainer_config,
            "--snapshot",
            snapshot,
            "--output",
            output,
        ],
        num_threads=num_threads,
    )


def inspect_checkpoint(
    binary: Path,
    *,
    checkpoint: Path,
    num_threads: int,
) -> dict:
    return run_helper(
        binary,
        ["inspect", "--checkpoint", checkpoint],
        num_threads=num_threads,
    )


def train_from_checkpoint(
    binary: Path,
    *,
    seed_backend: Path,
    checkpoint: Path,
    output_backend: Path,
    vocab_size: int,
    min_frequency: int,
    limit_alphabet: int,
    num_threads: int,
) -> dict:
    return run_helper(
        binary,
        [
            "train",
            "--tokenizer",
            seed_backend,
            "--checkpoint",
            checkpoint,
            "--output",
            output_backend,
            "--vocab-size",
            str(vocab_size),
            "--min-frequency",
            str(min_frequency),
            "--limit-alphabet",
            str(limit_alphabet),
        ],
        num_threads=num_threads,
    )


def nllb_from_trained_backend(path: Path, *, src_lang: str = "eng_Latn") -> object:
    """Attach the project NLLB wrapper and a fresh ID-correct post-processor."""
    from tokenizers import Tokenizer
    from transformers import NllbTokenizer

    backend = Tokenizer.from_file(str(path))
    return NllbTokenizer(
        tokenizer_object=backend,
        extra_special_tokens=list(PROJECT_LANGUAGES),
        src_lang=src_lang,
        legacy_behaviour=False,
    )
