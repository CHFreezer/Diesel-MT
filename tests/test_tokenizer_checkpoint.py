from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from tokenizer_checkpoint import (  # noqa: E402
    SNAPSHOT_MAGIC,
    inspect_length_prefixed_snapshot,
    write_length_prefixed_snapshot,
)
from train_tokenizer_checkpointed import training_provenance_from_state  # noqa: E402


def test_length_prefixed_snapshot_preserves_exact_python_strings(tmp_path: Path) -> None:
    batches = [
        [" leading space", "trailing space ", "中文"],
        ["日本語", "한국어", "embedded\rcharacter"],
    ]
    output = tmp_path / "corpus.snapshot"
    summary = write_length_prefixed_snapshot(
        batches,
        output,
        expected_records=6,
    )
    inspected = inspect_length_prefixed_snapshot(output)

    expected_digest = hashlib.sha256()
    for batch in batches:
        for text in batch:
            encoded = text.encode("utf-8")
            expected_digest.update(len(encoded).to_bytes(8, "big"))
            expected_digest.update(encoded)

    assert output.read_bytes().startswith(SNAPSHOT_MAGIC + (6).to_bytes(8, "big"))
    assert summary == inspected
    assert summary.records == 6
    assert summary.input_order_sha256 == expected_digest.hexdigest()


def test_snapshot_count_mismatch_is_not_published(tmp_path: Path) -> None:
    output = tmp_path / "corpus.snapshot"
    with pytest.raises(ValueError, match="expected 3"):
        write_length_prefixed_snapshot(
            [["one", "two"]],
            output,
            expected_records=3,
        )
    assert not output.exists()


def test_training_provenance_comes_from_checkpoint_state() -> None:
    snapshot = {"records": 42, "snapshot_sha256": "snapshot-hash"}
    state = {
        "config": {
            "seed": 20260713,
            "sample_fraction": 0.5,
            "sampling_algorithm": "seeded-sampling-v1",
            "balancing_algorithm": "character-budget-v1",
            "fingerprint": "state-fingerprint",
        },
        "snapshot": snapshot,
    }

    provenance = training_provenance_from_state(state)

    assert provenance == {
        "seed": 20260713,
        "sample_fraction": 0.5,
        "sampling_algorithm": "seeded-sampling-v1",
        "balancing_algorithm": "character-budget-v1",
        "checkpoint_config_fingerprint": "state-fingerprint",
        "snapshot": snapshot,
    }
    assert provenance["snapshot"] is not snapshot
