from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from build_micro_m2m100_checkpoint import (  # noqa: E402
    CheckpointBuildError,
    REPORT_PHASES,
    build_checkpoint,
    update_consolidated_report,
    validate_spec,
)
import tokenizer_utils  # noqa: E402


SPEC = ROOT / "configs" / "micro_m2m100_deployment.json"
TOKENIZER = ROOT / "artifacts" / "tokenizers" / "mvp-tokenizer-v0"


def test_consolidated_report_merges_phases(tmp_path: Path) -> None:
    output = tmp_path / "deployment-validation.json"

    partial = update_consolidated_report(
        output, {"td_01_hf_checkpoint": {"status": "passed", "value": 1}}
    )
    assert partial["status"] == "partial"

    remaining = {
        phase: {"status": "passed", "value": index}
        for index, phase in enumerate(REPORT_PHASES[1:], start=2)
    }
    complete = update_consolidated_report(output, remaining)

    assert complete["status"] == "passed"
    assert list(complete["phases"]) == list(REPORT_PHASES)
    assert complete["phases"]["td_01_hf_checkpoint"]["value"] == 1
    assert all("generated_at_utc" in record for record in complete["phases"].values())


def test_atomic_json_write_retries_transient_permission_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "record.json"
    original_replace = tokenizer_utils.os.replace
    attempts = 0

    def flaky_replace(source, target):  # type: ignore[no-untyped-def]
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise PermissionError("simulated transient file lock")
        original_replace(source, target)

    monkeypatch.setattr(tokenizer_utils.os, "replace", flaky_replace)
    monkeypatch.setattr(tokenizer_utils.time, "sleep", lambda _: None)

    tokenizer_utils.atomic_write_json(output, {"status": "passed"})

    assert attempts == 2
    assert json.loads(output.read_text(encoding="utf-8")) == {"status": "passed"}


def test_spec_rejects_untied_embeddings() -> None:
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    spec["model_config"]["tie_word_embeddings"] = False

    with pytest.raises(CheckpointBuildError, match="tied word embeddings"):
        validate_spec(spec)


def test_checkpoint_is_repeatable_and_offline_reloadable(tmp_path: Path) -> None:
    first = build_checkpoint(
        spec_path=SPEC,
        output_dir=tmp_path / "first",
        tokenizer_dir=TOKENIZER,
    )
    second = build_checkpoint(
        spec_path=SPEC,
        output_dir=tmp_path / "second",
        tokenizer_dir=TOKENIZER,
    )

    assert first["status"] == "passed"
    assert first["state_dict_sha256"] == second["state_dict_sha256"]
    assert first["checkpoint_manifest_sha256"] == second["checkpoint_manifest_sha256"]
    assert first["files"] == second["files"]
    assert set(first["dimensions"].values()) == {49_152, True}
    assert first["offline_model_reload"] is True
    assert first["offline_tokenizer_reload"] is True
    assert first["smoke_forward"]["logits_finite"] is True
    assert (tmp_path / "first" / "model.safetensors").is_file()
    assert (tmp_path / "first" / "tokenizer.json").is_file()
    assert (tmp_path / "first" / "checkpoint_manifest.json").is_file()


def test_checkpoint_refuses_to_replace_existing_output(tmp_path: Path) -> None:
    output = tmp_path / "checkpoint"
    output.mkdir()

    with pytest.raises(CheckpointBuildError, match="already exists"):
        build_checkpoint(
            spec_path=SPEC,
            output_dir=output,
            tokenizer_dir=TOKENIZER,
        )
