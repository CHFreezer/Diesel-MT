from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from build_micro_m2m100_checkpoint import build_checkpoint  # noqa: E402
from run_offline_ctranslate2_smoke import (  # noqa: E402
    OfflineSmokeError,
    verify_manifest as verify_offline_manifest,
)
from validate_ctranslate2_deployment import (  # noqa: E402
    DEPLOYMENT_MANIFEST,
    DeploymentValidationError,
    build_offline_package,
    convert_models,
    run_cpu_inference_smoke,
    validate_vocabulary_integrity,
    verify_payload_manifest,
)


SPEC = ROOT / "configs" / "micro_m2m100_deployment.json"
TOKENIZER = ROOT / "artifacts" / "tokenizers" / "mvp-tokenizer-v0"
OFFLINE_RUNNER = ROOT / "scripts" / "run_offline_ctranslate2_smoke.py"


def test_manifests_reject_path_traversal_and_incomplete_status(tmp_path: Path) -> None:
    package = tmp_path / "package"
    package.mkdir()
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"outside\n")
    manifest_path = package / DEPLOYMENT_MANIFEST
    traversal = {
        "schema_version": 1,
        "status": "complete",
        "files": [
            {
                "path": "../outside.bin",
                "bytes": outside.stat().st_size,
                "sha256": hashlib.sha256(outside.read_bytes()).hexdigest(),
            }
        ],
    }
    manifest_path.write_text(json.dumps(traversal), encoding="utf-8")

    with pytest.raises(DeploymentValidationError, match="invalid manifest path"):
        verify_payload_manifest(package, DEPLOYMENT_MANIFEST)
    with pytest.raises(OfflineSmokeError, match="invalid deployment manifest path"):
        verify_offline_manifest(package)

    manifest_path.write_text(
        json.dumps({"schema_version": 1, "status": "failed", "files": []}),
        encoding="utf-8",
    )
    with pytest.raises(DeploymentValidationError, match="manifest is not complete"):
        verify_payload_manifest(package, DEPLOYMENT_MANIFEST)
    with pytest.raises(OfflineSmokeError, match="manifest is not complete"):
        verify_offline_manifest(package)


@pytest.mark.slow
def test_full_ctranslate2_deployment_pipeline(tmp_path: Path) -> None:
    checkpoint = tmp_path / "hf-checkpoint"
    float32_dir = tmp_path / "ct2-float32"
    int8_dir = tmp_path / "ct2-int8"
    package_dir = tmp_path / "offline-package"

    checkpoint_record = build_checkpoint(
        spec_path=SPEC,
        output_dir=checkpoint,
        tokenizer_dir=TOKENIZER,
    )
    conversions = convert_models(
        checkpoint,
        float32_dir,
        int8_dir,
        overwrite=False,
    )
    repeated_conversions = convert_models(
        checkpoint,
        float32_dir,
        int8_dir,
        overwrite=True,
    )
    integrity = validate_vocabulary_integrity(
        TOKENIZER,
        checkpoint,
        float32_dir,
        int8_dir,
    )
    inference = run_cpu_inference_smoke(TOKENIZER, float32_dir, int8_dir)
    package = build_offline_package(
        TOKENIZER,
        int8_dir,
        package_dir,
        OFFLINE_RUNNER,
        overwrite=False,
    )

    assert checkpoint_record["status"] == "passed"
    assert conversions["models"]["float32"]["cpu_compute_type_actual"] == "float32"
    assert conversions["models"]["int8"]["cpu_compute_type_actual"] == "int8_float32"
    assert {
        label: model["conversion_manifest_sha256"]
        for label, model in conversions["models"].items()
    } == {
        label: model["conversion_manifest_sha256"]
        for label, model in repeated_conversions["models"].items()
    }
    assert conversions["input_checkpoint_state_dict_sha256"] == checkpoint_record[
        "state_dict_sha256"
    ]
    assert integrity["vocab_size"] == 49_152
    assert integrity["hf_tokenizer_exact_match"] is True
    assert {
        item["vocabulary_sequence_sha256"]
        for item in integrity["conversions"].values()
    } == {integrity["vocabulary_sequence_sha256"]}
    assert set(inference["source_languages_covered"]) == {
        "eng_Latn",
        "zho_Hans",
        "zho_Hant",
        "jpn_Jpan",
        "kor_Hang",
    }
    assert set(inference["target_languages_covered"]) == set(
        inference["source_languages_covered"]
    )
    assert all(
        len(model["cases"]) == 5 and model["compute_type_actual"]
        for model in inference["models"].values()
    )
    assert package["offline_subprocess"]["result"]["status"] == "passed"
    assert package["offline_subprocess"]["result"]["compute_type"] == "int8_float32"
    assert "socket" in package["offline_subprocess"]["result"]["network_guard"]

    (package_dir / "README.md").write_text("tampered\n", encoding="utf-8")
    with pytest.raises(DeploymentValidationError, match="byte count mismatch"):
        verify_payload_manifest(package_dir, DEPLOYMENT_MANIFEST)
