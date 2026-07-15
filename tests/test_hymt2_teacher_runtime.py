from __future__ import annotations

import copy
import hashlib
import socket
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from hymt2_teacher_runtime import (  # noqa: E402
    TeacherRuntimeError,
    blocked_network,
    resolve_runtime_root,
    validate_contract,
    verify_snapshot,
)


PROFILE = {
    "schema_version": 1,
    "identity": {
        "source_model": "tencent/Hy-MT2-7B",
        "selected_artifact": "tencent/Hy-MT2-7B-FP8",
        "selected_revision": "a" * 40,
        "artifact_format": "compressed-tensors-fp8",
        "backend": "transformers",
        "license": "Apache-2.0",
    },
    "runtime": {
        "default_root": "artifacts/model-training/runtime/teacher/test",
        "override_env": "DIESEL_MT_MODEL_RUNTIME",
        "override_subdir": "teacher/test",
        "snapshot_subdir": "snapshot",
        "overlay_subdir": "venv",
        "reports_subdir": "reports",
    },
    "environment": {},
    "loading": {"local_files_only": True, "trust_remote_code": False},
    "smoke": {
        "diagnostic_only": True,
        "required_tags": ["eng_Latn", "zho_Hans", "zho_Hant", "jpn_Jpan", "kor_Hang"],
        "probes": [
            {"target_tag": tag, "prompt": tag}
            for tag in ("eng_Latn", "zho_Hans", "zho_Hant", "jpn_Jpan", "kor_Hang")
        ],
    },
    "acceptance": {},
}


def make_lock(data: bytes = b"locked teacher fixture\n") -> dict[str, object]:
    return {
        "schema_version": 1,
        "selected": {
            "source_model": "tencent/Hy-MT2-7B",
            "repo_id": "tencent/Hy-MT2-7B-FP8",
            "revision": "a" * 40,
            "artifact_format": "compressed-tensors-fp8",
            "license": "Apache-2.0",
            "file_count": 1,
            "total_bytes": len(data),
            "files": [
                {
                    "path": "fixture.bin",
                    "bytes": len(data),
                    "sha256": hashlib.sha256(data).hexdigest(),
                    "role": "fixture",
                    "runtime_required": True,
                }
            ],
        },
        "remote_code_audit": {
            "python_files": [],
            "config_auto_map": None,
            "locked_runtime_trust_remote_code": False,
        },
    }


def test_repository_profile_and_lock_validate() -> None:
    import json
    import yaml

    profile = yaml.safe_load((ROOT / "configs/hymt2_teacher_runtime.yaml").read_text(encoding="utf-8"))
    lock = json.loads((ROOT / "configs/hymt2_teacher_artifact.lock.json").read_text(encoding="utf-8"))
    validate_contract(profile, lock)
    assert lock["selected"]["file_count"] == 14
    assert lock["selected"]["total_bytes"] == 8_046_445_711


def test_profile_rejects_floating_revision_and_remote_code() -> None:
    lock = make_lock()
    lock["selected"]["revision"] = "main"
    profile = copy.deepcopy(PROFILE)
    profile["identity"]["selected_revision"] = "main"
    with pytest.raises(TeacherRuntimeError, match="immutable"):
        validate_contract(profile, lock)
    lock = make_lock()
    lock["remote_code_audit"]["python_files"] = ["modeling.py"]
    with pytest.raises(TeacherRuntimeError, match="remote Python code"):
        validate_contract(PROFILE, lock)


def test_profile_rejects_non_offline_loading() -> None:
    profile = copy.deepcopy(PROFILE)
    profile["loading"]["local_files_only"] = False
    with pytest.raises(TeacherRuntimeError, match="local-only"):
        validate_contract(profile, make_lock())


def test_runtime_root_override_is_absolute_and_scoped(tmp_path: Path) -> None:
    assert resolve_runtime_root(PROFILE, ROOT, {}) == (
        ROOT / "artifacts/model-training/runtime/teacher/test"
    ).resolve()
    assert resolve_runtime_root(
        PROFILE, ROOT, {"DIESEL_MT_MODEL_RUNTIME": str(tmp_path)}
    ) == (tmp_path / "teacher/test").resolve()
    with pytest.raises(TeacherRuntimeError, match="absolute"):
        resolve_runtime_root(PROFILE, ROOT, {"DIESEL_MT_MODEL_RUNTIME": "relative"})


def test_snapshot_verification_is_byte_exact_and_rejects_extra_files(tmp_path: Path) -> None:
    data = b"locked teacher fixture\n"
    (tmp_path / "fixture.bin").write_bytes(data)
    verified = verify_snapshot(tmp_path, make_lock(data))
    assert verified[0]["sha256"] == hashlib.sha256(data).hexdigest()
    (tmp_path / "extra.py").write_text("raise RuntimeError\n", encoding="utf-8")
    with pytest.raises(TeacherRuntimeError, match="unexpected files"):
        verify_snapshot(tmp_path, make_lock(data))


def test_snapshot_verification_rejects_tampering(tmp_path: Path) -> None:
    data = b"locked teacher fixture\n"
    (tmp_path / "fixture.bin").write_bytes(data + b"tampered")
    with pytest.raises(TeacherRuntimeError, match="size mismatch"):
        verify_snapshot(tmp_path, make_lock(data))


def test_network_block_records_and_rejects_socket_attempt() -> None:
    with blocked_network() as attempts:
        with pytest.raises(TeacherRuntimeError, match="network access blocked"):
            socket.create_connection(("example.com", 443), timeout=1)
    assert attempts == ["('example.com', 443)"]
