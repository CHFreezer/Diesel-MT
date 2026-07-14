from __future__ import annotations

import copy
import hashlib
import io
import json
import sys
import tarfile
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from mt_evaluation_reference import (  # noqa: E402
    EvaluationReferenceError,
    load_lock,
    prepare_reference,
    validate_lock,
)


LOCK_PATH = ROOT / "configs" / "mvp_mt_evaluation.lock.json"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _fixture_lock_and_cache(tmp_path: Path) -> tuple[dict[str, object], Path]:
    lock = copy.deepcopy(load_lock(LOCK_PATH))
    cache = tmp_path / "cache"
    members: dict[str, bytes] = {}
    readme = b"fixture readme\n"
    lock["archive_readme"]["bytes"] = len(readme)
    lock["archive_readme"]["sha256"] = _sha(readme)
    lock["archive_readme"]["records"] = 1
    members[lock["archive_readme"]["member"]] = readme
    for index, record in enumerate(lock["files"]):
        data = f"locked evaluation sentence {index}\n".encode()
        record["bytes"] = len(data)
        record["sha256"] = _sha(data)
        record["records"] = 1
        members[record["member"]] = data
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for name, data in sorted(members.items()):
            member = tarfile.TarInfo(name)
            member.size = len(data)
            archive.addfile(member, io.BytesIO(data))
    archive_bytes = buffer.getvalue()
    lock["archive"]["bytes"] = len(archive_bytes)
    lock["archive"]["sha256"] = _sha(archive_bytes)
    cache.mkdir(parents=True)
    (cache / "flores200_dataset.tar.gz").write_bytes(archive_bytes)
    repo_cache = cache / "repo"
    repo_cache.mkdir()
    for index, record in enumerate(lock["repository"]["files"]):
        data = f"repository evidence {index}\n".encode()
        record["bytes"] = len(data)
        record["sha256"] = _sha(data)
        (repo_cache / str(record["path"]).replace("/", "__")).write_bytes(data)
    return lock, cache


def test_offline_reference_preparation_is_byte_stable(tmp_path: Path) -> None:
    lock, cache = _fixture_lock_and_cache(tmp_path)
    first = tmp_path / "first"
    second = tmp_path / "second"
    first_result = prepare_reference(lock, first, cache, offline=True)
    second_result = prepare_reference(lock, second, cache, offline=True)
    assert first_result["files"] == 10
    assert first_result["records"] == 10
    assert first_result["manifest_sha256"] == second_result["manifest_sha256"]
    first_dir = first / "raw" / "flores200-original"
    second_dir = second / "raw" / "flores200-original"
    assert {
        path.relative_to(first_dir).as_posix(): path.read_bytes()
        for path in first_dir.rglob("*")
        if path.is_file()
    } == {
        path.relative_to(second_dir).as_posix(): path.read_bytes()
        for path in second_dir.rglob("*")
        if path.is_file()
    }
    manifest = json.loads((first_dir / "reference-manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"
    assert "prohibited from training" in manifest["usage"]


def test_lock_rejects_path_traversal_and_incomplete_coverage() -> None:
    lock = load_lock(LOCK_PATH)
    unsafe = copy.deepcopy(lock)
    unsafe["files"][0]["member"] = "../eng_Latn.dev"
    with pytest.raises(EvaluationReferenceError, match="safe POSIX path"):
        validate_lock(unsafe)

    incomplete = copy.deepcopy(lock)
    incomplete["files"].pop()
    with pytest.raises(EvaluationReferenceError, match="five languages"):
        validate_lock(incomplete)
