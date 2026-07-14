#!/usr/bin/env python3
"""Prepare a locked FLORES-200 external evaluation contamination reference."""

from __future__ import annotations

import hashlib
import json
import re
import tarfile
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from model_data_pipeline import FetchError, SourceError, atomic_write_bytes, download_archive
from model_training_contract import LANGUAGE_TAGS, canonical_json_bytes


PIPELINE_VERSION = "td05-mt-evaluation-reference-v1"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
EXPECTED_SPLITS = ("dev", "devtest")


class EvaluationReferenceError(RuntimeError):
    """The evaluation lock or prepared reference is invalid."""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _exact_keys(value: Mapping[str, Any], required: set[str], context: str) -> None:
    missing = required - set(value)
    unknown = set(value) - required
    if missing:
        raise EvaluationReferenceError(f"{context} missing fields: {', '.join(sorted(missing))}")
    if unknown:
        raise EvaluationReferenceError(f"{context} unknown fields: {', '.join(sorted(unknown))}")


def _locked_file(value: Mapping[str, Any], context: str, *, path_field: str) -> None:
    _exact_keys(value, {path_field, "bytes", "sha256", "records"}, context)
    path = value[path_field]
    if not isinstance(path, str) or not path or "\\" in path or ".." in PurePosixPath(path).parts:
        raise EvaluationReferenceError(f"{context}.{path_field} must be a safe POSIX path")
    if not isinstance(value["bytes"], int) or value["bytes"] <= 0:
        raise EvaluationReferenceError(f"{context}.bytes must be positive")
    if not SHA256_RE.fullmatch(str(value["sha256"])):
        raise EvaluationReferenceError(f"{context}.sha256 is invalid")
    if not isinstance(value["records"], int) or value["records"] <= 0:
        raise EvaluationReferenceError(f"{context}.records must be positive")


def validate_lock(value: Mapping[str, Any]) -> dict[str, Any]:
    _exact_keys(
        value,
        {"schema_version", "identity", "archive", "repository", "archive_readme", "files"},
        "evaluation lock",
    )
    if value["schema_version"] != 1:
        raise EvaluationReferenceError("evaluation lock schema_version must be 1")
    identity = value["identity"]
    if not isinstance(identity, dict):
        raise EvaluationReferenceError("evaluation lock identity must be an object")
    _exact_keys(identity, {"benchmark_id", "scope", "training_use", "repository_revision"}, "identity")
    if identity != {
        "benchmark_id": "flores200-original-dev-devtest",
        "scope": "external-mt-evaluation-contamination-reference-only",
        "training_use": "prohibited",
        "repository_revision": "a6c830c6e1051fb4ac1a44b32358f00463f332bd",
    }:
        raise EvaluationReferenceError("evaluation identity changed")
    archive = value["archive"]
    if not isinstance(archive, dict):
        raise EvaluationReferenceError("archive must be an object")
    _exact_keys(archive, {"uri", "bytes", "sha256", "etag", "last_modified"}, "archive")
    if not str(archive["uri"]).startswith("https://dl.fbaipublicfiles.com/nllb/"):
        raise EvaluationReferenceError("archive URI must be the official immutable download")
    if not isinstance(archive["bytes"], int) or archive["bytes"] <= 0:
        raise EvaluationReferenceError("archive.bytes must be positive")
    if not SHA256_RE.fullmatch(str(archive["sha256"])):
        raise EvaluationReferenceError("archive.sha256 is invalid")
    repository = value["repository"]
    if not isinstance(repository, dict):
        raise EvaluationReferenceError("repository must be an object")
    _exact_keys(repository, {"uri", "revision", "files"}, "repository")
    if repository["uri"] != "https://github.com/facebookresearch/flores":
        raise EvaluationReferenceError("repository URI changed")
    if repository["revision"] != identity["repository_revision"]:
        raise EvaluationReferenceError("repository revision differs from identity")
    roles: set[str] = set()
    for index, record in enumerate(repository["files"]):
        if not isinstance(record, dict):
            raise EvaluationReferenceError(f"repository.files[{index}] must be an object")
        _exact_keys(record, {"path", "role", "bytes", "sha256"}, f"repository.files[{index}]")
        if record["role"] in roles:
            raise EvaluationReferenceError("repository file roles must be unique")
        roles.add(str(record["role"]))
        path = PurePosixPath(str(record["path"]))
        if path.is_absolute() or ".." in path.parts or "\\" in str(record["path"]):
            raise EvaluationReferenceError("repository paths must be safe POSIX paths")
        if not isinstance(record["bytes"], int) or record["bytes"] <= 0:
            raise EvaluationReferenceError("repository file bytes must be positive")
        if not SHA256_RE.fullmatch(str(record["sha256"])):
            raise EvaluationReferenceError("repository file SHA-256 is invalid")
    if roles != {"repository_readme", "benchmark_readme", "license"}:
        raise EvaluationReferenceError("repository must lock both readmes and the license")
    readme = value["archive_readme"]
    if not isinstance(readme, dict):
        raise EvaluationReferenceError("archive_readme must be an object")
    _locked_file(readme, "archive_readme", path_field="member")
    files = value["files"]
    if not isinstance(files, list) or len(files) != len(LANGUAGE_TAGS) * len(EXPECTED_SPLITS):
        raise EvaluationReferenceError("evaluation lock must contain five languages in both splits")
    coverage: set[tuple[str, str]] = set()
    for index, record in enumerate(files):
        if not isinstance(record, dict):
            raise EvaluationReferenceError(f"files[{index}] must be an object")
        _exact_keys(record, {"language", "split", "member", "bytes", "sha256", "records"}, f"files[{index}]")
        _locked_file(
            {key: record[key] for key in ("member", "bytes", "sha256", "records")},
            f"files[{index}]",
            path_field="member",
        )
        key = (str(record["language"]), str(record["split"]))
        if key in coverage or key[0] not in LANGUAGE_TAGS or key[1] not in EXPECTED_SPLITS:
            raise EvaluationReferenceError(f"invalid or duplicate evaluation coverage: {key}")
        expected_suffix = f"/{key[1]}/{key[0]}.{key[1]}"
        if not str(record["member"]).endswith(expected_suffix):
            raise EvaluationReferenceError(f"member does not match language/split: {record['member']}")
        coverage.add(key)
    expected = {(language, split) for split in EXPECTED_SPLITS for language in LANGUAGE_TAGS}
    if coverage != expected:
        raise EvaluationReferenceError("evaluation lock coverage is incomplete")
    return dict(value)


def load_lock(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise EvaluationReferenceError(f"cannot load evaluation lock {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise EvaluationReferenceError("evaluation lock root must be an object")
    return validate_lock(value)


def _verified_member(
    archive: tarfile.TarFile,
    record: Mapping[str, Any],
    *,
    allow_empty_rows: bool = False,
) -> bytes:
    member_name = str(record["member"])
    try:
        member = archive.getmember(member_name)
    except KeyError as exc:
        raise SourceError(f"locked FLORES member missing: {member_name}") from exc
    if not member.isfile() or member.size != int(record["bytes"]):
        raise SourceError(f"locked FLORES member has wrong type or size: {member_name}")
    handle = archive.extractfile(member)
    if handle is None:
        raise SourceError(f"cannot read locked FLORES member: {member_name}")
    data = handle.read()
    if len(data) != int(record["bytes"]) or sha256_bytes(data) != record["sha256"]:
        raise SourceError(f"locked FLORES member identity differs: {member_name}")
    try:
        rows = data.decode("utf-8", errors="strict").splitlines()
    except UnicodeError as exc:
        raise SourceError(f"locked FLORES member is not UTF-8: {member_name}") from exc
    if len(rows) != int(record["records"]) or (
        not allow_empty_rows and any(not row for row in rows)
    ):
        raise SourceError(f"locked FLORES row count/content differs: {member_name}")
    return data


def _repo_cache_name(record: Mapping[str, Any]) -> str:
    return str(record["path"]).replace("/", "__")


def prepare_reference(
    lock: Mapping[str, Any],
    out_root: Path,
    cache_root: Path,
    *,
    offline: bool,
    timeout: int = 120,
    retries: int = 4,
) -> dict[str, Any]:
    validated = validate_lock(lock)
    archive_record = validated["archive"]
    archive_path = cache_root / "flores200_dataset.tar.gz"
    repo_cache = cache_root / "repo"
    downloads: list[tuple[str, Path, int, str]] = [
        (
            str(archive_record["uri"]),
            archive_path,
            int(archive_record["bytes"]),
            str(archive_record["sha256"]),
        )
    ]
    revision = str(validated["repository"]["revision"])
    for record in validated["repository"]["files"]:
        downloads.append(
            (
                f"https://raw.githubusercontent.com/facebookresearch/flores/{revision}/{record['path']}",
                repo_cache / _repo_cache_name(record),
                int(record["bytes"]),
                str(record["sha256"]),
            )
        )
    for uri, path, byte_count, sha256 in downloads:
        valid = path.is_file() and path.stat().st_size == byte_count and sha256_bytes(path.read_bytes()) == sha256
        if valid:
            continue
        if offline:
            raise FetchError(f"offline mode requires a validated cache file: {path}")
        download_archive(uri, path, byte_count, sha256, timeout=timeout, retries=retries)

    output_dir = out_root / "raw" / "flores200-original"
    files: list[dict[str, Any]] = []
    with tarfile.open(archive_path, mode="r:gz") as archive:
        archive_readme = _verified_member(
            archive, validated["archive_readme"], allow_empty_rows=True
        )
        atomic_write_bytes(output_dir / "ARCHIVE_README", archive_readme)
        for record in validated["files"]:
            data = _verified_member(archive, record)
            relative = PurePosixPath(str(record["split"])) / PurePosixPath(str(record["member"])).name
            atomic_write_bytes(output_dir / Path(relative.as_posix()), data)
            files.append(
                {
                    "bytes": int(record["bytes"]),
                    "format": "text-lines",
                    "language": record["language"],
                    "path": relative.as_posix(),
                    "records": int(record["records"]),
                    "sha256": record["sha256"],
                    "split": record["split"],
                }
            )
    documentation: list[dict[str, Any]] = []
    for record in validated["repository"]["files"]:
        data = (repo_cache / _repo_cache_name(record)).read_bytes()
        output_name = f"{record['role']}{PurePosixPath(str(record['path'])).suffix or '.txt'}"
        atomic_write_bytes(output_dir / output_name, data)
        documentation.append(
            {
                "bytes": int(record["bytes"]),
                "path": output_name,
                "role": record["role"],
                "sha256": record["sha256"],
            }
        )
    manifest = {
        "archive": dict(validated["archive"]),
        "benchmark_id": validated["identity"]["benchmark_id"],
        "documentation": sorted(documentation, key=lambda row: row["role"]),
        "files": sorted(files, key=lambda row: (EXPECTED_SPLITS.index(row["split"]), LANGUAGE_TAGS.index(row["language"]))),
        "pipeline_version": PIPELINE_VERSION,
        "repository": {
            "revision": revision,
            "uri": validated["repository"]["uri"],
        },
        "schema_version": 1,
        "status": "complete",
        "usage": "external contamination reference and later MT evaluation only; prohibited from training",
    }
    manifest_bytes = canonical_json_bytes(manifest)
    atomic_write_bytes(output_dir / "reference-manifest.json", manifest_bytes)
    return {
        "benchmark_id": manifest["benchmark_id"],
        "files": len(files),
        "manifest": (output_dir / "reference-manifest.json").as_posix(),
        "manifest_bytes": len(manifest_bytes),
        "manifest_sha256": sha256_bytes(manifest_bytes),
        "records": sum(int(record["records"]) for record in files),
    }


def dry_run_plan(lock: Mapping[str, Any], out_root: Path, cache_root: Path, *, offline: bool) -> dict[str, Any]:
    validated = validate_lock(lock)
    return {
        "benchmark_id": validated["identity"]["benchmark_id"],
        "cache_root": cache_root.as_posix(),
        "files": len(validated["files"]),
        "languages": list(LANGUAGE_TAGS),
        "offline": offline,
        "out_root": out_root.as_posix(),
        "splits": list(EXPECTED_SPLITS),
        "status": "dry-run",
        "training_use": validated["identity"]["training_use"],
    }
