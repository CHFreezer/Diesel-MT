"""Shared byte-stable artifact I/O primitives for Diesel-MT workflows."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any


class ArtifactIOError(RuntimeError):
    """Raised when a structured artifact cannot be loaded safely."""


def canonical_json_bytes(value: Any, *, allow_nan: bool = True) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=allow_nan,
        ).encode("utf-8")
        + b"\n"
    )


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_bytes(path: Path, value: bytes, *, overwrite: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite artifact: {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_json(
    path: Path,
    value: Any,
    *,
    sort_keys: bool = True,
    allow_nan: bool = False,
    overwrite: bool = True,
) -> None:
    payload = (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=sort_keys,
            indent=2,
            allow_nan=allow_nan,
        ).encode("utf-8")
        + b"\n"
    )
    atomic_write_bytes(path, payload, overwrite=overwrite)


def write_json(path: Path, value: Any) -> None:
    """Write the legacy unsorted, indented JSON representation."""

    atomic_write_json(path, value, sort_keys=False, allow_nan=True)


def write_jsonl(
    path: Path, rows: Iterable[Mapping[str, Any]], *, allow_nan: bool = True
) -> tuple[int, str]:
    payload = b"".join(
        canonical_json_bytes(dict(row), allow_nan=allow_nan) for row in rows
    )
    atomic_write_bytes(path, payload)
    return payload.count(b"\n"), sha256_bytes(payload)


def read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ArtifactIOError(f"cannot load JSON object {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ArtifactIOError(f"expected JSON object: {path}")
    return value


def read_jsonl_objects(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ArtifactIOError(
                        f"non-object JSONL record: {path}:{line_number}"
                    )
                rows.append(value)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ArtifactIOError(f"cannot load JSONL {path}: {exc}") from exc
    return rows
