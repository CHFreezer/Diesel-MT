from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from artifact_io import (  # noqa: E402
    ArtifactIOError,
    atomic_write_json,
    canonical_json_bytes,
    read_json_object,
    read_jsonl_objects,
    sha256_file,
    write_jsonl,
)


def test_canonical_json_and_jsonl_are_byte_stable(tmp_path: Path) -> None:
    assert canonical_json_bytes({"b": 2, "a": "中"}) == (
        '{"a":"中","b":2}\n'.encode("utf-8")
    )
    path = tmp_path / "records.jsonl"
    records, digest = write_jsonl(path, [{"b": 2, "a": 1}, {"id": "x"}])
    assert records == 2
    assert digest == sha256_file(path)
    assert read_jsonl_objects(path) == [{"a": 1, "b": 2}, {"id": "x"}]


def test_atomic_json_preserves_existing_format_and_overwrite_gate(
    tmp_path: Path,
) -> None:
    path = tmp_path / "report.json"
    atomic_write_json(path, {"b": 2, "a": 1})
    assert path.read_bytes() == b'{\n  "a": 1,\n  "b": 2\n}\n'
    assert read_json_object(path) == {"a": 1, "b": 2}
    with pytest.raises(FileExistsError):
        atomic_write_json(path, {"a": 2}, overwrite=False)


def test_structured_readers_reject_non_objects(tmp_path: Path) -> None:
    json_path = tmp_path / "array.json"
    json_path.write_text(json.dumps([]), encoding="utf-8")
    with pytest.raises(ArtifactIOError, match="expected JSON object"):
        read_json_object(json_path)
    jsonl_path = tmp_path / "array.jsonl"
    jsonl_path.write_text("[]\n", encoding="utf-8")
    with pytest.raises(ArtifactIOError, match="non-object JSONL"):
        read_jsonl_objects(jsonl_path)
