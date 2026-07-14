#!/usr/bin/env python3
"""Run one self-contained offline Diesel-MT deployment smoke test."""
from __future__ import annotations

import argparse
import hashlib
import json
import socket
from pathlib import Path, PurePosixPath


MANIFEST_NAME = "deployment_manifest.json"


class OfflineSmokeError(RuntimeError):
    """Raised when an offline deployment invariant fails."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def validate_relative_path(relative: object) -> str:
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise OfflineSmokeError(f"invalid deployment manifest path: {relative!r}")
    path = PurePosixPath(relative)
    if (
        path.is_absolute()
        or path.as_posix() != relative
        or any(part in ("", ".", "..") for part in path.parts)
        or Path(relative).is_absolute()
    ):
        raise OfflineSmokeError(f"invalid deployment manifest path: {relative!r}")
    return relative


def verify_manifest(root: Path) -> dict:
    manifest_path = root / MANIFEST_NAME
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise OfflineSmokeError(f"cannot read deployment manifest: {error}") from error
    records = manifest.get("files")
    if manifest.get("schema_version") != 1 or not isinstance(records, list):
        raise OfflineSmokeError("unsupported deployment manifest schema")
    if manifest.get("status") != "complete":
        raise OfflineSmokeError("deployment manifest is not complete")
    expected: set[str] = set()
    for record in records:
        if not isinstance(record, dict):
            raise OfflineSmokeError("deployment manifest contains a non-object file record")
        relative = validate_relative_path(record.get("path"))
        if relative in expected:
            raise OfflineSmokeError(f"invalid deployment manifest path: {relative!r}")
        expected.add(relative)
        path = root / relative
        if path.is_symlink():
            raise OfflineSmokeError(f"deployment file cannot be a symlink: {relative}")
        if not path.is_file():
            raise OfflineSmokeError(f"deployment file is missing: {relative}")
        if path.stat().st_size != record.get("bytes"):
            raise OfflineSmokeError(f"deployment file size mismatch: {relative}")
        if sha256_file(path) != record.get("sha256"):
            raise OfflineSmokeError(f"deployment file SHA-256 mismatch: {relative}")
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.relative_to(root).as_posix() != MANIFEST_NAME
    }
    if actual != expected:
        raise OfflineSmokeError(
            f"deployment file set mismatch: missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )
    return manifest


def block_python_network() -> None:
    """Reject outbound Python socket connections in the offline child process."""
    original_socket = socket.socket

    class OfflineSocket(original_socket):
        def connect(self, address):  # type: ignore[no-untyped-def]
            raise OfflineSmokeError(f"network access is disabled: {address!r}")

        def connect_ex(self, address):  # type: ignore[no-untyped-def]
            raise OfflineSmokeError(f"network access is disabled: {address!r}")

    def blocked_create_connection(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise OfflineSmokeError("network access is disabled")

    socket.socket = OfflineSocket
    socket.create_connection = blocked_create_connection


def run_smoke(root: Path) -> dict:
    verify_manifest(root)
    block_python_network()

    import ctranslate2
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        root / "tokenizer", local_files_only=True
    )
    tokenizer.src_lang = "eng_Latn"
    source_ids = tokenizer(
        "A clean process loads only the offline deployment package.",
        add_special_tokens=True,
    )["input_ids"]
    source_tokens = tokenizer.convert_ids_to_tokens(source_ids)
    if source_tokens[0] != "eng_Latn" or source_tokens[-1] != "</s>":
        raise OfflineSmokeError(f"source token contract failed: {source_tokens}")

    translator = ctranslate2.Translator(
        str(root / "model"), device="cpu", compute_type="int8"
    )
    result = translator.translate_batch(
        [source_tokens],
        target_prefix=[["zho_Hans"]],
        beam_size=1,
        max_decoding_length=8,
        return_end_token=True,
    )[0]
    hypothesis = result.hypotheses[0]
    if not hypothesis or hypothesis[0] != "zho_Hans":
        raise OfflineSmokeError(f"target prefix contract failed: {hypothesis}")
    hypothesis_ids = tokenizer.convert_tokens_to_ids(hypothesis[1:])
    decoded = tokenizer.decode(hypothesis_ids, skip_special_tokens=True)
    return {
        "status": "passed",
        "compute_type": translator.compute_type,
        "source_tokens": source_tokens,
        "target_prefix": "zho_Hans",
        "hypothesis_tokens": hypothesis,
        "decoded_text": decoded,
        "network_guard": "Python socket connect/create_connection blocked",
        "warning": "Random model output; no translation quality is implied.",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deployment-root", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    print(json.dumps(run_smoke(args.deployment_root.resolve()), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
