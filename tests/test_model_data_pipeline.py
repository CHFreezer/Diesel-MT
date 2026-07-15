from __future__ import annotations

import copy
import gzip
import hashlib
import io
import json
import subprocess
import sys
import tarfile
from collections import Counter
from pathlib import Path
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import model_data_pipeline as pipeline  # noqa: E402
from model_data_pipeline import (  # noqa: E402
    FetchError,
    SourceError,
    archive_cache_path,
    build_model_data,
    download_archive,
    dry_run_plan,
    normalize_text,
    pair_rejection_reason,
    sha256_bytes,
)
from model_training_contract import (  # noqa: E402
    config_sha256,
    load_model_data_config,
    validate_source_lock,
)


CONFIG_PATH = ROOT / "configs" / "mvp_model_data.yaml"
FIXTURE_ROOT = ROOT / "tests" / "fixtures" / "model_data" / "massive-1.1"
MEMBER_PATHS = [
    "1.1/LICENSE",
    "1.1/NOTICE.md",
    "1.1/data/en-US.jsonl",
    "1.1/data/zh-CN.jsonl",
    "1.1/data/zh-TW.jsonl",
    "1.1/data/ja-JP.jsonl",
    "1.1/data/ko-KR.jsonl",
]
ROLES = {
    "1.1/LICENSE": "license",
    "1.1/NOTICE.md": "notice",
    "1.1/data/en-US.jsonl": "data:eng_Latn",
    "1.1/data/zh-CN.jsonl": "data:zho_Hans",
    "1.1/data/zh-TW.jsonl": "data:zho_Hant",
    "1.1/data/ja-JP.jsonl": "data:jpn_Jpan",
    "1.1/data/ko-KR.jsonl": "data:kor_Hang",
}


def _fixture_bytes(member_path: str) -> bytes:
    relative = member_path.removeprefix("1.1/")
    return (FIXTURE_ROOT / relative).read_bytes()


def _build_fixture_archive(path: Path) -> bytes:
    buffer = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=buffer, mtime=0) as compressed:
        with tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as archive:
            for member_path in MEMBER_PATHS:
                data = _fixture_bytes(member_path)
                member = tarfile.TarInfo(member_path)
                member.size = len(data)
                member.mtime = 0
                member.mode = 0o644
                member.uid = 0
                member.gid = 0
                member.uname = ""
                member.gname = ""
                archive.addfile(member, io.BytesIO(data))
    data = buffer.getvalue()
    path.write_bytes(data)
    return data


def fixture_inputs(
    tmp_path: Path, *, populate_cache: bool = True
) -> tuple[dict[str, Any], dict[str, Any], Path, Path, Path]:
    archive_path = tmp_path / "massive-fixture.tar.gz"
    archive_data = _build_fixture_archive(archive_path)
    config = copy.deepcopy(load_model_data_config(CONFIG_PATH))
    config["budgets"] = {
        "minimum_accepted_per_undirected_pair": {"train": 1, "dev": 1, "test": 1},
        "scan_limit_rows_per_locale": 3,
        "download_max_bytes": len(archive_data),
        "selected_extract_max_bytes": sum(len(_fixture_bytes(path)) for path in MEMBER_PATHS),
        "source_rows_per_locale": 3,
        "source_partition_rows_per_locale": {"train": 1, "dev": 1, "test": 1},
    }
    selected_files = [
        {
            "path": path,
            "role": ROLES[path],
            "bytes": len(_fixture_bytes(path)),
            "sha256": sha256_bytes(_fixture_bytes(path)),
        }
        for path in MEMBER_PATHS
    ]
    source = config["sources"][0]
    lock = {
        "schema_version": 2,
        "config_sha256": config_sha256(config),
        "source_order": ["massive-1.1"],
        "sources": [
            {
                "source_id": source["source_id"],
                "version": source["version"],
                "license": source["license"],
                "homepage": source["homepage"],
                "download_uri": source["download_uri"],
                "archive": {
                    "bytes": len(archive_data),
                    "sha256": hashlib.sha256(archive_data).hexdigest(),
                    "etag": "fixture-etag",
                    "last_modified": "fixture",
                },
                "selected_files": selected_files,
                "verification": {
                    "verified_on": "fixture",
                    "method": "deterministic unit-test tar archive",
                    "alignment_key": ["partition", "id"],
                    "rows_per_locale": 3,
                    "partition_rows_per_locale": {"train": 1, "dev": 1, "test": 1},
                    "selected_bytes": sum(record["bytes"] for record in selected_files),
                },
            }
        ],
    }
    validate_source_lock(lock, config)
    cache_root = tmp_path / "cache"
    if populate_cache:
        cache_path = archive_cache_path(cache_root, lock["sources"][0])
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(archive_data)
    return config, lock, archive_path, cache_root, tmp_path / "out"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_source_adapter_registry_and_dry_run_are_explicit_and_side_effect_free(
    tmp_path: Path,
) -> None:
    config, lock, _archive, cache_root, out_root = fixture_inputs(
        tmp_path, populate_cache=False
    )
    assert set(pipeline.SOURCE_ADAPTERS) == {"massive-1.1"}
    plan = dry_run_plan(
        config,
        lock,
        out_root,
        cache_root,
        offline=True,
        use_cache=True,
        resume=True,
    )
    assert plan["status"] == "dry-run"
    assert plan["maximum_canonical_samples"] == 30
    assert plan["sources"][0]["cache_status"] == "missing"
    assert plan["network_allowed"] is False
    assert not out_root.exists()
    assert not cache_root.exists()


def test_offline_build_is_byte_stable_and_preserves_group_and_provenance(
    tmp_path: Path,
) -> None:
    config, lock, _archive, cache_root, _out = fixture_inputs(tmp_path)
    first = tmp_path / "first"
    second = tmp_path / "second"
    first_result = build_model_data(
        config, lock, first, cache_root, offline=True, use_cache=True
    )
    second_result = build_model_data(
        config, lock, second, cache_root, offline=True, use_cache=True
    )

    first_corpus = first / "corpus" / "mvp" / "human_parallel.jsonl"
    second_corpus = second / "corpus" / "mvp" / "human_parallel.jsonl"
    first_manifest = first / "corpus" / "mvp" / "manifest.json"
    second_manifest = second / "corpus" / "mvp" / "manifest.json"
    assert first_corpus.read_bytes() == second_corpus.read_bytes()
    assert first_manifest.read_bytes() == second_manifest.read_bytes()
    assert first_result["manifest_sha256"] == second_result["manifest_sha256"]
    samples = _read_jsonl(first_corpus)
    assert len(samples) == 30
    assert len({sample["sample_id"] for sample in samples}) == 30
    group_counts = Counter(sample["sample_group_id"] for sample in samples)
    assert sorted(group_counts.values()) == [10, 10, 10]
    assert {sample["split"] for sample in samples} == {"train", "dev", "test"}
    assert all(sample["provenance"]["kind"] == "human_parallel" for sample in samples)
    assert all(sample["license"] == "CC-BY-4.0" for sample in samples)
    manifest = json.loads(first_manifest.read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"
    assert manifest["records"] == 30
    assert len(manifest["files"]) == 5
    report = json.loads((first / "reports" / "td03-build.json").read_text(encoding="utf-8"))
    assert len(report["directed_route_potential_counts"]) == 20
    assert set(first_result["resume_checkpoints_used"]["massive-1.1"].values()) == {False}


def test_cleaning_is_conservative_and_rejects_invalid_content() -> None:
    normalized, reason = normalize_text("  Cafe\u0301\tMIXED Case  ", "eng_Latn")
    assert normalized == "Café MIXED Case"
    assert reason is None
    traditional, reason = normalize_text("設定繁體中文鬧鐘", "zho_Hant")
    assert traditional == "設定繁體中文鬧鐘"
    assert reason is None
    assert normalize_text("<b>hello</b>", "eng_Latn")[1] == "html_residue"
    assert normalize_text("bad\x00text", "eng_Latn")[1] == "control_character"
    assert normalize_text("天气很好天气很好", "eng_Latn")[1] == "wrong_script_dominance"
    assert normalize_text("aaaaaaaaaaaaaaaa", "eng_Latn")[1] == "abnormal_repetition"
    assert pair_rejection_reason("short", "x" * 100) == "length_ratio"
    assert pair_rejection_reason("hello", "你好") is None


def test_resumable_download_uses_existing_partial_prefix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = b"locked archive bytes"
    destination = tmp_path / "archive.tar.gz"
    partial = destination.with_suffix(destination.suffix + ".part")
    partial.write_bytes(payload[:7])
    seen_ranges: list[str | None] = []

    class Response:
        status = 206

        def __init__(self, data: bytes) -> None:
            self._stream = io.BytesIO(data)

        def __enter__(self) -> "Response":
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def read(self, size: int = -1) -> bytes:
            return self._stream.read(size)

    def fake_urlopen(request: Any, timeout: int) -> Response:
        assert timeout == 3
        seen_ranges.append(request.get_header("Range"))
        return Response(payload[7:])

    monkeypatch.setattr(pipeline.urllib.request, "urlopen", fake_urlopen)
    download_archive(
        "https://fixture.invalid/archive.tar.gz",
        destination,
        len(payload),
        sha256_bytes(payload),
        timeout=3,
        retries=1,
    )
    assert seen_ranges == [f"bytes=7-{len(payload) - 1}"]
    assert destination.read_bytes() == payload
    assert not partial.exists()


def test_corrupt_offline_cache_fails_without_publishing_manifest(tmp_path: Path) -> None:
    config, lock, _archive, cache_root, out_root = fixture_inputs(tmp_path)
    archive_path = archive_cache_path(cache_root, lock["sources"][0])
    archive_path.write_bytes(b"corrupt")
    with pytest.raises(FetchError, match="missing or corrupt"):
        build_model_data(
            config, lock, out_root, cache_root, offline=True, use_cache=True
        )
    assert not (out_root / "corpus" / "mvp" / "manifest.json").exists()


def test_network_failure_is_explicit_and_publishes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, lock, _archive, cache_root, out_root = fixture_inputs(
        tmp_path, populate_cache=False
    )

    def fail_download(*_args: object, **_kwargs: object) -> None:
        raise FetchError("fixture network failure")

    monkeypatch.setattr(pipeline, "download_archive", fail_download)
    with pytest.raises(FetchError, match="fixture network failure"):
        build_model_data(
            config, lock, out_root, cache_root, offline=False, use_cache=False
        )
    assert not (out_root / "corpus" / "mvp" / "manifest.json").exists()


def test_resume_reuses_verified_locale_checkpoints(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, lock, _archive, cache_root, out_root = fixture_inputs(tmp_path)
    build_model_data(config, lock, out_root, cache_root, offline=True, use_cache=True)
    original_corpus = (out_root / "corpus" / "mvp" / "human_parallel.jsonl").read_bytes()
    original_manifest = (out_root / "corpus" / "mvp" / "manifest.json").read_bytes()

    def should_not_parse(*_args: object, **_kwargs: object) -> dict[str, Any]:
        raise AssertionError("verified checkpoint was not reused")

    monkeypatch.setitem(pipeline.SOURCE_ADAPTERS, "massive-1.1", should_not_parse)
    resumed = build_model_data(
        config,
        lock,
        out_root,
        cache_root,
        offline=True,
        use_cache=True,
        resume=True,
    )
    assert (out_root / "corpus" / "mvp" / "human_parallel.jsonl").read_bytes() == original_corpus
    assert (out_root / "corpus" / "mvp" / "manifest.json").read_bytes() == original_manifest
    assert set(resumed["resume_checkpoints_used"]["massive-1.1"].values()) == {True}


def test_interrupted_publication_never_leaves_completion_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, lock, _archive, cache_root, out_root = fixture_inputs(tmp_path)
    real_atomic_write = pipeline.atomic_write_bytes

    def fail_build_report(path: Path, data: bytes) -> None:
        if path.name == "td03-build.json":
            raise OSError("injected publication failure")
        real_atomic_write(path, data)

    monkeypatch.setattr(pipeline, "atomic_write_bytes", fail_build_report)
    with pytest.raises(OSError, match="injected publication failure"):
        build_model_data(
            config, lock, out_root, cache_root, offline=True, use_cache=True
        )
    assert not (out_root / "corpus" / "mvp" / "manifest.json").exists()


def test_selected_member_hash_mismatch_fails_before_publication(tmp_path: Path) -> None:
    config, lock, _archive, cache_root, out_root = fixture_inputs(tmp_path)
    lock["sources"][0]["selected_files"][2]["sha256"] = "0" * 64
    with pytest.raises(SourceError, match="member SHA-256 differs"):
        build_model_data(
            config, lock, out_root, cache_root, offline=True, use_cache=True
        )
    assert not (out_root / "corpus" / "mvp" / "manifest.json").exists()


def test_cli_dry_run_uses_real_lock_without_creating_runtime_directories(
    tmp_path: Path,
) -> None:
    out_root = tmp_path / "model-data"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/prepare_model_data.py",
            "--config",
            "configs/mvp_model_data.yaml",
            "--lock",
            "configs/mvp_model_data.lock.json",
            "--out",
            str(out_root),
            "--offline",
            "--dry-run",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    plan = json.loads(result.stdout)
    assert plan["status"] == "dry-run"
    assert plan["sources"][0]["archive_sha256"] == (
        "4cba5faa11c71437928e17cb1b9b3d8b8e727e7ea363a3a9a8045e19c0491577"
    )
    assert not out_root.exists()
