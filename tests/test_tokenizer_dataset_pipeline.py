from __future__ import annotations

import copy
import hashlib
import io
import json
import subprocess
import sys
from pathlib import Path

import pytest
import zstandard


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import tokenizer_dataset_pipeline as pipeline  # noqa: E402
from tokenizer_dataset_pipeline import (  # noqa: E402
    ConfigError,
    FetchError,
    PipelineError,
    build_corpus,
    cache_path_for,
    canonical_json_bytes,
    download_locked_prefix,
    load_config,
    load_lock,
    minhash_similarity,
    one_permutation_minhash,
    pack_minhash,
    packed_minhash_similarity,
    parse_map,
    resolve_lock,
    script_ratio,
    sha256_bytes,
    split_and_clean,
    validate_config,
)


CONFIG_PATH = ROOT / "configs" / "tokenizer_datasets_mvp.yaml"


def test_registry_and_profiles_are_explicit() -> None:
    config = load_config(CONFIG_PATH)
    enabled = [source for source in config["sources"] if source["enabled"]]
    assert [source["languages"]["output"] for source in enabled] == ["eng_Latn", "zho_Hans", "jpn_Jpan", "kor_Hang"]
    assert set(config["profiles"]) == {"smoke", "mvp"}
    assert 1_000_000_000 <= config["profiles"]["mvp"]["character_budget_per_language"] <= 2_000_000_000
    assert all(not source["enabled"] for source in config["sources"] if "backup" in source["source_id"])


def test_registry_missing_required_field_fails_fast() -> None:
    config = copy.deepcopy(load_config(CONFIG_PATH))
    del config["sources"][0]["license"]
    with pytest.raises(ConfigError, match="missing fields: license"):
        validate_config(config)


def test_map_parser_filters_and_uses_numeric_quality_order() -> None:
    data = b"https://x/lang/8_10.jsonl.zst\nhttps://x/lang/10_2.jsonl.zst\nhttps://x/lang/9_1.jsonl.zst\nhttps://x/lang/10_1.jsonl.zst\nhttps://x/lang/7_1.jsonl.zst\n"
    assert parse_map(data, 8, 10) == [
        (10, 1, "https://x/lang/10_1.jsonl.zst"),
        (10, 2, "https://x/lang/10_2.jsonl.zst"),
        (9, 1, "https://x/lang/9_1.jsonl.zst"),
        (8, 10, "https://x/lang/8_10.jsonl.zst"),
    ]


def test_resolve_lock_is_canonical_and_caches_map_metadata(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = copy.deepcopy(load_config(CONFIG_PATH))
    config["profiles"]["smoke"]["locked_prefix_bytes_per_shard"] = 64
    config["profiles"]["smoke"]["locked_prefix_bytes_by_language"] = {
        language: 64 for language in ("eng_Latn", "zho_Hans", "jpn_Jpan", "kor_Hang")
    }
    config_path = tmp_path / "resolve-config.json"
    config_path.write_text(json.dumps(config, sort_keys=True), encoding="utf-8")
    payloads: dict[str, bytes] = {}
    maps: dict[str, bytes] = {}
    md5_lists: dict[str, bytes] = {}
    for source in (item for item in config["sources"] if item["enabled"]):
        source_language = source["languages"]["source"]
        shard_url = f"https://fixture.invalid/{source_language}/10_1.jsonl.zst"
        payload = (source_language.encode("ascii") + b"-") * 64
        payloads[shard_url] = payload
        maps[source["download_uri"]] = (shard_url + "\n").encode("utf-8")
        md5_url = f"{config['dataset']['base_uri']}/{source_language}.md5"
        md5_lists[md5_url] = f"{hashlib.md5(payload).hexdigest()}  {source_language}/10_1.jsonl.zst\n".encode("ascii")  # noqa: S324

    def fake_request(url: str, **_kwargs: object) -> bytes:
        return maps[url] if url in maps else md5_lists[url]

    def fake_size(url: str, **_kwargs: object) -> int:
        return len(payloads[url])

    def fake_download(url: str, path: Path, target_bytes: int, **_kwargs: object) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payloads[url][:target_bytes])

    monkeypatch.setattr(pipeline, "request_bytes", fake_request)
    monkeypatch.setattr(pipeline, "remote_size", fake_size)
    monkeypatch.setattr(pipeline, "download_locked_prefix", fake_download)
    first_lock = tmp_path / "first.lock.json"
    second_lock = tmp_path / "second.lock.json"
    first_cache = tmp_path / "first-cache"
    second_cache = tmp_path / "second-cache"
    resolve_lock(config, config_path, first_lock, first_cache, "smoke")
    resolve_lock(config, config_path, second_lock, second_cache, "smoke")
    assert first_lock.read_bytes() == second_lock.read_bytes()
    assert load_lock(first_lock, config, "smoke", config_path)["sources"]
    for source in (item for item in config["sources"] if item["enabled"]):
        language = source["languages"]["source"]
        metadata = first_cache / "hplt3" / source["source_id"] / "metadata"
        assert (metadata / f"{language}.map").is_file()
        assert (metadata / f"{language}.md5").is_file()


def test_cleaning_is_conservative_and_reasoned() -> None:
    rules = load_config(CONFIG_PATH)["cleaning"]
    original = "Uppercase English 保留简体中文 かな 한글 표기를 그대로 유지합니다."
    assert list(split_and_clean(original, rules)) == [(original, None)]
    assert list(split_and_clean("<div>This HTML line must be rejected.</div>", rules))[0][1] == "html_residue"
    assert list(split_and_clean("Skip to: TopNavigation, MainContent, and Footer Navigation links.", rules))[0][1] == "template_residue"
    assert list(split_and_clean("美国买的商品相关站点推荐：这里是一串互不相关的搜索关键词。", rules))[0][1] == "template_residue"
    assert list(split_and_clean("カジノ 紹介コードと別のカジノ キーワードを不自然に連結した広告行です。", rules))[0][1] == "keyword_stuffing"
    assert list(split_and_clean("仮想通貨が使えるおすすめオンラインカジノ！2026年最新", rules))[0][1] == "template_residue"
    assert list(split_and_clean("プントバンコ カジノ 出金の公式案内です。", rules))[0][1] == "template_residue"
    assert list(split_and_clean("在线赌场注册送彩金官方入口，立即开户领取新人奖金。", rules))[0][1] == "template_residue"
    assert list(split_and_clean("引越し 安い日、格安、最安値、口コミ、一括見積もり、おすすめの業者を連結した広告です。", rules))[0][1] == "keyword_stuffing"
    assert list(split_and_clean("成人、色情、激情、巨乳、丝袜、少妇を並べただけの検索語一覧です。", rules))[0][1] == "keyword_stuffing"
    assert list(split_and_clean("FreshBet 보너스와 프로모션을 소개하는 모바일 카지노 특별 제안입니다.", rules))[0][1] == "template_residue"
    assert list(split_and_clean("This coherent casino article mentions one regulated venue and its history.", rules))[0][1] is None
    assert list(split_and_clean("住宅リフォームの記事で費用と工期を説明し、最後にリフォーム事例を一つ紹介します。", rules))[0][1] is None
    assert list(split_and_clean("tiny", rules))[0][1] == "too_short"
    assert list(split_and_clean("", rules))[0][1] == "empty"


def test_minhash_is_stable_and_detects_near_duplicate() -> None:
    parameters = load_config(CONFIG_PATH)["deduplication"]["minhash"]
    text = "A repeatable tokenizer dataset sentence with a stable ending and enough repeated context for similarity."
    left = one_permutation_minhash(text, parameters)
    same = one_permutation_minhash(text, parameters)
    near = one_permutation_minhash(text.replace("stable", "reliable"), parameters)
    assert left == same
    assert minhash_similarity(left, same) == 1.0
    assert minhash_similarity(left, near) >= parameters["threshold"]
    assert packed_minhash_similarity(pack_minhash(left), pack_minhash(same)) == 1.0


def fixture_build_inputs(tmp_path: Path) -> tuple[dict, Path, dict, Path, Path]:
    config = copy.deepcopy(load_config(CONFIG_PATH))
    config["profiles"]["smoke"]["character_budget_per_language"] = 110
    cache_root = tmp_path / "shared-cache"
    lock_sources = []
    max_size = 0
    for source in (item for item in config["sources"] if item["enabled"]):
        language = source["languages"]["output"]
        raw = (ROOT / "tests" / "fixtures" / "tokenizer_datasets" / f"{language}.jsonl").read_bytes()
        compressed = zstandard.ZstdCompressor(level=1).compress(raw)
        url = f"https://fixture.invalid/{language}/10_1.jsonl.zst"
        cache_path = cache_path_for(cache_root, source["source_id"], url)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(compressed)
        max_size = max(max_size, len(compressed))
        lock_sources.append(
            {
                "source_id": source["source_id"],
                "source_language": source["languages"]["source"],
                "output_language": language,
                "map_uri": source["download_uri"],
                "map_sha256": "1" * 64,
                "md5_uri": "https://fixture.invalid/list.md5",
                "shards": [
                    {
                        "logical_order": 0,
                        "wds": 10,
                        "shard_number": 1,
                        "url": url,
                        "remote_size": len(compressed),
                        "upstream_md5": hashlib.md5(compressed).hexdigest(),  # noqa: S324 - upstream compatibility field
                        "locked_bytes": len(compressed),
                        "sha256": sha256_bytes(compressed),
                        "sha256_scope": "bytes=0-(locked_bytes-1)",
                    }
                ],
            }
        )
    config["profiles"]["smoke"]["locked_prefix_bytes_per_shard"] = max_size
    config["profiles"]["smoke"]["locked_prefix_bytes_by_language"] = {
        language: max_size for language in ("eng_Latn", "zho_Hans", "jpn_Jpan", "kor_Hang")
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    lock = {
        "schema_version": 1,
        "dataset_name": config["dataset"]["name"],
        "version_or_snapshot": config["dataset"]["version_or_snapshot"],
        "resolved_profile": "smoke",
        "quality_wds": {"minimum": 8, "maximum": 10},
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "sources": sorted(lock_sources, key=lambda item: item["output_language"]),
    }
    lock_path = tmp_path / "lock.json"
    lock_path.write_bytes(canonical_json_bytes(lock))
    return config, config_path, lock, lock_path, cache_root


def deterministic_files(out_root: Path) -> dict[str, bytes]:
    paths = list((out_root / "corpus" / "smoke").glob("*.txt")) + [out_root / "corpus" / "smoke" / "manifest.jsonl"]
    result = {path.name: path.read_bytes() for path in paths}
    result["quality-report.md"] = (out_root / "reports" / "tokenizer_corpus_smoke.md").read_bytes()
    return result


def test_offline_fixture_build_is_byte_reproducible(tmp_path: Path) -> None:
    config, config_path, lock, lock_path, cache_root = fixture_build_inputs(tmp_path)
    first = tmp_path / "first"
    second = tmp_path / "second"
    result1 = build_corpus(config, config_path, lock, lock_path, first, cache_root, "smoke", 1234, offline=True, use_cache=True)
    result2 = build_corpus(config, config_path, lock, lock_path, second, cache_root, "smoke", 1234, offline=True, use_cache=True)
    assert deterministic_files(first) == deterministic_files(second)
    assert result1["manifest_sha256"] == result2["manifest_sha256"]
    assert result1["peak_main_rss_bytes"] > 0
    for language in ("eng_Latn", "zho_Hans", "jpn_Jpan", "kor_Hang"):
        content = (first / "corpus" / "smoke" / f"{language}.txt").read_bytes()
        assert content and content.endswith(b"\n") and not content.startswith(b"\xef\xbb\xbf")
        assert result1["outputs"][language]["characters"] <= 110
    assert (first / "reports" / "tokenizer_corpus_smoke.md").is_file()
    assert not list(first.rglob("*.sqlite3*"))
    assert len(list((first / "interim" / "smoke" / "ram-first" / "checkpoints").glob("*.json"))) == 4
    for checkpoint in (first / "interim" / "smoke" / "ram-first" / "checkpoints").glob("*.json"):
        state = json.loads(checkpoint.read_text(encoding="utf-8"))
        assert state["stats"]["documents_with_source_url"] == state["stats"]["documents"]
        assert all(len(item["source_url_sha256"]) == 64 for item in state["review"])


def test_different_worker_counts_and_staging_paths_are_byte_reproducible(tmp_path: Path) -> None:
    config, config_path, lock, lock_path, cache_root = fixture_build_inputs(tmp_path)
    first_config = copy.deepcopy(config)
    second_config = copy.deepcopy(config)
    first_config["profiles"]["smoke"]["concurrency"] = 1
    second_config["profiles"]["smoke"]["concurrency"] = 2
    first = tmp_path / "worker-1"
    second = tmp_path / "worker-2"
    build_corpus(
        first_config,
        config_path,
        lock,
        lock_path,
        first,
        cache_root,
        "smoke",
        1234,
        offline=True,
        use_cache=True,
        staging_root=tmp_path / "fast-stage-a",
    )
    build_corpus(
        second_config,
        config_path,
        lock,
        lock_path,
        second,
        cache_root,
        "smoke",
        1234,
        offline=True,
        use_cache=True,
        staging_root=tmp_path / "fast-stage-b",
    )
    assert deterministic_files(first) == deterministic_files(second)


def test_language_checkpoint_resume_skips_completed_language(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config, config_path, lock, lock_path, cache_root = fixture_build_inputs(tmp_path)
    config["profiles"]["smoke"]["concurrency"] = 1
    out = tmp_path / "resume-output"
    original = pipeline._build_language_ram_first
    calls: list[str] = []

    def fail_on_second_language(source: dict, *args: object, **kwargs: object) -> dict:
        language = source["output_language"]
        calls.append(language)
        if len(calls) == 2:
            raise PipelineError("injected interruption")
        return original(source, *args, **kwargs)

    monkeypatch.setattr(pipeline, "_build_language_ram_first", fail_on_second_language)
    with pytest.raises(PipelineError, match="injected interruption"):
        build_corpus(
            config,
            config_path,
            lock,
            lock_path,
            out,
            cache_root,
            "smoke",
            1234,
            offline=True,
            use_cache=True,
            staging_root=tmp_path / "resume-stage",
        )
    assert calls[0] == "eng_Latn"
    assert not (out / "corpus" / "smoke" / "manifest.jsonl").exists()

    resumed_calls: list[str] = []

    def record_rebuilds(source: dict, *args: object, **kwargs: object) -> dict:
        language = source["output_language"]
        resumed_calls.append(language)
        assert language != "eng_Latn"
        return original(source, *args, **kwargs)

    monkeypatch.setattr(pipeline, "_build_language_ram_first", record_rebuilds)
    build_corpus(
        config,
        config_path,
        lock,
        lock_path,
        out,
        cache_root,
        "smoke",
        1234,
        offline=True,
        use_cache=True,
        resume=True,
        staging_root=tmp_path / "resume-stage",
    )
    assert "eng_Latn" not in resumed_calls
    assert (out / "corpus" / "smoke" / "manifest.jsonl").is_file()


def test_cold_download_then_hot_offline_build_are_identical(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config, config_path, lock, lock_path, cache_root = fixture_build_inputs(tmp_path)
    payloads: dict[str, bytes] = {}
    for source in lock["sources"]:
        shard = source["shards"][0]
        path = cache_path_for(cache_root, source["source_id"], shard["url"])
        payloads[shard["url"]] = path.read_bytes()
        path.unlink()
    downloads: list[str] = []

    def fake_download(url: str, path: Path, target_bytes: int, **_kwargs: object) -> None:
        downloads.append(url)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payloads[url][:target_bytes])

    monkeypatch.setattr(pipeline, "download_locked_prefix", fake_download)
    cold = tmp_path / "cold"
    hot = tmp_path / "hot"
    config["profiles"]["smoke"]["concurrency"] = 1
    build_corpus(config, config_path, lock, lock_path, cold, cache_root, "smoke", 1234, offline=False, use_cache=False)
    assert len(downloads) == 4

    def network_forbidden(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("offline hot-cache build attempted network access")

    monkeypatch.setattr(pipeline.urllib.request, "urlopen", network_forbidden)
    config["profiles"]["smoke"]["concurrency"] = 2
    build_corpus(config, config_path, lock, lock_path, hot, cache_root, "smoke", 1234, offline=True, use_cache=True)
    assert deterministic_files(cold) == deterministic_files(hot)


def test_network_failure_is_explicit_and_publishes_no_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config, config_path, lock, lock_path, cache_root = fixture_build_inputs(tmp_path)
    for source in lock["sources"]:
        shard = source["shards"][0]
        cache_path_for(cache_root, source["source_id"], shard["url"]).unlink()

    def fail_download(url: str, *_args: object, **_kwargs: object) -> None:
        raise FetchError(f"injected network failure for {url}")

    monkeypatch.setattr(pipeline, "download_locked_prefix", fail_download)
    out = tmp_path / "network-failure"
    with pytest.raises(FetchError, match="injected network failure"):
        build_corpus(config, config_path, lock, lock_path, out, cache_root, "smoke", 1, offline=False, use_cache=False)
    assert not (out / "corpus" / "smoke" / "manifest.jsonl").exists()


def test_downloaded_content_mismatch_fails_before_processing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config, config_path, lock, lock_path, cache_root = fixture_build_inputs(tmp_path)
    first_source = lock["sources"][0]
    shard = first_source["shards"][0]
    path = cache_path_for(cache_root, first_source["source_id"], shard["url"])
    path.unlink()

    def wrong_download(_url: str, destination: Path, target_bytes: int, **_kwargs: object) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"x" * target_bytes)

    monkeypatch.setattr(pipeline, "download_locked_prefix", wrong_download)
    out = tmp_path / "remote-mismatch"
    with pytest.raises(FetchError, match="differs from source lock"):
        build_corpus(config, config_path, lock, lock_path, out, cache_root, "smoke", 1, offline=False, use_cache=False)
    assert not (out / "corpus" / "smoke" / "manifest.jsonl").exists()


def test_transfer_failure_does_not_publish_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config, config_path, lock, lock_path, cache_root = fixture_build_inputs(tmp_path)
    config["profiles"]["smoke"]["concurrency"] = 1
    original = pipeline._transfer_staged_file

    def fail_one(staged: Path, final: Path, expected_sha256: str, expected_bytes: int) -> str:
        if final.name == "zho_Hans.txt":
            raise PipelineError("injected transfer failure")
        return original(staged, final, expected_sha256, expected_bytes)

    monkeypatch.setattr(pipeline, "_transfer_staged_file", fail_one)
    out = tmp_path / "transfer-failure"
    with pytest.raises(PipelineError, match="injected transfer failure"):
        build_corpus(
            config,
            config_path,
            lock,
            lock_path,
            out,
            cache_root,
            "smoke",
            1,
            offline=True,
            use_cache=True,
            staging_root=tmp_path / "transfer-stage",
        )
    assert not (out / "corpus" / "smoke" / "manifest.jsonl").exists()
    assert not list(out.rglob("*.transfer.tmp"))


def test_corrupt_cache_fails_before_build(tmp_path: Path) -> None:
    config, config_path, lock, lock_path, cache_root = fixture_build_inputs(tmp_path)
    first_source = lock["sources"][0]
    shard = first_source["shards"][0]
    cache_path_for(cache_root, first_source["source_id"], shard["url"]).write_bytes(b"corrupt")
    with pytest.raises(FetchError, match="cache missing or corrupt"):
        build_corpus(config, config_path, lock, lock_path, tmp_path / "out", cache_root, "smoke", 1, offline=True, use_cache=True)


def test_range_download_resumes_partial_file_and_skips_completed_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    payload = (b"locked-prefix-test-" * 32) + b"end"
    target = tmp_path / "shard.prefix"
    partial = target.with_suffix(target.suffix + ".part")
    partial.write_bytes(payload[:37])
    requests: list[int] = []

    class Response(io.BytesIO):
        status = 206

        def __enter__(self) -> "Response":
            return self

        def __exit__(self, *_args: object) -> None:
            self.close()

    def fake_urlopen(request: object, timeout: int) -> Response:
        assert timeout > 0
        range_header = request.get_header("Range")  # type: ignore[attr-defined]
        start = int(range_header.split("=")[1].split("-")[0])
        requests.append(start)
        return Response(payload[start:])

    monkeypatch.setattr(pipeline.urllib.request, "urlopen", fake_urlopen)
    download_locked_prefix("https://fixture.invalid/shard", target, len(payload), retries=1)
    assert requests == [37]
    assert target.read_bytes() == payload
    download_locked_prefix("https://fixture.invalid/shard", target, len(payload), retries=1)
    assert requests == [37]


def test_lock_rejects_missing_checksum_and_small_prefix(tmp_path: Path) -> None:
    config, _, lock, lock_path, _ = fixture_build_inputs(tmp_path)
    del lock["sources"][0]["shards"][0]["sha256"]
    lock_path.write_bytes(canonical_json_bytes(lock))
    with pytest.raises(Exception, match="missing sha256"):
        load_lock(lock_path, config, "smoke")


def test_lock_rejects_config_fingerprint_mismatch(tmp_path: Path) -> None:
    config, config_path, _lock, lock_path, _cache_root = fixture_build_inputs(tmp_path)
    config_path.write_bytes(config_path.read_bytes() + b"\n")
    with pytest.raises(Exception, match="config SHA-256 differs"):
        load_lock(lock_path, config, "smoke", config_path)


def test_cli_dry_run_has_no_output_side_effect(tmp_path: Path) -> None:
    config, config_path, _, lock_path, cache_root = fixture_build_inputs(tmp_path)
    out = tmp_path / "dry-output"
    result = subprocess.run(
        [
            str(ROOT / ".conda" / "python.exe"),
            str(ROOT / "scripts" / "fetch_tokenizer_datasets.py"),
            "--config", str(config_path),
            "--lock", str(lock_path),
            "--out", str(out),
            "--cache-dir", str(cache_root),
            "--profile", "smoke",
            "--dry-run",
            "--offline",
            "--staging-dir", str(tmp_path / "fast-stage"),
            "--max-memory-gib", "2",
            "--min-available-memory-gib", "0.5",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    plan = json.loads(result.stdout)
    assert plan["network_allowed"] is False
    assert plan["quality_wds"] == {"minimum": 8, "maximum": 10}
    assert plan["ram_first"]["candidate_database"] is False
    assert plan["ram_first"]["background_transfers"] == 1
    assert plan["staging"] == str(tmp_path / "fast-stage")
    assert not out.exists()


def test_fixture_outputs_match_declared_language(tmp_path: Path) -> None:
    config, config_path, lock, lock_path, cache_root = fixture_build_inputs(tmp_path)
    config["profiles"]["smoke"]["concurrency"] = 1
    out = tmp_path / "language-check"
    build_corpus(config, config_path, lock, lock_path, out, cache_root, "smoke", 7, offline=True, use_cache=True)
    thresholds = config["quality"]["language_min_script_ratio"]
    for language in ("eng_Latn", "zho_Hans", "jpn_Jpan", "kor_Hang"):
        lines = (out / "corpus" / "smoke" / f"{language}.txt").read_text(encoding="utf-8").splitlines()
        assert lines
        assert all(script_ratio(language, line) >= thresholds[language] for line in lines)
