#!/usr/bin/env python3
"""Build a fixed multilingual evaluation set and assess tokenizer quality.

Corpus metrics and synthetic stress probes are reported separately.  Unknown
source characters are counted from fast-tokenizer offset mappings so a fused
``<unk>`` spanning several Unicode code points is not mistaken for one lost
character.
"""
from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import math
import os
import re
import sys
import tempfile
import time
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path


if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from tokenizer_utils import (  # noqa: E402
    TRAINING_LANGUAGES,
    atomic_write_json,
    reload_tokenizer,
    verify_tokenizer,
)


SAMPLE_SCHEMA_VERSION = 1
METRICS_SCHEMA_VERSION = 1
SAMPLING_ALGORITHM = "sha256-priority-with-long-quota-v1"
DEFAULT_SEED = 20260713
DEFAULT_SAMPLE_SIZE = 500
DEFAULT_LONG_QUOTA = 25
LONG_CHARACTER_THRESHOLD = 500
ENGLISH_WORD_RE = re.compile(r"[A-Za-z]+(?:['’-][A-Za-z]+)*")


STRESS_PROBES: dict[str, tuple[tuple[str, str], ...]] = {
    "eng_Latn": (
        ("daily", "I bought vegetables after work and cooked dinner with my family."),
        ("technical_news", "Researchers released version 2.1 of the multilingual inference engine on Monday."),
        ("mixed_language", "English 中文 日本語 한국어 mixed text, API-v2, GPU 16GB 🚀"),
        ("numeric_punctuation", "2026-07-13 12:34:56 +86-138-0000-0000 3.1415926 100%"),
        ("numeric_punctuation", "!@#$%^&*()_+-=[]{}|;:',.<>/?`~"),
        ("very_short", "OK"),
        ("empty", ""),
        ("rare_unicode", "𠀀 𪚥 㐀 𫠝 🧪🛰️ café naïve"),
    ),
    "zho_Hans": (
        ("daily", "下班以后我去市场买了蔬菜，晚上和家人一起做饭。"),
        ("technical_news", "研究团队周一发布了多语言推理引擎二点一版本，并公布了测试结果。"),
        ("mixed_language", "中文 English 日本語 한국어 混合文本，API-v2，GPU 16GB 🚀"),
        ("numeric_punctuation", "2026年07月13日 12:34:56，增长率100%，价格￥3.1415926。"),
        ("numeric_punctuation", "！@#￥%……&*（）——+-=[]{}；：，。《》？"),
        ("very_short", "好"),
        ("empty", ""),
        ("rare_unicode", "𠀀𪚥㐀𫠝，生僻姓名用字与表情🧪🛰️。"),
    ),
    "zho_Hant": (
        ("daily", "下班以後我去市場買了蔬菜，晚上和家人一起做飯。"),
        ("technical_news", "研究團隊週一發布了多語言推理引擎二點一版本，並公布了測試結果。"),
        ("mixed_language", "繁體中文 English 日本語 한국어 混合文字，API-v2，GPU 16GB 🚀"),
        ("numeric_punctuation", "2026年07月14日 12:34:56，增長率100%，價格￥3.1415926。"),
        ("numeric_punctuation", "！@#￥%……&*（）——+-=[]{}；：，。《》？"),
        ("very_short", "好"),
        ("empty", ""),
        ("rare_unicode", "𠀀𪚥㐀𫠝，罕見姓名用字與表情🧪🛰️。"),
    ),
    "jpn_Jpan": (
        ("daily", "仕事のあとで市場へ野菜を買いに行き、家族と夕食を作りました。"),
        ("technical_news", "研究チームは月曜日、多言語推論エンジンのバージョン2.1を公開しました。"),
        ("mixed_language", "日本語 中文 English 한국어 の混合テキスト、API-v2、GPU 16GB 🚀"),
        ("numeric_punctuation", "2026年07月13日 12:34:56、増加率100％、価格￥3.1415926。"),
        ("numeric_punctuation", "！＠＃＄％＾＆＊（）＿＋－＝［］｛｝；：、。＜＞？"),
        ("very_short", "はい"),
        ("empty", ""),
        ("rare_unicode", "𠀀𪚥㐀𫠝と異体字、絵文字🧪🛰️を確認します。"),
    ),
    "kor_Hang": (
        ("daily", "퇴근 후 시장에서 채소를 사고 가족과 함께 저녁을 만들었습니다."),
        ("technical_news", "연구팀은 월요일 다국어 추론 엔진 버전 2.1과 시험 결과를 공개했습니다."),
        ("mixed_language", "한국어 中文 English 日本語 혼합 텍스트, API-v2, GPU 16GB 🚀"),
        ("numeric_punctuation", "2026년 07월 13일 12:34:56, 증가율 100%, 가격 ₩3.1415926"),
        ("numeric_punctuation", "!@#$%^&*()_+-=[]{}|;:',.<>/?`~"),
        ("very_short", "네"),
        ("empty", ""),
        ("rare_unicode", "𠀀 𪚥 㐀 𫠝 희귀 문자와 이모지 🧪🛰️"),
    ),
}


class EvaluationError(RuntimeError):
    """Raised when evaluation inputs or tokenizer behavior are invalid."""


def progress(message: str) -> None:
    stamp = time.strftime("%H:%M:%S")
    print(f"[{stamp}] {message}", flush=True)


def sha256_file(path: Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _load_corpus_manifest(path: Path) -> dict[str, dict]:
    records: dict[str, dict] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise EvaluationError(f"invalid corpus manifest line {line_number}: {error}") from error
            language = record.get("language")
            if language in records:
                raise EvaluationError(f"duplicate corpus manifest language: {language!r}")
            records[language] = record
    if set(records) != set(TRAINING_LANGUAGES):
        raise EvaluationError(
            f"corpus manifest languages are {sorted(records)}, expected {sorted(TRAINING_LANGUAGES)}"
        )
    non_holdout = sorted(language for language, record in records.items() if record.get("split") != "holdout")
    if non_holdout:
        raise EvaluationError(
            f"evaluation input must be the independently generated holdout; non-holdout records: {non_holdout}"
        )
    return records


def _priority(seed: int, language: str, text_sha256: str) -> int:
    payload = f"{seed}\0{language}\0{text_sha256}".encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest(), "big")


def _keep_smallest(
    heap: list[tuple[int, int, int, str, str]],
    *,
    limit: int,
    priority: int,
    line_number: int,
    text: str,
    text_sha256: str,
) -> None:
    item = (-priority, -line_number, line_number, text, text_sha256)
    if len(heap) < limit:
        heapq.heappush(heap, item)
    elif item > heap[0]:
        heapq.heapreplace(heap, item)


def _scan_language_samples(
    corpus_path: Path,
    record: Mapping,
    *,
    language: str,
    seed: int,
    sample_size: int,
    long_quota: int,
) -> tuple[list[dict], dict]:
    overall: list[tuple[int, int, int, str, str]] = []
    long_lines: list[tuple[int, int, int, str, str]] = []
    file_digest = hashlib.sha256()
    observed_samples = 0
    observed_characters = 0
    started = time.monotonic()
    next_update = started + 10.0
    with corpus_path.open("rb") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            file_digest.update(raw_line)
            try:
                text = raw_line.decode("utf-8")
            except UnicodeDecodeError as error:
                raise EvaluationError(
                    f"{corpus_path} line {line_number} is not valid UTF-8: {error}"
                ) from error
            text = text.rstrip("\r\n")
            observed_samples += 1
            observed_characters += len(text)
            if text:
                text_sha256 = hashlib.sha256(text.encode("utf-8")).hexdigest()
                priority = _priority(seed, language, text_sha256)
                _keep_smallest(
                    overall,
                    limit=sample_size + long_quota,
                    priority=priority,
                    line_number=line_number,
                    text=text,
                    text_sha256=text_sha256,
                )
                if len(text) > LONG_CHARACTER_THRESHOLD:
                    _keep_smallest(
                        long_lines,
                        limit=long_quota,
                        priority=priority,
                        line_number=line_number,
                        text=text,
                        text_sha256=text_sha256,
                    )
            now = time.monotonic()
            if now >= next_update:
                progress(
                    f"SAMPLE {language}: lines={observed_samples:,}, "
                    f"chars={observed_characters:,}"
                )
                next_update = now + 10.0

    actual_sha256 = file_digest.hexdigest()
    expected_sha256 = record.get("sha256")
    if expected_sha256 and actual_sha256 != expected_sha256:
        raise EvaluationError(
            f"corpus hash mismatch for {language}: {actual_sha256} != {expected_sha256}"
        )
    for key, actual in (("samples", observed_samples), ("characters", observed_characters)):
        expected = record.get(key)
        if expected is not None and int(expected) != actual:
            raise EvaluationError(
                f"corpus {key} mismatch for {language}: {actual:,} != {int(expected):,}"
            )

    def unpack(item: tuple[int, int, int, str, str]) -> tuple[int, int, str, str]:
        negative_priority, _, line_number, text, text_sha256 = item
        return -negative_priority, line_number, text, text_sha256

    mandatory = sorted((unpack(item) for item in long_lines))
    candidates = sorted((unpack(item) for item in overall))
    selected: dict[str, tuple[int, int, str, str]] = {}
    for item in mandatory:
        selected.setdefault(item[3], item)
    for item in candidates:
        if len(selected) >= sample_size:
            break
        selected.setdefault(item[3], item)
    if len(selected) != sample_size:
        raise EvaluationError(
            f"could only select {len(selected)} unique rows for {language}; need {sample_size}"
        )
    ordered = sorted(selected.values())
    actual_long = sum(len(item[2]) > LONG_CHARACTER_THRESHOLD for item in ordered)
    if actual_long < long_quota:
        raise EvaluationError(
            f"selected only {actual_long} long rows for {language}; need {long_quota}"
        )

    rows: list[dict] = []
    for index, (_, line_number, text, text_sha256) in enumerate(ordered, start=1):
        rows.append(
            {
                "schema_version": SAMPLE_SCHEMA_VERSION,
                "sample_id": f"{language}-corpus-{index:04d}",
                "language": language,
                "category": "corpus_random",
                "source": "corpus",
                "line_number": line_number,
                "text_sha256": text_sha256,
                "text": text,
            }
        )
    for index, (category, text) in enumerate(STRESS_PROBES[language], start=1):
        rows.append(
            {
                "schema_version": SAMPLE_SCHEMA_VERSION,
                "sample_id": f"{language}-probe-{index:02d}",
                "language": language,
                "category": category,
                "source": "synthetic_probe",
                "line_number": None,
                "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "text": text,
            }
        )
    scan = {
        "source_file": record.get("file", corpus_path.name),
        "source_sha256": actual_sha256,
        "source_samples": observed_samples,
        "source_characters": observed_characters,
        "selected_corpus_samples": sample_size,
        "selected_long_samples": actual_long,
        "synthetic_probes": len(STRESS_PROBES[language]),
    }
    return rows, scan


def _sample_file_payload(rows: Sequence[Mapping]) -> str:
    return "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        for row in rows
    )


def _validate_existing_sample_set(
    sample_dir: Path,
    *,
    corpus_manifest_sha256: str,
    seed: int,
    sample_size: int,
    long_quota: int,
) -> dict | None:
    manifest_path = sample_dir / "manifest.json"
    if not manifest_path.is_file():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    expected = {
        "schema_version": SAMPLE_SCHEMA_VERSION,
        "sampling_algorithm": SAMPLING_ALGORITHM,
        "corpus_manifest_sha256": corpus_manifest_sha256,
        "seed": seed,
        "corpus_samples_per_language": sample_size,
        "long_quota_per_language": long_quota,
    }
    if any(manifest.get(key) != value for key, value in expected.items()):
        return None
    for language in TRAINING_LANGUAGES:
        record = (manifest.get("files") or {}).get(language) or {}
        path = sample_dir / record.get("path", "")
        if not path.is_file() or sha256_file(path) != record.get("sha256"):
            return None
    return manifest


def build_evaluation_sample_set(
    corpus_dir: Path,
    sample_dir: Path,
    *,
    seed: int = DEFAULT_SEED,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    long_quota: int = DEFAULT_LONG_QUOTA,
    rebuild: bool = False,
) -> dict:
    if sample_size < 1:
        raise ValueError("sample_size must be positive")
    if long_quota < 0 or long_quota > sample_size:
        raise ValueError("long_quota must be between zero and sample_size")
    corpus_dir = corpus_dir.resolve()
    sample_dir = sample_dir.resolve()
    corpus_manifest_path = corpus_dir / "manifest.jsonl"
    if not corpus_manifest_path.is_file():
        raise EvaluationError(f"missing corpus manifest: {corpus_manifest_path}")
    corpus_manifest_sha256 = sha256_file(corpus_manifest_path)
    if not rebuild:
        existing = _validate_existing_sample_set(
            sample_dir,
            corpus_manifest_sha256=corpus_manifest_sha256,
            seed=seed,
            sample_size=sample_size,
            long_quota=long_quota,
        )
        if existing is not None:
            progress(f"SAMPLE reuse validated fixed set: {sample_dir}")
            return existing

    records = _load_corpus_manifest(corpus_manifest_path)
    sample_dir.mkdir(parents=True, exist_ok=True)
    file_records: dict[str, dict] = {}
    scans: dict[str, dict] = {}
    for language in TRAINING_LANGUAGES:
        record = records[language]
        corpus_path = corpus_dir / record["file"]
        progress(f"SAMPLE {language}: scanning {corpus_path}")
        rows, scan = _scan_language_samples(
            corpus_path,
            record,
            language=language,
            seed=seed,
            sample_size=sample_size,
            long_quota=long_quota,
        )
        output = sample_dir / f"{language}.jsonl"
        atomic_write_text(output, _sample_file_payload(rows))
        file_records[language] = {
            "path": output.name,
            "rows": len(rows),
            "corpus_rows": sample_size,
            "synthetic_rows": len(STRESS_PROBES[language]),
            "bytes": output.stat().st_size,
            "sha256": sha256_file(output),
        }
        scans[language] = scan
        progress(f"SAMPLE {language}: wrote {len(rows):,} rows to {output}")
    manifest = {
        "schema_version": SAMPLE_SCHEMA_VERSION,
        "sampling_algorithm": SAMPLING_ALGORITHM,
        "corpus_manifest_path": corpus_manifest_path.name,
        "corpus_manifest_sha256": corpus_manifest_sha256,
        "seed": seed,
        "corpus_samples_per_language": sample_size,
        "long_character_threshold": LONG_CHARACTER_THRESHOLD,
        "long_quota_per_language": long_quota,
        "files": file_records,
        "source_scans": scans,
    }
    atomic_write_json(sample_dir / "manifest.json", manifest)
    progress(f"SAMPLE fixed evaluation set complete: {sample_dir}")
    return manifest


def load_evaluation_samples(sample_dir: Path, manifest: Mapping) -> dict[str, list[dict]]:
    result: dict[str, list[dict]] = {}
    seen_ids: set[str] = set()
    for language in TRAINING_LANGUAGES:
        record = manifest["files"][language]
        path = sample_dir / record["path"]
        rows: list[dict] = []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                row = json.loads(line)
                if row.get("language") != language:
                    raise EvaluationError(f"wrong language in {path}:{line_number}")
                if row.get("sample_id") in seen_ids:
                    raise EvaluationError(f"duplicate sample_id: {row.get('sample_id')}")
                if hashlib.sha256(row["text"].encode("utf-8")).hexdigest() != row.get("text_sha256"):
                    raise EvaluationError(f"text hash mismatch in {path}:{line_number}")
                seen_ids.add(row["sample_id"])
                rows.append(row)
        if len(rows) != record["rows"]:
            raise EvaluationError(f"row count mismatch for {path}: {len(rows)} != {record['rows']}")
        result[language] = rows
    return result


def _ratio(numerator: int | float, denominator: int | float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def _percentile(values: Sequence[int], percentile: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[index]


@dataclass
class TextAggregate:
    samples: int = 0
    source_characters: int = 0
    evaluated_characters: int = 0
    tokens: int = 0
    unk_tokens: int = 0
    unk_source_characters: int = 0
    unmapped_unk_tokens: int = 0
    exact_roundtrips: int = 0
    token_lengths: list[int] = field(default_factory=list)
    roundtrip_mismatches: list[dict] = field(default_factory=list)

    def add(self, item: Mapping) -> None:
        self.samples += 1
        self.source_characters += item["source_characters"]
        self.evaluated_characters += item["evaluated_characters"]
        self.tokens += item["tokens"]
        self.unk_tokens += item["unk_tokens"]
        self.unk_source_characters += item["unk_source_characters"]
        self.unmapped_unk_tokens += item["unmapped_unk_tokens"]
        self.exact_roundtrips += int(item["roundtrip_exact"])
        self.token_lengths.append(item["tokens"])
        if not item["roundtrip_exact"] and len(self.roundtrip_mismatches) < 5:
            self.roundtrip_mismatches.append(item["roundtrip_mismatch"])

    def as_dict(self) -> dict:
        return {
            "samples": self.samples,
            "source_characters": self.source_characters,
            "evaluated_non_whitespace_characters": self.evaluated_characters,
            "tokens": self.tokens,
            "unk_tokens": self.unk_tokens,
            "unk_token_ratio": _ratio(self.unk_tokens, self.tokens),
            "unk_source_characters": self.unk_source_characters,
            "source_character_loss_rate": _ratio(
                self.unk_source_characters, self.evaluated_characters
            ),
            "unmapped_unk_tokens": self.unmapped_unk_tokens,
            "mean_tokens_per_sample": _ratio(self.tokens, self.samples),
            "tokens_per_non_whitespace_character": _ratio(
                self.tokens, self.evaluated_characters
            ),
            "token_length_p50": _percentile(self.token_lengths, 0.50),
            "token_length_p95": _percentile(self.token_lengths, 0.95),
            "token_length_p99": _percentile(self.token_lengths, 0.99),
            "token_length_max": max(self.token_lengths, default=0),
            "exact_roundtrips": self.exact_roundtrips,
            "roundtrip_exact_rate": _ratio(self.exact_roundtrips, self.samples),
            "roundtrip_mismatches": self.roundtrip_mismatches,
        }


def _evaluate_text(tokenizer, row: Mapping) -> tuple[dict, set[int], list[int]]:
    text = row["text"]
    encoding = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
        truncation=False,
    )
    ids = list(encoding["input_ids"])
    offsets = list(encoding["offset_mapping"])
    if len(ids) != len(offsets):
        raise EvaluationError(f"token/offset length mismatch for {row['sample_id']}")
    unknown_positions: set[int] = set()
    unmapped_unk_tokens = 0
    for token_id, offset in zip(ids, offsets, strict=True):
        if token_id != tokenizer.unk_token_id:
            continue
        start, end = int(offset[0]), int(offset[1])
        mapped = False
        for index in range(max(0, start), min(len(text), end)):
            if not text[index].isspace():
                unknown_positions.add(index)
                mapped = True
        if not mapped:
            unmapped_unk_tokens += 1
    decoded = tokenizer.decode(
        ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    exact = decoded == text
    mismatch = {
        "sample_id": row["sample_id"],
        "source": text[:240],
        "decoded": decoded[:240],
    }
    item = {
        "source_characters": len(text),
        "evaluated_characters": sum(not character.isspace() for character in text),
        "tokens": len(ids),
        "unk_tokens": sum(token_id == tokenizer.unk_token_id for token_id in ids),
        "unk_source_characters": len(unknown_positions),
        "unmapped_unk_tokens": unmapped_unk_tokens,
        "roundtrip_exact": exact,
        "roundtrip_mismatch": mismatch,
    }
    return item, unknown_positions, ids


def classify_character(character: str) -> str:
    codepoint = ord(character)
    category = unicodedata.category(character)
    if 0x3400 <= codepoint <= 0x4DBF or 0x4E00 <= codepoint <= 0x9FFF or 0x20000 <= codepoint <= 0x323AF:
        return "Han"
    if 0x3040 <= codepoint <= 0x309F:
        return "Hiragana"
    if 0x30A0 <= codepoint <= 0x30FF or 0x31F0 <= codepoint <= 0x31FF:
        return "Katakana"
    if 0xAC00 <= codepoint <= 0xD7A3:
        return "Hangul syllable"
    if 0x1100 <= codepoint <= 0x11FF or 0x3130 <= codepoint <= 0x318F or 0xA960 <= codepoint <= 0xD7FF:
        return "Hangul Jamo"
    if character.isascii() and character.isalpha():
        return "Latin"
    if category.startswith("L"):
        return "Other letter"
    if category.startswith("N"):
        return "Number"
    if category.startswith("P"):
        return "Punctuation"
    if category.startswith("S"):
        return "Symbol/emoji"
    return "Other"


def _character_coverage(tokenizer, counter: Counter[str], examples: Mapping[str, dict]) -> dict:
    covered: dict[str, bool] = {}
    for character in counter:
        ids = tokenizer.encode(character, add_special_tokens=False)
        covered[character] = bool(ids) and tokenizer.unk_token_id not in ids
    covered_frequency = sum(count for character, count in counter.items() if covered[character])
    covered_unique = sum(covered.values())
    uncovered = []
    by_script: dict[str, dict[str, int]] = defaultdict(lambda: {"unique": 0, "frequency": 0})
    for character, count in counter.items():
        script = classify_character(character)
        if not covered[character]:
            by_script[script]["unique"] += 1
            by_script[script]["frequency"] += count
            uncovered.append(
                {
                    "char": character,
                    "codepoint": f"U+{ord(character):04X}",
                    "unicode_name": unicodedata.name(character, "UNNAMED"),
                    "script": script,
                    "count": count,
                    "example": examples.get(character),
                }
            )
    uncovered.sort(key=lambda item: (-item["count"], ord(item["char"])))
    return {
        "total_character_frequency": sum(counter.values()),
        "covered_character_frequency": covered_frequency,
        "frequency_weighted_coverage": _ratio(covered_frequency, sum(counter.values())),
        "total_unique_characters": len(counter),
        "covered_unique_characters": covered_unique,
        "unique_character_coverage": _ratio(covered_unique, len(counter)),
        "uncovered_by_script": dict(sorted(by_script.items())),
        "uncovered_characters": uncovered,
    }


def _example_context(text: str, index: int, radius: int = 40) -> str:
    start = max(0, index - radius)
    end = min(len(text), index + radius + 1)
    return text[start:end]


def _english_subword_metrics(tokenizer, word_counts: Counter[str]) -> dict:
    rows = []
    total_occurrences = sum(word_counts.values())
    weighted_pieces = 0
    one_piece = 0
    over_four = 0
    unknown = 0
    for word, count in word_counts.items():
        ids = tokenizer.encode(word, add_special_tokens=False)
        pieces = len(ids)
        has_unk = tokenizer.unk_token_id in ids
        weighted_pieces += pieces * count
        one_piece += int(pieces == 1 and not has_unk) * count
        over_four += int(pieces > 4) * count
        unknown += int(has_unk) * count
        rows.append({"word": word, "count": count, "pieces": pieces, "has_unk": has_unk})
    rows.sort(key=lambda item: (-item["pieces"], -item["count"], item["word"].casefold()))
    return {
        "word_occurrences": total_occurrences,
        "unique_words": len(word_counts),
        "mean_pieces_per_word": _ratio(weighted_pieces, total_occurrences),
        "single_piece_word_rate": _ratio(one_piece, total_occurrences),
        "over_four_piece_word_rate": _ratio(over_four, total_occurrences),
        "unknown_word_rate": _ratio(unknown, total_occurrences),
        "most_fragmented_words": rows[:20],
    }


def _cjk_shared_metrics(tokenizer, inventories: Mapping[str, Counter[str]]) -> dict:
    simplified = {char for char in inventories["zho_Hans"] if classify_character(char) == "Han"}
    traditional = {char for char in inventories["zho_Hant"] if classify_character(char) == "Han"}
    japanese = {char for char in inventories["jpn_Jpan"] if classify_character(char) == "Han"}
    shared = sorted(simplified & traditional & japanese, key=ord)
    simplified_traditional = simplified & traditional
    saved_src_lang = tokenizer.src_lang
    mismatches = []
    covered = 0
    piece_total = 0
    try:
        for character in shared:
            tokenizer.src_lang = "zho_Hans"
            zh_ids = tokenizer.encode(character, add_special_tokens=False)
            tokenizer.src_lang = "jpn_Jpan"
            ja_ids = tokenizer.encode(character, add_special_tokens=False)
            piece_total += len(zh_ids)
            if tokenizer.unk_token_id not in zh_ids:
                covered += 1
            if zh_ids != ja_ids and len(mismatches) < 20:
                mismatches.append(
                    {"char": character, "codepoint": f"U+{ord(character):04X}", "zh_ids": zh_ids, "ja_ids": ja_ids}
                )
    finally:
        tokenizer.src_lang = saved_src_lang
    return {
        "shared_unique_han": len(shared),
        "simplified_traditional_shared_unique_han": len(simplified_traditional),
        "covered_shared_han": covered,
        "shared_han_coverage": _ratio(covered, len(shared)),
        "mean_standalone_pieces": _ratio(piece_total, len(shared)),
        "source_language_split_mismatches": len(mismatches),
        "mismatch_examples": mismatches,
    }


def _chinese_sequence_parity(corpus_metrics: Mapping[str, Mapping]) -> dict:
    simplified = corpus_metrics["zho_Hans"]
    traditional = corpus_metrics["zho_Hant"]
    simplified_fertility = float(simplified["tokens_per_non_whitespace_character"])
    traditional_fertility = float(traditional["tokens_per_non_whitespace_character"])
    simplified_p95 = int(simplified["token_length_p95"])
    traditional_p95 = int(traditional["token_length_p95"])
    return {
        "simplified_tokens_per_character": simplified_fertility,
        "traditional_tokens_per_character": traditional_fertility,
        "traditional_to_simplified_fertility_ratio": _ratio(traditional_fertility, simplified_fertility),
        "traditional_minus_simplified_fertility": traditional_fertility - simplified_fertility,
        "simplified_token_length_p95": simplified_p95,
        "traditional_token_length_p95": traditional_p95,
        "traditional_to_simplified_p95_ratio": _ratio(traditional_p95, simplified_p95),
    }


def _korean_metrics(tokenizer, inventory: Counter[str]) -> dict:
    hangul = Counter(
        {character: count for character, count in inventory.items() if classify_character(character) in {"Hangul syllable", "Hangul Jamo"}}
    )
    covered_unique = 0
    covered_frequency = 0
    missing = []
    for character, count in hangul.items():
        ids = tokenizer.encode(character, add_special_tokens=False)
        if tokenizer.unk_token_id not in ids:
            covered_unique += 1
            covered_frequency += count
        else:
            missing.append({"char": character, "codepoint": f"U+{ord(character):04X}", "count": count})
    missing.sort(key=lambda item: (-item["count"], ord(item["char"])))
    return {
        "unique_hangul": len(hangul),
        "hangul_frequency": sum(hangul.values()),
        "unique_coverage": _ratio(covered_unique, len(hangul)),
        "frequency_weighted_coverage": _ratio(covered_frequency, sum(hangul.values())),
        "missing": missing[:50],
    }


def evaluate_tokenizer(
    tokenizer,
    samples: Mapping[str, Sequence[Mapping]],
    *,
    label: str,
    artifact_path: Path,
    training_metadata: Mapping | None = None,
    sample_manifest: Mapping | None = None,
) -> dict:
    verify_tokenizer(tokenizer)
    corpus_aggregates = {language: TextAggregate() for language in TRAINING_LANGUAGES}
    all_aggregates = {language: TextAggregate() for language in TRAINING_LANGUAGES}
    long_aggregates = {language: TextAggregate() for language in TRAINING_LANGUAGES}
    category_aggregates: dict[str, dict[str, TextAggregate]] = {
        language: defaultdict(TextAggregate) for language in TRAINING_LANGUAGES
    }
    inventories = {language: Counter() for language in TRAINING_LANGUAGES}
    inventory_examples: dict[str, dict[str, dict]] = {language: {} for language in TRAINING_LANGUAGES}
    english_words: Counter[str] = Counter()
    used_token_ids: set[int] = set()
    observed_unknowns: dict[str, Counter[str]] = {language: Counter() for language in TRAINING_LANGUAGES}

    for language in TRAINING_LANGUAGES:
        tokenizer.src_lang = language
        for row in samples[language]:
            item, unknown_positions, ids = _evaluate_text(tokenizer, row)
            all_aggregates[language].add(item)
            category_aggregates[language][row["category"]].add(item)
            if row["source"] != "corpus":
                continue
            corpus_aggregates[language].add(item)
            used_token_ids.update(ids)
            text = row["text"]
            if len(text) > LONG_CHARACTER_THRESHOLD:
                long_aggregates[language].add(item)
            for index, character in enumerate(text):
                if character.isspace():
                    continue
                inventories[language][character] += 1
                inventory_examples[language].setdefault(
                    character,
                    {"sample_id": row["sample_id"], "context": _example_context(text, index)},
                )
            for index in unknown_positions:
                observed_unknowns[language][text[index]] += 1
            if language == "eng_Latn":
                english_words.update(match.group(0) for match in ENGLISH_WORD_RE.finditer(text))

    character_coverage = {
        language: _character_coverage(tokenizer, inventories[language], inventory_examples[language])
        for language in TRAINING_LANGUAGES
    }
    total_aggregate = TextAggregate()
    for aggregate in corpus_aggregates.values():
        total_aggregate.samples += aggregate.samples
        total_aggregate.source_characters += aggregate.source_characters
        total_aggregate.evaluated_characters += aggregate.evaluated_characters
        total_aggregate.tokens += aggregate.tokens
        total_aggregate.unk_tokens += aggregate.unk_tokens
        total_aggregate.unk_source_characters += aggregate.unk_source_characters
        total_aggregate.unmapped_unk_tokens += aggregate.unmapped_unk_tokens
        total_aggregate.exact_roundtrips += aggregate.exact_roundtrips
        total_aggregate.token_lengths.extend(aggregate.token_lengths)
        total_aggregate.roundtrip_mismatches.extend(aggregate.roundtrip_mismatches)
    total_character_frequency = sum(
        coverage["total_character_frequency"] for coverage in character_coverage.values()
    )
    total_covered_frequency = sum(
        coverage["covered_character_frequency"] for coverage in character_coverage.values()
    )
    total_unique = sum(coverage["total_unique_characters"] for coverage in character_coverage.values())
    total_unique_covered = sum(
        coverage["covered_unique_characters"] for coverage in character_coverage.values()
    )
    corpus_metric_values = {
        language: corpus_aggregates[language].as_dict() for language in TRAINING_LANGUAGES
    }
    metrics = {
        "schema_version": METRICS_SCHEMA_VERSION,
        "label": label,
        "artifact_path": artifact_path.as_posix(),
        "vocab_size": len(tokenizer),
        "training_metadata": dict(training_metadata or {}),
        "sample_set": {
            "path": (sample_manifest or {}).get("sample_dir"),
            "corpus_manifest_sha256": (sample_manifest or {}).get("corpus_manifest_sha256"),
            "seed": (sample_manifest or {}).get("seed"),
            "corpus_samples_per_language": (sample_manifest or {}).get("corpus_samples_per_language"),
            "long_quota_per_language": (sample_manifest or {}).get("long_quota_per_language"),
        },
        "corpus_metrics": corpus_metric_values,
        "total_corpus_metrics": total_aggregate.as_dict(),
        "all_sample_metrics": {
            language: all_aggregates[language].as_dict() for language in TRAINING_LANGUAGES
        },
        "category_metrics": {
            language: {
                category: aggregate.as_dict()
                for category, aggregate in sorted(category_aggregates[language].items())
            }
            for language in TRAINING_LANGUAGES
        },
        "long_sentence_metrics": {
            language: long_aggregates[language].as_dict() for language in TRAINING_LANGUAGES
        },
        "character_coverage": character_coverage,
        "total_character_coverage": {
            "frequency_weighted_coverage": _ratio(total_covered_frequency, total_character_frequency),
            "unique_character_coverage": _ratio(total_unique_covered, total_unique),
            "total_character_frequency": total_character_frequency,
            "total_unique_characters_by_language": total_unique,
        },
        "observed_unknown_characters": {
            language: [
                {"char": char, "codepoint": f"U+{ord(char):04X}", "count": count, "script": classify_character(char)}
                for char, count in observed_unknowns[language].most_common(50)
            ]
            for language in TRAINING_LANGUAGES
        },
        "cjk_shared_han": _cjk_shared_metrics(tokenizer, inventories),
        "chinese_sequence_parity": _chinese_sequence_parity(corpus_metric_values),
        "korean_hangul": _korean_metrics(tokenizer, inventories["kor_Hang"]),
        "english_subwords": _english_subword_metrics(tokenizer, english_words),
        "vocab_utilization": {
            "used_non_special_ids": len(used_token_ids),
            "vocab_size": len(tokenizer),
            "ratio": _ratio(len(used_token_ids), len(tokenizer)),
        },
    }
    if metrics["total_corpus_metrics"]["unmapped_unk_tokens"]:
        raise EvaluationError(
            f"{label} produced unknown tokens without usable source offsets: "
            f"{metrics['total_corpus_metrics']['unmapped_unk_tokens']}"
        )
    return metrics


def _pct(value: float) -> str:
    return f"{value * 100:.6f}%"


def _number(value: float) -> str:
    return f"{value:.4f}"


def _escape_cell(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _markdown_table(headers: Sequence[str], rows: Iterable[Sequence[object]]) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(_escape_cell(value) for value in row) + " |" for row in rows)
    return lines


def render_candidate_report(metrics: Mapping) -> str:
    training = metrics.get("training_metadata") or {}
    sample_fraction = training.get("sample_fraction")
    sampled_corpus = sample_fraction is not None and float(sample_fraction) < 1.0
    lines = [
        f"# Tokenizer coverage report: {metrics['label']}",
        "",
    ]
    if sampled_corpus:
        lines.extend(
            [
                f"> Sampled-corpus result: this tokenizer was trained with sample_fraction={sample_fraction}. "
                "Compare it only with candidates that use the same training provenance; this report does not claim full-corpus coverage.",
                "",
            ]
        )
    lines.extend(
        [
            "## Provenance",
            "",
            f"- Artifact: `{metrics['artifact_path']}`",
            f"- Vocabulary size: {metrics['vocab_size']:,}",
            f"- Training sample fraction: {sample_fraction if sample_fraction is not None else 'unknown'}",
            f"- Evaluation seed: {metrics['sample_set'].get('seed')}",
            f"- Corpus samples per language: {metrics['sample_set'].get('corpus_samples_per_language')}",
            f"- Evaluation corpus manifest SHA-256: `{metrics['sample_set'].get('corpus_manifest_sha256')}`",
            "",
            "## Corpus metrics",
            "",
        ]
    )
    rows = []
    for language in TRAINING_LANGUAGES:
        item = metrics["corpus_metrics"][language]
        coverage = metrics["character_coverage"][language]
        rows.append(
            (
                language,
                item["samples"],
                _number(item["mean_tokens_per_sample"]),
                _number(item["tokens_per_non_whitespace_character"]),
                _pct(item["unk_token_ratio"]),
                _pct(item["source_character_loss_rate"]),
                _pct(coverage["frequency_weighted_coverage"]),
                _pct(coverage["unique_character_coverage"]),
                _pct(item["roundtrip_exact_rate"]),
            )
        )
    lines.extend(
        _markdown_table(
            [
                "Language",
                "Samples",
                "Mean tokens",
                "Tokens/char",
                "UNK tokens",
                "Source loss",
                "Freq coverage",
                "Unique coverage",
                "Exact roundtrip",
            ],
            rows,
        )
    )
    lines.extend(["", "## Token length distribution", ""])
    lines.extend(
        _markdown_table(
            ["Language", "P50", "P95", "P99", "Max", ">500-char samples", "Long-sample UNK"],
            (
                (
                    language,
                    metrics["corpus_metrics"][language]["token_length_p50"],
                    metrics["corpus_metrics"][language]["token_length_p95"],
                    metrics["corpus_metrics"][language]["token_length_p99"],
                    metrics["corpus_metrics"][language]["token_length_max"],
                    metrics["long_sentence_metrics"][language]["samples"],
                    _pct(metrics["long_sentence_metrics"][language]["unk_token_ratio"]),
                )
                for language in TRAINING_LANGUAGES
            ),
        )
    )
    lines.extend(["", "## Synthetic stress probes", ""])
    category_rows = []
    for language in TRAINING_LANGUAGES:
        for category, item in metrics["category_metrics"][language].items():
            if category == "corpus_random":
                continue
            category_rows.append(
                (
                    language,
                    category,
                    item["samples"],
                    item["tokens"],
                    _pct(item["unk_token_ratio"]),
                    _pct(item["source_character_loss_rate"]),
                    _pct(item["roundtrip_exact_rate"]),
                )
            )
    lines.extend(
        _markdown_table(
            ["Language", "Category", "Samples", "Tokens", "UNK tokens", "Source loss", "Exact roundtrip"],
            category_rows,
        )
    )
    lines.extend(["", "## Script-specific analysis", ""])
    cjk = metrics["cjk_shared_han"]
    chinese_parity = metrics["chinese_sequence_parity"]
    korean = metrics["korean_hangul"]
    english = metrics["english_subwords"]
    lines.extend(
        [
            f"- Shared Chinese/Japanese Han: {cjk['shared_unique_han']:,} unique, {_pct(cjk['shared_han_coverage'])} covered, "
            f"{cjk['source_language_split_mismatches']} standalone source-language split mismatches.",
            f"- Simplified/Traditional sequence parity: tokens/character "
            f"{_number(chinese_parity['simplified_tokens_per_character'])} vs "
            f"{_number(chinese_parity['traditional_tokens_per_character'])}; Traditional/Simplified ratio "
            f"{_number(chinese_parity['traditional_to_simplified_fertility_ratio'])}; P95 ratio "
            f"{_number(chinese_parity['traditional_to_simplified_p95_ratio'])}.",
            f"- Korean Hangul: {korean['unique_hangul']:,} unique syllables/Jamo, "
            f"{_pct(korean['frequency_weighted_coverage'])} frequency-weighted coverage and {_pct(korean['unique_coverage'])} unique coverage.",
            f"- English subwords: {_number(english['mean_pieces_per_word'])} pieces/word, "
            f"{_pct(english['single_piece_word_rate'])} one-piece words, {_pct(english['over_four_piece_word_rate'])} over four pieces.",
            f"- Evaluation vocabulary utilization: {metrics['vocab_utilization']['used_non_special_ids']:,}/{metrics['vocab_size']:,} "
            f"({_pct(metrics['vocab_utilization']['ratio'])}).",
            "",
            "## Highest-frequency uncovered characters",
            "",
        ]
    )
    for language in TRAINING_LANGUAGES:
        uncovered = metrics["character_coverage"][language]["uncovered_characters"][:15]
        lines.extend([f"### {language}", ""])
        if not uncovered:
            lines.extend(["No uncovered characters in the fixed corpus sample.", ""])
            continue
        lines.extend(
            _markdown_table(
                ["Character", "Code point", "Script", "Count", "Example"],
                (
                    (
                        item["char"],
                        item["codepoint"],
                        item["script"],
                        item["count"],
                        (item.get("example") or {}).get("context", "")[:100],
                    )
                    for item in uncovered
                ),
            )
        )
        lines.append("")
    lines.extend(
        [
            "## Metric definitions",
            "",
            "- Main metrics use only the fixed corpus sample; synthetic probes are reported separately.",
            "- Source-character loss counts non-whitespace Unicode code points covered by `<unk>` offsets. A fused unknown span may count as several lost characters.",
            "- Frequency-weighted and unique-character coverage encode each observed character independently; they are distinct from contextual source-character loss.",
            "- Tokens/char divides non-special tokens by non-whitespace source characters.",
            "- Exact roundtrip requires `decode(encode(text)) == text` with no cleanup or normalization.",
            "",
        ]
    )
    return "\n".join(lines)


def build_comparison(metrics_by_label: Mapping[str, Mapping]) -> dict:
    return {
        "schema_version": METRICS_SCHEMA_VERSION,
        "candidates": {
            label: {
                "vocab_size": metrics["vocab_size"],
                "training_sample_fraction": (metrics.get("training_metadata") or {}).get("sample_fraction"),
                "total_corpus_metrics": metrics["total_corpus_metrics"],
                "total_character_coverage": metrics["total_character_coverage"],
                "vocab_utilization": metrics["vocab_utilization"],
                "languages": {
                    language: {
                        "corpus_metrics": metrics["corpus_metrics"][language],
                        "character_coverage": {
                            key: metrics["character_coverage"][language][key]
                            for key in (
                                "frequency_weighted_coverage",
                                "unique_character_coverage",
                                "total_unique_characters",
                            )
                        },
                    }
                    for language in TRAINING_LANGUAGES
                },
            }
            for label, metrics in metrics_by_label.items()
        },
    }


def render_comparison_report(metrics_by_label: Mapping[str, Mapping]) -> str:
    lines = ["# Tokenizer candidate comparison", ""]
    if any(float((metrics.get("training_metadata") or {}).get("sample_fraction", 1.0)) < 1.0 for metrics in metrics_by_label.values()):
        lines.extend(
            [
                "> Sampled-corpus comparison. At least one candidate used sample_fraction < 1. "
                "Compare candidates only when their training provenance matches; this report does not claim full-corpus coverage.",
                "",
            ]
        )
    rows = []
    for label, metrics in metrics_by_label.items():
        total = metrics["total_corpus_metrics"]
        coverage = metrics["total_character_coverage"]
        rows.append(
            (
                label,
                metrics["vocab_size"],
                (metrics.get("training_metadata") or {}).get("sample_fraction", "unknown"),
                _number(total["tokens_per_non_whitespace_character"]),
                _pct(total["unk_token_ratio"]),
                _pct(total["source_character_loss_rate"]),
                _pct(coverage["frequency_weighted_coverage"]),
                _pct(coverage["unique_character_coverage"]),
                _pct(metrics["vocab_utilization"]["ratio"]),
            )
        )
    lines.extend(
        _markdown_table(
            [
                "Candidate",
                "Vocab",
                "Train fraction",
                "Tokens/char",
                "UNK tokens",
                "Source loss",
                "Freq coverage",
                "Unique coverage",
                "Vocab used",
            ],
            rows,
        )
    )
    lines.extend(["", "## Per-language comparison", ""])
    rows = []
    for language in TRAINING_LANGUAGES:
        for label, metrics in metrics_by_label.items():
            item = metrics["corpus_metrics"][language]
            coverage = metrics["character_coverage"][language]
            rows.append(
                (
                    language,
                    label,
                    _number(item["mean_tokens_per_sample"]),
                    _number(item["tokens_per_non_whitespace_character"]),
                    _pct(item["unk_token_ratio"]),
                    _pct(item["source_character_loss_rate"]),
                    _pct(coverage["frequency_weighted_coverage"]),
                    _pct(coverage["unique_character_coverage"]),
                )
            )
    lines.extend(
        _markdown_table(
            ["Language", "Candidate", "Mean tokens", "Tokens/char", "UNK tokens", "Source loss", "Freq coverage", "Unique coverage"],
            rows,
        )
    )
    lines.append("")
    return "\n".join(lines)


def parse_candidate(value: str) -> tuple[str, Path]:
    label, separator, path = value.partition("=")
    if not separator or not label.strip() or not path.strip():
        raise argparse.ArgumentTypeError("candidate must use LABEL=PATH")
    if not re.fullmatch(r"[A-Za-z0-9._-]+", label):
        raise argparse.ArgumentTypeError("candidate label may contain only letters, digits, dot, underscore, and dash")
    return label, Path(path)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-dir", type=Path, default=Path("data/tokenizer/holdout/mvp"))
    parser.add_argument("--sample-dir", type=Path, default=Path("data/tokenizer/evaluation/mvp-v0"))
    parser.add_argument("--report-dir", type=Path, default=Path("artifacts/tokenizers/reports"))
    parser.add_argument("--candidate", action="append", type=parse_candidate, required=True, metavar="LABEL=PATH")
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    parser.add_argument("--long-quota", type=int, default=DEFAULT_LONG_QUOTA)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--rebuild-samples", action="store_true")
    return parser.parse_args(argv)


def summarize_training_metadata(metadata: Mapping, *, metadata_sha256: str | None = None) -> dict:
    """Keep evaluation provenance portable and free of machine-local paths."""
    summary = {
        key: metadata.get(key)
        for key in (
            "schema_version",
            "vocab_size",
            "random_seed",
            "sample_fraction",
            "sampling_algorithm",
            "balancing_algorithm",
            "min_frequency",
            "limit_alphabet",
            "initial_alphabet_size",
            "initial_alphabet_sha256",
            "batch_size",
            "input_order_sha256",
            "total_training_lines",
            "total_training_characters",
            "corpus_unique_characters",
            "missing_corpus_characters",
            "versions",
        )
        if key in metadata
    }
    corpus_manifest = metadata.get("corpus_manifest") or {}
    if corpus_manifest.get("sha256"):
        summary["corpus_manifest_sha256"] = corpus_manifest["sha256"]
    if metadata_sha256 is not None:
        summary["training_meta_sha256"] = metadata_sha256
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    candidates = dict(args.candidate)
    if len(candidates) != len(args.candidate):
        raise EvaluationError("candidate labels must be unique")
    manifest = build_evaluation_sample_set(
        args.corpus_dir,
        args.sample_dir,
        seed=args.seed,
        sample_size=args.sample_size,
        long_quota=args.long_quota,
        rebuild=args.rebuild_samples,
    )
    manifest = dict(manifest)
    manifest["sample_dir"] = args.sample_dir.as_posix()
    samples = load_evaluation_samples(args.sample_dir.resolve(), manifest)
    report_dir = args.report_dir.resolve()
    report_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict] = {}
    for label, artifact_path in candidates.items():
        resolved_artifact_path = artifact_path.resolve()
        progress(f"EVALUATE {label}: loading {resolved_artifact_path}")
        tokenizer = reload_tokenizer(resolved_artifact_path)
        training_meta_path = resolved_artifact_path / "training_meta.json"
        if training_meta_path.is_file():
            raw_training_metadata = json.loads(training_meta_path.read_text(encoding="utf-8"))
            training_metadata = summarize_training_metadata(
                raw_training_metadata,
                metadata_sha256=sha256_file(training_meta_path),
            )
        else:
            training_metadata = {}
        metrics = evaluate_tokenizer(
            tokenizer,
            samples,
            label=label,
            artifact_path=artifact_path,
            training_metadata=training_metadata,
            sample_manifest=manifest,
        )
        atomic_write_json(report_dir / f"coverage-{label}.json", metrics)
        atomic_write_text(report_dir / f"coverage-{label}.md", render_candidate_report(metrics))
        results[label] = metrics
        progress(f"EVALUATE {label}: reports complete")
    comparison = build_comparison(results)
    atomic_write_json(report_dir / "comparison.json", comparison)
    atomic_write_text(report_dir / "comparison.md", render_comparison_report(results))
    progress(f"EVALUATE all candidates complete: {report_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
