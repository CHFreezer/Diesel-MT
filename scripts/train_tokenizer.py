#!/usr/bin/env python3
"""Train reproducible Diesel-MT NLLB BPE tokenizers from balanced corpora.

The corpus is read from the data disk once.  Selected text stays in RAM and is
fed to ``NllbTokenizer.train_new_from_iterator`` in deterministic multilingual
batches.  Progress is reported with newline-delimited messages, including a
separate heartbeat process that remains visible while the native Rust trainer
is blocking the caller.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import queue
import random
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from functools import cached_property
from pathlib import Path


if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from tokenizer_utils import (  # noqa: E402
    PROJECT_LANGUAGES,
    TRAINING_LANGUAGES,
    TokenizerValidationError,
    atomic_write_json,
    backend_pipeline,
    build_language_mapping,
    create_seed_tokenizer,
    reload_tokenizer,
    save_language_mapping,
    verify_tokenizer,
)


GIB = 1024**3
ARTIFACT_MANIFEST_NAME = "artifact_manifest.json"
CORPUS_MANIFEST_NAME = "manifest.jsonl"
SAMPLING_ALGORITHM = "per-language-seeded-bernoulli-v1"
BALANCING_ALGORITHM = "equal-selected-unicode-characters-v1"
VOCAB_CANDIDATE_NAMES = {
    32768: "mvp-32k",
    49152: "mvp-48k",
}


class CorpusValidationError(RuntimeError):
    """Raised when corpus files do not match their manifest."""


class MemorySafetyError(RuntimeError):
    """Raised before the process crosses a configured memory guard."""


def progress(message: str) -> None:
    """Print one timestamped, immediately flushed progress line."""
    stamp = datetime.now().astimezone().strftime("%H:%M:%S")
    print(f"[{stamp}] {message}", flush=True)


def sha256_file(path: Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def stable_seed(seed: int, namespace: str) -> int:
    digest = hashlib.sha256(f"{seed}\0{namespace}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def _legacy_cjk_characters(encoding: str) -> set[str]:
    """Return CJK characters represented by a stable standard legacy codec."""
    result: set[str] = set()
    for first in range(256):
        for second in range(256):
            try:
                decoded = bytes((first, second)).decode(encoding)
            except UnicodeDecodeError:
                continue
            if len(decoded) == 1 and "\u3400" <= decoded <= "\u9fff":
                result.add(decoded)
    return result


def build_must_cover_alphabet() -> frozenset[str]:
    """Build the fixed project alphabet passed directly to ``BpeTrainer``.

    The set deliberately reserves real vocabulary capacity for BPE merges.  It
    includes printable ASCII/Latin-1, common Chinese/Japanese Han characters,
    full kana ranges, all modern Hangul syllables and Jamo, CJK punctuation,
    full-width forms, and the Metaspace marker.  Rare CJK extensions remain an
    explicit coverage-report concern because byte fallback is disabled.
    """
    characters: set[str] = {"▁"}
    characters.update(chr(codepoint) for codepoint in range(0x21, 0x7F))
    characters.update(chr(codepoint) for codepoint in range(0xA1, 0x100))
    characters.update(chr(codepoint) for codepoint in range(0x3000, 0x3040))
    characters.update(chr(codepoint) for codepoint in range(0x3040, 0x3100))
    characters.update(chr(codepoint) for codepoint in range(0x1100, 0x1200))
    characters.update(chr(codepoint) for codepoint in range(0x3130, 0x3190))
    characters.update(chr(codepoint) for codepoint in range(0xA960, 0xA980))
    characters.update(chr(codepoint) for codepoint in range(0xAC00, 0xD7A4))
    characters.update(chr(codepoint) for codepoint in range(0xD7B0, 0xD800))
    characters.update(chr(codepoint) for codepoint in range(0xFF01, 0xFFA0))
    characters.update(_legacy_cjk_characters("gb2312"))
    characters.update(_legacy_cjk_characters("shift_jis"))
    characters.update(
        "臺灣香港澳門國學體醫藥衞門風雲龍廣東萬與專業叢絲兩嚴喪個豐臨為麗舉麼義烏樂喬習鄉書買亂爭於虧亞產畝親億僅從侖倉儀們價眾優會傘偉傳傷倫偽佇佈佔來侶俠係俁倆倉個們倫偉側偵儉債傾儲兒兌黨蘭關興養獸內岡冊寫軍農馮沖決況凍淨涼減湊凜幾鳳憑凱擊鑿劃劉則剛創刪別剮製剎劑勁動務勛勝勞勢勳勵勸勻匭匯區協單賣盧衛卻廠廳歷厲壓厭廁廂廈廚廢廣歸當錄彙徑復徵德憶應懷態慣慘慶憂戲戶撲執擴掃揚換據擇擔擬擁攔攙攝攜擺搖摺敵數齊齋鬥斷無舊時晉暫術樸機殺雜權條來楊極樓標樣樹橋檔檢櫃欄歡歐殲殼毀氣漢湯溝滅滬潔淚潛澤濟濤灣濕濺燈靈災爐爭愛爺牆獨獲環現產電畫異疊療瘋癡發盜盞監盤睜瞞矚礦碼磚禮禍禦種稱穀積穩窩窪窮竄競筆筍築範簽簡糧糾紀約紅級紋納純紙紛紡紐線練組細終紹經結給絡絕統絲綠維綱網綜緊緒線緣編緩緬縣縱總績織繼續纖罰罵職聯聰肅腦腳脫臉臘臺舊艦艙藝節範薦蘇處虛號蟲蠶補裝裡製複見規覺覽觀觸譽計訂認討訓議訊記講許論訴診詞該詳語誤說請諸諾讀課調談謀謂謝證識譜警譯讓貝貞負財貢貧貨販貪貫責貴貸貿費賀賊賓賜賞賠賢賣賦質賴贊趕趙跡踐車軌軍軒軟轉輪輯輸辦邊遼達遷過運還這進遠違連遲適選遺郵鄧鄭醜醫釋鐘鋼錢錄鍋錯鎖鎮鏡鐵鑄長門閃閉問閒間聞閩閥閣閱隊陽陰陣階際陸陳險隨隱難雜雞離電靜頂項順須預領頭頻題顏額風飛飯飲飾餅館馬馳駕駛驚驗髮鬚鬥魚鳥鳴鴨鷹麥黃點黨齒齡"
    )
    return frozenset(characters)


MUST_COVER_ALPHABET = build_must_cover_alphabet()


@dataclass
class MemoryStatus:
    rss_bytes: int | None
    peak_rss_bytes: int | None
    available_bytes: int | None


def memory_status() -> MemoryStatus:
    """Read process and system memory using only the standard library."""
    rss: int | None = None
    peak: int | None = None
    available: int | None = None
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class ProcessMemoryCountersEx(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
                ("PrivateUsage", ctypes.c_size_t),
            ]

        class MemoryStatusEx(ctypes.Structure):
            _fields_ = [
                ("dwLength", wintypes.DWORD),
                ("dwMemoryLoad", wintypes.DWORD),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        kernel32 = ctypes.windll.kernel32
        psapi = ctypes.windll.psapi
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(ProcessMemoryCountersEx),
            wintypes.DWORD,
        ]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        counters = ProcessMemoryCountersEx()
        counters.cb = ctypes.sizeof(counters)
        if psapi.GetProcessMemoryInfo(
            kernel32.GetCurrentProcess(),
            ctypes.byref(counters),
            counters.cb,
        ):
            rss = int(counters.WorkingSetSize)
            peak = int(counters.PeakWorkingSetSize)
        system = MemoryStatusEx()
        system.dwLength = ctypes.sizeof(system)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(system)):
            available = int(system.ullAvailPhys)
    else:
        try:
            values: dict[str, int] = {}
            for line in Path("/proc/self/status").read_text(encoding="ascii").splitlines():
                if line.startswith(("VmRSS:", "VmHWM:")):
                    key, value, _unit = line.split()
                    values[key.rstrip(":")] = int(value) * 1024
            rss = values.get("VmRSS")
            peak = values.get("VmHWM")
            for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines():
                if line.startswith("MemAvailable:"):
                    available = int(line.split()[1]) * 1024
                    break
        except OSError:
            pass
    return MemoryStatus(rss_bytes=rss, peak_rss_bytes=peak, available_bytes=available)


def format_gib(value: int | None) -> str:
    return "unknown" if value is None else f"{value / GIB:.2f} GiB"


def process_rss_bytes(process_id: int) -> int | None:
    """Return another process's RSS, or ``None`` after it exits."""
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        kernel32 = ctypes.windll.kernel32
        psapi = ctypes.windll.psapi
        kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel32.OpenProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(ProcessMemoryCounters),
            wintypes.DWORD,
        ]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        handle = kernel32.OpenProcess(0x1000 | 0x0400, False, process_id)
        if not handle:
            return None
        try:
            counters = ProcessMemoryCounters()
            counters.cb = ctypes.sizeof(counters)
            if not psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
                return 0
            return int(counters.WorkingSetSize)
        finally:
            kernel32.CloseHandle(handle)
    try:
        for line in (Path("/proc") / str(process_id) / "status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    except OSError:
        return None
    return 0


def enforce_memory_guards(*, max_memory_gib: float | None, min_available_memory_gib: float) -> None:
    status = memory_status()
    if max_memory_gib is not None and status.rss_bytes is not None:
        if status.rss_bytes > max_memory_gib * GIB:
            raise MemorySafetyError(
                f"RSS {format_gib(status.rss_bytes)} exceeded --max-memory-gib={max_memory_gib}"
            )
    if min_available_memory_gib > 0 and status.available_bytes is not None:
        if status.available_bytes < min_available_memory_gib * GIB:
            raise MemorySafetyError(
                f"available RAM {format_gib(status.available_bytes)} fell below "
                f"--min-available-memory-gib={min_available_memory_gib}"
            )


@dataclass
class LanguageCorpusStats:
    language: str
    file: str
    sha256: str
    bytes: int
    scanned_lines: int
    scanned_characters: int
    sampled_lines_before_balance: int
    sampled_characters_before_balance: int
    training_lines: int
    training_characters: int


@dataclass
class LoadedCorpus:
    lines_by_language: dict[str, list[str]]
    character_counts: Counter[str]
    language_stats: dict[str, LanguageCorpusStats]
    manifest_path: Path
    manifest_sha256: str
    seed: int
    sample_fraction: float
    load_elapsed_s: float

    @cached_property
    def total_lines(self) -> int:
        return sum(len(lines) for lines in self.lines_by_language.values())

    @cached_property
    def total_characters(self) -> int:
        return sum(sum(map(len, lines)) for lines in self.lines_by_language.values())


def read_corpus_manifest(corpus_dir: Path) -> tuple[Path, dict[str, dict]]:
    manifest_path = corpus_dir / CORPUS_MANIFEST_NAME
    if not manifest_path.is_file():
        raise CorpusValidationError(f"corpus manifest not found: {manifest_path}")
    records: dict[str, dict] = {}
    with manifest_path.open(encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            if not raw.strip():
                continue
            try:
                record = json.loads(raw)
            except json.JSONDecodeError as error:
                raise CorpusValidationError(
                    f"invalid JSON in {manifest_path}:{line_number}: {error}"
                ) from error
            language = record.get("language")
            if not isinstance(language, str) or language in records:
                raise CorpusValidationError(
                    f"invalid or duplicate language in {manifest_path}:{line_number}: {language!r}"
                )
            records[language] = record
    missing = sorted(set(TRAINING_LANGUAGES) - set(records))
    if missing:
        raise CorpusValidationError(f"manifest is missing training languages: {missing}")
    return manifest_path, records


def _trim_to_character_budget(lines: list[str], budget: int) -> tuple[list[str], int]:
    kept: list[str] = []
    used = 0
    for line in lines:
        length = len(line)
        if used + length <= budget:
            kept.append(line)
            used += length
    if not kept:
        raise CorpusValidationError(f"character budget {budget} retained no training lines")
    return kept, used


def load_balanced_corpus(
    corpus_dir: Path,
    *,
    seed: int,
    sample_fraction: float = 1.0,
    progress_interval_s: float = 10.0,
    max_memory_gib: float | None = None,
    min_available_memory_gib: float = 0.0,
) -> LoadedCorpus:
    """Read each corpus file once, deterministically sample, then char-balance."""
    if not 0 < sample_fraction <= 1:
        raise ValueError("sample_fraction must be in the interval (0, 1]")
    corpus_dir = corpus_dir.resolve()
    manifest_path, manifest = read_corpus_manifest(corpus_dir)
    started = time.perf_counter()
    lines_by_language: dict[str, list[str]] = {}
    counters_by_language: dict[str, Counter[str]] = {}
    stats: dict[str, LanguageCorpusStats] = {}

    progress(
        f"LOAD start: languages={list(TRAINING_LANGUAGES)}, "
        f"sample_fraction={sample_fraction:.3f}, manifest={manifest_path}"
    )
    for language in TRAINING_LANGUAGES:
        record = manifest[language]
        expected_file = record.get("file")
        if expected_file != f"{language}.txt":
            raise CorpusValidationError(
                f"manifest file for {language} is {expected_file!r}, expected {language + '.txt'!r}"
            )
        path = corpus_dir / expected_file
        if not path.is_file():
            raise CorpusValidationError(f"corpus file not found: {path}")
        expected_bytes = int(record["bytes"])
        expected_lines = int(record["samples"])
        expected_characters = int(record["characters"])
        expected_sha256 = str(record["sha256"])
        rng = random.Random(stable_seed(seed, f"sample:{language}"))
        selected: list[str] = []
        character_counts: Counter[str] = Counter()
        digest = hashlib.sha256()
        scanned_bytes = 0
        scanned_lines = 0
        scanned_characters = 0
        selected_characters = 0
        file_started = time.perf_counter()
        next_report = file_started + max(progress_interval_s, 0.1)
        progress(f"LOAD {language}: reading {expected_bytes / GIB:.2f} GiB sequentially")
        with path.open("rb", buffering=8 * 1024 * 1024) as handle:
            for raw_line in handle:
                digest.update(raw_line)
                scanned_bytes += len(raw_line)
                scanned_lines += 1
                payload = raw_line.rstrip(b"\r\n")
                if not payload:
                    raise CorpusValidationError(f"empty line at {path}:{scanned_lines}")
                try:
                    line = payload.decode("utf-8", errors="strict")
                except UnicodeDecodeError as error:
                    raise CorpusValidationError(
                        f"invalid UTF-8 at {path}:{scanned_lines}: {error}"
                    ) from error
                scanned_characters += len(line)
                take = sample_fraction == 1.0 or rng.random() < sample_fraction
                if take:
                    selected.append(line)
                    selected_characters += len(line)
                    character_counts.update(line)
                if scanned_lines % 8192 == 0:
                    enforce_memory_guards(
                        max_memory_gib=max_memory_gib,
                        min_available_memory_gib=min_available_memory_gib,
                    )
                    now = time.perf_counter()
                    if now >= next_report:
                        elapsed = now - file_started
                        percent = 100 * scanned_bytes / expected_bytes if expected_bytes else 100.0
                        status = memory_status()
                        progress(
                            f"LOAD {language}: {percent:5.1f}% "
                            f"({scanned_bytes / GIB:.2f}/{expected_bytes / GIB:.2f} GiB), "
                            f"lines={scanned_lines:,}, selected={len(selected):,}, "
                            f"rate={scanned_bytes / max(elapsed, 1e-9) / 1024**2:.1f} MiB/s, "
                            f"rss={format_gib(status.rss_bytes)}"
                        )
                        next_report = now + max(progress_interval_s, 0.1)
        actual_sha256 = digest.hexdigest()
        errors: list[str] = []
        if scanned_bytes != expected_bytes:
            errors.append(f"bytes={scanned_bytes}, expected={expected_bytes}")
        if scanned_lines != expected_lines:
            errors.append(f"lines={scanned_lines}, expected={expected_lines}")
        if scanned_characters != expected_characters:
            errors.append(f"characters={scanned_characters}, expected={expected_characters}")
        if actual_sha256 != expected_sha256:
            errors.append(f"sha256={actual_sha256}, expected={expected_sha256}")
        if errors:
            raise CorpusValidationError(f"corpus validation failed for {path}: {'; '.join(errors)}")
        if not selected:
            raise CorpusValidationError(
                f"sampling selected zero lines for {language}; increase --sample-fraction"
            )
        random.Random(stable_seed(seed, f"shuffle:{language}")).shuffle(selected)
        lines_by_language[language] = selected
        counters_by_language[language] = character_counts
        stats[language] = LanguageCorpusStats(
            language=language,
            file=expected_file,
            sha256=actual_sha256,
            bytes=scanned_bytes,
            scanned_lines=scanned_lines,
            scanned_characters=scanned_characters,
            sampled_lines_before_balance=len(selected),
            sampled_characters_before_balance=selected_characters,
            training_lines=0,
            training_characters=0,
        )
        progress(
            f"LOAD {language}: verified sha256, selected={len(selected):,} lines / "
            f"{selected_characters:,} chars"
        )

    target_characters = min(
        item.sampled_characters_before_balance for item in stats.values()
    )
    final_counts: Counter[str] = Counter()
    for language in TRAINING_LANGUAGES:
        original_lines = lines_by_language[language]
        retained, retained_characters = _trim_to_character_budget(
            original_lines, target_characters
        )
        if len(retained) != len(original_lines):
            retained_counter: Counter[str] = Counter()
            for line in retained:
                retained_counter.update(line)
            counters_by_language[language] = retained_counter
        lines_by_language[language] = retained
        stats[language].training_lines = len(retained)
        stats[language].training_characters = retained_characters
        final_counts.update(counters_by_language[language])
        progress(
            f"BALANCE {language}: lines={len(retained):,}, chars={retained_characters:,}, "
            f"target={target_characters:,}"
        )

    elapsed = time.perf_counter() - started
    loaded = LoadedCorpus(
        lines_by_language=lines_by_language,
        character_counts=final_counts,
        language_stats=stats,
        manifest_path=manifest_path,
        manifest_sha256=sha256_file(manifest_path),
        seed=seed,
        sample_fraction=sample_fraction,
        load_elapsed_s=elapsed,
    )
    progress(
        f"LOAD complete: lines={loaded.total_lines:,}, chars={loaded.total_characters:,}, "
        f"unique_chars={len(loaded.character_counts):,}, elapsed={elapsed:.1f}s"
    )
    return loaded


class BalancedBatchIterator:
    """Yield deterministic, round-robin multilingual batches from RAM."""

    def __init__(
        self,
        corpus: LoadedCorpus,
        *,
        batch_size: int,
        label: str,
        progress_interval_s: float,
        max_memory_gib: float | None = None,
        min_available_memory_gib: float = 0.0,
    ) -> None:
        if batch_size < len(TRAINING_LANGUAGES):
            raise ValueError(
                f"batch_size must be at least {len(TRAINING_LANGUAGES)}"
            )
        self.corpus = corpus
        self.batch_size = batch_size
        self.label = label
        self.progress_interval_s = progress_interval_s
        self.max_memory_gib = max_memory_gib
        self.min_available_memory_gib = min_available_memory_gib
        self.order_sha256: str | None = None
        self.yielded_lines = 0

    def __iter__(self) -> Iterator[list[str]]:
        positions = {language: 0 for language in TRAINING_LANGUAGES}
        total = self.corpus.total_lines
        digest = hashlib.sha256()
        started = time.perf_counter()
        next_report = started + max(self.progress_interval_s, 0.1)
        batch: list[str] = []
        remaining = total
        while remaining:
            made_progress = False
            for language in TRAINING_LANGUAGES:
                position = positions[language]
                lines = self.corpus.lines_by_language[language]
                if position >= len(lines):
                    continue
                line = lines[position]
                positions[language] = position + 1
                remaining -= 1
                made_progress = True
                encoded = line.encode("utf-8")
                digest.update(len(encoded).to_bytes(8, "big"))
                digest.update(encoded)
                batch.append(line)
                self.yielded_lines += 1
                if len(batch) >= self.batch_size:
                    yield batch
                    batch = []
                    enforce_memory_guards(
                        max_memory_gib=self.max_memory_gib,
                        min_available_memory_gib=self.min_available_memory_gib,
                    )
                    now = time.perf_counter()
                    if now >= next_report:
                        progress(
                            f"FEED {self.label}: {self.yielded_lines:,}/{total:,} lines "
                            f"({100 * self.yielded_lines / total:.1f}%) passed to Rust trainer"
                        )
                        next_report = now + max(self.progress_interval_s, 0.1)
            if not made_progress:
                raise RuntimeError("balanced iterator stopped before consuming all lines")
        if batch:
            yield batch
        self.order_sha256 = digest.hexdigest()
        progress(f"FEED {self.label}: 100.0% of lines passed to Rust trainer")


HEARTBEAT_CODE = r"""
import ctypes, os, pathlib, sys, time
pid = int(sys.argv[1]); interval = float(sys.argv[2]); label = sys.argv[3]
started = time.monotonic()
def status():
    if os.name == 'nt':
        from ctypes import wintypes
        class PMC(ctypes.Structure):
            _fields_=[('cb',wintypes.DWORD),('PageFaultCount',wintypes.DWORD),('PeakWorkingSetSize',ctypes.c_size_t),('WorkingSetSize',ctypes.c_size_t),('a',ctypes.c_size_t),('b',ctypes.c_size_t),('c',ctypes.c_size_t),('d',ctypes.c_size_t),('e',ctypes.c_size_t),('f',ctypes.c_size_t)]
        kernel32=ctypes.windll.kernel32; psapi=ctypes.windll.psapi
        kernel32.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD]
        kernel32.OpenProcess.restype=wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes=[wintypes.HANDLE,ctypes.POINTER(PMC),wintypes.DWORD]
        psapi.GetProcessMemoryInfo.restype=wintypes.BOOL
        handle=kernel32.OpenProcess(0x1000|0x0400,False,pid)
        if not handle: return None
        counters=PMC(); counters.cb=ctypes.sizeof(counters)
        ok=psapi.GetProcessMemoryInfo(handle,ctypes.byref(counters),counters.cb)
        kernel32.CloseHandle(handle)
        return int(counters.WorkingSetSize) if ok else 0
    path=pathlib.Path('/proc')/str(pid)/'status'
    try:
        for line in path.read_text().splitlines():
            if line.startswith('VmRSS:'): return int(line.split()[1])*1024
    except OSError: return None
    return 0
while True:
    time.sleep(interval)
    rss=status()
    if rss is None: break
    rss_text='unknown' if not rss else f'{rss/1024**3:.2f} GiB'
    print(f'[{time.strftime("%H:%M:%S")}] HEARTBEAT {label}: still running, elapsed={time.monotonic()-started:.1f}s, rss={rss_text}', flush=True)
"""


class NativeTrainingHeartbeat:
    """Emit progress from another process while native training blocks Python."""

    def __init__(self, label: str, interval_s: float) -> None:
        self.label = label
        self.interval_s = interval_s
        self.process: subprocess.Popen | None = None
        self.relay_thread: threading.Thread | None = None

    def _relay_output(self) -> None:
        if self.process is None or self.process.stdout is None:
            return
        for line in self.process.stdout:
            print(line.rstrip("\r\n"), flush=True)

    def __enter__(self) -> "NativeTrainingHeartbeat":
        if self.interval_s <= 0:
            return self
        creationflags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        try:
            self.process = subprocess.Popen(
                [
                    sys.executable,
                    "-u",
                    "-c",
                    HEARTBEAT_CODE,
                    str(os.getpid()),
                    str(self.interval_s),
                    self.label,
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                creationflags=creationflags,
            )
            self.relay_thread = threading.Thread(
                target=self._relay_output,
                name=f"heartbeat-relay-{self.label}",
                daemon=True,
            )
            self.relay_thread.start()
        except OSError as error:
            progress(f"WARNING: could not start independent heartbeat: {error}")
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self.process is None:
            return
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=3)
        if self.relay_thread is not None:
            self.relay_thread.join(timeout=3)
        if self.process.stdout is not None:
            self.process.stdout.close()


@dataclass(frozen=True)
class TrainingConfig:
    vocab_size: int
    min_frequency: int
    limit_alphabet: int | None
    num_threads: int
    batch_size: int
    heartbeat_interval_s: float
    native_progress: bool


def effective_alphabet_limit(vocab_size: int, requested: int | None, initial_size: int) -> int:
    special_count = 5 + len(PROJECT_LANGUAGES)
    merge_reserve = min(1024, max(32, vocab_size // 8))
    maximum = vocab_size - special_count - merge_reserve
    if maximum < initial_size:
        minimum_vocab = initial_size + special_count + merge_reserve
        raise ValueError(
            f"vocab_size={vocab_size} is too small for initial_alphabet={initial_size}; "
            f"use at least {minimum_vocab}"
        )
    value = min(maximum, initial_size + 2048) if requested is None else requested
    if value < initial_size:
        raise ValueError(
            f"limit_alphabet={value} is smaller than initial_alphabet={initial_size}"
        )
    if value > maximum:
        raise ValueError(
            f"limit_alphabet={value} leaves fewer than {merge_reserve} BPE merge slots; "
            f"maximum is {maximum} for vocab_size={vocab_size}"
        )
    return value


def configure_tokenizer_threads(num_threads: int) -> dict[str, str]:
    if num_threads < 1:
        raise ValueError("num_threads must be at least 1")
    desired = str(num_threads)
    existing = os.environ.get("RAYON_NUM_THREADS")
    if existing is not None and existing != desired:
        raise RuntimeError(
            f"RAYON_NUM_THREADS is already {existing}; cannot safely change the global "
            f"tokenizers pool to {desired} in this process"
        )
    os.environ["RAYON_NUM_THREADS"] = desired
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    return {
        "RAYON_NUM_THREADS": os.environ["RAYON_NUM_THREADS"],
        "TOKENIZERS_PARALLELISM": os.environ["TOKENIZERS_PARALLELISM"],
    }


def character_is_covered(tokenizer, character: str) -> bool:
    ids = tokenizer.encode(character, add_special_tokens=False)
    return bool(ids) and tokenizer.unk_token_id not in ids


def build_alphabet_audit(
    tokenizer,
    character_counts: Mapping[str, int],
    initial_alphabet: frozenset[str],
) -> dict:
    all_characters = sorted(set(character_counts) | set(initial_alphabet), key=ord)
    rows = []
    missing_initial: list[str] = []
    missing_corpus: list[str] = []
    for character in all_characters:
        covered = character_is_covered(tokenizer, character)
        in_initial = character in initial_alphabet
        count = int(character_counts.get(character, 0))
        if in_initial and not covered:
            missing_initial.append(character)
        if count and not covered:
            missing_corpus.append(character)
        rows.append(
            {
                "char": character,
                "codepoint": f"U+{ord(character):04X}",
                "count": count,
                "in_initial_alphabet": in_initial,
                "covered_without_unk": covered,
            }
        )
    return {
        "initial_alphabet_size": len(initial_alphabet),
        "corpus_unique_characters": len(character_counts),
        "missing_initial_alphabet": missing_initial,
        "missing_corpus_characters": missing_corpus,
        "characters": rows,
    }


def _artifact_file_records(directory: Path) -> list[dict]:
    records: list[dict] = []
    for path in sorted(item for item in directory.rglob("*") if item.is_file()):
        if path.name == ARTIFACT_MANIFEST_NAME:
            continue
        records.append(
            {
                "path": path.relative_to(directory).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return records


def write_artifact_manifest(directory: Path) -> dict:
    payload = {
        "schema_version": 1,
        "files": _artifact_file_records(directory),
    }
    atomic_write_json(directory / ARTIFACT_MANIFEST_NAME, payload)
    return payload


def _copy_and_verify(source: Path, destination: Path, manifest: dict) -> None:
    destination.mkdir(parents=True, exist_ok=False)
    records = {record["path"]: record for record in manifest["files"]}
    for relative, record in records.items():
        source_path = source / relative
        destination_path = destination / relative
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, destination_path)
        actual = sha256_file(destination_path)
        if actual != record["sha256"]:
            raise RuntimeError(
                f"artifact copy hash mismatch for {relative}: {actual} != {record['sha256']}"
            )
        progress(f"PUBLISH copied+verified {relative} ({record['bytes']:,} bytes)")
    shutil.copy2(source / ARTIFACT_MANIFEST_NAME, destination / ARTIFACT_MANIFEST_NAME)


def publish_artifact(staged_directory: Path, output_directory: Path, manifest: dict) -> None:
    """Copy to a hidden incoming directory, verify, then swap directories."""
    output_directory = output_directory.resolve()
    output_directory.parent.mkdir(parents=True, exist_ok=True)
    incoming = output_directory.parent / f".{output_directory.name}.incoming-{uuid.uuid4().hex}"
    backup = output_directory.parent / f".{output_directory.name}.backup-{uuid.uuid4().hex}"
    progress(f"PUBLISH staging {staged_directory} -> {incoming}")
    try:
        _copy_and_verify(staged_directory, incoming, manifest)
        if output_directory.exists():
            os.replace(output_directory, backup)
        try:
            os.replace(incoming, output_directory)
        except BaseException:
            if backup.exists() and not output_directory.exists():
                os.replace(backup, output_directory)
            raise
        if backup.exists():
            try:
                shutil.rmtree(backup)
            except OSError as error:
                progress(f"WARNING: new artifact is live but old backup cleanup failed: {error}")
    except BaseException:
        if incoming.exists():
            shutil.rmtree(incoming, ignore_errors=True)
        raise
    progress(f"PUBLISH complete: {output_directory}")


def train_candidate(
    corpus: LoadedCorpus,
    config: TrainingConfig,
    *,
    output_dir: Path,
    staging_dir: Path,
    initial_alphabet: frozenset[str] = MUST_COVER_ALPHABET,
    max_memory_gib: float | None = None,
    min_available_memory_gib: float = 0.0,
) -> object:
    """Train, validate, stage and atomically publish one vocabulary candidate."""
    import tokenizers
    import transformers

    enforce_memory_guards(
        max_memory_gib=max_memory_gib,
        min_available_memory_gib=min_available_memory_gib,
    )
    parallelism = configure_tokenizer_threads(config.num_threads)
    alphabet_limit = effective_alphabet_limit(
        config.vocab_size, config.limit_alphabet, len(initial_alphabet)
    )
    label = f"vocab-{config.vocab_size}"
    progress(
        f"TRAIN {label} start: lines={corpus.total_lines:,}, "
        f"chars={corpus.total_characters:,}, initial_alphabet={len(initial_alphabet):,}, "
        f"limit_alphabet={alphabet_limit:,}, threads={config.num_threads}"
    )
    seed_tokenizer = create_seed_tokenizer()
    pre_training_backend = backend_pipeline(seed_tokenizer)
    iterator = BalancedBatchIterator(
        corpus,
        batch_size=config.batch_size,
        label=label,
        progress_interval_s=config.heartbeat_interval_s,
        max_memory_gib=max_memory_gib,
        min_available_memory_gib=min_available_memory_gib,
    )
    started = time.perf_counter()
    nested_heartbeat_interval = (
        0.0
        if os.environ.get("DIESEL_MT_TOKENIZER_SUPERVISED") == "1"
        else config.heartbeat_interval_s
    )
    with NativeTrainingHeartbeat(label, nested_heartbeat_interval):
        tokenizer = seed_tokenizer.train_new_from_iterator(
            iterator,
            vocab_size=config.vocab_size,
            length=corpus.total_lines,
            min_frequency=config.min_frequency,
            limit_alphabet=alphabet_limit,
            initial_alphabet=sorted(initial_alphabet, key=ord),
            show_progress=config.native_progress,
        )
    train_elapsed = time.perf_counter() - started
    progress(f"TRAIN {label} native call returned after {train_elapsed:.1f}s; validating")
    verify_tokenizer(tokenizer, expected_vocab_size=config.vocab_size)
    post_training_backend = backend_pipeline(tokenizer)
    alphabet_audit = build_alphabet_audit(
        tokenizer, corpus.character_counts, initial_alphabet
    )
    if alphabet_audit["missing_initial_alphabet"]:
        missing = alphabet_audit["missing_initial_alphabet"]
        raise TokenizerValidationError(
            f"{len(missing)} must-cover characters encode as <unk>: {missing[:20]!r}"
        )

    staging_dir = staging_dir.resolve()
    staging_dir.mkdir(parents=True, exist_ok=True)
    working = Path(
        tempfile.mkdtemp(prefix=f"diesel-mt-{config.vocab_size}-", dir=staging_dir)
    )
    try:
        tokenizer.src_lang = "eng_Latn"
        tokenizer.save_pretrained(str(working))
        shutil.copy2(corpus.manifest_path, working / "corpus_manifest.jsonl")
        language_mapping = build_language_mapping(tokenizer)
        save_language_mapping(language_mapping, working / "language_map.json")
        atomic_write_json(working / "alphabet_audit.json", alphabet_audit)

        reloaded = reload_tokenizer(working)
        verify_tokenizer(reloaded, expected_vocab_size=config.vocab_size)
        if reloaded.get_vocab() != tokenizer.get_vocab():
            raise TokenizerValidationError("vocabulary changed after save/reload")
        if backend_pipeline(reloaded) != backend_pipeline(tokenizer):
            raise TokenizerValidationError("backend pipeline changed after save/reload")

        memory = memory_status()
        model_summary = {
            key: post_training_backend["model"].get(key)
            for key in (
                "type",
                "dropout",
                "unk_token",
                "continuing_subword_prefix",
                "end_of_word_suffix",
                "fuse_unk",
                "byte_fallback",
            )
        }
        seed_model_summary = {
            key: pre_training_backend["model"].get(key)
            for key in model_summary
        }
        metadata = {
            "schema_version": 1,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "vocab_size": config.vocab_size,
            "random_seed": corpus.seed,
            "sample_fraction": corpus.sample_fraction,
            "sampling_algorithm": SAMPLING_ALGORITHM,
            "balancing_algorithm": BALANCING_ALGORITHM,
            "min_frequency": config.min_frequency,
            "limit_alphabet": alphabet_limit,
            "initial_alphabet_size": len(initial_alphabet),
            "initial_alphabet_sha256": hashlib.sha256(
                "".join(sorted(initial_alphabet, key=ord)).encode("utf-8")
            ).hexdigest(),
            "batch_size": config.batch_size,
            "input_order_sha256": iterator.order_sha256,
            "total_training_lines": corpus.total_lines,
            "total_training_characters": corpus.total_characters,
            "corpus_unique_characters": len(corpus.character_counts),
            "missing_corpus_characters": len(alphabet_audit["missing_corpus_characters"]),
            "load_elapsed_s": round(corpus.load_elapsed_s, 3),
            "train_elapsed_s": round(train_elapsed, 3),
            "training_characters_per_second": round(
                corpus.total_characters / max(train_elapsed, 1e-9), 3
            ),
            "rss_bytes_at_save": memory.rss_bytes,
            "peak_rss_bytes": memory.peak_rss_bytes,
            "available_memory_bytes_at_save": memory.available_bytes,
            "parallelism": parallelism,
            "versions": {
                "python": platform.python_version(),
                "transformers": transformers.__version__,
                "tokenizers": tokenizers.__version__,
                "platform": platform.platform(),
            },
            "corpus_manifest": {
                "source": str(corpus.manifest_path),
                "sha256": corpus.manifest_sha256,
                "snapshot_file": "corpus_manifest.jsonl",
            },
            "languages": {
                language: asdict(corpus.language_stats[language])
                for language in TRAINING_LANGUAGES
            },
            "language_token_ids": language_mapping,
            "backend": {
                "model": model_summary,
                "normalizer": post_training_backend.get("normalizer"),
                "pre_tokenizer": post_training_backend.get("pre_tokenizer"),
                "post_processor": post_training_backend.get("post_processor"),
                "decoder": post_training_backend.get("decoder"),
            },
            "seed_backend_model": seed_model_summary,
        }
        atomic_write_json(working / "training_meta.json", metadata)
        artifact_manifest = write_artifact_manifest(working)
        progress(
            f"VALIDATE {label} complete: vocab={len(reloaded):,}, "
            f"missing_must_cover=0, files={len(artifact_manifest['files'])}"
        )
        publish_artifact(working, output_dir, artifact_manifest)
    finally:
        shutil.rmtree(working, ignore_errors=True)

    final_tokenizer = reload_tokenizer(output_dir)
    verify_tokenizer(final_tokenizer, expected_vocab_size=config.vocab_size)
    progress(f"DONE {label}: artifact={output_dir.resolve()}")
    return final_tokenizer


def default_num_threads() -> int:
    return max(1, min(8, (os.cpu_count() or 2) // 2))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus-dir",
        type=Path,
        default=Path("data/tokenizer/corpus/mvp"),
    )
    parser.add_argument(
        "--vocab-size",
        action="append",
        type=int,
        required=True,
        help="Repeat to train multiple candidates from one in-memory corpus load.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--staging-dir", type=Path, default=None)
    parser.add_argument("--sample-fraction", type=float, default=1.0)
    parser.add_argument("--min-frequency", type=int, default=2)
    parser.add_argument("--limit-alphabet", type=int, default=None)
    parser.add_argument("--num-threads", type=int, default=default_num_threads())
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--heartbeat-interval", type=float, default=10.0)
    parser.add_argument("--load-progress-interval", type=float, default=10.0)
    parser.add_argument("--max-memory-gib", type=float, default=96.0)
    parser.add_argument("--min-available-memory-gib", type=float, default=8.0)
    parser.add_argument(
        "--no-native-progress",
        action="store_true",
        help="Disable the Rust carriage-return progress bar; newline heartbeats remain enabled.",
    )
    parser.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def run_worker(args: argparse.Namespace) -> int:
    vocab_sizes = list(dict.fromkeys(args.vocab_size))
    if any(size <= 0 for size in vocab_sizes):
        raise ValueError("all vocabulary sizes must be positive")
    if args.min_frequency < 1:
        raise ValueError("min_frequency must be at least 1")
    configure_tokenizer_threads(args.num_threads)
    output_root = args.output_dir.resolve()
    staging_root = (
        args.staging_dir.resolve()
        if args.staging_dir is not None
        else Path(tempfile.gettempdir()).resolve() / "diesel-mt-tokenizer-staging"
    )
    progress("=== Diesel-MT tokenizer training ===")
    progress(
        f"CONFIG corpus={args.corpus_dir.resolve()}, vocab_sizes={vocab_sizes}, "
        f"sample_fraction={args.sample_fraction:.3f}, threads={args.num_threads}, "
        f"staging={staging_root}, output={output_root}"
    )
    corpus = load_balanced_corpus(
        args.corpus_dir,
        seed=args.seed,
        sample_fraction=args.sample_fraction,
        progress_interval_s=args.load_progress_interval,
        max_memory_gib=args.max_memory_gib,
        min_available_memory_gib=args.min_available_memory_gib,
    )
    multiple = len(vocab_sizes) > 1
    for vocab_size in vocab_sizes:
        if multiple:
            suffix = VOCAB_CANDIDATE_NAMES.get(vocab_size, f"vocab-{vocab_size}")
            candidate_output = output_root / suffix
        else:
            candidate_output = output_root
        config = TrainingConfig(
            vocab_size=vocab_size,
            min_frequency=args.min_frequency,
            limit_alphabet=args.limit_alphabet,
            num_threads=args.num_threads,
            batch_size=args.batch_size,
            heartbeat_interval_s=args.heartbeat_interval,
            native_progress=not args.no_native_progress,
        )
        train_candidate(
            corpus,
            config,
            output_dir=candidate_output,
            staging_dir=staging_root,
            max_memory_gib=args.max_memory_gib,
            min_available_memory_gib=args.min_available_memory_gib,
        )
    progress("All requested tokenizer candidates completed successfully")
    return 0


def supervise_worker(raw_args: Sequence[str], args: argparse.Namespace) -> int:
    """Run all mutable work in one child while this process owns terminal output."""
    command = [
        sys.executable,
        "-u",
        str(Path(__file__).resolve()),
        *raw_args,
        "--_worker",
    ]
    environment = os.environ.copy()
    environment["PYTHONIOENCODING"] = "utf-8"
    environment["PYTHONUNBUFFERED"] = "1"
    environment["DIESEL_MT_TOKENIZER_SUPERVISED"] = "1"
    creationflags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
    progress("SUPERVISOR starting isolated tokenizer worker")
    worker = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        env=environment,
        creationflags=creationflags,
    )
    if worker.stdout is None:
        worker.terminate()
        raise RuntimeError("worker stdout pipe was not created")
    messages: queue.Queue[str | None] = queue.Queue()

    def read_worker_output() -> None:
        try:
            for line in worker.stdout:
                messages.put(line)
        finally:
            messages.put(None)

    reader = threading.Thread(
        target=read_worker_output,
        name="tokenizer-worker-output",
        daemon=True,
    )
    reader.start()
    started = time.monotonic()
    heartbeat_interval = max(args.heartbeat_interval, 0.1)
    next_heartbeat = started + heartbeat_interval
    output_closed = False
    last_worker_output = started
    try:
        while True:
            now = time.monotonic()
            timeout = min(0.5, max(0.0, next_heartbeat - now))
            try:
                message = messages.get(timeout=timeout)
            except queue.Empty:
                message = ""
            if message is None:
                output_closed = True
            elif message:
                print(message, end="", flush=True)
                last_worker_output = time.monotonic()
            now = time.monotonic()
            if now >= next_heartbeat and worker.poll() is None:
                rss = process_rss_bytes(worker.pid)
                progress(
                    f"SUPERVISOR heartbeat: worker_pid={worker.pid}, "
                    f"elapsed={now - started:.1f}s, rss={format_gib(rss)}, "
                    f"last_worker_output={now - last_worker_output:.1f}s ago"
                )
                next_heartbeat = now + heartbeat_interval
            if worker.poll() is not None and output_closed and messages.empty():
                break
    except KeyboardInterrupt:
        progress("SUPERVISOR interrupted; terminating tokenizer worker")
        worker.terminate()
        try:
            worker.wait(timeout=10)
        except subprocess.TimeoutExpired:
            worker.kill()
            worker.wait(timeout=5)
        return 130
    finally:
        reader.join(timeout=5)
        worker.stdout.close()
    return_code = worker.wait()
    if return_code:
        progress(f"SUPERVISOR worker failed with exit code {return_code}")
        return return_code
    progress(f"SUPERVISOR worker completed in {time.monotonic() - started:.1f}s")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    raw_args = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(raw_args)
    if args._worker:
        return run_worker(args)
    return supervise_worker(raw_args, args)


if __name__ == "__main__":
    raise SystemExit(main())
