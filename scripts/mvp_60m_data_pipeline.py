"""Ability-first source, anchor, teacher, and mixed-corpus helpers.

This module implements the v3 data path reopened after the old MASSIVE-only M0
failed.  It deliberately keeps source auditing and deterministic publication
separate from teacher inference so that TD-02/TD-03 evidence can be reproduced
without loading the teacher or touching evaluation data.
"""

from __future__ import annotations

import hashlib
import html
import json
import os
import re
import tempfile
import unicodedata
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter
from dataclasses import asdict, dataclass
import heapq
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence


PIPELINE_VERSION = "mvp-60m-data-v1"
MODEL_TAGS = ("eng_Latn", "zho_Hans", "zho_Hant", "jpn_Jpan", "kor_Hang")

_SPACE_RE = re.compile(r"\s+", flags=re.UNICODE)
_URL_RE = re.compile(r"(?:https?://|www\.)\S+", flags=re.IGNORECASE)
_EMAIL_RE = re.compile(r"\b[^\s@]+@[^\s@]+\.[^\s@]+\b")
_HTML_RE = re.compile(r"</?[A-Za-z][^>]{0,200}>")
_TEMPLATE_RE = re.compile(r"\{\{[^{}]{0,300}\}\}|\{%[^%]{0,300}%\}")
_REPEATED_RE = re.compile(r"(.)\1{5,}")
_MARKDOWN_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]*\)")
_INLINE_CODE_RE = re.compile(r"`[^`]*`")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[。！？!?；;])\s*|\n+")
_CANTONESE_MARKERS = frozenset(
    ("唔", "嘅", "喺", "咁", "冇", "佢", "啲", "咗", "嗰", "嚟", "喎", "囉", "乜", "點解", "邊度", "而家")
)


class AbilityDataError(RuntimeError):
    """Raised when locked input or publication invariants fail."""


@dataclass(frozen=True)
class TextCandidate:
    source_id: str
    source_record_id: str
    language_tag: str
    domain: str
    text: str


@dataclass(frozen=True)
class QualityDecision:
    accepted: bool
    text: str
    reason: str | None
    characters: int
    student_tokens: int | None


@dataclass(frozen=True)
class ParallelGroup:
    source_id: str
    source_group_id: str
    texts: Mapping[str, str]


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8") + b"\n"


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def stable_rank(seed: str, *values: str) -> str:
    payload = "\x1f".join((seed, *values)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def normalized_identity(text: str) -> str:
    normalized = _SPACE_RE.sub(" ", unicodedata.normalize("NFC", text)).strip().casefold()
    return sha256_bytes(normalized.encode("utf-8"))


def near_identity(text: str) -> str:
    normalized = unicodedata.normalize("NFC", text).casefold()
    normalized = "".join(
        character
        for character in normalized
        if not unicodedata.category(character).startswith(("P", "S", "Z"))
    )
    return sha256_bytes(normalized.encode("utf-8"))


def normalize_text(value: Any) -> tuple[str, str | None]:
    if not isinstance(value, str):
        return "", "non_string"
    text = html.unescape(unicodedata.normalize("NFC", value))
    for character in text:
        if unicodedata.category(character) == "Cc" and character not in "\t\n\r":
            return text, "control_character"
    text = _SPACE_RE.sub(" ", text).strip()
    if not text:
        return text, "empty"
    if "\ufffd" in text:
        return text, "unicode_replacement"
    if _URL_RE.search(text) or _EMAIL_RE.search(text):
        return text, "url_or_email"
    if _HTML_RE.search(text) or _TEMPLATE_RE.search(text):
        return text, "html_or_template"
    if _REPEATED_RE.search(text):
        return text, "mechanical_repetition"
    return text, None


def _script_counts(text: str) -> Counter[str]:
    result: Counter[str] = Counter()
    for character in text:
        codepoint = ord(character)
        if 0x0041 <= codepoint <= 0x024F:
            result["latin"] += 1
        elif 0x3040 <= codepoint <= 0x30FF or 0x31F0 <= codepoint <= 0x31FF:
            result["kana"] += 1
        elif 0xAC00 <= codepoint <= 0xD7AF or 0x1100 <= codepoint <= 0x11FF:
            result["hangul"] += 1
        elif 0x3400 <= codepoint <= 0x4DBF or 0x4E00 <= codepoint <= 0x9FFF or 0xF900 <= codepoint <= 0xFAFF:
            result["han"] += 1
    return result


def wrong_script(text: str, language_tag: str) -> bool:
    counts = _script_counts(text)
    total = sum(counts.values())
    if total < 4:
        return True
    if language_tag == "eng_Latn":
        expected = counts["latin"]
    elif language_tag == "jpn_Jpan":
        expected = counts["kana"] + counts["han"]
    elif language_tag == "kor_Hang":
        expected = counts["hangul"]
    elif language_tag in {"zho_Hans", "zho_Hant"}:
        expected = counts["han"]
    else:
        raise AbilityDataError(f"unsupported language tag: {language_tag}")
    return expected / total < 0.55


def written_cantonese(text: str) -> bool:
    hits = {marker for marker in _CANTONESE_MARKERS if marker in text}
    return len(hits) >= 2


def quality_decision(
    candidate: TextCandidate,
    *,
    token_count: int | None = None,
    minimum_characters: int = 20,
    maximum_characters: int = 256,
    minimum_tokens: int = 4,
    maximum_tokens: int = 256,
) -> QualityDecision:
    text, reason = normalize_text(candidate.text)
    if reason is None and len(text) < minimum_characters:
        reason = "too_short"
    if reason is None and len(text) > maximum_characters:
        reason = "too_long"
    if reason is None and wrong_script(text, candidate.language_tag):
        reason = "wrong_script"
    if reason is None and candidate.language_tag == "zho_Hant" and written_cantonese(text):
        reason = "written_cantonese"
    if reason is None and token_count is not None and token_count < minimum_tokens:
        reason = "too_few_tokens"
    if reason is None and token_count is not None and token_count > maximum_tokens:
        reason = "token_overflow"
    return QualityDecision(
        accepted=reason is None,
        text=text,
        reason=reason,
        characters=len(text),
        student_tokens=token_count,
    )


def split_prose(value: str) -> Iterator[str]:
    for paragraph in _SENTENCE_SPLIT_RE.split(value):
        text = _SPACE_RE.sub(" ", paragraph).strip(" \t\r\n-*#>：:")
        if text:
            yield text


def parse_massive_hant(archive: Path) -> Iterator[TextCandidate]:
    member = "1.1/data/zh-TW.jsonl"
    import tarfile

    with tarfile.open(archive, "r:gz") as handle:
        extracted = handle.extractfile(member)
        if extracted is None:
            raise AbilityDataError(f"MASSIVE member missing: {member}")
        for line_number, raw_line in enumerate(extracted, 1):
            row = json.loads(raw_line)
            if row.get("partition") != "train":
                continue
            yield TextCandidate(
                source_id="massive-1.1-route-control",
                source_record_id=f"train:{row['id']}",
                language_tag="zho_Hant",
                domain="daily_and_dialogue",
                text=str(row["utt"]),
            )


def parse_moj_hant(paths: Sequence[Path]) -> Iterator[TextCandidate]:
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        rows = payload.get("Laws") if isinstance(payload, Mapping) else payload
        if not isinstance(rows, list):
            raise AbilityDataError(f"MOJ payload has no Laws list: {path}")
        for law_index, law in enumerate(rows):
            pcode = str(law.get("PCode") or law.get("LawURL") or law_index)
            for article_index, article in enumerate(law.get("LawArticles") or []):
                article_no = str(article.get("ArticleNo") or article_index)
                content = str(article.get("ArticleContent") or "")
                for part_index, part in enumerate(split_prose(content)):
                    if not part.endswith(("。", "！", "？", "；", ";")):
                        continue
                    yield TextCandidate(
                        source_id="taiwan-moj-law-api-20260710",
                        source_record_id=f"{path.stem}:{pcode}:{article_no}:{part_index}",
                        language_tag="zho_Hant",
                        domain="legal_and_government",
                        text=part,
                    )


def parse_hkel_hant(archive: Path) -> Iterator[TextCandidate]:
    with zipfile.ZipFile(archive) as handle:
        for member in sorted(name for name in handle.namelist() if name.endswith(".xml")):
            root = ET.fromstring(handle.read(member))
            ordinal = 0
            for element in root.iter():
                local_name = element.tag.rsplit("}", 1)[-1]
                if local_name != "content":
                    continue
                value = "".join(element.itertext())
                for part in split_prose(value):
                    if not part.endswith(("。", "！", "？", "；", ";")):
                        continue
                    yield TextCandidate(
                        source_id="hkel-current-legislation",
                        source_record_id=f"{member}:{ordinal}",
                        language_tag="zho_Hant",
                        domain="legal_and_government",
                        text=part,
                    )
                    ordinal += 1


def _clean_markdown_lines(value: str) -> Iterator[str]:
    in_front_matter = value.startswith("---\n")
    in_fence = False
    for line in value.splitlines():
        stripped = line.strip()
        if in_front_matter:
            if stripped == "---":
                in_front_matter = False
            continue
        if stripped.startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence or not stripped:
            continue
        if stripped.startswith(("{{", "{%", "<", "[!")) or "`" in stripped or "**" in stripped:
            continue
        stripped = re.sub(r"^#{1,6}\s+", "", stripped)
        stripped = re.sub(r"^[-*+]\s+", "", stripped)
        stripped = re.sub(r"^>\s?", "", stripped)
        stripped = _MARKDOWN_LINK_RE.sub(r"\1", stripped)
        for part in split_prose(stripped):
            yield part


def parse_mdn_hant(archive: Path) -> Iterator[TextCandidate]:
    with zipfile.ZipFile(archive) as handle:
        members = sorted(
            name
            for name in handle.namelist()
            if "/files/zh-tw/" in name.lower() and name.lower().endswith(".md")
        )
        for member in members:
            value = handle.read(member).decode("utf-8", errors="strict")
            for ordinal, part in enumerate(_clean_markdown_lines(value)):
                if not part.endswith(("。", "！", "？")):
                    continue
                if part[0].isascii() or "、、、" in part or "（）" in part:
                    continue
                yield TextCandidate(
                    source_id="mdn-translated-content-zh-tw",
                    source_record_id=f"{member}:{ordinal}",
                    language_tag="zho_Hant",
                    domain="technical",
                    text=part,
                )


def parse_tldr_hant(archive: Path) -> Iterator[TextCandidate]:
    with zipfile.ZipFile(archive) as handle:
        members = sorted(
            name
            for name in handle.namelist()
            if "/pages.zh_tw/" in name.lower() and name.lower().endswith(".md")
        )
        for member in members:
            value = handle.read(member).decode("utf-8", errors="strict")
            ordinal = 0
            for line in value.splitlines():
                stripped = line.strip()
                if not stripped or stripped.startswith(("#", "`")):
                    continue
                if stripped.startswith("> 更多資訊"):
                    continue
                if stripped.startswith(">") or stripped.startswith("-"):
                    stripped = re.sub(r"^[>-]\s*", "", stripped)
                    stripped = _MARKDOWN_LINK_RE.sub(r"\1", stripped)
                    for part in split_prose(stripped):
                        yield TextCandidate(
                            source_id="tldr-pages-zh-tw",
                            source_record_id=f"{member}:{ordinal}",
                            language_tag="zho_Hant",
                            domain="technical",
                            text=part,
                        )
                        ordinal += 1


def parse_ud_chinese_hk(archive: Path) -> Iterator[TextCandidate]:
    with zipfile.ZipFile(archive) as handle:
        members = [name for name in handle.namelist() if name.endswith(".conllu")]
        if len(members) != 1:
            raise AbilityDataError("UD Chinese-HK archive must contain one CoNLL-U file")
        text = handle.read(members[0]).decode("utf-8", errors="strict")
        sent_id = 0
        for line in text.splitlines():
            if line.startswith("# sent_id = "):
                sent_id = int(line.partition("= ")[2])
            if line.startswith("# text = "):
                sentence = line.partition("= ")[2]
                yield TextCandidate(
                    source_id="ud-chinese-hk",
                    source_record_id=str(sent_id),
                    language_tag="zho_Hant",
                    domain=("daily_and_dialogue" if sent_id <= 650 else "legal_and_government"),
                    text=sentence,
                )


def iter_hant_candidates(inputs: Mapping[str, Any]) -> Iterator[TextCandidate]:
    yield from parse_massive_hant(Path(inputs["massive"]))
    yield from parse_moj_hant([Path(path) for path in inputs["moj"]])
    yield from parse_hkel_hant(Path(inputs["hkel_hant"]))
    yield from parse_mdn_hant(Path(inputs["mdn"]))
    yield from parse_tldr_hant(Path(inputs["tldr"]))
    yield from parse_ud_chinese_hk(Path(inputs["ud_hk"]))


def tokenizer_lengths(tokenizer: Any, texts: Sequence[str], *, batch_size: int = 1024) -> list[int]:
    lengths: list[int] = []
    for offset in range(0, len(texts), batch_size):
        encoded = tokenizer(
            list(texts[offset : offset + batch_size]),
            add_special_tokens=True,
            truncation=False,
            padding=False,
        )
        lengths.extend(len(ids) for ids in encoded["input_ids"])
    return lengths


def audit_candidates(
    candidates: Iterable[TextCandidate],
    *,
    tokenizer: Any,
    contamination_texts: Iterable[str] = (),
    batch_size: int = 2048,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    contamination_values = list(contamination_texts)
    contamination = {normalized_identity(text) for text in contamination_values}
    contamination_near = {near_identity(text) for text in contamination_values}
    scanned = list(candidates)
    preliminary = [quality_decision(candidate) for candidate in scanned]
    token_inputs = [decision.text for decision in preliminary if decision.reason is None]
    token_lengths = iter(tokenizer_lengths(tokenizer, token_inputs, batch_size=batch_size))
    accepted: list[dict[str, Any]] = []
    rejection_counts: Counter[str] = Counter()
    source_stats: dict[str, Counter[str]] = {}
    seen: set[str] = set()
    seen_near: set[str] = set()
    for candidate, decision in zip(scanned, preliminary, strict=True):
        stats = source_stats.setdefault(candidate.source_id, Counter())
        stats["scanned"] += 1
        token_count = next(token_lengths) if decision.reason is None else None
        final = quality_decision(candidate, token_count=token_count)
        identity = normalized_identity(final.text)
        near = near_identity(final.text)
        reason = final.reason
        if reason is None and (identity in contamination or near in contamination_near):
            reason = "flores_dev_contamination"
        if reason is None and identity in seen:
            reason = "exact_duplicate"
        if reason is None and near in seen_near:
            reason = "near_duplicate"
        if reason is not None:
            rejection_counts[reason] += 1
            stats[f"rejected:{reason}"] += 1
            continue
        seen.add(identity)
        seen_near.add(near)
        stats["accepted_before_domain_caps"] += 1
        accepted.append(
            {
                **asdict(candidate),
                "text": final.text,
                "characters": final.characters,
                "student_tokens": final.student_tokens,
                "normalized_sha256": identity,
                "selection_rank": stable_rank(
                    "diesel-mt-td02-hant-quality", candidate.source_id, candidate.source_record_id, identity
                ),
            }
        )
    report = {
        "pipeline_version": PIPELINE_VERSION,
        "scanned_candidates": len(scanned),
        "accepted_before_domain_caps": len(accepted),
        "rejection_counts": dict(sorted(rejection_counts.items())),
        "sources": {
            key: dict(sorted(value.items())) for key, value in sorted(source_stats.items())
        },
    }
    return accepted, report


def apply_domain_ceilings(
    records: Sequence[Mapping[str, Any]],
    *,
    technical_ceiling: float = 0.15,
    legal_ceiling: float = 0.20,
) -> list[dict[str, Any]]:
    pools: dict[str, list[Mapping[str, Any]]] = {
        "base": [],
        "technical": [],
        "legal_and_government": [],
    }
    for record in records:
        domain = str(record["domain"])
        key = domain if domain in {"technical", "legal_and_government"} else "base"
        pools[key].append(record)
    for pool in pools.values():
        pool.sort(key=lambda item: (str(item["selection_rank"]), str(item["source_record_id"])))

    base_count = len(pools["base"])
    technical_count = len(pools["technical"])
    legal_count = len(pools["legal_and_government"])
    while True:
        total = base_count + technical_count + legal_count
        next_technical = min(len(pools["technical"]), int(technical_ceiling * total))
        next_legal = min(len(pools["legal_and_government"]), int(legal_ceiling * total))
        if (next_technical, next_legal) == (technical_count, legal_count):
            break
        technical_count, legal_count = next_technical, next_legal
    selected = (
        pools["base"]
        + pools["technical"][:technical_count]
        + pools["legal_and_government"][:legal_count]
    )
    return [dict(record) for record in sorted(selected, key=lambda item: str(item["selection_rank"]))]


def atomic_write_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def write_json(path: Path, value: Any) -> None:
    atomic_write_bytes(path, json.dumps(value, ensure_ascii=False, indent=2).encode("utf-8") + b"\n")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> tuple[int, str]:
    payload = b"".join(canonical_json_bytes(dict(row)) for row in rows)
    atomic_write_bytes(path, payload)
    return payload.count(b"\n"), sha256_bytes(payload)


def read_parallel_lines(
    source_id: str,
    paths: Mapping[str, Path],
) -> list[ParallelGroup]:
    handles = {language: path.open(encoding="utf-8") for language, path in paths.items()}
    groups: list[ParallelGroup] = []
    try:
        iterators = [iter(handle) for handle in handles.values()]
        import itertools

        for index, values in enumerate(itertools.zip_longest(*iterators)):
            if any(value is None for value in values):
                raise AbilityDataError(f"parallel line count differs for {source_id}")
            texts = {
                language: str(value).rstrip("\r\n")
                for language, value in zip(handles, values, strict=True)
            }
            groups.append(ParallelGroup(source_id, str(index), texts))
    finally:
        for handle in handles.values():
            handle.close()
    return groups


def parse_alt_groups(en_ja_archive: Path, en_zh_archive: Path) -> list[ParallelGroup]:
    def sides(path: Path, suffixes: tuple[str, str]) -> tuple[list[str], list[str]]:
        with zipfile.ZipFile(path) as handle:
            left = handle.read(suffixes[0]).decode("utf-8").splitlines()
            right = handle.read(suffixes[1]).decode("utf-8").splitlines()
        if len(left) != len(right):
            raise AbilityDataError(f"ALT side counts differ: {path}")
        return left, right

    en_ja, ja = sides(en_ja_archive, ("ALT.en-ja.en", "ALT.en-ja.ja"))
    en_zh, zh = sides(en_zh_archive, ("ALT.en-zh.en", "ALT.en-zh.zh"))
    ja_by_en: dict[str, list[str]] = {}
    zh_by_en: dict[str, list[str]] = {}
    for english, japanese in zip(en_ja, ja, strict=True):
        ja_by_en.setdefault(english, []).append(japanese)
    for english, chinese in zip(en_zh, zh, strict=True):
        zh_by_en.setdefault(english, []).append(chinese)
    groups: list[ParallelGroup] = []
    for english in sorted(set(ja_by_en) & set(zh_by_en)):
        if len(ja_by_en[english]) != 1 or len(zh_by_en[english]) != 1:
            continue
        group_hash = sha256_bytes(english.encode("utf-8"))[:24]
        groups.append(
            ParallelGroup(
                "alt-v20191206-en-ja-zh",
                group_hash,
                {
                    "eng_Latn": english,
                    "zho_Hans": zh_by_en[english][0],
                    "jpn_Jpan": ja_by_en[english][0],
                },
            )
        )
    return groups


def parse_massive_groups(archive: Path) -> list[ParallelGroup]:
    import tarfile

    locale_by_tag = {
        "eng_Latn": "en-US",
        "zho_Hans": "zh-CN",
        "zho_Hant": "zh-TW",
        "jpn_Jpan": "ja-JP",
        "kor_Hang": "ko-KR",
    }
    by_language: dict[str, dict[str, str]] = {}
    with tarfile.open(archive, "r:gz") as handle:
        for language, locale in locale_by_tag.items():
            member = handle.extractfile(f"1.1/data/{locale}.jsonl")
            if member is None:
                raise AbilityDataError(f"MASSIVE locale missing: {locale}")
            values: dict[str, str] = {}
            for line in member:
                row = json.loads(line)
                if row.get("partition") == "train":
                    values[str(row["id"])] = str(row["utt"])
            by_language[language] = values
    ids = set.intersection(*(set(values) for values in by_language.values()))
    return [
        ParallelGroup(
            "massive-1.1-route-control",
            f"train:{record_id}",
            {language: values[record_id] for language, values in by_language.items()},
        )
        for record_id in sorted(ids, key=lambda value: int(value))
    ]


def _pcode(url: str) -> str:
    return url.partition("pcode=")[2].lower()


def parse_moj_parallel_groups(chinese_paths: Sequence[Path], english_paths: Sequence[Path]) -> list[ParallelGroup]:
    def laws(paths: Sequence[Path], *, english: bool) -> dict[str, Mapping[str, Any]]:
        result: dict[str, Mapping[str, Any]] = {}
        for path in paths:
            payload = json.loads(path.read_text(encoding="utf-8-sig"))
            for law in payload["Laws"]:
                url_key = "EngLawURL" if english else "LawURL"
                code = _pcode(str(law.get(url_key, "")))
                if code:
                    result[code] = law
        return result

    zh_laws = laws(chinese_paths, english=False)
    en_laws = laws(english_paths, english=True)
    groups: list[ParallelGroup] = []
    for code in sorted(set(zh_laws) & set(en_laws)):
        zh_articles = [
            item for item in zh_laws[code].get("LawArticles", []) if item.get("ArticleType") == "A"
        ]
        en_articles = [
            item for item in en_laws[code].get("EngLawArticles", []) if item.get("EngArticleType") == "A"
        ]
        for ordinal, (zh_article, en_article) in enumerate(zip(zh_articles, en_articles)):
            zh_parts = list(split_prose(str(zh_article.get("ArticleContent", ""))))
            en_parts = list(split_prose(str(en_article.get("EngArticleContent", ""))))
            if len(zh_parts) != 1 or len(en_parts) != 1:
                continue
            article_id = str(zh_article.get("ArticleNo") or ordinal)
            group_id = f"{code}:{article_id}"
            groups.append(
                ParallelGroup(
                    "taiwan-moj-law-api-20260710",
                    group_id,
                    {"zho_Hant": zh_parts[0], "eng_Latn": en_parts[0]},
                )
            )
    return groups


def filter_parallel_groups(
    groups: Sequence[ParallelGroup],
    *,
    tokenizer: Any,
    contamination_exact: set[str],
    contamination_near: set[str],
) -> tuple[list[ParallelGroup], dict[str, int]]:
    preliminary: list[tuple[ParallelGroup, dict[str, QualityDecision]]] = []
    token_texts: list[str] = []
    rejected: Counter[str] = Counter()
    for group in groups:
        decisions = {
            language: quality_decision(
                TextCandidate(group.source_id, group.source_group_id, language, "parallel", text)
            )
            for language, text in group.texts.items()
        }
        reason = next((decision.reason for decision in decisions.values() if decision.reason), None)
        if reason:
            rejected[str(reason)] += 1
            continue
        preliminary.append((group, decisions))
        token_texts.extend(decision.text for decision in decisions.values())
    lengths = iter(tokenizer_lengths(tokenizer, token_texts))
    accepted: list[ParallelGroup] = []
    for group, decisions in preliminary:
        normalized: dict[str, str] = {}
        reason: str | None = None
        for language, decision in decisions.items():
            final = quality_decision(
                TextCandidate(group.source_id, group.source_group_id, language, "parallel", decision.text),
                token_count=next(lengths),
            )
            if final.reason:
                reason = final.reason
            normalized[language] = final.text
        identities = [normalized_identity(text) for text in normalized.values()]
        near = [near_identity(text) for text in normalized.values()]
        if reason is None and any(value in contamination_exact for value in identities):
            reason = "flores_dev_contamination"
        if reason is None and any(value in contamination_near for value in near):
            reason = "flores_dev_contamination"
        lengths_without_space = [max(1, len(text.replace(" ", ""))) for text in normalized.values()]
        if reason is None and max(lengths_without_space) / min(lengths_without_space) > 3.5:
            reason = "alignment_length_ratio"
        if reason is None and len(set(identities)) != len(identities):
            reason = "cross_language_copy"
        if reason:
            rejected[reason] += 1
            continue
        accepted.append(ParallelGroup(group.source_id, group.source_group_id, normalized))
    return accepted, dict(sorted(rejected.items()))


def _ranked_groups(groups: Sequence[ParallelGroup], seed: str) -> list[ParallelGroup]:
    return sorted(
        groups,
        key=lambda group: stable_rank(seed, group.source_id, group.source_group_id),
    )


def select_group_role(
    groups: Sequence[ParallelGroup],
    *,
    count: int,
    seed: str,
    languages: Sequence[str],
    used_groups: set[tuple[str, str]],
    used_exact: set[str],
    used_near: set[str],
    forbidden_groups: set[tuple[str, str]] = frozenset(),
) -> list[ParallelGroup]:
    selected: list[ParallelGroup] = []
    for group in _ranked_groups(groups, seed):
        key = (group.source_id, group.source_group_id)
        if key in used_groups or key in forbidden_groups:
            continue
        texts = [group.texts[language] for language in languages]
        exact = [normalized_identity(text) for text in texts]
        near = [near_identity(text) for text in texts]
        if any(value in used_exact for value in exact) or any(value in used_near for value in near):
            continue
        selected.append(group)
        used_groups.add(key)
        used_exact.update(exact)
        used_near.update(near)
        if len(selected) == count:
            break
    return selected


def select_unpc_side(
    path: Path,
    *,
    tokenizer: Any,
    language_tag: str,
    count: int,
    seed: str,
    used_exact: set[str],
    used_near: set[str],
    contamination_exact: set[str],
    contamination_near: set[str],
    excluded_record_ids: set[str] = frozenset(),
    scan_limit: int = 1_000_000,
    candidate_capacity: int = 90_000,
) -> list[dict[str, Any]]:
    heap: list[tuple[int, int, str, str, str]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle):
            if line_number >= scan_limit:
                break
            if str(line_number) in excluded_record_ids:
                continue
            candidate = TextCandidate(
                "unpc-v1.0-en-zho_hans",
                str(line_number),
                language_tag,
                "formal",
                line.rstrip("\r\n"),
            )
            decision = quality_decision(candidate)
            if not decision.accepted:
                continue
            exact = normalized_identity(decision.text)
            near = near_identity(decision.text)
            if exact in contamination_exact or near in contamination_near:
                continue
            rank = int(stable_rank(seed, str(line_number), exact), 16)
            item = (-rank, line_number, decision.text, exact, near)
            if len(heap) < candidate_capacity:
                heapq.heappush(heap, item)
            elif rank < -heap[0][0]:
                heapq.heapreplace(heap, item)
    candidates = sorted(heap, key=lambda item: (-item[0], item[1]))
    original_language = getattr(tokenizer, "src_lang", None)
    try:
        tokenizer.src_lang = language_tag
        token_counts = tokenizer_lengths(tokenizer, [item[2] for item in candidates])
    finally:
        tokenizer.src_lang = original_language
    selected: list[dict[str, Any]] = []
    for item, token_count in zip(candidates, token_counts, strict=True):
        _neg_rank, line_number, text, exact, near = item
        if token_count > 256 or token_count < 4 or exact in used_exact or near in used_near:
            continue
        used_exact.add(exact)
        used_near.add(near)
        selected.append(
            {
                "source_id": "unpc-v1.0-en-zho_hans",
                "source_record_id": str(line_number),
                "language_tag": language_tag,
                "domain": "formal",
                "text": text,
                "characters": len(text),
                "student_tokens": token_count,
                "normalized_sha256": exact,
            }
        )
        if len(selected) == count:
            break
    if len(selected) != count:
        raise AbilityDataError(
            f"UNPC/{language_tag} selected {len(selected)} records, expected {count}"
        )
    return selected


def select_unpc_hans(
    path: Path,
    *,
    tokenizer: Any,
    count: int,
    seed: str,
    used_exact: set[str],
    used_near: set[str],
    contamination_exact: set[str],
    contamination_near: set[str],
    scan_limit: int = 1_000_000,
    candidate_capacity: int = 90_000,
) -> list[dict[str, Any]]:
    """Backward-compatible Simplified-Chinese UNPC selector."""

    return select_unpc_side(
        path,
        tokenizer=tokenizer,
        language_tag="zho_Hans",
        count=count,
        seed=seed,
        used_exact=used_exact,
        used_near=used_near,
        contamination_exact=contamination_exact,
        contamination_near=contamination_near,
        scan_limit=scan_limit,
        candidate_capacity=candidate_capacity,
    )


def source_rows(groups: Sequence[ParallelGroup], language: str) -> list[dict[str, Any]]:
    return [
        {
            "record_id": f"src-{sha256_bytes(canonical_json_bytes([group.source_id, group.source_group_id, language]))[:24]}",
            "semantic_group_id": f"grp-{sha256_bytes(canonical_json_bytes([group.source_id, group.source_group_id]))[:24]}",
            "source_id": group.source_id,
            "source_record_id": group.source_group_id,
            "language_tag": language,
            "domain": "general_parallel_side",
            "text": group.texts[language],
            "normalized_sha256": normalized_identity(group.texts[language]),
        }
        for group in groups
    ]


def anchor_rows(groups: Sequence[ParallelGroup], languages: Sequence[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for group in groups:
        semantic_id = f"grp-{sha256_bytes(canonical_json_bytes([group.source_id, group.source_group_id]))[:24]}"
        for source in languages:
            for target in languages:
                if source == target:
                    continue
                rows.append(
                    {
                        "record_id": f"human-{sha256_bytes(canonical_json_bytes([group.source_id, group.source_group_id, source, target]))[:24]}",
                        "semantic_group_id": semantic_id,
                        "source_id": group.source_id,
                        "source_record_id": group.source_group_id,
                        "src_lang": source,
                        "tgt_lang": target,
                        "source_text": group.texts[source],
                        "target_text": group.texts[target],
                        "provenance": "human_parallel",
                    }
                )
    return rows
