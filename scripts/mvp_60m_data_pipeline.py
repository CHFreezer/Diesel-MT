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
