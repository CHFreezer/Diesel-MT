"""Independent DeepSeek batch review for TD-04 teacher translations.

The reviewer never edits the teacher corpus.  It writes identity-bound response
files and a derived decision/report layer under the supplied runtime root.  A
known-error calibration must qualify before a full review can start.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import random
import re
import tempfile
import threading
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence

import yaml


PROMPT_VERSION = "deepseek-translation-fidelity-review-v6-thinking"
SYSTEM_PROMPT = """You are an independent multilingual translation fidelity auditor.
Review every supplied source/candidate pair. Do not translate or rewrite the text.
Judge whether the candidate preserves meaning, entities, numbers, dates, polarity,
relations, terminology, and requested target language/script.

Be recall-oriented and audit each pair independently. Fluency is never evidence of
fidelity. Before assigning a verdict, apply this checklist to every pair:
1. Account for every number, unit, date, clock time, time-zone offset, quantity, and
   polarity in the source. A missing or changed detail is an error.
2. Account for every person, place, organization, era name, work/title, and product.
   A conventional target-language name or faithful transliteration is acceptable;
   replacing a proper name/title with a descriptive gloss is not.
3. Reject corrupted or mixed-script entity fragments (for example, combining a
   translated prefix with an untranslated substring inside one name).
4. Check that no material clause, relation, qualifier, or limitation was omitted or
   added, and that the candidate is actually in the requested target language/script.
5. If a detail cannot be verified confidently, use warning rather than pass. Never
   pass merely because the candidate sounds plausible.

Use verdicts exactly as follows:
- pass: all meaning and material details pass the checklist; harmless style or faithful romanization variants are acceptable.
- warning: probably usable but ambiguous, mildly incomplete, or uncertain; it needs human review.
- reject: a material semantic, entity, numeric, date, omission, addition, polarity, terminology, or language/script error exists.

Return one JSON object with an `items` array, in exactly the same order and with
exactly the same IDs as the input. For pass, categories/evidence/note must be empty.
For warning/reject, categories and note must be non-empty. Evidence must be short
verbatim substrings copied from the supplied source/candidate, not a new translation.
Never output a corrected translation. Output JSON only.

JSON example:
{"items":[{"id":"x","verdict":"reject","categories":["numeric_error"],"confidence":0.98,"source_evidence":"135,000","target_evidence":"35,000","note":"A leading digit was lost."}]}
"""

_ENV_ASSIGNMENT_RE = re.compile(
    r"\$env:([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(['\"])(.*?)\2",
    flags=re.DOTALL,
)


class TranslationReviewError(RuntimeError):
    """Raised when an input, API response, or review invariant fails."""


class ResponseContractError(TranslationReviewError):
    """Raised for deterministic response-shape errors that must not be retried."""


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


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TranslationReviewError(f"expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TranslationReviewError(f"non-object JSONL: {path}:{line_number}")
            rows.append(value)
    return rows


def _atomic_replace(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    _atomic_replace(path, json.dumps(value, ensure_ascii=False, indent=2).encode("utf-8") + b"\n")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> tuple[int, str]:
    payload = b"".join(canonical_json_bytes(row) for row in rows)
    _atomic_replace(path, payload)
    return payload.count(b"\n"), sha256_bytes(payload)


def load_config(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise TranslationReviewError("translation review config schema differs")
    if value.get("identity", {}).get("status") != "frozen":
        raise TranslationReviewError("translation review config is not frozen")
    inputs = value.get("inputs", {})
    if inputs.get("formal_test_access") != "prohibited":
        raise TranslationReviewError("formal test access must be prohibited")
    if inputs.get("formal_devtest_access") != "prohibited":
        raise TranslationReviewError("formal devtest access must be prohibited")
    api = value.get("api", {})
    if api.get("model") != "deepseek-v4-flash":
        raise TranslationReviewError("only the frozen deepseek-v4-flash reviewer is supported")
    if api.get("thinking") not in {"enabled", "disabled"}:
        raise TranslationReviewError("thinking mode must be explicit")
    if (
        int(api.get("batch_max_records", 0))
        * int(value.get("pricing", {}).get("estimated_output_tokens_per_record", 0))
        > int(api.get("max_output_tokens", 0))
    ):
        raise TranslationReviewError("estimated batch output exceeds API max_output_tokens")
    if api.get("response_format") != "json_object":
        raise TranslationReviewError("JSON output must be enabled")
    return value


def load_api_key(*, env_name: str, auth_script: Path | None) -> str:
    direct = os.environ.get(env_name, "").strip()
    if direct:
        return direct
    for fallback in ("DEEPSEEK_API_KEY", "ANTHROPIC_AUTH_TOKEN"):
        value = os.environ.get(fallback, "").strip()
        if value:
            return value
    if auth_script is None:
        raise TranslationReviewError(
            f"API key not found in {env_name}, DEEPSEEK_API_KEY, or ANTHROPIC_AUTH_TOKEN"
        )
    text = auth_script.resolve().read_text(encoding="utf-8-sig")
    assignments = {
        match.group(1): match.group(3) for match in _ENV_ASSIGNMENT_RE.finditer(text)
    }
    for candidate in (env_name, "DEEPSEEK_API_KEY", "ANTHROPIC_AUTH_TOKEN"):
        value = assignments.get(candidate, "").strip()
        if value:
            return value
    raise TranslationReviewError(f"API key assignment not found in auth script: {auth_script}")


def _is_han(character: str) -> bool:
    codepoint = ord(character)
    return (
        0x3400 <= codepoint <= 0x4DBF
        or 0x4E00 <= codepoint <= 0x9FFF
        or 0xF900 <= codepoint <= 0xFAFF
    )


def _is_kana_or_hangul(character: str) -> bool:
    codepoint = ord(character)
    return (
        0x3040 <= codepoint <= 0x30FF
        or 0x31F0 <= codepoint <= 0x31FF
        or 0xAC00 <= codepoint <= 0xD7AF
        or 0x1100 <= codepoint <= 0x11FF
    )


def estimate_text_tokens(text: str) -> int:
    """Conservative offline estimate based on DeepSeek's documented ratios."""

    units = 0.0
    for character in text:
        if _is_han(character):
            units += 0.6
        elif _is_kana_or_hangul(character):
            units += 1.0
        elif ord(character) < 0x0250:
            units += 0.3
        else:
            units += 0.7
    return max(1, math.ceil(units))


def _source_provenance(runtime_root: Path, config: Mapping[str, Any]) -> tuple[dict[tuple[str, str], dict[str, str]], dict[str, dict[str, str]]]:
    path = runtime_root / PurePosixPath(config["inputs"]["source_bank"])
    exact: dict[tuple[str, str], dict[str, str]] = {}
    by_group: dict[str, dict[str, str]] = {}
    for row in read_jsonl(path):
        group = str(row["semantic_group_id"])
        language = str(row["language_tag"])
        compact = {
            "source_id": str(row["source_id"]),
            "source_record_id": str(row["source_record_id"]),
        }
        exact[(group, language)] = compact
        by_group.setdefault(group, compact)
    return exact, by_group


def _review_item(
    row: Mapping[str, Any],
    *,
    kind: str,
    exact_provenance: Mapping[tuple[str, str], Mapping[str, str]],
    group_provenance: Mapping[str, Mapping[str, str]],
) -> dict[str, Any]:
    source_language = str(row["src_lang"])
    target_language = str(row["tgt_lang"])
    group = str(row["semantic_group_id"])
    provenance = exact_provenance.get((group, source_language)) or group_provenance.get(group) or {}
    input_id = str(row.get("teacher_job_id") or row.get("record_id") or row.get("forward_job_id"))
    if not input_id:
        raise TranslationReviewError("review record lacks stable ID")
    return {
        "id": input_id,
        "kind": kind,
        "route": f"{source_language}->{target_language}",
        "source_language": source_language,
        "target_language": target_language,
        "source_text": str(row["source_text"]),
        "candidate_translation": str(row["target_text"]),
        "semantic_group_id": group,
        "source_id": str(provenance.get("source_id", kind)),
        "source_record_id": str(provenance.get("source_record_id", "")),
    }


def load_full_review_items(runtime_root: Path, config: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    inputs = config["inputs"]
    manifest_path = runtime_root / PurePosixPath(inputs["td04_manifest"])
    manifest = read_json(manifest_path)
    if manifest.get("status") != "complete":
        raise TranslationReviewError("TD-04 generation manifest must be complete")
    accepted_path = runtime_root / PurePosixPath(inputs["accepted_teacher"])
    reverse_path = runtime_root / PurePosixPath(inputs["reverse_pairs"])
    if sha256_file(accepted_path) != manifest["accepted_teacher"]["sha256"]:
        raise TranslationReviewError("accepted teacher hash drift")
    if sha256_file(reverse_path) != manifest["reverse_pairs"]["sha256"]:
        raise TranslationReviewError("reverse pair hash drift")
    exact, by_group = _source_provenance(runtime_root, config)
    items = [
        _review_item(row, kind="teacher", exact_provenance=exact, group_provenance=by_group)
        for row in read_jsonl(accepted_path)
    ]
    items.extend(
        _review_item(row, kind="reverse", exact_provenance=exact, group_provenance=by_group)
        for row in read_jsonl(reverse_path)
    )
    ids = [str(item["id"]) for item in items]
    if len(ids) != len(set(ids)):
        raise TranslationReviewError("full review IDs are not unique")
    evidence = {
        "td04_manifest_sha256": sha256_file(manifest_path),
        "generation_identity": manifest["generation_identity"],
        "accepted_teacher_sha256": manifest["accepted_teacher"]["sha256"],
        "reverse_pairs_sha256": manifest["reverse_pairs"]["sha256"],
        "records": len(items),
    }
    return items, evidence


def load_calibration_items(
    runtime_root: Path, config: Mapping[str, Any], repo_root: Path
) -> tuple[list[dict[str, Any]], dict[str, dict[str, str]]]:
    queue_path = runtime_root / PurePosixPath(config["inputs"]["manual_review_queue"])
    queue = {str(row["review_id"]): row for row in read_jsonl(queue_path)}
    calibration_path = repo_root / PurePosixPath(config["inputs"]["calibration"])
    calibration = yaml.safe_load(calibration_path.read_text(encoding="utf-8"))
    if not isinstance(calibration, dict) or calibration.get("schema_version") != 1:
        raise TranslationReviewError("calibration schema differs")
    if calibration.get("identity", {}).get("status") != "frozen":
        raise TranslationReviewError("calibration is not frozen")
    expectations: dict[str, dict[str, str]] = {}
    items: list[dict[str, Any]] = []
    for case in calibration["cases"]:
        review_id = str(case["review_id"])
        if review_id not in queue or review_id in expectations:
            raise TranslationReviewError(f"calibration review ID mismatch: {review_id}")
        row = queue[review_id]
        if row.get("kind") != "accepted":
            raise TranslationReviewError("calibration only supports accepted teacher candidates")
        source_language, target_language = str(row["route"]).split("->", 1)
        items.append(
            {
                "id": review_id,
                "kind": "calibration",
                "route": str(row["route"]),
                "source_language": source_language,
                "target_language": target_language,
                "source_text": str(row["source_text"]),
                "candidate_translation": str(row["target_text"]),
                "semantic_group_id": "",
                "source_id": "calibration",
                "source_record_id": str(row["input_record_id"]),
            }
        )
        expectations[review_id] = {
            "expected": str(case["expected"]),
            "category": str(case.get("category", "")),
            "note": str(case.get("note", "")),
        }
    return items, expectations


def api_item(item: Mapping[str, Any]) -> dict[str, str]:
    return {
        "id": str(item["id"]),
        "source_language": str(item["source_language"]),
        "target_language": str(item["target_language"]),
        "source": str(item["source_text"]),
        "candidate": str(item["candidate_translation"]),
    }


def user_payload(
    items: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> str:
    value = {
        "task": "translation_fidelity_review",
        "allowed_categories": list(config["review"]["categories"]),
        "category_rule": "Use only allowed_categories. Use uncertain when no more specific category applies.",
        "required_json_shape": {
            "items": [
                {
                    "id": "same input ID",
                    "verdict": "pass|warning|reject",
                    "categories": [],
                    "confidence": 0.0,
                    "source_evidence": "",
                    "target_evidence": "",
                    "note": "",
                }
            ]
        },
        "items": [api_item(item) for item in items],
    }
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def make_batches(items: Sequence[Mapping[str, Any]], config: Mapping[str, Any]) -> list[dict[str, Any]]:
    maximum_records = int(config["api"]["batch_max_records"])
    maximum_tokens = int(config["api"]["batch_max_estimated_input_tokens"])
    prompt_tokens = estimate_text_tokens(SYSTEM_PROMPT)
    batches: list[dict[str, Any]] = []
    current: list[Mapping[str, Any]] = []
    current_tokens = prompt_tokens
    for item in items:
        item_tokens = estimate_text_tokens(json.dumps(api_item(item), ensure_ascii=False)) + 8
        if current and (len(current) >= maximum_records or current_tokens + item_tokens > maximum_tokens):
            batches.append({"items": list(current), "estimated_input_tokens": current_tokens})
            current = []
            current_tokens = prompt_tokens
        if item_tokens + prompt_tokens > maximum_tokens:
            raise TranslationReviewError(f"single review item exceeds batch token budget: {item['id']}")
        current.append(item)
        current_tokens += item_tokens
    if current:
        batches.append({"items": list(current), "estimated_input_tokens": current_tokens})
    for index, batch in enumerate(batches):
        identity = sha256_bytes(
            canonical_json_bytes(
                [
                    PROMPT_VERSION,
                    config["api"]["model"],
                    config["api"]["thinking"],
                    [api_item(item) for item in batch["items"]],
                ]
            )
        )
        batch["index"] = index
        batch["identity"] = identity
    return batches


def estimate_cost(batches: Sequence[Mapping[str, Any]], config: Mapping[str, Any]) -> dict[str, Any]:
    input_tokens = sum(int(batch["estimated_input_tokens"]) for batch in batches)
    records = sum(len(batch["items"]) for batch in batches)
    output_tokens = records * int(config["pricing"]["estimated_output_tokens_per_record"])
    prices = config["pricing"]["per_million_tokens"]
    cache_miss_cost = input_tokens / 1_000_000 * float(prices["input_cache_miss"])
    output_cost = output_tokens / 1_000_000 * float(prices["output"])
    return {
        "records": records,
        "batches": len(batches),
        "estimated_input_tokens": input_tokens,
        "estimated_output_tokens": output_tokens,
        "estimated_cache_miss_input_usd": round(cache_miss_cost, 6),
        "estimated_output_usd": round(output_cost, 6),
        "estimated_total_usd": round(cache_miss_cost + output_cost, 6),
        "pricing_retrieved_on": str(config["pricing"]["retrieved_on"]),
    }


def _validate_decisions(
    raw: Any, items: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> list[dict[str, Any]]:
    if not isinstance(raw, dict) or not isinstance(raw.get("items"), list):
        raise TranslationReviewError("review response lacks JSON items array")
    values = raw["items"]
    expected_ids = [str(item["id"]) for item in items]
    actual_ids = [str(value.get("id", "")) for value in values if isinstance(value, dict)]
    if actual_ids != expected_ids or len(values) != len(items):
        raise TranslationReviewError("review response IDs/order do not match request")
    verdicts = set(config["review"]["verdicts"])
    allowed_categories = set(config["review"]["categories"])
    maximum_evidence = int(config["review"]["maximum_evidence_characters"])
    maximum_note = int(config["review"]["maximum_note_characters"])
    decisions: list[dict[str, Any]] = []
    for value, item in zip(values, items, strict=True):
        if not isinstance(value, dict):
            raise TranslationReviewError("review item is not an object")
        verdict = str(value.get("verdict", ""))
        categories = value.get("categories")
        confidence = value.get("confidence")
        source_evidence = value.get("source_evidence", "")
        target_evidence = value.get("target_evidence", "")
        note = value.get("note", "")
        if verdict not in verdicts:
            raise TranslationReviewError(f"unsupported verdict for {item['id']}")
        if not isinstance(categories, list) or any(str(category) not in allowed_categories for category in categories):
            raise TranslationReviewError(f"unsupported categories for {item['id']}")
        if not isinstance(confidence, (int, float)) or isinstance(confidence, bool) or not 0 <= float(confidence) <= 1:
            raise TranslationReviewError(f"invalid confidence for {item['id']}")
        if not all(isinstance(value, str) for value in (source_evidence, target_evidence, note)):
            raise TranslationReviewError(f"review evidence must be strings for {item['id']}")
        normalized_categories = [str(category) for category in categories]
        if verdict == "pass" and (normalized_categories or source_evidence or target_evidence or note):
            raise TranslationReviewError(f"pass decision contains error claims for {item['id']}")
        if verdict != "pass" and (not normalized_categories or not note):
            raise TranslationReviewError(f"flagged decision lacks category/note for {item['id']}")
        discarded_evidence: list[str] = []
        truncated_fields: list[str] = []
        if verdict != "pass":
            if len(note) > maximum_note:
                note = note[:maximum_note].rstrip()
                truncated_fields.append("note")
            evidence_pairs = (
                ("source", source_evidence, str(item["source_text"])),
                ("target", target_evidence, str(item["candidate_translation"])),
            )
            normalized_evidence: dict[str, str] = {}
            wrappers = " \t\r\n\"'`“”‘’「」『』《》〈〉()[]{}<>"
            for name, evidence, supplied_text in evidence_pairs:
                grounded = evidence
                if grounded and grounded not in supplied_text:
                    unwrapped = grounded.strip(wrappers)
                    if unwrapped and unwrapped in supplied_text:
                        grounded = unwrapped
                    else:
                        grounded = ""
                        discarded_evidence.append(name)
                if len(grounded) > maximum_evidence:
                    grounded = grounded[:maximum_evidence].rstrip()
                    truncated_fields.append(f"{name}_evidence")
                normalized_evidence[name] = grounded
            source_evidence = normalized_evidence["source"]
            target_evidence = normalized_evidence["target"]
        decisions.append(
            {
                "id": str(item["id"]),
                "verdict": verdict,
                "categories": normalized_categories,
                "confidence": round(float(confidence), 6),
                "source_evidence": source_evidence,
                "target_evidence": target_evidence,
                "note": note,
                "discarded_ungrounded_evidence": discarded_evidence,
                "truncated_response_fields": truncated_fields,
            }
        )
    return decisions


class DeepSeekClient:
    def __init__(self, config: Mapping[str, Any], api_key: str):
        self.config = config
        self.api_key = api_key
        self.random = random.Random(20260718)
        self.random_lock = threading.Lock()

    def _retry_delay(self, attempt: int) -> float:
        base = float(self.config["api"]["retry_base_seconds"])
        with self.random_lock:
            jitter = self.random.random() * 0.25
        return min(base * (2 ** (attempt - 1)) + jitter, 30.0)

    def review_batch(self, items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        api = self.config["api"]
        url = str(api["base_url"]).rstrip("/") + "/" + str(api["endpoint"]).lstrip("/")
        payload: dict[str, Any] = {
            "model": api["model"],
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_payload(items, self.config)},
            ],
            "thinking": {"type": api["thinking"]},
            "response_format": {"type": api["response_format"]},
            "max_tokens": int(api["max_output_tokens"]),
            "stream": False,
            "user_id": str(api["user_id"]),
        }
        if api["thinking"] == "disabled":
            payload["temperature"] = 0
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        attempts = int(api["request_attempts"])
        last_error: Exception | None = None
        for attempt in range(1, attempts + 1):
            request = urllib.request.Request(
                url,
                data=body,
                method="POST",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                },
            )
            started = time.perf_counter()
            try:
                with urllib.request.urlopen(
                    request, timeout=float(api["request_timeout_seconds"])
                ) as response:
                    response_body = response.read()
                decoded = json.loads(response_body)
                choice = decoded["choices"][0]
                finish_reason = str(choice["finish_reason"])
                content = choice["message"].get("content")
                if finish_reason != "stop" or not isinstance(content, str) or not content.strip():
                    raise TranslationReviewError(
                        f"incomplete API response: finish_reason={finish_reason!r}"
                    )
                parsed = json.loads(content)
                try:
                    decisions = _validate_decisions(parsed, items, self.config)
                except TranslationReviewError as error:
                    raise ResponseContractError(str(error)) from error
                usage = decoded.get("usage", {})
                return {
                    "api_response_id": str(decoded.get("id", "")),
                    "model": str(decoded.get("model", api["model"])),
                    "system_fingerprint": str(decoded.get("system_fingerprint", "")),
                    "finish_reason": finish_reason,
                    "latency_seconds": round(time.perf_counter() - started, 6),
                    "request_attempts": attempt,
                    "usage": {
                        "prompt_tokens": int(usage.get("prompt_tokens", 0)),
                        "prompt_cache_hit_tokens": int(usage.get("prompt_cache_hit_tokens", 0)),
                        "prompt_cache_miss_tokens": int(usage.get("prompt_cache_miss_tokens", usage.get("prompt_tokens", 0))),
                        "completion_tokens": int(usage.get("completion_tokens", 0)),
                        "reasoning_tokens": int(usage.get("completion_tokens_details", {}).get("reasoning_tokens", 0)),
                        "total_tokens": int(usage.get("total_tokens", 0)),
                    },
                    "decisions": decisions,
                }
            except urllib.error.HTTPError as error:
                detail = error.read(1024).decode("utf-8", errors="replace")
                last_error = TranslationReviewError(f"HTTP {error.code}: {detail}")
                if error.code not in {408, 409, 429, 500, 502, 503, 504}:
                    break
            except ResponseContractError:
                raise
            except (
                OSError,
                urllib.error.URLError,
                json.JSONDecodeError,
                KeyError,
                IndexError,
                TypeError,
                ValueError,
                TranslationReviewError,
            ) as error:
                last_error = error
            if attempt < attempts:
                time.sleep(self._retry_delay(attempt))
        raise TranslationReviewError(f"DeepSeek review failed after {attempts} attempts: {last_error}")


def _response_path(response_root: Path, batch: Mapping[str, Any]) -> Path:
    return response_root / f"batch-{int(batch['index']):06d}-{str(batch['identity'])[:16]}.json"


def _load_existing_response(
    path: Path, batch: Mapping[str, Any], config: Mapping[str, Any]
) -> dict[str, Any] | None:
    if not path.exists():
        return None
    value = read_json(path)
    if value.get("status") != "complete" or value.get("batch_identity") != batch["identity"]:
        raise TranslationReviewError(f"response identity drift: {path}")
    _validate_decisions(
        {"items": value.get("decisions")}, batch["items"], config
    )
    return value


def _actual_cost(responses: Sequence[Mapping[str, Any]], config: Mapping[str, Any]) -> dict[str, Any]:
    usage = Counter()
    for response in responses:
        usage.update({key: int(value) for key, value in response["usage"].items()})
    prices = config["pricing"]["per_million_tokens"]
    hit = usage["prompt_cache_hit_tokens"]
    miss = usage["prompt_cache_miss_tokens"]
    output = usage["completion_tokens"]
    cost = (
        hit / 1_000_000 * float(prices["input_cache_hit"])
        + miss / 1_000_000 * float(prices["input_cache_miss"])
        + output / 1_000_000 * float(prices["output"])
    )
    return {"usage": dict(sorted(usage.items())), "actual_cost_usd": round(cost, 6)}


def run_batches(
    batches: Sequence[Mapping[str, Any]],
    *,
    response_root: Path,
    client: DeepSeekClient,
    config: Mapping[str, Any],
    concurrency: int,
) -> list[dict[str, Any]]:
    response_root.mkdir(parents=True, exist_ok=True)
    completed: dict[str, dict[str, Any]] = {}
    pending: list[Mapping[str, Any]] = []
    for batch in batches:
        path = _response_path(response_root, batch)
        existing = _load_existing_response(path, batch, config)
        if existing is None:
            pending.append(batch)
        else:
            completed[str(batch["identity"])] = existing

    def worker(batch: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
        response = client.review_batch(batch["items"])
        record = {
            "schema_version": 1,
            "status": "complete",
            "prompt_version": PROMPT_VERSION,
            "batch_index": int(batch["index"]),
            "batch_identity": str(batch["identity"]),
            "records": len(batch["items"]),
            **response,
        }
        write_json(_response_path(response_root, batch), record)
        return str(batch["identity"]), record

    if pending:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, concurrency)) as executor:
            futures = {executor.submit(worker, batch): batch for batch in pending}
            for future in concurrent.futures.as_completed(futures):
                identity, record = future.result()
                completed[identity] = record
    return [completed[str(batch["identity"])] for batch in batches if str(batch["identity"]) in completed]


def _decision_rows(
    batches: Sequence[Mapping[str, Any]], responses: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    by_identity = {str(response["batch_identity"]): response for response in responses}
    rows: list[dict[str, Any]] = []
    for batch in batches:
        response = by_identity.get(str(batch["identity"]))
        if response is None:
            continue
        for item, decision in zip(batch["items"], response["decisions"], strict=True):
            rows.append(
                {
                    "review_id": str(item["id"]),
                    "kind": str(item["kind"]),
                    "route": str(item["route"]),
                    "source_id": str(item["source_id"]),
                    "semantic_group_id": str(item["semantic_group_id"]),
                    **decision,
                    "batch_identity": str(batch["identity"]),
                }
            )
    return rows


def _aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    verdicts = Counter(str(row["verdict"]) for row in rows)
    categories = Counter(
        str(category) for row in rows for category in row.get("categories", [])
    )
    route_verdicts = Counter(f"{row['route']}|{row['verdict']}" for row in rows)
    source_route_verdicts = Counter(
        f"{row['source_id']}|{row['route']}|{row['verdict']}" for row in rows
    )
    return {
        "verdict_counts": dict(sorted(verdicts.items())),
        "category_counts": dict(sorted(categories.items())),
        "route_verdict_counts": dict(sorted(route_verdicts.items())),
        "source_route_verdict_counts": dict(sorted(source_route_verdicts.items())),
    }


def _write_plan(
    path: Path,
    *,
    mode: str,
    batches: Sequence[Mapping[str, Any]],
    config_path: Path,
    evidence: Mapping[str, Any],
    estimate: Mapping[str, Any],
) -> None:
    write_json(
        path,
        {
            "schema_version": 1,
            "status": "frozen",
            "mode": mode,
            "prompt_version": PROMPT_VERSION,
            "config_sha256": sha256_file(config_path),
            "input_evidence": dict(evidence),
            "estimate": dict(estimate),
            "batches": [
                {
                    "index": int(batch["index"]),
                    "identity": str(batch["identity"]),
                    "records": len(batch["items"]),
                    "estimated_input_tokens": int(batch["estimated_input_tokens"]),
                    "first_id": str(batch["items"][0]["id"]),
                    "last_id": str(batch["items"][-1]["id"]),
                }
                for batch in batches
            ],
            "formal_test_accessed": False,
            "formal_devtest_accessed": False,
        },
    )


def estimate_action(
    runtime_root: Path, config: Mapping[str, Any], config_path: Path
) -> dict[str, Any]:
    items, evidence = load_full_review_items(runtime_root, config)
    batches = make_batches(items, config)
    estimate = estimate_cost(batches, config)
    output = runtime_root / PurePosixPath(config["outputs"]["root"]) / config["outputs"]["estimate"]
    report = {
        "schema_version": 1,
        "status": "estimated",
        "prompt_version": PROMPT_VERSION,
        "config_sha256": sha256_file(config_path),
        "input_evidence": evidence,
        **estimate,
        "formal_test_accessed": False,
        "formal_devtest_accessed": False,
    }
    write_json(output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def calibrate_action(
    runtime_root: Path,
    repo_root: Path,
    config: Mapping[str, Any],
    config_path: Path,
    client: DeepSeekClient,
    *,
    cost_ceiling: float,
) -> dict[str, Any]:
    items, expectations = load_calibration_items(runtime_root, config, repo_root)
    calibration_path = repo_root / PurePosixPath(config["inputs"]["calibration"])
    calibration_sha = sha256_file(calibration_path)
    batches = make_batches(items, config)
    estimate = estimate_cost(batches, config)
    if float(estimate["estimated_total_usd"]) > cost_ceiling:
        raise TranslationReviewError("calibration estimate exceeds cost ceiling")
    output_root = runtime_root / PurePosixPath(config["outputs"]["calibration_root"])
    _write_plan(
        output_root / config["outputs"]["plan"],
        mode="calibration",
        batches=batches,
        config_path=config_path,
        evidence={
            "queue_sha256": sha256_file(runtime_root / PurePosixPath(config["inputs"]["manual_review_queue"])),
            "calibration_definition_sha256": calibration_sha,
            "records": len(items),
        },
        estimate=estimate,
    )
    responses = run_batches(
        batches,
        response_root=output_root / config["outputs"]["responses"],
        client=client,
        config=config,
        concurrency=1,
    )
    rows = _decision_rows(batches, responses)
    expected_flags = [review_id for review_id, value in expectations.items() if value["expected"] == "flag"]
    expected_passes = [review_id for review_id, value in expectations.items() if value["expected"] == "pass"]
    actual = {str(row["review_id"]): row for row in rows}
    detected_flags = [review_id for review_id in expected_flags if actual.get(review_id, {}).get("verdict") in {"warning", "reject"}]
    clean_passes = [review_id for review_id in expected_passes if actual.get(review_id, {}).get("verdict") == "pass"]
    error_recall = len(detected_flags) / len(expected_flags) if expected_flags else 1.0
    clean_pass_rate = len(clean_passes) / len(expected_passes) if expected_passes else 1.0
    complete = len(rows) == len(items)
    qualified = (
        complete
        and error_recall >= float(config["review"]["calibration_minimum_error_flag_recall"])
        and clean_pass_rate >= float(config["review"]["calibration_minimum_clean_pass_rate"])
    )
    _, decisions_sha = write_jsonl(output_root / config["outputs"]["decisions"], rows)
    report = {
        "schema_version": 1,
        "status": "qualified" if qualified else "rejected",
        "task": "TD-04-independent-translation-review-calibration",
        "prompt_version": PROMPT_VERSION,
        "config_sha256": sha256_file(config_path),
        "calibration_definition_sha256": calibration_sha,
        "records": len(items),
        "decisions_sha256": decisions_sha,
        "expected_error_records": len(expected_flags),
        "detected_error_records": len(detected_flags),
        "error_flag_recall": round(error_recall, 6),
        "expected_clean_records": len(expected_passes),
        "clean_pass_records": len(clean_passes),
        "clean_pass_rate": round(clean_pass_rate, 6),
        "missed_error_ids": sorted(set(expected_flags) - set(detected_flags)),
        "false_flag_ids": sorted(set(expected_passes) - set(clean_passes)),
        "aggregate": _aggregate(rows),
        "cost": _actual_cost(responses, config),
        "formal_test_accessed": False,
        "formal_devtest_accessed": False,
    }
    write_json(output_root / config["outputs"]["report"], report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not qualified:
        raise TranslationReviewError("DeepSeek calibration did not qualify for full review")
    return report


def review_action(
    runtime_root: Path,
    config: Mapping[str, Any],
    config_path: Path,
    client: DeepSeekClient,
    *,
    cost_ceiling: float,
    maximum_batches: int | None,
    concurrency: int,
) -> dict[str, Any]:
    if config["review"]["full_review_requires_qualified_calibration"]:
        calibration_report = read_json(
            runtime_root
            / PurePosixPath(config["outputs"]["calibration_root"])
            / config["outputs"]["report"]
        )
        if calibration_report.get("status") != "qualified":
            raise TranslationReviewError("qualified calibration is required before full review")
        if calibration_report.get("prompt_version") != PROMPT_VERSION:
            raise TranslationReviewError("calibration prompt identity drift")
        if calibration_report.get("config_sha256") != sha256_file(config_path):
            raise TranslationReviewError("calibration config identity drift")
        calibration_path = (
            config_path.parent.parent
            / PurePosixPath(config["inputs"]["calibration"])
        )
        if calibration_report.get("calibration_definition_sha256") != sha256_file(calibration_path):
            raise TranslationReviewError("calibration definition identity drift")
    items, evidence = load_full_review_items(runtime_root, config)
    all_batches = make_batches(items, config)
    batches = all_batches[:maximum_batches] if maximum_batches is not None else all_batches
    estimate = estimate_cost(batches, config)
    if float(estimate["estimated_total_usd"]) > cost_ceiling:
        raise TranslationReviewError(
            f"full review estimate ${estimate['estimated_total_usd']:.6f} exceeds ceiling ${cost_ceiling:.6f}"
        )
    output_root = runtime_root / PurePosixPath(config["outputs"]["root"])
    _write_plan(
        output_root / config["outputs"]["plan"],
        mode="full" if maximum_batches is None else "bounded-pilot",
        batches=batches,
        config_path=config_path,
        evidence=evidence,
        estimate=estimate,
    )
    responses = run_batches(
        batches,
        response_root=output_root / config["outputs"]["responses"],
        client=client,
        config=config,
        concurrency=concurrency,
    )
    rows = _decision_rows(batches, responses)
    complete = maximum_batches is None and len(rows) == len(items)
    _, decisions_sha = write_jsonl(output_root / config["outputs"]["decisions"], rows)
    report = {
        "schema_version": 1,
        "status": "complete" if complete else "in_progress",
        "task": "TD-04-independent-translation-review",
        "prompt_version": PROMPT_VERSION,
        "config_sha256": sha256_file(config_path),
        "generation_identity": evidence["generation_identity"],
        "records_total": len(items),
        "records_reviewed": len(rows),
        "batches_total": len(all_batches),
        "batches_selected": len(batches),
        "batches_complete": len(responses),
        "decisions_sha256": decisions_sha,
        "aggregate": _aggregate(rows),
        "cost": _actual_cost(responses, config),
        "formal_test_accessed": False,
        "formal_devtest_accessed": False,
    }
    write_json(output_root / config["outputs"]["report"], report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("estimate", "calibrate", "review"))
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/deepseek_translation_review.yaml")
    )
    parser.add_argument("--api-key-env", default="DEEPSEEK_API_KEY")
    parser.add_argument("--auth-script", type=Path)
    parser.add_argument("--max-cost-usd", type=float)
    parser.add_argument("--max-batches", type=int)
    parser.add_argument("--concurrency", type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    config_path = args.config.resolve()
    config = load_config(config_path)
    runtime_root = args.runtime_root.resolve()
    if args.action == "estimate":
        estimate_action(runtime_root, config, config_path)
        return 0
    api_key = load_api_key(env_name=args.api_key_env, auth_script=args.auth_script)
    client = DeepSeekClient(config, api_key)
    ceiling = (
        float(args.max_cost_usd)
        if args.max_cost_usd is not None
        else float(config["pricing"]["default_cost_ceiling_usd"])
    )
    if ceiling <= 0:
        raise TranslationReviewError("cost ceiling must be positive")
    if args.action == "calibrate":
        calibrate_action(
            runtime_root,
            repo_root,
            config,
            config_path,
            client,
            cost_ceiling=ceiling,
        )
    else:
        concurrency = args.concurrency or int(config["api"]["concurrency"])
        if concurrency <= 0:
            raise TranslationReviewError("concurrency must be positive")
        if args.max_batches is not None and args.max_batches <= 0:
            raise TranslationReviewError("max batches must be positive")
        review_action(
            runtime_root,
            config,
            config_path,
            client,
            cost_ceiling=ceiling,
            maximum_batches=args.max_batches,
            concurrency=concurrency,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
